
import os
from pprint import pformat
import time

from transformers import PreTrainedModel, VisionEncoderDecoderModel, ViTMAEModel, ConditionalDetrModel
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrMLPPredictionHead,
    ConditionalDetrModelOutput,
    inverse_sigmoid,
)
from configuration_magiv2 import Magiv2Config
from module.crop_embedding.default import CropEmbeddingService
from module.detection.microsoft_conditional_detr_resnet_50 import MicrosoftConditionalDetrResnet50Detector
from module.knowledge.type import Language
from module.ocr.Japanese_OCR.japanese_ocr import JapaneseOCR
from module.ocr.default import OCRService

from module.ocr.llm_ocr import LLM_OCR
from module.tail_classification.model import TailClassifier
from processing_magiv2 import Magiv2Processor
from torch import nn
from typing import Optional, List
import torch
from einops import rearrange, repeat
from settings import BASE_DIR, BATCH_SIZE_CROP_EMBEDDING, BATCH_SIZE_DETECTION, BATCH_SIZE_OCR, CHARACTER_CHARACTER_MATCHING_THRESHOLD, CHARACTER_DETECTION_THRESHOLD, DEBUG, TAIL_DETECTION_THRESHOLD, TEXT_CHARACTER_MATCHING_THRESHOLD, TEXT_CLASSIFICATION_THRESHOLD, TEXT_TAIL_MATCHING_THRESHOLD
from utils.text import classify_text
from utils.utils import visualise_single_image_prediction, sort_panels, sort_text_boxes_in_reading_order
from utils.common import move_to_device
from transformers.image_transforms import center_to_corners_format
from utils.utils import UnionFind, sort_panels, sort_text_boxes_in_reading_order
import pulp
import scipy
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.log import setup_logger
from utils.visualize import draw_detections
from tqdm import tqdm
logger = setup_logger(__name__)


class Magiv2Model(PreTrainedModel):
    config_class = Magiv2Config

    def __init__(self, config):
        super().__init__(config)
        print("using local Magiv2Model")
        self.config = config
        self.processor = Magiv2Processor(config)
        if not config.disable_detections:
            # xác định số lượng token đặc biệt được sử dụng cho các nhiệm vụ khác ngoài việc phát hiện đối tượng, cho phép mô hình không chỉ phát hiện các thành phần trong truyện tranh mà còn hiểu được mối quan hệ giữa chúng.
            self.num_non_obj_tokens = 5

            # Mô hình này dùng để dự đoán bounding box cho các đối tượng
            self.bbox_predictor = ConditionalDetrMLPPredictionHead(
                input_dim=config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=4, num_layers=3
            )
            #  Xác định mối quan hệ giữa các nhân vật (cùng một nhân vật xuất hiện ở nhiều nơi)
            self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim=3 * config.detection_model_config.d_model +
                (2 * config.crop_embedding_model_config.hidden_size if not config.disable_crop_embeddings else 0),
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            #  Liên kết text box với nhân vật (ai đang nói) bằng cách tính độ tương đồng giữa các token đại diện cho text và nhân vật
            self.text_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim=3 * config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            # Liên kết text box với đuôi bong bóng thoại bằng cách tính độ tương đồng giữa các token đại diện cho text và đuôi bong bóng thoại
            self.text_tail_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim=2 * config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            # Phân loại các đối tượng thành các lớp (panel, text, character, tail)
            self.class_labels_classifier = nn.Linear(
                config.detection_model_config.d_model, config.detection_model_config.num_labels
            )
            # Xác định xem một text box có phải là đoạn hội thoại hay không
            self.is_this_text_a_dialogue = nn.Linear(
                config.detection_model_config.d_model, 1
            )

            #  Sử dụng thuật toán Hungarian để ghép cặp giữa dự đoán và ground truth trong quá trình huấn luyện
            # Tính toán chi phí dựa trên lỗi phân loại, lỗi bounding box và lỗi IoU
            self.matcher = ConditionalDetrHungarianMatcher(
                class_cost=config.detection_model_config.class_cost,
                bbox_cost=config.detection_model_config.bbox_cost,
                giou_cost=config.detection_model_config.giou_cost
            )

    def move_to_device(self, input):
        return move_to_device(input, self.device)

    @torch.no_grad()
    async def do_chapter_wide_prediction(self, pages_in_order, character_bank, source_language, eta=0.75):
        logger.info("========= run do_chapter_wide_prediction ==========")
        texts = []
        characters = []
        character_clusters = []
        iterator = tqdm(range(0, len(pages_in_order), BATCH_SIZE_DETECTION))

        per_page_results = []
        start_time = time.time()
        # STEP 1: Detection and Association.
        logger.info("STEP1: start detection loop")
        for i in iterator:
            logger.info(f"STEP1: batch {i} start")
            pages = pages_in_order[i:i+BATCH_SIZE_DETECTION]
            results = self.predict_detections_and_associations(pages, 
                                                               character_character_matching_threshold=CHARACTER_CHARACTER_MATCHING_THRESHOLD,
                                                               text_character_matching_threshold=TEXT_CHARACTER_MATCHING_THRESHOLD,
                                                               character_detection_threshold=CHARACTER_DETECTION_THRESHOLD, text_classification_threshold=TEXT_CLASSIFICATION_THRESHOLD,
                                                               text_tail_matching_threshold=TEXT_TAIL_MATCHING_THRESHOLD,
                                                               tail_detection_threshold=TAIL_DETECTION_THRESHOLD)
            per_page_results.extend([result for result in results])
            logger.info(f"STEP1: batch {i} done")
        end_time_step1 = time.time()
        logger.info(
            f"Time taken for detection and association: {end_time_step1 - start_time} seconds")

        texts = [result["texts"] for result in per_page_results]
        characters = [result["characters"] for result in per_page_results]
        character_clusters = [result["character_cluster_labels"]
                              for result in per_page_results]

        # STEP 2 :  Chapter-Wide Character Naming.
        logger.info("STEP2: start character naming")
        assigned_character_names = self.assign_names_to_characters(
            pages_in_order, characters, character_bank, character_clusters, eta=eta, batch_size=BATCH_SIZE_CROP_EMBEDDING)
        end_time_step2 = time.time()
        logger.info(
            f"Time taken for chapter-wide character naming: {end_time_step2 - end_time_step1} seconds")

        # STEP 3 : Transcript Generation.
        logger.info("STEP3: start OCR")
        ocr = await self.predict_ocr(pages_in_order, texts, source_language, batch_size=BATCH_SIZE_OCR)
        end_time_step3 = time.time()
        offset_characters = 0
        iteration_over = zip(per_page_results, ocr)
        for iter in iteration_over:
            result, ocr_for_page = iter
            result["ocr"] = ocr_for_page
           
            result["character_names"] = assigned_character_names[offset_characters:
                                                                 offset_characters + len(result["characters"])]
            offset_characters += len(result["characters"])
            logger.info(
                f"Time taken for OCR: {end_time_step3 - end_time_step2} seconds")
        
        ocrs = [result["ocr"] for result in per_page_results]
        text_speech_types = [result["text_speech_type"] for result in per_page_results]
        new_text_speech_types = []
        

        for i, result in enumerate(per_page_results):
            ocr_list = result.get("ocr", [])
            labels = result.get("text_speech_type", [])
            if ocr_list and labels and len(ocr_list) == len(labels):
                result["text_speech_type"] = [classify_text(ocr, label) for ocr, label in zip(ocr_list, labels)]

        # STEP 4: Assign dialogue targets (người nghe)
        previous_speakers = {}  # Lưu trữ người nói cuối cùng trong mỗi panel
        for page_idx, result in enumerate(per_page_results):
            # Lấy dữ liệu cần thiết cho mỗi trang
            panel_bboxes = result["panels"]
            texts_bboxes = result["texts"]
            characters_bboxes = result["characters"]
            character_names = result["character_names"]
            text_character_associations = result["text_character_associations"]
            ocr_texts = result.get("ocr", None)

            # Xác định người nghe cho mỗi đoạn hội thoại
            text_to_target = self.assign_target_to_dialogue(
                panel_bboxes=panel_bboxes,
                texts=texts_bboxes,
                characters_bboxes=characters_bboxes,
                character_names=character_names,
                text_character_associations=text_character_associations,
                ocr_texts=ocr_texts,
                previous_speakers=previous_speakers
            )

            # Lưu kết quả vào result
            result["dialogue_targets"] = text_to_target
        end_time_step4 = time.time()
        logger.info(
            f"Time taken for assign dialogue targets: {end_time_step4 - end_time_step3} seconds")

        # Cập nhật previous_speakers cho trang tiếp theo
        # (previous_speakers được cập nhật bên trong hàm assign_target_to_dialogue)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return per_page_results

    def assign_target_to_dialogue(self, panel_bboxes, texts, characters_bboxes, character_names, text_character_associations, ocr_texts=None, previous_speakers=None):
        """
        Xác định người nghe (target) cho mỗi đoạn hội thoại.

        Args:
            panel_bboxes: List các bounding box của panels [x1, y1, x2, y2]
            texts: List các bounding box của text boxes [x1, y1, x2, y2]
            characters_bboxes: List các bounding box của nhân vật [x1, y1, x2, y2]
            character_names: List tên các nhân vật tương ứng với characters_bboxes
            text_character_associations: List các cặp [text_idx, character_idx] xác định ai đang nói
            ocr_texts: List các văn bản đã được OCR từ text boxes (tùy chọn)
            previous_speakers: Dictionary lưu người nói trước đó trong mỗi panel (tùy chọn)

        Returns:
            Dictionary ánh xạ từ chỉ số text box đến tên nhân vật đích (người nghe)
        """
        import numpy as np
        from scipy.spatial.distance import cdist
        logger.info("==========================================")
        # Khởi tạo kết quả
        text_to_target = {}
        logger.debug(f"Len of characters_bboxes: {len(characters_bboxes)}")

        # Xác định panel chứa mỗi text box và character
        text_in_panel = self._assign_elements_to_panels(texts, panel_bboxes)
        character_in_panel = self._assign_elements_to_panels(
            characters_bboxes, panel_bboxes)

        # Tạo dictionary ánh xạ từ text_idx đến character_idx (người nói)
        speakers = {}
        for text_idx, char_idx in text_character_associations:
            speakers[text_idx] = char_idx

        logger.debug(f"speakers: {speakers}")

        # Lưu người nói cuối cùng trong mỗi panel
        last_speaker_in_panel = {} if previous_speakers is None else previous_speakers.copy()

        # Xử lý từng text box
        for text_idx in range(len(texts)):
            # Bỏ qua nếu không phải hội thoại hoặc không biết người nói
            if text_idx not in speakers:
                logger.debug(
                    f"Không xác định được người nói của text: {text_idx}")
                text_to_target[text_idx] = None
                continue

            speaker_idx = speakers[text_idx]
            panel_idx = text_in_panel[text_idx]

            # Lấy danh sách nhân vật trong cùng panel
            characters_in_same_panel = [idx for idx, p_idx in enumerate(
                character_in_panel) if p_idx == panel_idx]
            logger.debug(
                f"characters_in_same_panel: {characters_in_same_panel}")

            # Loại bỏ người nói khỏi danh sách tiềm năng
            potential_targets = [
                idx for idx in characters_in_same_panel if idx != speaker_idx]
            logger.debug(f"potential_targets: {potential_targets}")

            # Trường hợp 1: Chỉ có 2 nhân vật trong panel (người nói và một người khác)
            if len(characters_in_same_panel) == 2 and len(potential_targets) == 1:
                text_to_target[text_idx] = character_names[potential_targets[0]]
                last_speaker_in_panel[panel_idx] = speaker_idx
                logger.debug(f"text_to_target: {text_to_target[text_idx]}")
                continue

            # Trường hợp 2: Kiểm tra xưng hô số nhiều
            if ocr_texts and text_idx < len(ocr_texts):
                text_content = ocr_texts[text_idx].lower()
                plural_pronouns = ["you guys", "các cậu", "các bạn",
                                   "everyone", "mọi người", "các người", "tất cả"]

                if any(pronoun in text_content for pronoun in plural_pronouns):
                    # Đánh dấu là nói với tất cả mọi người trong panel (trừ người nói)
                    if potential_targets:
                        target_names = [character_names[idx]
                                        for idx in potential_targets]
                        text_to_target[text_idx] = "Group: " + \
                            ", ".join(target_names)
                        last_speaker_in_panel[panel_idx] = speaker_idx
                        continue

            # Trường hợp 3: Nhiều nhân vật và chọn người gần nhất với text box
            if len(potential_targets) > 0:
                # Tính tâm của text box
                text_center = np.array([(texts[text_idx][0] + texts[text_idx][2])/2,
                                        (texts[text_idx][1] + texts[text_idx][3])/2])

                # Tính tâm của các nhân vật tiềm năng
                character_centers = []
                for idx in potential_targets:
                    char_center = np.array([(characters_bboxes[idx][0] + characters_bboxes[idx][2])/2,
                                            (characters_bboxes[idx][1] + characters_bboxes[idx][3])/2])
                    character_centers.append(char_center)

                # Tính khoảng cách từ text box đến các nhân vật
                distances = cdist([text_center], character_centers)[0]

                # Chọn nhân vật gần nhất
                nearest_char_idx = potential_targets[np.argmin(distances)]
                text_to_target[text_idx] = character_names[nearest_char_idx]
                last_speaker_in_panel[panel_idx] = speaker_idx
                continue

            # Trường hợp 4: Dựa vào người nói trước đó
            if panel_idx in last_speaker_in_panel and last_speaker_in_panel[panel_idx] != speaker_idx:
                text_to_target[text_idx] = character_names[last_speaker_in_panel[panel_idx]]
                last_speaker_in_panel[panel_idx] = speaker_idx
                continue

            # Trường hợp 5: Fallback - không xác định được người nghe
            text_to_target[text_idx] = None
            last_speaker_in_panel[panel_idx] = speaker_idx
        logger.info("==========================================")
        return text_to_target

    def _assign_elements_to_panels(self, element_bboxes, panel_bboxes):
        """
        Xác định panel chứa mỗi phần tử (text box hoặc character)

        Args:
            element_bboxes: List các bounding box của phần tử [x1, y1, x2, y2]
            panel_bboxes: List các bounding box của panels [x1, y1, x2, y2]

        Returns:
            List chỉ số panel chứa mỗi phần tử
        """
        element_in_panel = []

        for elem_bbox in element_bboxes:
            # Tính tâm của phần tử
            elem_center_x = (elem_bbox[0] + elem_bbox[2]) / 2
            elem_center_y = (elem_bbox[1] + elem_bbox[3]) / 2

            # Tìm panel chứa tâm của phần tử
            found_panel = False
            for panel_idx, panel_bbox in enumerate(panel_bboxes):
                if (panel_bbox[0] <= elem_center_x <= panel_bbox[2] and
                        panel_bbox[1] <= elem_center_y <= panel_bbox[3]):
                    element_in_panel.append(panel_idx)
                    found_panel = True
                    break

            # Nếu không tìm thấy panel nào chứa phần tử, gán cho panel gần nhất
            if not found_panel:
                min_distance = float('inf')
                nearest_panel = 0

                for panel_idx, panel_bbox in enumerate(panel_bboxes):
                    # Tính khoảng cách từ tâm phần tử đến tâm panel
                    panel_center_x = (panel_bbox[0] + panel_bbox[2]) / 2
                    panel_center_y = (panel_bbox[1] + panel_bbox[3]) / 2

                    distance = ((elem_center_x - panel_center_x) ** 2 +
                                (elem_center_y - panel_center_y) ** 2) ** 0.5

                    if distance < min_distance:
                        min_distance = distance
                        nearest_panel = panel_idx

                element_in_panel.append(nearest_panel)

        return element_in_panel

    def assign_names_to_characters(self, images, character_bboxes, character_bank, character_clusters, eta=0.75, batch_size=240):
        if len(character_bank["images"]) == 0:
            return ["Other" for bboxes_for_image in character_bboxes for bbox in bboxes_for_image]
        
        chapter_wide_char_embeddings = self.predict_crop_embeddings(
            images, character_bboxes, batch_size=batch_size)
        chapter_wide_char_embeddings = torch.cat(
            chapter_wide_char_embeddings, dim=0)
        chapter_wide_char_embeddings = torch.nn.functional.normalize(
            chapter_wide_char_embeddings, p=2, dim=1).cpu().numpy()
        
        # create must-link and cannot link constraints from character_clusters
        must_link = []
        cannot_link = []
        offset = 0
        for clusters_per_image in character_clusters:
            for i in range(len(clusters_per_image)):
                for j in range(i+1, len(clusters_per_image)):
                    if clusters_per_image[i] == clusters_per_image[j]:
                        must_link.append((offset + i, offset + j))
                    else:
                        cannot_link.append((offset + i, offset + j))
            offset += len(clusters_per_image)
            
        character_bank_for_this_chapter = self.predict_crop_embeddings(
            character_bank["images"], [[[0, 0, x.shape[1], x.shape[0]]] for x in character_bank["images"]])
        character_bank_for_this_chapter = torch.cat(
            character_bank_for_this_chapter, dim=0)
        character_bank_for_this_chapter = torch.nn.functional.normalize(
            character_bank_for_this_chapter, p=2, dim=1).cpu().numpy()
        
        # Tính khoảng cách Euclidean giữa các embedding của các nhân vật trong chapter và các nhân vật trong character_bank là ma trận [ số nhân vật trong trang truyện x số nhân vật trong character_bank ]
        costs = scipy.spatial.distance.cdist(
            chapter_wide_char_embeddings, character_bank_for_this_chapter)
        
        logger.debug(f"costs:\n{pformat(costs)}")
        
        none_of_the_above = eta * np.ones((costs.shape[0], 1))
        costs = np.concatenate([costs, none_of_the_above], axis=1)
        logger.debug(f"costs:\n{pformat(costs)}")
        
        # Bài toán tối thiểu hóa
        sense = pulp.LpMinimize
        num_supply, num_demand = costs.shape
        problem = pulp.LpProblem("Optimal_Transport_Problem", sense)
        x = pulp.LpVariable.dicts("x", ((i, j) for i in range(
            num_supply) for j in range(num_demand)), cat='Binary')
        # Objective Function to minimize
        problem += pulp.lpSum([costs[i][j] * x[(i, j)]
                              for i in range(num_supply) for j in range(num_demand)])
        # each crop must be assigned to exactly one character
        for i in range(num_supply):
            problem += pulp.lpSum([x[(i, j)] for j in range(num_demand)]
                                  ) == 1, f"Supply_{i}_Total_Assignment"
        # cannot link constraints
        for j in range(num_demand-1):
            for (s1, s2) in cannot_link:
                problem += x[(s1, j)] + x[(s2, j)
                                          ] <= 1, f"Exclusion_{s1}_{s2}_Demand_{j}"
        # must link constraints
        for j in range(num_demand):
            for (s1, s2) in must_link:
                problem += x[(s1, j)] - x[(s2, j)
                                          ] == 0, f"Inclusion_{s1}_{s2}_Demand_{j}"
        problem.solve()
        assignments = []
        for v in problem.variables():
            if v.varValue is not None and v.varValue > 0:
                index, assignment = v.name.split(
                    "(")[1].split(")")[0].split(",")
                assignment = assignment[1:]
                assignments.append((int(index), int(assignment)))

        labels = np.zeros(num_supply)
        for i, j in assignments:
            labels[i] = j

        return [character_bank["names"][int(i)] if i < len(character_bank["names"]) else "Other" for i in labels]

    def predict_detections_and_associations(
        self,
        images,
        move_to_device_fn=None,
        character_detection_threshold=0.1,
        panel_detection_threshold=0.2,
        text_detection_threshold=0.2,
        tail_detection_threshold=0.34,
        character_character_matching_threshold=0.65,
        text_character_matching_threshold=0.35,
        text_tail_matching_threshold=0.3,
        text_classification_threshold=0.05,
    ):
        """
        # Dự đoán các đối tượng và liên kết trong một batch các hình ảnh.
        Args:
            images: Danh sách các hình ảnh để dự đoán các đối tượng và liên kết.
            move_to_device_fn: Hàm để chuyển dữ liệu sang thiết bị tính toán (như GPU).
            character_detection_threshold: Ngưỡng để phát hiện các đối tượng nhân vật.
            panel_detection_threshold: Ngưỡng để phát hiện các khung hình.
            text_detection_threshold: Ngưỡng để phát hiện các đối tượng văn bản.
        Returns:
            dict gồm các key:
                - panels: bounding box của các panel
                - characters: bounding box của các nhân vật
                - texts: bounding box của các văn bản
                - tails: bounding box của các đuôi
                - character_character_affinity_matrices: ma trận khoảng cách giữa các nhân vật
                - text_character_affinity_matrices: ma trận khoảng cách giữa các văn bản và các nhân vật
                - text_tail_affinity_matrices: ma trận khoảng cách giữa các văn bản và các đuôi
                - is_this_text_a_dialogue: ma trận xác định xem các văn bản có phải là đoạn truyện hay không
                - character_cluster_labels: nhãn của các nhóm nhân vật
                - text_character_associations: liên kết giữa các văn bản và các nhân vật
                - text_tail_associations: liên kết giữa các văn bản và các đuôi
                - text_speech_type: speaking hoặc thinking
        """
        assert not self.config.disable_detections
        logger.info(f"[PREDICT] start predict_detections_and_associations, num_images={len(images)}")
        # Xác định hàm để chuyển dữ liệu sang thiết bị tính toán (như GPU)
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        # Tiền xử lý hình ảnh đầu vào cho mô hình detection
        logger.debug("[PREDICT] preprocess_inputs_for_detection...")
        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(
            images)
        inputs_to_detection_transformer = move_to_device_fn(
            inputs_to_detection_transformer)

        # detect các đối tượng (panel, text, character, tail) trong hình ảnh
        logger.debug("[PREDICT] _get_detection_transformer_output...")
        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer)

        # trích xuất điểm của mỗi class dự đoán trong 1 đối tượng và bbox của các đối tượng
        logger.debug("[PREDICT] _get_predicted_bboxes_and_classes...")
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(
            detection_transformer_output)

        # image.shape[:2] là chiều cao và chiều rộng của hình ảnh (H,W)
        original_image_sizes = torch.stack([torch.tensor(
            # (B, 2)
            img.shape[:2]) for img in images], dim=0).to(predicted_bboxes.device)

        #  Lấy điểm số cao nhất và nhãn tương ứng cho mỗi dự đoán.
        batch_scores, batch_labels = predicted_class_scores.max(-1)
        batch_scores = batch_scores.sigmoid()
        batch_labels = batch_labels.long()

        # Chuyển đổi bounding box từ định dạng center-size (x_center, y_center, width, height) sang định dạng corners (x_min, y_min, x_max, y_max)
        batch_bboxes = center_to_corners_format(predicted_bboxes)  # (B, 4)

        # scale the bboxes back to the original image size
        if isinstance(original_image_sizes, List):
            img_h = torch.Tensor([i[0] for i in original_image_sizes])
            img_w = torch.Tensor([i[1] for i in original_image_sizes])
        else:
            img_h, img_w = original_image_sizes.unbind(1)
        scale_fct = torch.stack(
            [img_w, img_h, img_w, img_h], dim=1).to(batch_bboxes.device)
        # chuyển đổi từ tọa độ chuẩn hóa [0, 1] về tọa độ pixel trong hình ảnh gốc
        batch_bboxes = batch_bboxes * scale_fct[:, None, :]

        # Lọc các đối tượng dựa trên ngưỡng điểm số (panel_detection_threshold)
        logger.debug("[PREDICT] selecting indices (panels/characters/text/tails)...")
        batch_panel_indices = self.processor._get_indices_of_panels_to_keep(
            batch_scores, batch_labels, batch_bboxes, panel_detection_threshold)
        batch_character_indices = self.processor._get_indices_of_characters_to_keep(
            batch_scores, batch_labels, batch_bboxes, character_detection_threshold)
        batch_text_indices = self.processor._get_indices_of_texts_to_keep(
            batch_scores, batch_labels, batch_bboxes, text_detection_threshold)
        batch_tail_indices = self.processor._get_indices_of_tails_to_keep(
            batch_scores, batch_labels, batch_bboxes, tail_detection_threshold)

        # Lấy token đại diện cho các đối tượng, mối quan hệ giữa text và character, mối quan hệ giữa character và character
        logger.debug("[PREDICT] _get_predicted_obj_tokens / t2c / c2c...")
        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
            detection_transformer_output)
        predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(
            detection_transformer_output)
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
            detection_transformer_output)

        # Tính toán mối quan hệ giữa các đối tượng văn bản và nhân vật
        logger.debug("[PREDICT] _get_text_character_affinity_matrices...")
        text_character_affinity_matrices = self._get_text_character_affinity_matrices(
            character_obj_tokens_for_batch=[x[i] for x, i in zip(
                predicted_obj_tokens_for_batch, batch_character_indices)],
            text_obj_tokens_for_this_batch=[x[i] for x, i in zip(
                predicted_obj_tokens_for_batch, batch_text_indices)],
            t2c_tokens_for_batch=predicted_t2c_tokens_for_batch,
            apply_sigmoid=True,
        )

        character_bboxes_in_batch = [batch_bboxes[i][j]
                                     for i, j in enumerate(batch_character_indices)]
        logger.debug("[PREDICT] _get_character_character_affinity_matrices (with crop embeddings)...")
        character_character_affinity_matrices = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=[x[i] for x, i in zip(
                predicted_obj_tokens_for_batch, batch_character_indices)],
            crop_embeddings_for_batch=self.predict_crop_embeddings(
                images, character_bboxes_in_batch, move_to_device_fn),
            c2c_tokens_for_batch=predicted_c2c_tokens_for_batch,
            apply_sigmoid=True,
        )

        logger.debug("[PREDICT] _get_text_tail_affinity_matrices...")
        text_tail_affinity_matrices = self._get_text_tail_affinity_matrices(
            text_obj_tokens_for_this_batch=[x[i] for x, i in zip(
                predicted_obj_tokens_for_batch, batch_text_indices)],
            tail_obj_tokens_for_batch=[x[i] for x, i in zip(
                predicted_obj_tokens_for_batch, batch_tail_indices)],
            apply_sigmoid=True,
        )

        logger.debug("[PREDICT] _get_text_classification...")
        is_this_text_a_dialogue = self._get_text_classification(
            [x[i] for x, i in zip(predicted_obj_tokens_for_batch, batch_text_indices)])

        results = []
        logger.debug("[PREDICT] building per-image results...")
        for batch_index in range(len(batch_scores)):
            panel_indices = batch_panel_indices[batch_index]
            character_indices = batch_character_indices[batch_index]
            text_indices = batch_text_indices[batch_index]
            tail_indices = batch_tail_indices[batch_index]

            character_bboxes = batch_bboxes[batch_index][character_indices]
            panel_bboxes = batch_bboxes[batch_index][panel_indices]
            text_bboxes = batch_bboxes[batch_index][text_indices]
            tail_bboxes = batch_bboxes[batch_index][tail_indices]

            # Nới rộng bbox cho text để bao trọn nội dung
            img_h, img_w = images[batch_index].shape[:2]
            if text_bboxes.numel() > 0:
                w = text_bboxes[:, 2] - text_bboxes[:, 0]
                h = text_bboxes[:, 3] - text_bboxes[:, 1]
                fx, fy = 0.5, 0.1  # ngang 12%, dọc 6% (tùy chỉnh theo dataset)
                text_bboxes = text_bboxes.clone()
                text_bboxes[:, 0] = (text_bboxes[:, 0] -
                                     fx*w).clamp(0, img_w-1)
                text_bboxes[:, 1] = (text_bboxes[:, 1] -
                                     fy*h).clamp(0, img_h-1)
                text_bboxes[:, 2] = (text_bboxes[:, 2] + fx*w).clamp(
                    0, img_w-1)  # Không áp dụng nớ cho bên phải
                text_bboxes[:, 3] = (text_bboxes[:, 3] +
                                     fy*h).clamp(0, img_h-1)

            local_sorted_panel_indices = sort_panels(panel_bboxes)
            panel_bboxes = panel_bboxes[local_sorted_panel_indices]
            local_sorted_text_indices = sort_text_boxes_in_reading_order(
                text_bboxes, panel_bboxes)
            text_bboxes = text_bboxes[local_sorted_text_indices]

            character_character_matching_scores = character_character_affinity_matrices[
                batch_index]
            text_character_matching_scores = text_character_affinity_matrices[
                batch_index][local_sorted_text_indices]
            text_tail_matching_scores = text_tail_affinity_matrices[
                batch_index][local_sorted_text_indices]

            is_essential_text = is_this_text_a_dialogue[batch_index][
                local_sorted_text_indices] > text_classification_threshold
            character_cluster_labels = UnionFind.from_adj_matrix(
                character_character_matching_scores > character_character_matching_threshold
            ).get_labels_for_connected_components()

            if 0 in text_character_matching_scores.shape:
                text_character_associations = torch.zeros(
                    (0, 2), dtype=torch.long)
            else:
                most_likely_speaker_for_each_text = torch.argmax(
                    text_character_matching_scores, dim=1)
                text_indices = torch.arange(len(text_bboxes)).type_as(
                    most_likely_speaker_for_each_text)
                text_character_associations = torch.stack(
                    [text_indices, most_likely_speaker_for_each_text], dim=1)
                to_keep = text_character_matching_scores.max(
                    dim=1).values > text_character_matching_threshold
                text_character_associations = text_character_associations[to_keep]

            if 0 in text_tail_matching_scores.shape:
                text_tail_associations = torch.zeros((0, 2), dtype=torch.long)
            else:
                most_likely_tail_for_each_text = torch.argmax(
                    text_tail_matching_scores, dim=1)
                text_indices = torch.arange(len(text_bboxes)).type_as(
                    most_likely_tail_for_each_text)
                text_tail_associations = torch.stack(
                    [text_indices, most_likely_tail_for_each_text], dim=1)
                to_keep = text_tail_matching_scores.max(
                    dim=1).values > text_tail_matching_threshold
                text_tail_associations = text_tail_associations[to_keep]
                
            # Gắn nhãn speaking/thinking cho mỗi text dựa trên phân loại tail
            text_speech_type = []
            for i in range(len(text_bboxes)):
                text_speech_type.append("speaking")  # Mặc định là speaking
            
            # Nếu có liên kết text-tail, sử dụng phân loại tail để xác định speech type
            if text_tail_associations.numel() > 0:
                try:
                    tail_classifier = TailClassifier(model_path=os.path.join(BASE_DIR, "module", "tail_classification", "best_tail_classifier.pth"))

                    # Lấy ảnh gốc để crop tail
                    original_image = images[batch_index]
                    
                    for text_idx, tail_idx in text_tail_associations.tolist():
                        # Crop tail từ ảnh gốc
                        tail_bbox = tail_bboxes[tail_idx].tolist()
                        x1, y1, x2, y2 = [int(coord) for coord in tail_bbox]
                        
                        # Đảm bảo coordinates hợp lệ
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(original_image.shape[1], x2)
                        y2 = min(original_image.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:  # Valid tail crop
                            tail_crop = original_image[y1:y2, x1:x2]
                            
                            # Phân loại tail
                            try:
                                tail_np = tail_crop.astype('uint8')
                                tail_class, confidence = tail_classifier.predict_single(tail_np)
                                
                                # Nếu confidence cao, cập nhật speech type
                                if confidence > 0.5:  # Threshold for confidence
                                    if tail_class == "thinking":
                                        text_speech_type[text_idx] = "thinking"
                                    else:  # speaking
                                        text_speech_type[text_idx] = "speaking"
                                        
                            except Exception as e:
                                logger.debug(f"Error classifying tail for text {text_idx}: {e}")
                                # Giữ nguyên mặc định "speaking"
                                
                except ImportError:
                    logger.warning("TailInference not available, using default speech type assignment")
                except Exception as e:
                    logger.warning(f"Error in tail classification: {e}")

            # Fallback: nếu text không gán được character nhưng có tail, gán character gần tail nhất (ưu tiên cùng panel)
            if len(text_bboxes) > 0 and len(character_bboxes) > 0 and len(tail_bboxes) > 0:
                # Chuyển các bbox sang list để dùng hàm _assign_elements_to_panels
                panel_bboxes_list = panel_bboxes.tolist()
                text_bboxes_list = text_bboxes.tolist()
                character_bboxes_list = character_bboxes.tolist()
                tail_bboxes_list = tail_bboxes.tolist()

                # Ánh xạ panel cho text/character/tail
                text_in_panel = self._assign_elements_to_panels(text_bboxes_list, panel_bboxes_list)
                character_in_panel = self._assign_elements_to_panels(character_bboxes_list, panel_bboxes_list)
                tail_in_panel = self._assign_elements_to_panels(tail_bboxes_list, panel_bboxes_list)

                existing_texts_with_speaker = set()
                if text_character_associations.numel() > 0:
                    existing_texts_with_speaker = set(text_character_associations[:, 0].tolist())

                # Tạo dict từ text -> tail_idx (chỉ 1 tail tốt nhất cho mỗi text sau lọc)
                text_to_tail = {}
                if text_tail_associations.numel() > 0:
                    for pair in text_tail_associations.tolist():
                        t_idx, tail_idx = pair
                        # Nếu có nhiều hơn 1, giữ cái đầu tiên (điểm cao nhất theo argmax trước đó)
                        if t_idx not in text_to_tail:
                            text_to_tail[t_idx] = tail_idx

                fallback_pairs = []
                for t_idx in range(len(text_bboxes_list)):
                    if t_idx in existing_texts_with_speaker:
                        continue
                    if t_idx not in text_to_tail:
                        continue

                    tail_idx = text_to_tail[t_idx]
                    # Tính tâm tail
                    tail = tail_bboxes_list[tail_idx]
                    tail_cx = (tail[0] + tail[2]) / 2.0
                    tail_cy = (tail[1] + tail[3]) / 2.0

                    # Ưu tiên character trong cùng panel với tail
                    tail_panel = tail_in_panel[tail_idx]
                    candidate_chars = [ci for ci, p in enumerate(character_in_panel) if p == tail_panel]
                    if len(candidate_chars) == 0:
                        candidate_chars = list(range(len(character_bboxes_list)))

                    # Chọn character gần tail nhất
                    best_char = None
                    best_dist = None
                    for ci in candidate_chars:
                        cb = character_bboxes_list[ci]
                        cx = (cb[0] + cb[2]) / 2.0
                        cy = (cb[1] + cb[3]) / 2.0
                        dist = (cx - tail_cx) * (cx - tail_cx) + (cy - tail_cy) * (cy - tail_cy)
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_char = ci

                    if best_char is not None:
                        fallback_pairs.append([t_idx, best_char])

                if len(fallback_pairs) > 0:
                    fallback_tensor = torch.tensor(fallback_pairs, dtype=torch.long, device=text_bboxes.device)
                    if text_character_associations.numel() == 0:
                        text_character_associations = fallback_tensor
                    else:
                        text_character_associations = torch.cat([text_character_associations, fallback_tensor], dim=0)

            result = {
                "panels": panel_bboxes.tolist(),
                "texts": text_bboxes.tolist(),
                "characters": character_bboxes.tolist(),
                "tails": tail_bboxes.tolist(),
                "text_character_associations": text_character_associations.tolist(),
                "text_tail_associations": text_tail_associations.tolist(),
                "text_speech_type": text_speech_type,
                "character_cluster_labels": character_cluster_labels,
                "is_essential_text": is_essential_text.tolist(),
            }
            results.append(result)

            if DEBUG:
                debug_path = os.path.join(BASE_DIR, "debug", f"detection_{batch_index}.png")
                draw_detections(images[batch_index], result, debug_path)

        logger.info(f"[PREDICT] done predict_detections_and_associations, num_results={len(results)}")
        return results

    def get_affinity_matrices_given_annotations(
            self, images, annotations, move_to_device_fn=None, apply_sigmoid=True
    ):
        assert not self.config.disable_detections
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        character_bboxes_in_batch = [[bbox for bbox, label in zip(
            a["bboxes_as_x1y1x2y2"], a["labels"]) if label == 0] for a in annotations]
        crop_embeddings_for_batch = self.predict_crop_embeddings(
            images, character_bboxes_in_batch, move_to_device_fn)

        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(
            images, annotations)
        inputs_to_detection_transformer = move_to_device_fn(
            inputs_to_detection_transformer)
        processed_targets = inputs_to_detection_transformer.pop("labels")

        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer)
        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
            detection_transformer_output)
        predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(
            detection_transformer_output)
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
            detection_transformer_output)

        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(
            detection_transformer_output)
        matching_dict = {
            "logits": predicted_class_scores,
            "pred_boxes": predicted_bboxes,
        }
        indices = self.matcher(matching_dict, processed_targets)

        matched_char_obj_tokens_for_batch = []
        matched_text_obj_tokens_for_batch = []
        matched_tail_obj_tokens_for_batch = []
        t2c_tokens_for_batch = []
        c2c_tokens_for_batch = []

        for j, (pred_idx, tgt_idx) in enumerate(indices):
            target_idx_to_pred_idx = {tgt.item(): pred.item()
                                      for pred, tgt in zip(pred_idx, tgt_idx)}
            targets_for_this_image = processed_targets[j]
            indices_of_text_boxes_in_annotation = [i for i, label in enumerate(
                targets_for_this_image["class_labels"]) if label == 1]
            indices_of_char_boxes_in_annotation = [i for i, label in enumerate(
                targets_for_this_image["class_labels"]) if label == 0]
            indices_of_tail_boxes_in_annotation = [i for i, label in enumerate(
                targets_for_this_image["class_labels"]) if label == 3]
            predicted_text_indices = [target_idx_to_pred_idx[i]
                                      for i in indices_of_text_boxes_in_annotation]
            predicted_char_indices = [target_idx_to_pred_idx[i]
                                      for i in indices_of_char_boxes_in_annotation]
            predicted_tail_indices = [target_idx_to_pred_idx[i]
                                      for i in indices_of_tail_boxes_in_annotation]
            matched_char_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_char_indices])
            matched_text_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_text_indices])
            matched_tail_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_tail_indices])
            t2c_tokens_for_batch.append(predicted_t2c_tokens_for_batch[j])
            c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])

        text_character_affinity_matrices = self._get_text_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            text_obj_tokens_for_this_batch=matched_text_obj_tokens_for_batch,
            t2c_tokens_for_batch=t2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        character_character_affinity_matrices = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            crop_embeddings_for_batch=crop_embeddings_for_batch,
            c2c_tokens_for_batch=c2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        character_character_affinity_matrices_crop_only = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            crop_embeddings_for_batch=crop_embeddings_for_batch,
            c2c_tokens_for_batch=c2c_tokens_for_batch,
            crop_only=True,
            apply_sigmoid=apply_sigmoid,
        )

        text_tail_affinity_matrices = self._get_text_tail_affinity_matrices(
            text_obj_tokens_for_this_batch=matched_text_obj_tokens_for_batch,
            tail_obj_tokens_for_batch=matched_tail_obj_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        is_this_text_a_dialogue = self._get_text_classification(
            matched_text_obj_tokens_for_batch, apply_sigmoid=apply_sigmoid)

        return {
            "text_character_affinity_matrices": text_character_affinity_matrices,
            "character_character_affinity_matrices": character_character_affinity_matrices,
            "character_character_affinity_matrices_crop_only": character_character_affinity_matrices_crop_only,
            "text_tail_affinity_matrices": text_tail_affinity_matrices,
            "is_this_text_a_dialogue": is_this_text_a_dialogue,
        }

    def predict_crop_embeddings(self, images, crop_bboxes, move_to_device_fn=None, mask_ratio=0.0, batch_size=10):
        if self.config.disable_crop_embeddings:
            return None

        assert isinstance(
            crop_bboxes, List), "please provide a list of bboxes for each image to get embeddings for"

        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        weight_path = os.path.join(
            BASE_DIR, "models", "crop_embedding.safetensors")
        crop_embedding_model = CropEmbeddingService.from_safetensors(
            self.config.crop_embedding_model_config, self.processor, weight_path)

        crop_embeddings_for_batch = crop_embedding_model.predict(
            images, crop_bboxes, move_to_device_fn=move_to_device_fn, mask_ratio=mask_ratio, batch_size=batch_size)

        crop_embedding_model.unload()

        return crop_embeddings_for_batch

    async def predict_ocr(self, images, crop_bboxes, source_language: Language, move_to_device_fn=None, batch_size=32, max_new_tokens=64):
        match source_language:
            case Language.JAPANESE:
                ocr_model = JapaneseOCR(self.processor)
            case Language.ENGLISH:
                weight_path = os.path.join(BASE_DIR, "models", "ocr.safetensors")
                ocr_model = OCRService.from_safetensors(
                    self.processor, self.config.ocr_model_config, weight_path)
            case _:
                ocr_model = LLM_OCR(self.processor)
        
        texts = await ocr_model.predict(images, crop_bboxes, move_to_device_fn=move_to_device_fn, batch_size=batch_size, max_new_tokens=max_new_tokens)

        ocr_model.unload()
        return texts

    def visualise_single_image_prediction(
            self, image_as_np_array, predictions, filename=None
    ):
        return visualise_single_image_prediction(image_as_np_array, predictions, filename)

    @torch.no_grad()
    def _get_detection_transformer_output(
            self,
            pixel_values: torch.FloatTensor,
            pixel_mask: Optional[torch.LongTensor] = None
    ):
        """
        Mô hình này phát hiện các đối tượng (panel, text, character, tail) trong hình ảnh
        """
        if self.config.disable_detections:
            raise ValueError(
                "Detection model is disabled. Set disable_detections=False in the config.")
        weight_path = os.path.join(
            BASE_DIR, "models", "detection_transformer.safetensors")
        with MicrosoftConditionalDetrResnet50Detector.from_safetensors(self.processor, self.config.detection_model_config, weight_path) as detection_transformer:
            output = detection_transformer.predict(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask
            )
            detection_transformer.unload()
        return output

    def _get_predicted_obj_tokens(
            self,
            detection_transformer_output: ConditionalDetrModelOutput
    ):
        # Token đại diện cho các đối tượng
        return detection_transformer_output.last_hidden_state[:, :-self.num_non_obj_tokens]

    def _get_predicted_c2c_tokens(
            self,
            detection_transformer_output: ConditionalDetrModelOutput
    ):
        # Token đại diện cho mối quan hệ giữa các nhân vật
        return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens]

    def _get_predicted_t2c_tokens(
            self,
            detection_transformer_output: ConditionalDetrModelOutput
    ):
        # Token đại diện cho mối quan hệ giữa các nhân vật và văn bản
        return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens+1]

    def _get_predicted_bboxes_and_classes(
            self,
            detection_transformer_output: ConditionalDetrModelOutput,
    ):
        if self.config.disable_detections:
            raise ValueError(
                "Detection model is disabled. Set disable_detections=False in the config.")

        obj = self._get_predicted_obj_tokens(detection_transformer_output)
        print(f"_get_predicted_obj_tokens: {obj.shape}")

        predicted_class_scores = self.class_labels_classifier(obj)
        print(f"predicted_class_scores: {predicted_class_scores.shape}")
        reference = detection_transformer_output.reference_points[:-
                                                                  self.num_non_obj_tokens]
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)
        predicted_boxes = self.bbox_predictor(obj)
        predicted_boxes[..., :2] += reference_before_sigmoid
        predicted_boxes = predicted_boxes.sigmoid()
        print(f"predicted_boxes: {predicted_boxes.shape}")

        return predicted_class_scores, predicted_boxes

    def _get_text_classification(
            self,
            text_obj_tokens_for_batch: List[torch.FloatTensor],
            apply_sigmoid=True,
    ):
        assert not self.config.disable_detections
        is_this_text_a_dialogue = []
        for text_obj_tokens in text_obj_tokens_for_batch:       
            if text_obj_tokens.shape[0] == 0:
                is_this_text_a_dialogue.append(
                    torch.tensor([], dtype=torch.bool))
                continue
            classification = self.is_this_text_a_dialogue(
                text_obj_tokens).squeeze(-1)
            if apply_sigmoid:
                classification = classification.sigmoid()
            is_this_text_a_dialogue.append(classification)
        return is_this_text_a_dialogue

    def _get_character_character_affinity_matrices(
            self,
            character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
            crop_embeddings_for_batch: List[torch.FloatTensor] = None,
            c2c_tokens_for_batch: List[torch.FloatTensor] = None,
            crop_only=False,
            apply_sigmoid=True,
    ):
        assert self.config.disable_detections or (
            character_obj_tokens_for_batch is not None and c2c_tokens_for_batch is not None)
        assert self.config.disable_crop_embeddings or crop_embeddings_for_batch is not None
        assert not self.config.disable_detections or not self.config.disable_crop_embeddings

        if crop_only:
            affinity_matrices = []
            for crop_embeddings in crop_embeddings_for_batch:
                crop_embeddings = crop_embeddings / \
                    crop_embeddings.norm(dim=-1, keepdim=True)
                affinity_matrix = crop_embeddings @ crop_embeddings.T
                affinity_matrices.append(affinity_matrix)
            return affinity_matrices
        affinity_matrices = []
        for batch_index, (character_obj_tokens, c2c) in enumerate(zip(character_obj_tokens_for_batch, c2c_tokens_for_batch)):
            if character_obj_tokens.shape[0] == 0:
                affinity_matrices.append(torch.zeros(
                    0, 0).type_as(character_obj_tokens))
                continue
            if not self.config.disable_crop_embeddings:
                crop_embeddings = crop_embeddings_for_batch[batch_index]
                assert character_obj_tokens.shape[0] == crop_embeddings.shape[0]
                character_obj_tokens = torch.cat(
                    [character_obj_tokens, crop_embeddings], dim=-1)
            char_i = repeat(character_obj_tokens, "i d -> i repeat d",
                            repeat=character_obj_tokens.shape[0])
            char_j = repeat(character_obj_tokens, "j d -> repeat j d",
                            repeat=character_obj_tokens.shape[0])
            char_ij = rearrange([char_i, char_j], "two i j d -> (i j) (two d)")
            c2c = repeat(c2c, "d -> repeat d", repeat=char_ij.shape[0])
            char_ij_c2c = torch.cat([char_ij, c2c], dim=-1)
            character_character_affinities = self.character_character_matching_head(
                char_ij_c2c)
            character_character_affinities = rearrange(
                character_character_affinities, "(i j) 1 -> i j", i=char_i.shape[0])
            character_character_affinities = (
                character_character_affinities + character_character_affinities.T) / 2
            if apply_sigmoid:
                character_character_affinities = character_character_affinities.sigmoid()
            affinity_matrices.append(character_character_affinities)
        return affinity_matrices

    def _get_text_character_affinity_matrices(
            self,
            character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
            text_obj_tokens_for_this_batch: List[torch.FloatTensor] = None,
            t2c_tokens_for_batch: List[torch.FloatTensor] = None,
            apply_sigmoid=True,
    ):
        """
        tính toán ma trận độ tương đồng (affinity) giữa các text box và nhân vật, giúp xác định mối quan hệ "ai đang nói gì" trong truyện tranh.
        Args:
            character_obj_tokens_for_batch: Token đại diện cho các đối tượng nhân vật.
            text_obj_tokens_for_this_batch: Token đại diện cho các đối tượng văn bản.
            t2c_tokens_for_batch: Token đại diện cho mối quan hệ giữa các đối tượng văn bản và nhân vật.
            apply_sigmoid: Áp dụng hàm sigmoid để chuyển đổi kết quả thành độ tương đồng.
        Returns:
            affinity_matrices: Ma trận độ tương đồng giữa các đối tượng văn bản và nhân vật.
        """
        assert not self.config.disable_detections
        assert character_obj_tokens_for_batch is not None and text_obj_tokens_for_this_batch is not None and t2c_tokens_for_batch is not None
        affinity_matrices = []
        for (
            character_obj_tokens,  # [num_texts, embedding_dim]
            text_obj_tokens,  # [num_characters, embedding_dim]
            t2c  # [num_characters, embedding_dim]
        ) in zip(
                character_obj_tokens_for_batch,
                text_obj_tokens_for_this_batch,
                t2c_tokens_for_batch):
            # Xử lý trường hợp đặc biệt khi không có nhân vật hoặc text box thì bỏ qua
            if character_obj_tokens.shape[0] == 0 or text_obj_tokens.shape[0] == 0:
                affinity_matrices.append(torch.zeros(
                    text_obj_tokens.shape[0], character_obj_tokens.shape[0]).type_as(character_obj_tokens))
                continue
            # [num_texts, num_characters, embedding_dim]
            text_i = repeat(text_obj_tokens, "i d -> i repeat d",
                            repeat=character_obj_tokens.shape[0])
            # [num_characters, num_texts, embedding_dim]
            char_j = repeat(character_obj_tokens, "j d -> repeat j d",
                            repeat=text_obj_tokens.shape[0])
            # [num_texts * num_characters, embedding_dim]
            text_char = rearrange(
                [text_i, char_j], "two i j d -> (i j) (two d)")
            # [num_texts * num_characters, embedding_dim]
            t2c = repeat(t2c, "d -> repeat d", repeat=text_char.shape[0])
            text_char_t2c = torch.cat([text_char, t2c], dim=-1)
            text_character_affinities = self.text_character_matching_head(
                text_char_t2c)
            text_character_affinities = rearrange(
                text_character_affinities, "(i j) 1 -> i j", i=text_i.shape[0])
            if apply_sigmoid:
                text_character_affinities = text_character_affinities.sigmoid()
            affinity_matrices.append(text_character_affinities)
        return affinity_matrices

    def _get_text_tail_affinity_matrices(
            self,
            text_obj_tokens_for_this_batch: List[torch.FloatTensor] = None,
            tail_obj_tokens_for_batch: List[torch.FloatTensor] = None,
            apply_sigmoid=True,
    ):
        assert not self.config.disable_detections
        assert tail_obj_tokens_for_batch is not None and text_obj_tokens_for_this_batch is not None
        affinity_matrices = []
        for tail_obj_tokens, text_obj_tokens in zip(tail_obj_tokens_for_batch, text_obj_tokens_for_this_batch):
            if tail_obj_tokens.shape[0] == 0 or text_obj_tokens.shape[0] == 0:
                affinity_matrices.append(torch.zeros(
                    text_obj_tokens.shape[0], tail_obj_tokens.shape[0]).type_as(tail_obj_tokens))
                continue
            text_i = repeat(text_obj_tokens, "i d -> i repeat d",
                            repeat=tail_obj_tokens.shape[0])
            tail_j = repeat(tail_obj_tokens, "j d -> repeat j d",
                            repeat=text_obj_tokens.shape[0])
            text_tail = rearrange(
                [text_i, tail_j], "two i j d -> (i j) (two d)")
            text_tail_affinities = self.text_tail_matching_head(text_tail)
            text_tail_affinities = rearrange(
                text_tail_affinities, "(i j) 1 -> i j", i=text_i.shape[0])
            if apply_sigmoid:
                text_tail_affinities = text_tail_affinities.sigmoid()
            affinity_matrices.append(text_tail_affinities)
        return affinity_matrices

# Copied from transformers.models.detr.modeling_detr._upcast


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(
            f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(
            f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher with DeformableDetr->ConditionalDetr
class ConditionalDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(
            0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * \
            ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - \
            neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(
            out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + \
            self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(
            cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
