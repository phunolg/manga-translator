import os
from transformers import ConditionalDetrImageProcessor, TrOCRProcessor, ViTImageProcessor
import torch
from typing import List
from shapely.geometry import box
from utils.utils import x1y1x2y2_to_xywh
import numpy as np
import cv2
from settings import BASE_DIR
class Magiv2Processor():
    def __init__(self, config):
        self.config = config
        self.detection_image_preprocessor = None
        self.ocr_preprocessor = None
        self.crop_embedding_image_preprocessor = None
        if not config.disable_detections:
            assert config.detection_image_preprocessing_config is not None
            self.detection_image_preprocessor =  ConditionalDetrImageProcessor.from_dict(config.detection_image_preprocessing_config)
        if not config.disable_ocr:
            assert config.ocr_pretrained_processor_path is not None
            self.ocr_preprocessor = TrOCRProcessor.from_pretrained(config.ocr_pretrained_processor_path)
        if not config.disable_crop_embeddings:
            assert config.crop_embedding_image_preprocessing_config is not None
            self.crop_embedding_image_preprocessor = ViTImageProcessor.from_dict(config.crop_embedding_image_preprocessing_config)

    def preprocess_inputs_for_detection(self, images, annotations=None):
        """
        Tiền xử lý hình ảnh đầu vào cho mô hình detection.
        Args:
            images: Danh sách các hình ảnh để tiền xử lý.
            annotations: Danh sách các annotations để tiền xử lý.
        Returns:
            inputs: Đầu vào cho mô hình detection.
        """
        images = list(images)
        assert isinstance(images[0], np.ndarray)
        annotations = self._convert_annotations_to_coco_format(annotations)
        inputs = self.detection_image_preprocessor(images, annotations=annotations, return_tensors="pt")
        return inputs

    def preprocess_inputs_for_ocr(self, images):
        """
        Tiền xử lý hình ảnh đầu vào cho mô hình OCR.
        Args:
            images: Danh sách các hình ảnh để tiền xử lý.
        Returns:
            inputs: Đầu vào cho mô hình OCR.
        """
        images = list(images)
        assert isinstance(images[0], np.ndarray)
        return self.ocr_preprocessor(images, return_tensors="pt").pixel_values
    
    def preprocess_inputs_for_crop_embeddings(self, images):
        """
        Tiền xử lý hình ảnh đầu vào cho mô hình crop embedding. Chuẩn bị các vùng ảnh đã cắt (crops) để đưa vào mô hình tạo embedding, có thể được sử dụng cho các tác vụ như phân loại, so sánh, hoặc tìm kiếm tương tự.
        Args:
            images: Danh sách các hình ảnh để tiền xử lý.
        Returns:
            inputs: Đầu vào cho mô hình crop embedding.
        """
        images = list(images)
        assert isinstance(images[0], np.ndarray)
        return self.crop_embedding_image_preprocessor(images, return_tensors="pt").pixel_values
    
    def postprocess_ocr_tokens(self, generated_ids, skip_special_tokens=True):
        """
        Hậu xử lý kết quả từ mô hình OCR, chuyển đổi các ID token thành văn bản có thể đọc được
        Args:
            generated_ids: Kết quả từ mô hình OCR.
            skip_special_tokens: Bỏ qua các token đặc biệt.
        Returns:
            generated_texts: Kết quả từ mô hình OCR.
        """
        return self.ocr_preprocessor.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
    
    def crop_image(
        self, 
        image: np.ndarray, 
        bboxes: List[List[int]]
    ) -> List[np.ndarray]:
        """
        Cắt các đối tượng từ hình ảnh dựa trên các bounding box.
        Args:
            image: Hình ảnh để cắt các đối tượng.
            bboxes: Danh sách các bounding box để cắt các đối tượng.
        Returns:
            crops_for_image: Danh sách các đối tượng đã được cắt.
        """

        crops_for_image = []
        img_h, img_w = image.shape[:2]
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox

            # fix the bounding box in case it is out of bounds or too small
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # đảm bảo x1 < x2 và y1 < y2
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

            # đảm bảo bbox không vượt quá kích thước hình ảnh
            x1, y1 = max(0, x1), max(0, y1)
            x1, y1 = min(img_w, x1), min(img_h, y1)
            x2, y2 = max(0, x2), max(0, y2)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Đảm bảo bounding box có kích thước tối thiểu là 10x10 pixel.
            if x2 - x1 < 10:
                if img_w - x1 > 10:
                    x2 = x1 + 10
                else:
                    x1 = x2 - 10
            if y2 - y1 < 10:
                if img_h - y1 > 10:
                    y2 = y1 + 10
                else:
                    y1 = y2 - 10

            crop = image[y1:y2, x1:x2]
            crops_for_image.append(crop)

        return crops_for_image

    def _get_indices_of_characters_to_keep(self, batch_scores, batch_labels, batch_bboxes, character_detection_threshold):
        return self._get_indices_of_objects_to_keep(
            batch_scores, batch_labels, batch_bboxes, character_detection_threshold, target_label=0, IoU_threshold=0.5
        )

    def _get_indices_of_texts_to_keep(self, batch_scores, batch_labels, batch_bboxes, text_detection_threshold):
        return self._get_indices_of_objects_to_keep(
            batch_scores, batch_labels, batch_bboxes, text_detection_threshold, target_label=1
        )

    def _get_indices_of_panels_to_keep(self, batch_scores, batch_labels, batch_bboxes, panel_detection_threshold):
        return self._get_indices_of_objects_to_keep(
            batch_scores, batch_labels, batch_bboxes, panel_detection_threshold, target_label=2
        )

    def _get_indices_of_tails_to_keep(self, batch_scores, batch_labels, batch_bboxes, text_detection_threshold):
        return self._get_indices_of_objects_to_keep(
            batch_scores, batch_labels, batch_bboxes, text_detection_threshold, target_label=3
        )
        
    def _get_indices_of_objects_to_keep(
        self, 
        batch_scores: torch.Tensor, 
        batch_labels: torch.Tensor, 
        batch_bboxes: torch.Tensor, 
        detection_threshold: float, 
        target_label: int,
        IoU_threshold = 0.1  # IoU threshold để loại bỏ các đối tượng trùng nhau
    ) -> List[List[int]]:
        """
        Lọc các đối tượng dựa trên nhãn và ngưỡng điểm số.
        
        Args:
            batch_scores: Điểm số của các đối tượng
            batch_labels: Nhãn của các đối tượng
            batch_bboxes: Bounding box của các đối tượng
            detection_threshold: Ngưỡng để phát hiện các đối tượng
            target_label: Nhãn đối tượng cần lọc (0: character, 1: text, 2: panel, 3: tail)
        
        Returns:
            indices_of_objects_to_keep: Danh sách chỉ số của các bbox được lọc
        """
        indices_of_objects_to_keep = []
        
        for scores, labels, bboxes in zip(batch_scores, batch_labels, batch_bboxes):
            indices = torch.where((labels == target_label) & (scores > detection_threshold))[0]

            # Xử lý đặc biệt cho characters (label 0)
            if target_label == 0:
                indices_of_objects_to_keep.append(indices)
                continue
                    
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            
            if len(indices) == 0:
                indices_of_objects_to_keep.append([])
                continue
                
            scores, labels, indices, bboxes = zip(*sorted(zip(scores, labels, indices, bboxes), reverse=True))
            objects_to_keep = []
            
            # Xử lý đặc biệt cho panels (label 2)
            if target_label == 2:
                union_of_panels_so_far = box(0, 0, 0, 0)
                for ps, pb, pl, pi in zip(scores, bboxes, labels, indices):
                    minx, miny, maxx, maxy = pb
                    obj_polygon = box(minx, miny, maxx, maxy)
                    if ps < detection_threshold:
                        continue
                    if union_of_panels_so_far.intersection(obj_polygon).area / obj_polygon.area > IoU_threshold:
                        continue
                    objects_to_keep.append((ps, pl, pb, pi))
                    union_of_panels_so_far = union_of_panels_so_far.union(obj_polygon)
            else:
                # Xử lý cho text (label 1) và tails (label 3)
                objects_to_keep_as_shapely = []
                for ps, pb, pl, pi in zip(scores, bboxes, labels, indices):
                    minx, miny, maxx, maxy = pb
                    obj_polygon = box(minx, miny, maxx, maxy)
                    should_append = True
                    for t in objects_to_keep_as_shapely:
                        if t.intersection(obj_polygon).area / t.union(obj_polygon).area > IoU_threshold:
                            should_append = False
                            break
                    if should_append:
                        objects_to_keep.append((ps, pl, pb, pi))
                        objects_to_keep_as_shapely.append(obj_polygon)
                        
            indices_of_objects_to_keep.append([indice.item() for _, _, _, indice in objects_to_keep])
        
        return indices_of_objects_to_keep  # (B, N)
        
    def _convert_annotations_to_coco_format(self, annotations):
        if annotations is None:
            return None
        self._verify_annotations_are_in_correct_format(annotations)
        coco_annotations = []
        for annotation in annotations:
            coco_annotation = {
                "image_id": annotation["image_id"],
                "annotations": [],
            }
            for bbox, label in zip(annotation["bboxes_as_x1y1x2y2"], annotation["labels"]):
                xmin, ymin, xmax, ymax = bbox
                area = (xmax - xmin) * (ymax - ymin)
                coco_annotation["annotations"].append({
                    "bbox": x1y1x2y2_to_xywh(bbox),
                    "category_id": label,
                    "area": area,
                })
            coco_annotations.append(coco_annotation)
        return coco_annotations
    
    def _verify_annotations_are_in_correct_format(self, annotations):
        error_msg = """
        Annotations must be in the following format:
        [
            {
                "image_id": 0,
                "bboxes_as_x1y1x2y2": [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
                "labels": [0, 1, 2],
            },
            ...
        ]
        Labels: 0 for characters, 1 for text, 2 for panels.
        """
        if annotations is None:
            return
        if not isinstance(annotations, List) and not isinstance(annotations, tuple):
            raise ValueError(
                f"{error_msg} Expected a List/Tuple, found {type(annotations)}."
            )
        if len(annotations) == 0:
            return
        if not isinstance(annotations[0], dict):
            raise ValueError(
                f"{error_msg} Expected a List[Dicct], found {type(annotations[0])}."
            )
        if "image_id" not in annotations[0]:
            raise ValueError(
                f"{error_msg} Dict must contain 'image_id'."
            )
        if "bboxes_as_x1y1x2y2" not in annotations[0]:
            raise ValueError(
                f"{error_msg} Dict must contain 'bboxes_as_x1y1x2y2'."
            )
        if "labels" not in annotations[0]:
            raise ValueError(
                f"{error_msg} Dict must contain 'labels'."
            )
