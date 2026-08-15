from pprint import pformat
import cv2
import copy
from typing import List, Tuple
import langdetect
from langdetect.detector_factory import os
from shapely import Polygon
from module.utils.textblock import TextBlock
from settings import BASE_DIR, DEBUG
from utils.geometry import create_oval_mask
import numpy as np
import re
from typing import List
from pyvi import ViTokenizer
from utils.log import setup_logger

logger = setup_logger(__name__)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', "\""}


def seg_vietnamese(text: str) -> List[str]:
    """
    Phân đoạn văn bản tiếng Việt thành các từ sử dụng underthesea

    Args:
        text: Văn bản tiếng Việt cần phân đoạn

    Returns:
        Danh sách các từ đã được phân đoạn
    """
    # Xử lý các ký tự đặc biệt và dấu câu
    tokenzized_text = ViTokenizer.tokenize(text)
    tokens = tokenzized_text.split()
    result = []
    # Danh sách các dấu câu phổ biến
    punctuation = {'.', ',', '?', '!', ':', ';', ')',
                   '(', '}', '{', '"', "'", '…', '-', '–', '—', '“', '”', '’', '‘', '[', ']', '...', '…', '/', '\\'}

    for token in tokens:
        if '_' in token:
            word = token.replace('_', ' ')
        else:
            word = token

        # Nếu là dấu câu và đã có phần tử trước đó, ghép vào phần tử trước
        if word in punctuation and len(result) > 0:
            result[-1] = result[-1] + word
        else:
            result.append(word.upper())

    return result


def seg_eng(text: str) -> List[str]:
    """
    Extracts every word from text parameter
    """
    # TODO: replace with regexes

    text = text.strip().upper().replace(
        '  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = ''

    # dumb way to ensure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
        if c in PUNSET_RIGHT_ENG and ii < text_len - 1:
            next_c = text[ii + 1]
            if next_c.isalpha() or next_c.isnumeric():
                processed_text += c + ' '
            else:
                processed_text += c
        else:
            processed_text += c

    word_list = processed_text.split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
    for ii, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue
        if len(word) < 3:
            append_left, append_right = False, False
            len_word, len_next, len_prev = len(word), -1, -1
            if ii < word_num - 1:
                len_next = len(word_list[ii + 1])
            if ii > 0:
                len_prev = len(words[-1])
            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1
            if len_next > 0 and len_prev > 0:
                if len_next < len_prev:
                    append_right = cond_next
                else:
                    append_left = cond_prev
            elif len_next > 0:
                append_right = cond_next
            elif len_prev:
                append_left = cond_prev

            if append_left:
                words[-1] = words[-1] + ' ' + word
            elif append_right:
                words.append(word + ' ' + word_list[ii + 1])
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)
    return words


def visualize_text_blocks(text_blocks: List[TextBlock], img: np.ndarray, name: str):
    img = img.copy()
    for idx, bbox in enumerate(text_blocks):
        for line in bbox.lines:
            pts = np.array(line, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(
                0, 0, 255), thickness=2)
            # Vẽ index lên bbox
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(img, str(idx), tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Lưu ảnh debug
    debug_dir = os.path.join(BASE_DIR, "debug", "debug_text_bboxes")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, name)
    cv2.imwrite(debug_path, img)
    logger.debug(
        f"Đã lưu ảnh debug text_bboxes tại {debug_path}")


def assign_translations_to_regions(
    text_bboxes: List[TextBlock],
    text_regions: List[TextBlock],
    min_overlap_ratio: float = 0.9,
    lang: str = "vi",
) -> List[TextBlock]:
    """
    Gán chuỗi translation từ các bbox OCR vào những vùng chèn (text_regions)
    nằm bên trong mỗi bbox. Chuỗi được chia theo tỉ lệ diện tích vùng và thứ tự đọc.

    - text_bboxes: mỗi phần tử có translation và một polygon tại lines[0]
    - text_regions: các vùng đích (không có translation ban đầu)
    - min_overlap_ratio: ngưỡng tỉ lệ chồng lấp tối thiểu (diện tích giao/diện tích vùng) để xem là "nằm trong"
    """
    region_to_bbox_mapping = {}  # mapping từ region_idx đến bbox_idx
    # Duyệt từng bbox có translation
    for index, bbox in enumerate(text_bboxes):
        try:
            bbox_poly = Polygon(bbox.lines[0])
        except Exception:
            continue

        # Tìm các vùng nằm trong bbox theo tỉ lệ chồng lấp
        candidate_regions = []
        for region in text_regions:
            try:
                region_poly = Polygon(region.lines[0])
            except Exception:
                continue

            inter_area = bbox_poly.intersection(region_poly).area
            region_area = max(region_poly.area, 1e-6)
            overlap_ratio = inter_area / region_area
            if overlap_ratio >= min_overlap_ratio:
                candidate_regions.append((region, region_poly.area))

        if not candidate_regions:
            continue

        # Sắp xếp vùng theo thứ tự đọc cơ bản
        regions_sorted: List[TextBlock] = [r for r, _ in sorted(
            candidate_regions, key=lambda ra: _reading_order_key(ra[0]))]
        # Lấy font size trung bình của các vùng
        avg_font_size = int(
            sum([r.font_size for r in regions_sorted]) / len(regions_sorted))

        # Chuẩn bị token để chia đoạn văn
        full_text = str(bbox.translation).strip()
        if not full_text:
            continue

        tokens = full_text.split()

        n_tokens = len(tokens)
        # Trọng số theo diện tích vùng với mũ > 1 để "phạt thêm chữ" cho vùng lớn
        area_by_region = {r: a for r, a in candidate_regions}
        # >1 sẽ ưu tiên vùng có diện tích lớn hơn => vùng càng lớn thì càng bị phạt thêm từ
        bonus_exponent = 2.3
        weights = {r: (max(a, 1e-6) ** bonus_exponent)
                   for r, a in area_by_region.items()}
        total_weight = sum(weights.values()) or 1.0

        # Phân bổ theo phương pháp "largest remainder" với trọng số đã phạt vùng lớn
        desired = [weights[r] / total_weight *
                   n_tokens for r in regions_sorted]
        base_counts = [int(x) for x in desired]
        remainders = [x - int(x) for x in desired]
        allocated = sum(base_counts)
        remaining = max(0, n_tokens - allocated)

        # Thêm mapping
        for region in regions_sorted:
            region_idx = text_regions.index(region)
            region_to_bbox_mapping[region_idx] = index

        # Cộng 1 lần lượt cho các remainder lớn nhất
        for idx in np.argsort(remainders)[::-1][:remaining]:
            base_counts[idx] += 1

        # Đảm bảo tối thiểu 1 từ mỗi vùng nếu đủ số từ tổng
        if n_tokens >= len(regions_sorted):
            zeros = [i for i, c in enumerate(base_counts) if c == 0]
            for zi in zeros:
                # tìm vùng đang có nhiều từ nhất để nhường bớt (ưu tiên vùng nhỏ hơn bằng cách chọn idx có base_counts lớn nhất)
                donors = [(i, c) for i, c in enumerate(
                    base_counts) if c > 1 and i != zi]
                if not donors:
                    break
                # chọn donor có c lớn nhất; nếu bằng nhau, chọn vùng có trọng số nhỏ hơn
                donors.sort(key=lambda ic: (
                    ic[1], -(weights[regions_sorted[ic[0]]])), reverse=True)
                di, _ = donors[0]
                base_counts[di] -= 1
                base_counts[zi] += 1

        # Đảm bảo tổng đúng n_tokens
        if sum(base_counts) != n_tokens:
            diff = n_tokens - sum(base_counts)
            if base_counts:
                base_counts[-1] += diff

        # Gán chuỗi cho từng vùng theo thứ tự đọc
        start = 0

        for region, count in zip(regions_sorted, base_counts):
            end = start + max(0, count)
            slice_tokens = tokens[start:end]
            region.font_size = avg_font_size
            region_text = (' '.join(slice_tokens)).strip()
            region.translation = region_text
            region.target_lang = lang
            start = end
            print(
                f"region {region} in bbox {index} has translation {region.translation}")

    return text_regions, region_to_bbox_mapping


def _reading_order_key(region: TextBlock):
    """Khóa sắp xếp vùng theo thứ tự đọc cơ bản (trên→dưới, trái→phải)."""
    poly = Polygon(region.lines[0])
    c = poly.centroid
    return (c.y, c.x)


def merge_regions_into_oval(
    text_regions: List[TextBlock],
    text_bboxes: List[TextBlock],
    region_to_bbox_mapping: dict,
    padding: int = 0,
    original_img: np.ndarray = None
):
    """
    Gộp các text_region trong cùng một text_box thành một hình oval

    Args:
        text_regions: Danh sách các vùng văn bản
        text_bboxes: Danh sách các bounding box của văn bản
        region_to_bbox_mapping: Mapping từ region_idx đến bbox_idx
        padding: Khoảng cách lề cho hình oval

    Returns:
        merged_regions: Danh sách các vùng văn bản đã gộp
        oval_masks: Danh sách các mask hình oval tương ứng
    """
    # Tạo dictionary để lưu các region thuộc cùng một bbox
    bbox_to_regions = {}
    for region_idx, bbox_idx in region_to_bbox_mapping.items():
        if bbox_idx not in bbox_to_regions:
            bbox_to_regions[bbox_idx] = []
        bbox_to_regions[bbox_idx].append(region_idx)

    # Danh sách kết quả
    merged_regions = []
    oval_masks = []

    # Tạo bản sao ảnh để vẽ oval (nếu có)
    img_with_ovals = None
    if original_img is not None:
        img_with_ovals = original_img.copy()
        if img_with_ovals.ndim == 2:
            img_with_ovals = cv2.cvtColor(img_with_ovals, cv2.COLOR_GRAY2BGR)
        elif img_with_ovals.shape[2] == 1:
            img_with_ovals = cv2.cvtColor(img_with_ovals, cv2.COLOR_GRAY2BGR)

    # Xử lý từng bbox
    for bbox_idx, region_indices in bbox_to_regions.items():
        if not region_indices:
            continue

        # Lấy bbox hiện tại
        bbox = text_bboxes[bbox_idx]

        # Lấy tất cả các region trong bbox này
        regions = [text_regions[idx] for idx in region_indices]

        # Tính toán bounding box chứa tất cả các region
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for region in regions:
            try:
                x, y, w, h = region.xywh
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
            except Exception:
                continue

        if min_x == float('inf'):
            continue

        # Thêm padding cho hình oval
        min_x = max(0, int(min_x - padding))
        min_y = max(0, int(min_y - padding))
        max_x = min(original_img.shape[1], int(max_x + padding))
        max_y = min(original_img.shape[0], int(max_y + padding))

        # Tạo một region mới từ việc gộp các region
        merged_region = copy.deepcopy(regions[0])

        # Gộp văn bản từ tất cả các region
        translations = [r.translation for r in regions if r.translation]
        merged_region.translation = " ".join(translations)

        # Cập nhật vị trí và kích thước
        merged_region.xywh = np.array(
            [min_x, min_y, max_x - min_x, max_y - min_y])

        # Tính toán font size trung bình
        merged_region.font_size = int(
            sum(r.font_size for r in regions) / len(regions))

        # Tạo mask hình oval
        mask_h = int(max_y - min_y)
        mask_w = int(max_x - min_x)

        # Tạo mask hình oval
        oval_mask = create_oval_mask(mask_w, mask_h, padding=padding)

        # Vẽ oval lên ảnh kiểm tra
        if img_with_ovals is not None:
            center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
            axes = (int((max_x - min_x) / 2), int((max_y - min_y) / 2))
            color = (0, 0, 255)  # Đỏ
            cv2.ellipse(img_with_ovals, center, axes, 0, 0, 360, color, 2)

        # Thêm vào kết quả
        merged_regions.append(merged_region)
        oval_masks.append((oval_mask, [min_x, min_y, max_x, max_y]))

    # Thêm các region không thuộc mapping
    for i, region in enumerate(text_regions):
        if i not in [idx for indices in bbox_to_regions.values() for idx in indices]:
            merged_regions.append(region)

            # Tạo mask hình oval cho region này
            x, y, w, h = region.xywh
            # Thêm padding
            x = max(0, int(x - padding))
            y = max(0, int(y - padding))
            w = min(original_img.shape[1] - x, int(w + 2 * padding))
            h = min(original_img.shape[0] - y, int(h + 2 * padding))

            mask = create_oval_mask(int(w), int(h), padding=padding//2)
            oval_masks.append((mask, [x, y, x + w, y + h]))

            # Vẽ oval lên ảnh kiểm tra
            if img_with_ovals is not None:
                center = (int(x + w / 2), int(y + h / 2))
                axes = (int(w / 2), int(h / 2))
                color = (0, 255, 0)  # Xanh lá
                cv2.ellipse(img_with_ovals, center, axes, 0, 0, 360, color, 2)

    # Nếu có ảnh kiểm tra, lưu ra file để kiểm tra
    if img_with_ovals is not None:
        cv2.imwrite(
            os.path.join(BASE_DIR, "debug", "debug_ovals.png"), img_with_ovals)
        print("Đã lưu ảnh debug_ovals.png với các oval đã vẽ.")

    return oval_masks


def merge_regions_into_polygon(
    text_regions: List[TextBlock],
    text_bboxes: List[TextBlock],
    region_to_bbox_mapping: dict,
    padding: int = 5,
    original_img: np.ndarray = None
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Gộp tất cả `lines` của các `text_regions` thuộc cùng một `bbox_idx`
    thành MỘT khối duy nhất. Trả về danh sách mask đã crop và bbox tương ứng.

    Args:
        text_regions: Danh sách các vùng văn bản (từng dòng văn bản)
        text_bboxes: Danh sách các bounding box của cả văn bản (không dùng để gán dịch ở đây)
        region_to_bbox_mapping: Mapping từ region_idx đến bbox_idx
        padding: Khoảng cách lề (dilation trên mask)
        original_img: Ảnh gốc (dùng lấy kích thước H, W)

    Returns:
        polygon_masks: List các tuple (mask_crop, (x1, y1, x2, y2))
    """
    H, W = original_img.shape[:2]

    # Gom các region theo bbox như merge_regions_into_oval
    bbox_to_regions = {}
    # {bbox_idx: [region_idx, region_idx, ...]}
    for region_idx, bbox_idx in region_to_bbox_mapping.items():
        logger.debug(
            f"region_idx: {region_idx} have {len(text_regions[region_idx].lines)} lines")
        if bbox_idx not in bbox_to_regions:
            bbox_to_regions[bbox_idx] = []
        bbox_to_regions[bbox_idx].append(region_idx)
    polygon_masks = []

    for bbox_idx, region_indices in bbox_to_regions.items():
        if not region_indices:
            continue

        # Lấy các region thuộc cùng bbox
        regions = [text_regions[idx] for idx in region_indices]

        # Tạo mask gộp theo line của từng region
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for r in regions:
            for poly in r.lines:
                try:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(union_mask, [pts], 255)
                except Exception:
                    # Bỏ qua polygon lỗi
                    pass

        # Nối các cụm mask rời trong cùng bbox_idx bằng "cầu" ngắn nhất
        # Ý tưởng: tìm các contour ngoài; nếu >1, lặp: nối hai cụm gần nhất bằng cv2.line

        def _find_contours(img):
            cnts, _ = cv2.findContours(
                img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # bỏ các contour quá nhỏ
            return [c for c in cnts if c is not None and len(c) > 0]

        contours = _find_contours(union_mask)
        # Loop an toàn (giới hạn số lần) để tránh vô hạn khi khoảng cách cực lớn
        safe_guard = 0
        while len(contours) > 1 and safe_guard < 64:
            safe_guard += 1
            # Chọn hai cụm gần nhất theo tâm để giảm chi phí, rồi refine bằng điểm gần nhất
            centers = [c.reshape(-1, 2).mean(axis=0) for c in contours]
            centers = [np.array(c, dtype=np.float32) for c in centers]
            min_pair = (0, 1)
            min_dc = float('inf')
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dc = float(np.sum((centers[i] - centers[j]) ** 2))
                    if dc < min_dc:
                        min_dc = dc
                        min_pair = (i, j)
            i, j = min_pair
            # Lấy bounding boxes của hai cụm
            xA, yA, wA, hA = cv2.boundingRect(contours[i])
            xB, yB, wB, hB = cv2.boundingRect(contours[j])

            # Xác định cụm trên/dưới theo toạ độ y
            if yA <= yB:
                up_x, up_y, up_w, up_h = xA, yA, wA, hA
                dn_x, dn_y, dn_w, dn_h = xB, yB, wB, hB
            else:
                up_x, up_y, up_w, up_h = xB, yB, wB, hB
                dn_x, dn_y, dn_w, dn_h = xA, yA, wA, hA

            # Chiều ngang cầu nối = min(độ dài 2 line) ~ dùng bwidth của bbox
            rect_w = int(max(1, 0.8 * min(up_w, dn_w)))
            # Chiều cao = khoảng cách giữa 2 line
            gap = int(max(1, dn_y - (up_y + up_h)))

            # Tâm x: trung bình tâm hai bbox
            cx_up = up_x + up_w / 2.0
            cx_dn = dn_x + dn_w / 2.0
            cx = (cx_up + cx_dn) / 2.0
            rect_x1 = int(round(cx - rect_w / 2.0))
            rect_y1 = int(round(up_y + up_h))

            # Clip biên
            rect_x1 = max(0, min(W - 1, rect_x1))
            if rect_x1 + rect_w > W:
                rect_w = max(1, W - rect_x1)
            rect_y1 = max(0, min(H - 1, rect_y1))
            if rect_y1 + gap > H:
                gap = max(1, H - rect_y1)

            # Vẽ hình chữ nhật cầu nối
            cv2.rectangle(union_mask, (rect_x1, rect_y1),
                          (rect_x1 + rect_w, rect_y1 + gap), 255, thickness=-1)
            contours = _find_contours(union_mask)

        # Lấy bbox bao toàn bộ union để gộp tất cả line thành 1 khối duy nhất
        non_zero = cv2.findNonZero(union_mask)
        if non_zero is None:
            continue
        x, y, w, h = cv2.boundingRect(non_zero)
        crop = union_mask[y:y+h, x:x+w].copy()

        # (Tùy chọn) đóng kín nhẹ để lấp khe hở nhỏ
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, se)

        polygon_masks.append((crop, (x, y, x + w, y + h)))

    # # Thêm các region không thuộc mapping (mask theo chính polygon của vùng)
    # mapped_indices = {idx for indices in bbox_to_regions.values()
    #                   for idx in indices}
    # for i, region in enumerate(text_regions):
    #     if i in mapped_indices:
    #         continue
    #     # Tạo mask riêng cho region (không dùng xywh)
    #     mask = np.zeros((H, W), dtype=np.uint8)
    #     for poly in region.lines:
    #         try:
    #             pts = np.array(poly, dtype=np.int32)
    #             cv2.fillPoly(mask, [pts], 255)
    #         except Exception:
    #             pass
    #     if kernel is not None:
    #         mask = cv2.dilate(mask, kernel)
    #     contours, _ = cv2.findContours(
    #         mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     for contour in contours:
    #         if contour is None or len(contour) == 0:
    #             continue
    #         x, y, w, h = cv2.boundingRect(contour)
    #         crop = mask[y:y+h, x:x+w].copy()
    #         polygon_masks.append((crop, [x, y, x + w, y + h]))

    # Debug: vẽ polygon lên ảnh gốc
    if DEBUG:
        try:
            img_poly = original_img.copy()
            debug_dir = os.path.join(BASE_DIR, "debug", "debug_polygons")
            os.makedirs(debug_dir, exist_ok=True)

            for idx, poly in enumerate(polygon_masks):
                mask, (x1, y1, x2, y2) = poly
                roi = img_poly[y1:y2, x1:x2]
                overlay = roi.copy()
                color = (0, 255, 255)  # BGR
                overlay[mask > 0] = color
                alpha = 0.4
                roi[:] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)

            out_path = os.path.join(debug_dir, "polygons_on_img.png")
            cv2.imwrite(out_path, img_poly)
            logger.debug(f"Saved polygon visualization to {out_path}")
        except Exception:
            logger.warning("Failed to save polygon debug image", exc_info=True)

    return polygon_masks
