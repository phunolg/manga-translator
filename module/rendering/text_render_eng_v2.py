import logging
import os
from pprint import pformat
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from module.rendering.ballon_extractor import extract_ballon_region
from module.rendering.common import (
    assign_translations_to_regions,
    merge_regions_into_oval,
    merge_regions_into_polygon,
    seg_eng,
    seg_vietnamese,
    visualize_text_blocks,
)
from module.rendering.text_render import calc_vertical, get_char_glyph, get_font_line_metrics, put_char_horizontal, add_color, get_pair_kerning
from module.utils.textblock import TextBlock, rect_distance
from settings import BASE_DIR, DEBUG
from utils.visualize import save_polygons_image
from utils.log import setup_logger


LOG = setup_logger(__name__)


class Textline:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = self.text + delimiter + word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = word + delimiter + self.text
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0


class TextRenderEng:
    def __init__(self, text_regions: List[TextBlock], text_bboxes: List[TextBlock], original_img: np.ndarray, cleaned_img: np.ndarray):
        self.inpainted_image = Image.fromarray(cleaned_img)
        self.text_regions = text_regions
        self.text_bboxes = text_bboxes
        self.original_img = original_img

        LOG.debug("init TextRenderEng: \n %s", pformat({
            "text_regions": len(text_regions),
            "text_bboxes": len(text_bboxes),
        }))

    @staticmethod
    def _calculate_font_values(font_size: int, words: List[str], delimiter: str, stroke_width: float) -> Tuple[int, int, int, int, int, List[int]]:
        """
        Args:
            font_size: kích thước font chữ ban đầu (đọc từ text_region.font_size)
            words: danh sách từ khi được phân đoạn từ translation của text_region (thông qua seg_vietnamese hoặc seg_eng)
            delimiter: ký tự phân cách giữa các từ, mặc định là ' '
            stroke_width: độ dày viền chữ
        Returns:
            tuple chứa:
                - font_size: kích thước font chữ sau khi tính toán
                - stroke_width: độ dày viền chữ
                - line_height: chiều cao dòng
                - delimiter_len: độ dài của ký tự phân cách
                - base_length: độ dài từ dài nhất
        """
        font_size = int(font_size)
        sw = int(font_size * stroke_width)  # bề dày viền/đổ bóng (pixel).
        try:
            line_height, asc, desc = get_font_line_metrics(
                font_size, direction=0)
        except Exception as e:
            LOG.error("Lỗi khi tính toán line_height: %s, error: %s",
                      pformat(locals()), e, exc_info=True)
            line_height = calc_vertical(
                font_size, words[0][0], max_height=font_size)[1][0]

        delimiter_glyph = get_char_glyph(delimiter, font_size, 0)
        delimiter_len = delimiter_glyph.advance.x >> 6
        base_length = -1
        word_lengths: List[int] = []
        for word in words:
            word_length = 0
            prev_c = None
            for cdpt in word:
                if prev_c is not None:
                    word_length += get_pair_kerning(prev_c, cdpt, font_size, 0)
                glyph = get_char_glyph(cdpt, font_size, 0)
                char_offset_x = glyph.metrics.horiAdvance >> 6
                word_length += char_offset_x
                prev_c = cdpt
            word_lengths.append(word_length)
            if word_length > base_length:
                base_length = word_length
        return font_size, sw, line_height, delimiter_len, base_length, word_lengths

    @staticmethod
    def _update_enlarged_xyxy(region) -> None:
        region.enlarged_xyxy = region.xyxy.copy()
        w_diff, h_diff = (
            (region.xywh[2:] * region.enlarge_ratio) - region.xywh[2:].astype(np.float64)) // 2
        region.enlarged_xyxy[0] -= w_diff
        region.enlarged_xyxy[2] += w_diff
        region.enlarged_xyxy[1] -= h_diff
        region.enlarged_xyxy[3] += h_diff

    def render_lines(
        self,
        textlines: List[Textline],
        canvas_h: int,
        canvas_w: int,
        font_size: int,
        stroke_width: int,
        line_spacing: int = 0.001,
        fg: Tuple[int] = (0, 0, 0),
        bg: Tuple[int] = (255, 255, 255)
    ) -> Image.Image:

        bg_size = stroke_width
        # line_spacing ở đây được hiểu là số pixel giữa các dòng
        spacing_y = int(line_spacing)

        # Dùng line height thật (FreeType) để tính kích thước canvas theo chiều dọc
        max_line_width = max([l.length for l in textlines]) if textlines else 0
        canvas_w = max_line_width + (font_size + bg_size) * 2
        try:
            estimated_line_height, _, _ = get_font_line_metrics(
                font_size, direction=0)
            estimated_line_height = max(1, int(estimated_line_height))
        except Exception:
            estimated_line_height = max(1, int(font_size))
        canvas_h = estimated_line_height * \
            len(textlines) + spacing_y * (len(textlines) - 1) + \
            (estimated_line_height + bg_size) * 2
        canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas_border = canvas_text.copy()

        pen_orig = [font_size + bg_size, estimated_line_height + bg_size]

        for line in textlines:
            pen_line = pen_orig.copy()
            pen_line[0] += line.pos_x
            for c in line.text:
                offset_x = put_char_horizontal(
                    font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
                pen_line[0] += offset_x
            pen_orig[1] += spacing_y + estimated_line_height

        canvas_border = np.clip(canvas_border, 0, 255)
        line_box = add_color(canvas_text, fg, canvas_border, bg)

        x, y, width, height = cv2.boundingRect(canvas_border)
        return Image.fromarray(line_box[y:y+height, x:x+width])

    # ===== Helper: Region initialization and overlap handling =====
    def _initialize_regions(self, lang: str):
        # Debug: Gán enlarge_ratio và enlarged_xyxy cho từng region
        if DEBUG:
            for idx, region in enumerate(self.text_regions):
                region.enlarge_ratio = 1
                region.enlarged_xyxy = region.xyxy.copy()
                LOG.debug(
                    f"Region {idx} enlarge_ratio=1, enlarged_xyxy={region.enlarged_xyxy}")

            visualize_text_blocks(
                self.text_regions, self.original_img, "text_bboxes_init.png")

        text_regions, region_to_bbox_mapping = assign_translations_to_regions(
            self.text_bboxes, self.text_regions, lang=lang
        )
        if DEBUG:
            visualize_text_blocks(
                text_regions, self.original_img, "text_bboxes_after_assign.png")

            LOG.debug(
                f"region_to_bbox_mapping: {region_to_bbox_mapping}")
            for idx, region in enumerate(text_regions):
                LOG.debug(
                    f"Region {idx} translation: {region.translation}")

        # Debug: Gộp region thành oval và lấy mask
        oval_masks = merge_regions_into_polygon(
            self.text_regions,
            self.text_bboxes,
            region_to_bbox_mapping,
            original_img=self.original_img
        )
        if DEBUG:
            LOG.debug(
                f"Số lượng oval_masks: {len(oval_masks)}")
            for idx, (region, oval) in enumerate(zip(text_regions, oval_masks)):
                if oval is not None and isinstance(oval, tuple) and len(oval) == 2:
                    mask, xyxy = oval
                    LOG.debug(
                        f"Region {idx} oval mask shape: {mask.shape}, xyxy: {xyxy}")
                else:
                    LOG.debug(
                        f"Region {idx} oval mask: {oval}")
        # Nới khung ban đầu theo tỉ lệ khung chữ
        # for region in text_regions:
        #     if region.enlarge_ratio == 1:
        #         region.enlarge_ratio = min(
        #             max(region.xywh[2] / region.xywh[3],
        #                 region.xywh[3] / region.xywh[2]) * 1.5, 3
        #         )
        #         self._update_enlarged_xyxy(region)

        #     for region2 in text_regions:
        #         if region is region2:
        #             continue
        #         if rect_distance(*region.enlarged_xyxy, *region2.enlarged_xyxy) == 0:
        #             d = rect_distance(*region.xyxy, *region2.xyxy)
        #             l1 = (region.xywh[2] + region.xywh[3]) / 2
        #             l2 = (region2.xywh[2] + region2.xywh[3]) / 2
        #             region.enlarge_ratio = d / (2 * l1) + 1
        #             region2.enlarge_ratio = d / (2 * l2) + 1
        #             self._update_enlarged_xyxy(region)
        #             self._update_enlarged_xyxy(region2)
        if DEBUG:
            # Vẽ lại các text_regions sau khi đã cập nhật enlarge_ratio và enlarged_xyxy để debug
            debug_img = self.original_img.copy()
            for idx, region in enumerate(text_regions):
                # Vẽ bounding box màu xanh lá cây enlarged_xyxy
                x1, y1, x2, y2 = map(int, region.enlarged_xyxy)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Vẽ index màu đỏ lên bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(debug_img, str(idx), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            debug_dir = os.path.join(
                BASE_DIR, "debug", "debug_enlarged_text_bboxes")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "text_bboxes_enlarged.png")
            cv2.imwrite(debug_path, debug_img)
            LOG.debug(
                f"Đã lưu ảnh debug text_regions đã enlarge tại {debug_path}")

        return text_regions, oval_masks

    # ===== Helper: Balloon polygon extraction for visualization =====
    @staticmethod
    def _add_balloon_polygon(ballon_mask: np.ndarray, xyxy: Tuple[int, int, int, int],
                             ballon_polygons: List[np.ndarray], ballon_labels: List[str], ri: int) -> None:
        """
        Trích đa giác (polygon) bao ngoài lớn nhất từ một mask (ô thoại), đổi tọa độ polygon từ hệ cục bộ của mask sang tọa độ ảnh gốc, rồi lưu vào danh sách để vẽ/hiển thị.`
        """
        contours, _ = cv2.findContours(
            ballon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygon = largest_contour.reshape(-1, 2)
            polygon[:, 0] += xyxy[0]
            polygon[:, 1] += xyxy[1]
            ballon_polygons.append(polygon)
            ballon_labels.append(f"Region {ri}")

    # ===== Helper: Rotation handling =====
    @staticmethod
    def _maybe_rotate_mask(region, ballon_mask: np.ndarray):
        rotated, rx, ry = False, 0, 0
        region_angle_sin, region_angle_cos = 0.0, 1.0
        if abs(region.angle) > 3:
            rotated = True
            region_angle_rad = np.deg2rad(region.angle)
            region_angle_sin = float(np.sin(region_angle_rad))
            region_angle_cos = float(np.cos(region_angle_rad))
            rotated_ballon_mask = Image.fromarray(
                ballon_mask).rotate(region.angle, expand=True)
            rotated_ballon_mask = np.array(rotated_ballon_mask)
            region.angle %= 360
            if region.angle > 0 and region.angle <= 90:
                ry = abs(ballon_mask.shape[1] * region_angle_sin)
            elif region.angle > 90 and region.angle <= 180:
                rx = abs(ballon_mask.shape[1] * region_angle_cos)
                ry = rotated_ballon_mask.shape[0]
            elif region.angle > 180 and region.angle <= 270:
                ry = abs(ballon_mask.shape[0] * region_angle_cos)
                rx = rotated_ballon_mask.shape[1]
            else:
                rx = abs(ballon_mask.shape[0] * region_angle_sin)
            ballon_mask = rotated_ballon_mask
        return rotated, ballon_mask, rx, ry, region_angle_sin, region_angle_cos

    # ===== Helper: Compute region rect from mask =====
    @staticmethod
    def _compute_region_rect(ballon_mask: np.ndarray):
        """
        Tìm hình chữ nhật bao trục (axis-aligned bounding box) nhỏ nhất bao hết các pixel khác 0 trong ballon_mask.
        """
        non_zero = cv2.findNonZero(ballon_mask)
        if non_zero is None:
            return None
        return cv2.boundingRect(non_zero)

    # ===== Helper: Downscale font if needed =====
    def _maybe_downscale_font(self,
                              font_size: int,
                              words: List[str],
                              delimiter: str,
                              stroke_width: float,
                              region_w: int,
                              lines_needed: float,
                              lines_available: int,
                              downscale_constraint: float):
        # Tính thông số hiện tại
        font_size_c, sw_c, line_height_c, delimiter_len_c, base_length_c, word_lengths_c = self._calculate_font_values(
            font_size, words, delimiter, stroke_width
        )
        # Ước lượng hệ số co
        font_size_multiplier = max(
            min(region_w / (base_length_c + 2*sw_c),
                lines_available / lines_needed), downscale_constraint
        )
        if font_size_multiplier < 1:
            new_font = int(font_size_c * font_size_multiplier)
            return self._calculate_font_values(new_font, words, delimiter, stroke_width)
        return font_size_c, sw_c, line_height_c, delimiter_len_c, base_length_c, word_lengths_c

    # ===== Helper: Build canvas extents and lines map =====
    @staticmethod
    def _build_canvas_and_lines_map(textlines: List[Textline], ballon_mask: np.ndarray, sw: int, line_height: int,
                                    y_offset: int):
        lines_x1, lines_x2 = [], []
        for line in textlines:
            lines_x1.append(line.pos_x)
            lines_x2.append(max(line.pos_x, 0) + line.length)
        lines_x1 = np.array(lines_x1)
        lines_x2 = np.array(lines_x2)

        canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
        canvas_y1, canvas_y2 = textlines[0].pos_y - \
            sw, textlines[-1].pos_y + line_height + sw
        canvas_h = int(canvas_y2 - canvas_y1)
        canvas_w = int(canvas_x2 - canvas_x1)

        lines_map = np.zeros_like(ballon_mask, dtype=np.uint8)
        for line in textlines:
            cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_offset),
                          (line.pos_x + line.length + sw, line.pos_y + line_height), 255, -1)
            line.pos_x -= canvas_x1
            line.pos_y -= canvas_y1

        return canvas_x1, canvas_x2, canvas_y1, canvas_y2, canvas_h, canvas_w, lines_map

    # ===== Helper: Compute valid lines ratio and resize factor =====
    @staticmethod
    def _compute_resize_ratio(lines_map: np.ndarray, ballon_mask: np.ndarray,
                              canvas_x1: int, canvas_y1: int, canvas_x2: int, canvas_y2: int,
                              region_x: int, region_y: int, region_w: int, region_h: int,
                              allow_overflow: bool, max_overflow_ratio: float, downscale_constraint: float) -> Tuple[float, float]:
        lines_area = np.sum(lines_map)
        canvas_h = int(canvas_y2 - canvas_y1)
        canvas_w = int(canvas_x2 - canvas_x1)
        lines_area += (max(0, region_y - canvas_y1) + max(0, canvas_y2 - region_h - region_y)) * canvas_w * 255 \
            + (max(0, region_x - canvas_x1) +
               max(0, canvas_x2 - region_w - region_x)) * canvas_h * 255
        denom = np.sum(cv2.bitwise_and(lines_map, ballon_mask))
        valid_lines_ratio = lines_area / denom if denom else 0.0
        resize_ratio = 1.0
        if not allow_overflow and valid_lines_ratio > 1:
            resize_ratio = min(resize_ratio * valid_lines_ratio,
                               (1 / downscale_constraint) ** 2)
        elif allow_overflow and valid_lines_ratio > max_overflow_ratio:
            resize_ratio = min(resize_ratio * (valid_lines_ratio /
                               max_overflow_ratio), (1 / downscale_constraint) ** 2)
        return valid_lines_ratio, resize_ratio

    # ===== Helper: Compute relative center =====
    @staticmethod
    def _compute_rel_center(canvas_x1: float, canvas_x2: float, canvas_y1: float, canvas_y2: float,
                            rx: float, ry: float, y_offset: int, resize_ratio: float) -> Tuple[float, float]:
        rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
        rel_cy = ((canvas_y1 + canvas_y2) / 2 - ry + y_offset) / resize_ratio
        return rel_cx, rel_cy

    # ===== Helper: Apply rotation to image and center =====
    @staticmethod
    def _apply_rotation_to_image_and_center(textlines_image: Image.Image,
                                            rel_cx: float, rel_cy: float,
                                            region_angle_sin: float, region_angle_cos: float,
                                            angle_deg: float):
        rcx = rel_cx * region_angle_cos - rel_cy * region_angle_sin
        rcy = rel_cx * region_angle_sin + rel_cy * region_angle_cos
        rel_cx = rcx
        rel_cy = rcy
        textlines_image = textlines_image.rotate(
            -angle_deg, expand=True, resample=Image.BILINEAR)
        textlines_image = textlines_image.crop(textlines_image.getbbox())
        return textlines_image, rel_cx, rel_cy

    def layout_lines_aligncenter(
        self,
        mask: np.ndarray,
        words: List[str],
        word_lengths: List[int],
        delimiter_len: int,
        line_height: int,
        spacing: int = 0,
        delimiter: str = ' ',
        max_central_width: float = np.inf,
        word_break: bool = False,
        allow_overflow: bool = True
    ) -> List[Textline]:

        m = cv2.moments(mask)
        mask = 255 - mask
        centroid_y = int(m['m01'] / m['m00'])
        centroid_x = int(m['m10'] / m['m00'])

        num_words = len(words)
        len_left, len_right = [], []
        wlst_left, wlst_right = [], []
        sum_left, sum_right = 0, 0
        if num_words > 1:
            wl_array = np.array(word_lengths, dtype=np.float64)
            wl_cumsums = np.cumsum(wl_array)
            wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
            central_index = np.argmin(np.abs(wl_cumsums))
            if central_index > 0:
                wlst_left = words[:central_index]
                len_left = word_lengths[:central_index]
                sum_left = np.sum(len_left)
            if central_index < num_words - 1:
                wlst_right = words[central_index + 1:]
                len_right = word_lengths[central_index + 1:]
                sum_right = np.sum(len_right)
        else:
            central_index = 0

        pos_y = centroid_y - line_height // 2
        pos_x = centroid_x - word_lengths[central_index] // 2

        bh, bw = mask.shape[:2]
        central_line = Textline(
            words[central_index], pos_x, pos_y, word_lengths[central_index], spacing)
        line_bottom = pos_y + line_height
        while sum_left > 0 or sum_right > 0:
            left_valid, right_valid = False, False

            if sum_left > 0:
                new_len_l = central_line.length + len_left[-1] + delimiter_len
                new_x_l = centroid_x - new_len_l // 2
                new_r_l = new_x_l + new_len_l
                if (new_x_l > 0 and new_r_l < bw):
                    if allow_overflow or mask[pos_y: line_bottom, new_x_l].sum() == 0 and mask[pos_y: line_bottom, new_r_l].sum() == 0:
                        left_valid = True
            if sum_right > 0:
                new_len_r = central_line.length + len_right[0] + delimiter_len
                new_x_r = centroid_x - new_len_r // 2
                new_r_r = new_x_r + new_len_r
                if (new_x_r > 0 and new_r_r < bw):
                    if allow_overflow or mask[pos_y: line_bottom, new_x_r].sum() == 0 and mask[pos_y: line_bottom, new_r_r].sum() == 0:
                        right_valid = True

            insert_left = False
            if left_valid and right_valid:
                if sum_left > sum_right:
                    insert_left = True
            elif left_valid:
                insert_left = True
            elif not right_valid:
                break

            if insert_left:
                central_line.append_left(
                    wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
                sum_left -= len_left.pop(-1)
                central_line.pos_x = new_x_l
            else:
                central_line.append_right(wlst_right.pop(
                    0), len_right[0] + delimiter_len, delimiter)
                sum_right -= len_right.pop(0)
                central_line.pos_x = new_x_r
            if central_line.length > max_central_width:
                break

        central_line.strip_spacing()
        lines = [central_line]

        if sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            pos_x = centroid_x - wl // 2
            pos_y = centroid_y + line_height // 2
            line_bottom = pos_y + line_height
            line = Textline(w, pos_x, pos_y, wl, spacing)
            lines.append(line)
            sum_right -= wl
            while sum_right > 0:
                w, wl = wlst_right.pop(0), len_right.pop(0)
                sum_right -= wl
                new_len = line.length + wl + delimiter_len
                new_x = centroid_x - new_len // 2
                right_x = new_x + new_len
                if new_x <= 0 or right_x >= bw:
                    line_valid = False
                elif mask[pos_y: line_bottom, new_x].sum() > 0 or \
                        mask[pos_y: line_bottom, right_x].sum() > 0:
                    line_valid = False
                else:
                    line_valid = True
                if line_valid:
                    line.append_right(w, wl+delimiter_len, delimiter)
                    line.pos_x = new_x
                    if new_len > max_central_width:
                        line_valid = False
                        if sum_right > 0:
                            w, wl = wlst_right.pop(0), len_right.pop(0)
                            sum_right -= wl
                        else:
                            line.strip_spacing()
                            break

                if not line_valid:
                    pos_x = centroid_x - wl // 2
                    pos_y = line_bottom
                    line_bottom += line_height
                    line.strip_spacing()
                    line = Textline(w, pos_x, pos_y, wl, spacing)
                    lines.append(line)

        if sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            pos_x = centroid_x - wl // 2
            pos_y = centroid_y - line_height // 2 - line_height
            line_bottom = pos_y + line_height
            line = Textline(w, pos_x, pos_y, wl, spacing)
            lines.insert(0, line)
            sum_left -= wl
            while sum_left > 0:
                w, wl = wlst_left.pop(-1), len_left.pop(-1)
                sum_left -= wl
                new_len = line.length + wl + delimiter_len
                new_x = centroid_x - new_len // 2
                right_x = new_x + new_len
                if new_x <= 0 or right_x >= bw:
                    line_valid = False
                elif mask[pos_y: line_bottom, new_x].sum() > 0 or \
                        mask[pos_y: line_bottom, right_x].sum() > 0:
                    line_valid = False
                else:
                    line_valid = True
                if line_valid:
                    line.append_left(w, wl+delimiter_len, delimiter)
                    line.pos_x = new_x
                    if new_len > max_central_width:
                        line_valid = False
                        if sum_left > 0:
                            w, wl = wlst_left.pop(-1), len_left.pop(-1)
                            sum_left -= wl
                        else:
                            line.strip_spacing()
                            break

                if not line_valid:
                    pos_x = centroid_x - wl // 2
                    pos_y -= line_height
                    line_bottom = pos_y + line_height
                    line.strip_spacing()
                    line = Textline(w, pos_x, pos_y, wl, spacing)
                    lines.insert(0, line)

        return lines

    def render_textblock_list_eng(
        self,
        font_color=(0, 0, 0),
        stroke_color=(255, 255, 255),
        delimiter: str = ' ',
        line_spacing: int = 2,
        stroke_width: float = 0.1,
        size_tol: float = 1.0,
        ballonarea_thresh: float = 2,
        downscale_constraint: float = 0.95,
        disable_font_border: bool = False,
        allow_overflow: bool = False,
        max_overflow_ratio: float = 1.4,
        lang: str = "vi",
    ) -> np.ndarray:

        text_regions, ballon_polygons = self._initialize_regions(lang)
        # ovel_mask là tuple (mask: np.array, xyxy: Tuple[int, int, int, int]) tương ứng với từng region
        LOG.debug(
            f"Text_regions sau khi _initialize_regions: \n {pformat(text_regions)}")
        LOG.debug(
            f"Ballon_polygons sau khi _initialize_regions: \n {pformat(ballon_polygons)}")

        # ballon_polygons: List[np.ndarray] = []
        ballon_labels: List[str] = []

        for text_region, ballon_polygon in zip(text_regions, ballon_polygons):
            LOG.debug("=== DEBUG REGION ===")
            LOG.debug("translation: %r", text_region.translation)

            words = seg_vietnamese(text_region.translation)
            LOG.debug("Words sau khi seg_eng: %s", words)
            if not words:
                continue

            font_size, sw, line_height, delimiter_len, base_length, word_lengths = self._calculate_font_values(
                text_region.font_size, words, delimiter, stroke_width
            )
            LOG.debug("font_size: %s", pformat({
                "font_size": font_size,
                "sw": sw,
                "line_height": line_height,
                "delimiter_len": delimiter_len,
                "base_length": base_length,
                "word_lengths": word_lengths
            }))

            ballon_mask, xyxy = ballon_polygon

            rotated, ballon_mask, rx, ry, region_angle_sin, region_angle_cos = self._maybe_rotate_mask(
                text_region, ballon_mask)

            line_width = sum(word_lengths) + delimiter_len * \
                (len(word_lengths) - 1)
            region_area = line_width * line_height + \
                delimiter_len * (len(words) - 1) * line_height
            ballon_area = (ballon_mask > 0).sum()
            area_ratio = ballon_area / region_area if region_area else 0
            resize_ratio = 1

            rect = self._compute_region_rect(ballon_mask)
            if rect is None:
                continue
            region_x, region_y, region_w, region_h = rect

            # Tìm từ dài nhất theo pixel (advance), vừa lấy giá trị vừa lấy chỉ số
            longest_idx = int(np.argmax(word_lengths)) if len(
                word_lengths) > 0 else -1
            base_length_word = words[longest_idx] if longest_idx >= 0 else ""
            if len(base_length_word) == 0:
                continue

            # Tính effective width trừ stroke + inner padding (dựa trên sw)
            inner_padding = max(1, sw)
            effective_w = max(1, region_w - 2 * (sw + inner_padding))

            # Suy ra spacing_px theo line_height (không dùng centroid để canh dọc)
            spacing_px = int(line_height * (line_spacing or 0))

            # Binary search để fit font-size theo effective_w và lines_available
            def wrap_lines_needed(delim_len: int, wl: List[int], max_w: int) -> int:
                lines = 1
                current = 0
                for wlen in wl:
                    # Khi mở dòng mới, không cộng delimiter_len (tránh cộng thừa)
                    extra = delim_len if current > 0 else 0
                    if current + extra + wlen <= max_w:
                        current += extra + wlen
                    else:
                        lines += 1
                        current = wlen  # không cộng extra ở đầu dòng mới
                return max(1, lines)

            low, high = max(
                6, int(font_size * downscale_constraint)), int(font_size)
            best = (font_size, sw, line_height,
                    delimiter_len, base_length, word_lengths)
            while low <= high:
                mid = (low + high) // 2
                fz, sw_m, lh_m, delim_m, base_m, wl_m = self._calculate_font_values(
                    mid, words, delimiter, stroke_width)
                eff_w_m = max(1, region_w - 2 * (sw_m + max(1, sw_m)))
                spacing_m = int(lh_m * (line_spacing or 0))
                denom_m = max(1, lh_m + spacing_m)
                lines_avail_m = max(1, region_h // denom_m)
                lines_need_m = wrap_lines_needed(delim_m, wl_m, eff_w_m)
                if lines_need_m <= lines_avail_m:
                    best = (fz, sw_m, lh_m, delim_m, base_m, wl_m)
                    # có thể tăng kích thước hơn
                    low = mid + 1
                else:
                    high = mid - 1

            font_size, sw, line_height, delimiter_len, base_length, word_lengths = best
            spacing_px = int(line_height * (line_spacing or 0))

            textlines = self.layout_lines_aligncenter(
                ballon_mask,
                words,
                word_lengths,
                delimiter_len,
                line_height,
                delimiter=delimiter,
                allow_overflow=allow_overflow,
            )

            # Căn ngang theo trọng tâm mask theo từng dòng (scanline COM)
            if len(textlines) > 0:
                for ln in textlines:
                    y1 = max(0, ln.pos_y)
                    y2 = min(ballon_mask.shape[0], ln.pos_y + line_height)
                    if y2 > y1:
                        row = ballon_mask[y1:y2]
                        m = cv2.moments(255 - row)
                        if m['m00'] != 0:
                            cx = int(m['m10'] / m['m00'])
                            # dịch để tâm dòng trùng COM theo hàng
                            ln.pos_x = int(cx - ln.length / 2)

            # Canh dọc block theo tổng chiều cao dòng thay vì centroid
            if len(textlines) > 0:
                current_top = min(line.pos_y for line in textlines)
                block_h = len(textlines) * line_height + \
                    (len(textlines) - 1) * spacing_px
                target_top = region_y + max(0, (region_h - block_h) // 2)
                y_offset = int(round(target_top - current_top))
            else:
                y_offset = 0

            canvas_x1, canvas_x2, canvas_y1, canvas_y2, canvas_h, canvas_w, lines_map = self._build_canvas_and_lines_map(
                textlines, ballon_mask, sw, line_height, y_offset
            )

            # Chọn màu theo độ tương phản nền local (median trong mask)
            region_font_color, region_stroke_color = text_region.get_font_colors()
            try:
                # Tính median màu nền trong vùng mask
                roi = self.original_img[xyxy[1]:xyxy[1] +
                                        ballon_mask.shape[0], xyxy[0]:xyxy[0]+ballon_mask.shape[1]]
                inv = (ballon_mask == 0)
                if inv.any():
                    bg_median = np.median(roi[inv], axis=0)
                    # Tính luminance và chọn màu tương phản (đơn giản: đen/trắng hoặc đảo)
                    lum = 0.2126 * bg_median[2] + 0.7152 * \
                        bg_median[1] + 0.0722 * bg_median[0]
                    if lum > 140:
                        region_font_color = (0, 0, 0)
                        region_stroke_color = (255, 255, 255)
                    else:
                        region_font_color = (255, 255, 255)
                        region_stroke_color = (0, 0, 0)
            except Exception:
                pass
            if len(textlines) == 0:
                continue

            textlines_image = self.render_lines(
                textlines,
                canvas_h,
                canvas_w,
                font_size,
                sw,
                spacing_px,
                region_font_color,
                region_stroke_color,
            )

            rel_cx, rel_cy = self._compute_rel_center(
                canvas_x1, canvas_x2, canvas_y1, canvas_y2, rx, ry, y_offset, resize_ratio)

            valid_lines_ratio, resize_ratio = self._compute_resize_ratio(
                lines_map, ballon_mask, canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                region_x, region_y, region_w, region_h, allow_overflow, max_overflow_ratio, downscale_constraint
            )
            LOG.debug("valid_lines_ratio: %s", valid_lines_ratio)
            LOG.debug("resize_ratio: %s", resize_ratio)
            if resize_ratio != 1:
                textlines_image = textlines_image.resize(
                    (int(textlines_image.width / resize_ratio), int(textlines_image.height / resize_ratio)))

            if rotated:
                textlines_image, rel_cx, rel_cy = self._apply_rotation_to_image_and_center(
                    textlines_image, rel_cx, rel_cy, region_angle_sin, region_angle_cos, text_region.angle
                )

            abs_cx = rel_cx + xyxy[0]
            abs_cy = rel_cy + xyxy[1]
            abs_x = int(abs_cx - textlines_image.width / 2)
            abs_y = int(abs_cy - textlines_image.height / 2)
            # Hợp nhất alpha fg ∪ stroke để tránh dày alpha do chồng lấn nhiều lần
            tl_rgba = np.array(textlines_image)
            if tl_rgba.shape[2] == 4:
                # Alpha tiền xử lý: nhị phân mềm -> tránh tích lũy alpha nhiều lần
                alpha = tl_rgba[:, :, 3]
                # Không thay đổi màu, chỉ đảm bảo alpha hợp nhất
                tl_rgba[:, :, 3] = alpha
                textlines_image = Image.fromarray(tl_rgba)

            self.inpainted_image.paste(
                textlines_image, (abs_x, abs_y), mask=textlines_image)
        return np.array(self.inpainted_image)
