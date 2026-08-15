import copy
import json
from pprint import pformat
import cv2
import langdetect
import numpy as np
from PIL import Image
from typing import List, Tuple
from module.rendering.common import assign_translations_to_regions, merge_regions_into_oval, merge_regions_into_polygon, seg_eng, seg_vietnamese
from utils.geometry import create_oval_mask
from utils.visualize import save_polygons_image
import os
from utils.log import setup_logger
from shapely import Polygon

from module.utils.generic import Quadrilateral
from settings import BASE_DIR, DEBUG
from module.rendering.text_render import get_char_glyph, put_char_horizontal, add_color
from module.rendering.ballon_extractor import extract_ballon_region
from module.utils.textblock import TextBlock, rect_distance
logger = setup_logger(__name__)


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
        logger.debug("init TextRenderEng: \n %s", pformat({
            "text_regions": len(text_regions),
            "text_bboxes": len(text_bboxes),
        }))

    def render_lines(
        self,
        textlines: List[Textline],
        canvas_h: int,
        canvas_w: int,
        font_size: int,
        stroke_width: int,
        line_spacing: int = 0.01,
        fg: Tuple[int] = (0, 0, 0),
        bg: Tuple[int] = (255, 255, 255)
    ) -> Image.Image:

        # bg_size = int(max(font_size * 0.1, 1)) if bg is not None else 0
        bg_size = stroke_width
        spacing_y = int(font_size * (line_spacing or 0.01))

        # make large canvas
        canvas_w = max([l.length for l in textlines]) + \
            (font_size + bg_size) * 2
        canvas_h = font_size * \
            len(textlines) + spacing_y * \
            (len(textlines) - 1) + (font_size + bg_size) * 2
        canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas_border = canvas_text.copy()

        # pen (x, y)
        pen_orig = [font_size + bg_size, font_size + bg_size]

        # write stuff
        for line in textlines:
            pen_line = pen_orig.copy()
            pen_line[0] += line.pos_x  # center
            for c in line.text:
                offset_x = put_char_horizontal(
                    font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
                pen_line[0] += offset_x
            pen_orig[1] += spacing_y + font_size

        # colorize
        canvas_border = np.clip(canvas_border, 0, 255)
        line_box = add_color(canvas_text, fg, canvas_border, bg)

        # rect
        x, y, width, height = cv2.boundingRect(canvas_border)
        return Image.fromarray(line_box[y:y+height, x:x+width])

        # c = Image.new('RGBA', (canvas_w, canvas_h), color = (0, 0, 0, 0))
        # d = ImageDraw.Draw(c)
        # d.fontmode = 'L'
        # for line in lines:
        #     d.text((line.pos_x, line.pos_y), line.text, font=font, fill=font_color, stroke_width=font_size, stroke_fill=stroke_color)
        # return c

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
        allow_overflow: bool = True          # Cho phép text tràn ra khỏi region
    ) -> List[Textline]:

        m = cv2.moments(mask)
        mask = 255 - mask
        centroid_y = int(m['m01'] / m['m00'])
        centroid_x = int(m['m10'] / m['m00'])

        # layout the central line, the center word is approximately aligned with the centroid of the mask
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

        # layout bottom half
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
                elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
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

        # layout top half
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
                elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
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

        # rbgmsk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.circle(rbgmsk, (centroid_x, centroid_y), 10, (255, 0, 0))
        # for line in lines:
        #     cv2.rectangle(rbgmsk, (line.pos_x, line.pos_y), (line.pos_x + line.length, line.pos_y + line_height), (0, 255, 0))
        # cv2.imshow('mask', rbgmsk)
        # cv2.waitKey(0)

        return lines

    def render_textblock_list_eng(
        self,
        font_color=(0, 0, 0),               # Màu chữ (mặc định: đen)
        stroke_color=(255, 255, 255),       # Màu viền chữ (mặc định: trắng)
        delimiter: str = ' ',                 # Ký tự phân cách giữa các từ
        line_spacing: int = 0.01,             # Khoảng cách giữa các dòng
        stroke_width: float = 0.1,            # Độ dày viền chữ
        size_tol: float = 1.0,                # Dung sai kích thước
        ballonarea_thresh: float = 2,         # Ngưỡng diện tích bong bóng thoại
        downscale_constraint: float = 0.1,    # Giới hạn thu nhỏ tỷ lệ
        disable_font_border: bool = False,    # Tắt viền chữ
        allow_overflow: bool = False,
        max_overflow_ratio: float = 1.4,
        lang: str = "vi",
    ) -> np.ndarray:
        r"""
        Hàm render danh sách các khối văn bản vào hình ảnh, đặt văn bản vào trong các bong bóng thoại

        Args:
            downscale_constraint (float, optional): Tỷ lệ thu nhỏ tối thiểu, ngăn văn bản render quá nhỏ
            ref_textballon (bool, optional): Sử dụng bong bóng thoại làm tham chiếu cho bố cục văn bản
            original_img (np.ndarray, optional): Hình ảnh gốc dùng để trích xuất bong bóng thoại
        """


        def calculate_font_values(font_size: int, words: List[str]):
            """
            Tính toán các giá trị liên quan đến font chữ và kích thước của từng từ

            Args:
                font_size: Kích thước font chữ
                words: Danh sách các từ cần tính toán kích thước

            Returns:
                Tuple chứa: font_size, độ dày viền, chiều cao dòng, độ dài ký tự phân cách,
                độ dài từ dài nhất, và danh sách độ dài của từng từ
            """
            font_size = int(font_size)
            sw = int(font_size * stroke_width)  # Độ dày viền chữ
            line_height = int(font_size * 0.8)  # Chiều cao dòng
            # Lấy glyph của ký tự phân cách
            delimiter_glyph = get_char_glyph(delimiter, font_size, 0)
            # Độ dài của ký tự phân cách (bit shift để chuyển đơn vị)
            delimiter_len = delimiter_glyph.advance.x >> 6
            base_length = -1  # Độ dài từ dài nhất
            word_lengths = []  # Danh sách độ dài của từng từ

            # Tính toán độ dài của từng từ
            for word in words:
                word_length = 0
                for cdpt in word:
                    # Lấy glyph của ký tự
                    glyph = get_char_glyph(cdpt, font_size, 0)
                    char_offset_x = glyph.metrics.horiAdvance >> 6  # Độ dài của ký tự
                    word_length += char_offset_x
                word_lengths.append(word_length)
                if word_length > base_length:
                    base_length = word_length  # Cập nhật độ dài từ dài nhất

            return font_size, sw, line_height, delimiter_len, base_length, word_lengths

        def update_enlarged_xyxy(region):
            """
            Cập nhật tọa độ vùng phóng to dựa trên tỷ lệ phóng to

            Args:
                region: Vùng văn bản cần cập nhật
            """
            region.enlarged_xyxy = region.xyxy.copy()
            # Tính toán chênh lệch chiều rộng và chiều cao khi phóng to
            w_diff, h_diff = (
                (region.xywh[2:] * region.enlarge_ratio) - region.xywh[2:].astype(np.float64)) // 2
            # Cập nhật tọa độ vùng phóng to
            region.enlarged_xyxy[0] -= w_diff  # x1 giảm
            region.enlarged_xyxy[2] += w_diff  # x2 tăng
            region.enlarged_xyxy[1] -= h_diff  # y1 giảm
            region.enlarged_xyxy[3] += h_diff  # y2 tăng

        # Khởi tạo tỷ lệ phóng to cho các vùng văn bản
        for region in self.text_regions:
            region.enlarge_ratio = 1  # Tỷ lệ phóng to ban đầu là 1
            region.enlarged_xyxy = region.xyxy.copy()  # Sao chép tọa độ gốc

        text_regions, region_to_bbox_mapping = assign_translations_to_regions(
            self.text_bboxes, self.text_regions, lang=lang)
        # oval_masks = merge_regions_into_oval(
        #     text_regions, self.text_bboxes, region_to_bbox_mapping, original_img=self.original_img)
        print("text_regions after assign translations:", text_regions)
        if DEBUG:
            logger.debug(
                "Số lượng text region sau khi assign translations: %s", len(text_regions))

        # Điều chỉnh tỷ lệ phóng to giữa các vùng văn bản để giảm thiểu sự chồng lấn
        for region in text_regions:
            # Nếu tỷ lệ phóng to chưa được thay đổi
            if region.enlarge_ratio == 1:
                # Tỷ lệ khung hình càng lớn thì càng nên phóng to bong bóng
                # Tính tỷ lệ dựa trên tỷ lệ khung hình (lấy giá trị lớn hơn giữa width/height và height/width)
                region.enlarge_ratio = min(
                    max(region.xywh[2] / region.xywh[3], region.xywh[3] / region.xywh[2]) * 1.7, 3)
                update_enlarged_xyxy(region)

            # Kiểm tra chồng lấn với các vùng khác
            for region2 in text_regions:
                if region is region2:  # Bỏ qua nếu là cùng một vùng
                    continue

                # Nếu hai vùng phóng to giao nhau
                if rect_distance(*region.enlarged_xyxy, *region2.enlarged_xyxy) == 0:  # Nếu giao nhau
                    # Lấy khoảng cách ban đầu và điều chỉnh tỷ lệ phóng to tương ứng
                    # Khoảng cách giữa hai vùng gốc
                    d = rect_distance(*region.xyxy, *region2.xyxy)
                    l1 = (region.xywh[2] + region.xywh[3]) / \
                        2  # Kích thước trung bình của vùng 1
                    # Kích thước trung bình của vùng 2
                    l2 = (region2.xywh[2] + region2.xywh[3]) / 2

                    # Điều chỉnh tỷ lệ phóng to để tránh chồng lấn
                    region.enlarge_ratio = d / (2 * l1) + 1
                    region2.enlarge_ratio = d / (2 * l2) + 1

                    # Cập nhật tọa độ vùng phóng to
                    update_enlarged_xyxy(region)
                    update_enlarged_xyxy(region2)
                    # print('Reducing enlarge ratio to prevent intersection')
                    # print(region.translation, region.enlarged_xyxy, region.enlarge_ratio)
                    # print('>->', region2.translation, region2.enlarged_xyxy, region2.enlarge_ratio)
        ballon_polygons = []
        ballon_labels = []
        # Xử lý từng vùng văn bản
        for ri, region in enumerate(text_regions):
            print(f"\n=== DEBUG REGION {ri} ===")
            print("translation:", repr(region.translation))

            # Phân đoạn văn bản thành các từ
            words = seg_vietnamese(region.translation)
            print(f"Words sau khi seg_eng: {words}")
            if not words:  # Bỏ qua nếu không có từ nào
                continue

            # Tính toán các giá trị liên quan đến font chữ
            # The above code in Python is attempting to create a variable named `font_size`, but it
            # seems to be incomplete or missing the assignment of a value to the variable.
            # The above code in Python is attempting to create a variable named `font_size`, but it is
            # missing the assignment of a value to the variable. In Python, variables need to be
            # assigned a value before they can be used.
            font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(
                region.font_size, words)
            print(f"||1|| font_size: {font_size}, stroke_width: {sw}, line_height: {line_height}, delimiter_len: {delimiter_len}, base_length: {base_length}, word_lengths: {word_lengths}")

            # Trích xuất vùng bong bóng thoại
            # Sử dụng phương pháp không dựa trên deep learning để phân đoạn bong bóng thoại
            # ballon_mask, xyxy = extract_ballon_region(original_img, region.xywh, enlarge_ratio=region.enlarge_ratio)

            ballon_mask, xyxy = extract_ballon_region(
                self.original_img, region.xywh, enlarge_ratio=region.enlarge_ratio)
            if ballon_mask is None:  # Bỏ qua nếu không trích xuất được bong bóng thoại
                continue

            # # Lưu mask bong bóng thoại để debug
            # cv2.imwrite(_result_path(os.path.join(BASE_DIR, "debug", f"region_{ri}_ballon_mask.png")), ballon_mask)
            # cv2.imwrite(os.path.join(BASE_DIR, "debug", f"region_{ri}_ballon_mask.png"), ballon_mask)

            # Tìm đường viền của ballon_mask
            contours, _ = cv2.findContours(
                ballon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Lấy đường viền lớn nhất
                largest_contour = max(contours, key=cv2.contourArea)
                # Chuyển đổi contour thành polygon và điều chỉnh tọa độ theo vị trí thực trong ảnh gốc
                polygon = largest_contour.reshape(-1, 2)
                polygon[:, 0] += xyxy[0]  # Cộng x offset
                polygon[:, 1] += xyxy[1]  # Cộng y offset
                ballon_polygons.append(polygon)
                ballon_labels.append(f"Region {ri}")

            # Tính diện tích bong bóng thoại
            ballon_area = (ballon_mask > 0).sum()

            # Khởi tạo biến cho trường hợp xoay
            rotated, rx, ry = False, 0, 0

            # Xử lý trường hợp vùng văn bản bị xoay
            if abs(region.angle) > 3:  # Nếu góc xoay lớn hơn 3 độ
                rotated = True
                # Chuyển đổi góc từ độ sang radian
                region_angle_rad = np.deg2rad(region.angle)
                region_angle_sin = np.sin(region_angle_rad)
                region_angle_cos = np.cos(region_angle_rad)

                # Xoay mask bong bóng thoại theo góc của vùng văn bản
                rotated_ballon_mask = Image.fromarray(
                    ballon_mask).rotate(region.angle, expand=True)
                rotated_ballon_mask = np.array(rotated_ballon_mask)

                # Chuẩn hóa góc về khoảng [0, 360)
                region.angle %= 360

                # Tính toán offset dựa trên góc xoay
                if region.angle > 0 and region.angle <= 90:  # Góc phần tư thứ nhất
                    ry = abs(ballon_mask.shape[1] * region_angle_sin)
                elif region.angle > 90 and region.angle <= 180:  # Góc phần tư thứ hai
                    rx = abs(ballon_mask.shape[1] * region_angle_cos)
                    ry = rotated_ballon_mask.shape[0]
                elif region.angle > 180 and region.angle <= 270:  # Góc phần tư thứ ba
                    ry = abs(ballon_mask.shape[0] * region_angle_cos)
                    rx = rotated_ballon_mask.shape[1]
                else:  # Góc phần tư thứ tư
                    rx = abs(ballon_mask.shape[0] * region_angle_sin)

                # Sử dụng mask đã xoay
                ballon_mask = rotated_ballon_mask

            # Tính toán chiều rộng tổng của văn bản và diện tích vùng văn bản
            # Tổng độ dài các từ và khoảng cách
            line_width = sum(word_lengths) + delimiter_len * \
                (len(word_lengths) - 1)
            region_area = line_width * line_height + delimiter_len * \
                (len(words) - 1) * line_height  # Diện tích vùng văn bản
            area_ratio = ballon_area / region_area  # Tỷ lệ diện tích bong bóng/văn bản
            resize_ratio = 1  # Tỷ lệ thay đổi kích thước ban đầu

            # Tìm hình chữ nhật bao quanh vùng bong bóng thoại
            region_x, region_y, region_w, region_h = cv2.boundingRect(
                cv2.findNonZero(ballon_mask))

            # Lấy từ dài nhất để làm cơ sở tính toán
            base_length_word = words[max(
                enumerate(word_lengths), key=lambda x: x[1])[0]]
            if len(base_length_word) == 0:
                continue

            # Tính số dòng cần thiết và số dòng có thể có
            # Ước lượng số dòng cần thiết
            lines_needed = len(region.translation) / len(base_length_word)
            # Số dòng có thể có trong vùng bong bóng
            lines_available = abs(xyxy[3] - xyxy[1]) // line_height + 1

            # Tính hệ số nhân kích thước font để vừa với bong bóng thoại
            # Lấy giá trị nhỏ nhất giữa:
            # 1. Tỷ lệ chiều rộng vùng/độ dài từ dài nhất (có tính đến viền)
            # 2. Tỷ lệ số dòng có thể có/số dòng cần thiết
            # Nhưng không nhỏ hơn giới hạn thu nhỏ
            font_size_multiplier = max(min(
                region_w / (base_length + 2*sw), lines_available / lines_needed), downscale_constraint)

            # Nếu cần thu nhỏ font
            if font_size_multiplier < 1:
                # Giảm kích thước font
                font_size = int(font_size * font_size_multiplier)
                # Tính toán lại các giá trị liên quan đến font với kích thước mới
                font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(
                    font_size, words)
            # elif font_size_multiplier > 1.5 and font_size_multiplier < 1.9:
            #     font_size = int(font_size * (font_size_multiplier - 0.42))
                # font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(
                #     font_size, words)

            # Bố trí các dòng văn bản căn giữa trong bong bóng thoại
            textlines = self.layout_lines_aligncenter(
                ballon_mask, words, word_lengths, delimiter_len, line_height, delimiter=delimiter)

            # Tính toán vị trí trung tâm theo chiều dọc của văn bản và vùng bong bóng
            line_cy = np.array([line.pos_y for line in textlines]).mean(
            ) + line_height / 2  # Tâm dọc của văn bản
            region_cy = region_y + region_h / 2  # Tâm dọc của vùng bong bóng

            # Tính offset theo chiều dọc để căn giữa văn bản trong bong bóng
            # Giới hạn offset trong khoảng [-line_height, line_height] để tránh văn bản bị đẩy ra quá xa
            y_offset = int(
                round(np.clip(region_cy - line_cy, -line_height, line_height)))

            # Tính toán kích thước canvas để vẽ văn bản
            lines_x1, lines_x2 = [], []  # Lưu tọa độ x bên trái và bên phải của mỗi dòng
            for line in textlines:
                lines_x1.append(line.pos_x)  # Tọa độ x bên trái
                # Tọa độ x bên phải
                lines_x2.append(max(line.pos_x, 0) + line.length)

            lines_x1 = np.array(lines_x1)
            lines_x2 = np.array(lines_x2)

            # Tính toán kích thước canvas, thêm khoảng đệm stroke_width ở cả hai bên
            canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
            canvas_y1, canvas_y2 = textlines[0].pos_y - \
                sw, textlines[-1].pos_y + line_height + sw
            canvas_h = int(canvas_y2 - canvas_y1)
            canvas_w = int(canvas_x2 - canvas_x1)

            # Tạo bản đồ các dòng văn bản
            lines_map = np.zeros_like(ballon_mask, dtype=np.uint8)
            for line in textlines:
                # Vẽ hình chữ nhật đại diện cho mỗi dòng văn bản
                cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_offset),
                              (line.pos_x + line.length + sw, line.pos_y + line_height), 255, -1)

                # Điều chỉnh tọa độ của dòng để phù hợp với canvas mới
                line.pos_x -= canvas_x1
                line.pos_y -= canvas_y1

            # Lấy màu chữ và màu viền từ vùng văn bản
            region_font_color, region_stroke_color = region.get_font_colors()

            # Kiểm tra nếu không có dòng nào được tạo ra
            if len(textlines) == 0:
                continue

            # Render các dòng văn bản thành hình ảnh
            textlines_image = self.render_lines(
                textlines,
                canvas_h,
                canvas_w,
                font_size, sw,
                line_spacing,
                region_font_color,
                region_stroke_color
            )

            # Tính toán tọa độ tâm tương đối của canvas văn bản
            rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
            rel_cy = ((canvas_y1 + canvas_y2) / 2 -
                      ry + y_offset) / resize_ratio

            # Tính toán diện tích của các dòng văn bản
            lines_area = np.sum(lines_map)

            # Thêm diện tích khoảng trống xung quanh văn bản
            # Tính phần diện tích thừa theo chiều dọc và chiều ngang
            lines_area += (max(0, region_y - canvas_y1) + max(0, canvas_y2 - region_h - region_y)) * canvas_w * 255 \
                + (max(0, region_x - canvas_x1) +
                   max(0, canvas_x2 - region_w - region_x)) * canvas_h * 255

            # Tính tỷ lệ diện tích văn bản so với diện tích bong bóng thoại
            valid_lines_ratio = lines_area / \
                np.sum(cv2.bitwise_and(lines_map, ballon_mask))

            # Nếu diện tích văn bản lớn hơn diện tích bong bóng, cần thu nhỏ
            # Xử lý tùy theo cài đặt tràn
            if not allow_overflow and valid_lines_ratio > 1:
                # Không cho phép tràn, thu nhỏ text
                resize_ratio = min(
                    resize_ratio * valid_lines_ratio, (1 / downscale_constraint) ** 2)
            elif allow_overflow and valid_lines_ratio > max_overflow_ratio:
                # Cho phép tràn nhưng vẫn giới hạn mức độ tràn
                resize_ratio = min(resize_ratio * (valid_lines_ratio /
                                   max_overflow_ratio), (1 / downscale_constraint) ** 2)
            print(f"valid_lines_ratio: {valid_lines_ratio}")
            print(f"resize_ratio: {resize_ratio}")
            if resize_ratio != 1:
                textlines_image = textlines_image.resize(
                    (int(textlines_image.width / resize_ratio), int(textlines_image.height / resize_ratio)))

            # Xử lý trường hợp văn bản bị xoay
            if rotated:
                # Chuyển đổi tọa độ tương đối trong không gian xoay
                rcx = rel_cx * region_angle_cos - rel_cy * region_angle_sin
                rcy = rel_cx * region_angle_sin + rel_cy * region_angle_cos
                rel_cx = rcx
                rel_cy = rcy

                # Xoay hình ảnh văn bản ngược lại để hiển thị đúng
                textlines_image = textlines_image.rotate(
                    -region.angle, expand=True, resample=Image.BILINEAR)
                # Cắt bớt phần trong suốt sau khi xoay
                textlines_image = textlines_image.crop(
                    textlines_image.getbbox())

            # Chuyển đổi tọa độ tương đối thành tọa độ tuyệt đối trong hình ảnh gốc
            abs_cx = rel_cx + xyxy[0]
            abs_cy = rel_cy + xyxy[1]
            # Tính toán vị trí để paste hình ảnh văn bản vào hình ảnh gốc
            abs_x = int(abs_cx - textlines_image.width / 2)
            abs_y = int(abs_cy - textlines_image.height / 2)

            # Paste hình ảnh văn bản vào hình ảnh gốc, sử dụng mask để giữ tính trong suốt
            self.inpainted_image.paste(
                textlines_image, (abs_x, abs_y), mask=textlines_image)

        # if ballon_polygons:
        #     # Tạo đường dẫn lưu file
        #     output_path = os.path.join(BASE_DIR, "debug", "balloon_visualization.png")
        #     # Lưu hình ảnh với các ballon được vẽ
        #     save_polygons_image(
        #         image=self.original_img.copy(),  # Sử dụng ảnh gốc
        #         polygons=ballon_polygons,
        #         out_path=output_path,
        #         edge_color=(0, 255, 0),  # Viền xanh lá
        #         fill_color=(0, 255, 0),  # Fill xanh lá
        #         alpha=0.25,  # Độ trong suốt
        #         thickness=2,
        #         labels=ballon_labels  # Hiển thị nhãn cho mỗi ballon
        #     )
        #     print(f"Đã lưu hình ảnh visualization tại: {output_path}")
        return np.array(self.inpainted_image)
