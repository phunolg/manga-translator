import re
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, box
import networkx as nx
from copy import deepcopy
from itertools import groupby
from typing import List
from settings import DEVICE



class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_components = n

    @classmethod
    def from_adj_matrix(cls, adj_matrix):
        ufds = cls(adj_matrix.shape[0])
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    ufds.unite(i, j)
        return ufds
    
    @classmethod
    def from_adj_list(cls, adj_list):
        ufds = cls(len(adj_list))
        for i in range(len(adj_list)):
            for j in adj_list[i]:
                ufds.unite(i, j)
        return ufds
    
    @classmethod
    def from_edge_list(cls, edge_list, num_nodes):
        ufds = cls(num_nodes)
        for edge in edge_list:
            ufds.unite(edge[0], edge[1])
        return ufds

    def find(self, x):
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            if self.size[x] < self.size[y]:
                x, y = y, x
            self.parent[y] = x
            self.size[x] += self.size[y]
            self.num_components -= 1
    
    def get_components_of(self, x):
        x = self.find(x)
        return [i for i in range(len(self.parent)) if self.find(i) == x]
    
    def are_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_size(self, x):
        return self.size[self.find(x)]

    def get_num_components(self):
        return self.num_components
    
    def get_labels_for_connected_components(self):
        map_parent_to_label = {}
        labels = []
        for i in range(len(self.parent)):
            parent = self.find(i)
            if parent not in map_parent_to_label:
                map_parent_to_label[parent] = len(map_parent_to_label)
            labels.append(map_parent_to_label[parent])
        return labels


def bbox_to_quad(bboxes: List[List[float]]) -> List[List[List[float]]]:
    """
    Chuyển đổi bbox (xmin, ymin, xmax, ymax) thành quad [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] tương ứng với toạn độ 4 góc
    Args:
        bboxes: list of bboxes (N,4) 
    Returns:
        list of quads (N,4,2) 
    """
    xyxy = np.array(bboxes)  # (N,4) với cột [xmin, ymin, xmax, ymax]
    quads = np.stack([
        xyxy[:, [0, 1]],
        xyxy[:, [2, 1]],
        xyxy[:, [2, 3]],
        xyxy[:, [0, 3]],
    ], axis=1)  # (N,4,2)
    return quads
    
def plot_bboxes(subplot, bboxes, color="red", visibility=None):
    if visibility is None:
        visibility = [1] * len(bboxes)
    for id, bbox in enumerate(bboxes):
        if visibility[id] == 0:
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=1, edgecolor=color, facecolor="none", linestyle="solid"
        )
        subplot.add_patch(rect)

def visualise_single_image_prediction(image_as_np_array, predictions, filename):
    h, w = image_as_np_array.shape[:2]
    dpi = 200  # tăng DPI để khớp mật độ pixel cao
    figure = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    subplot = figure.add_axes([0, 0, 1, 1])
    subplot.axis("off")
    subplot.imshow(image_as_np_array, interpolation="nearest")
    full_text_bboxes = predictions["texts"]
    text_is_essential_bboxes = [text for text, is_essential in zip(predictions["texts"], predictions["is_essential_text"]) if is_essential]

    plot_bboxes(subplot, predictions["panels"], color="green")
    plot_bboxes(subplot, text_is_essential_bboxes, color="red")
    
    # Thêm nhiễu nhỏ cho full_text_bboxes để tránh chồng lấn lên text_is_essential_bboxes
    np.random.seed(42)
    noise_level = 4  # pixel, có thể điều chỉnh nếu cần
    noisy_full_text_bboxes = []
    for bbox in full_text_bboxes:
        # bbox: [xmin, ymin, xmax, ymax]
        # Thêm nhiễu nhỏ cho mỗi tọa độ
        noisy_bbox = [
            bbox[0] + np.random.uniform(-noise_level, noise_level),
            bbox[1] + np.random.uniform(-noise_level, noise_level),
            bbox[2] + np.random.uniform(-noise_level, noise_level),
            bbox[3] + np.random.uniform(-noise_level, noise_level),
        ]
        noisy_full_text_bboxes.append(noisy_bbox)
    plot_bboxes(subplot, noisy_full_text_bboxes, color="black")

    plot_bboxes(subplot, predictions["characters"], color="blue")
    plot_bboxes(subplot, predictions["tails"], color="purple")

    for i, name in enumerate(predictions["character_names"]):
        char_bbox = predictions["characters"][i]
        x1, y1, x2, y2 = char_bbox
        subplot.text(
            x1, y1 - 2, name,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='blue', alpha=1, edgecolor='none'),
            color='white', fontsize=8
        )

    COLOURS = ["#b7ff51", "#f50a8f", "#4b13b6", "#ddaa34", "#bea2a2"]
    colour_index = 0
    character_cluster_labels = predictions["character_cluster_labels"]
    unique_label_sorted_by_frequency = sorted(
        list(set(character_cluster_labels)),
        key=lambda x: character_cluster_labels.count(x),
        reverse=True
    )
    for label in unique_label_sorted_by_frequency:
        root = None
        others = []
        for i in range(len(predictions["characters"])):
            if character_cluster_labels[i] == label:
                if root is None:
                    root = i
                else:
                    others.append(i)
        if colour_index >= len(COLOURS):
            random_colour = COLOURS[0]
            while random_colour in COLOURS:
                random_colour = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        else:
            random_colour = COLOURS[colour_index]
            colour_index += 1
        bbox_i = predictions["characters"][root]
        x1 = bbox_i[0] + (bbox_i[2] - bbox_i[0]) / 2
        y1 = bbox_i[1] + (bbox_i[3] - bbox_i[1]) / 2
        subplot.plot([x1], [y1], color=random_colour, marker="o", markersize=5)
        for j in others:
            bbox_j = predictions["characters"][j]
            x1 = bbox_i[0] + (bbox_i[2] - bbox_i[0]) / 2
            y1 = bbox_i[1] + (bbox_i[3] - bbox_i[1]) / 2
            x2 = bbox_j[0] + (bbox_j[2] - bbox_j[0]) / 2
            y2 = bbox_j[1] + (bbox_j[3] - bbox_j[1]) / 2
            subplot.plot([x1, x2], [y1, y2], color=random_colour, linewidth=2)
            subplot.plot([x2], [y2], color=random_colour, marker="o", markersize=5)

    for (i, j) in predictions["text_character_associations"]:
        bbox_i = predictions["texts"][i]
        bbox_j = predictions["characters"][j]
        if not predictions["is_essential_text"][i]:
            continue
        x1 = bbox_i[0] + (bbox_i[2] - bbox_i[0]) / 2
        y1 = bbox_i[1] + (bbox_i[3] - bbox_i[1]) / 2
        x2 = bbox_j[0] + (bbox_j[2] - bbox_j[0]) / 2
        y2 = bbox_j[1] + (bbox_j[3] - bbox_j[1]) / 2
        subplot.plot([x1, x2], [y1, y2], color="red", linewidth=2, linestyle="dashed")

    for (i, j) in predictions["text_tail_associations"]:
        bbox_i = predictions["texts"][i]
        bbox_j = predictions["tails"][j]
        x1 = bbox_i[0] + (bbox_i[2] - bbox_i[0]) / 2
        y1 = bbox_i[1] + (bbox_i[3] - bbox_i[1]) / 2
        x2 = bbox_j[0] + (bbox_j[2] - bbox_j[0]) / 2
        y2 = bbox_j[1] + (bbox_j[3] - bbox_j[1]) / 2
        subplot.plot([x1, x2], [y1, y2], color="purple", linewidth=2, linestyle="dashed")

    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches=None, pad_inches=0)

    figure.canvas.draw()
    image = np.array(figure.canvas.renderer._renderer)
    plt.close()
    return image

def sort_panels(rects):
    """
    Sort panels in reading order.
    Args:
        rects: list of tuples (x1, y1, x2, y2)
    Returns:
        list of indices of the panels in reading order
    """
    before_rects = convert_to_list_of_lists(rects)
    # slightly erode all rectangles initially to account for imperfect detections
    rects = [erode_rectangle(rect, 0.05) for rect in before_rects]
    G = nx.DiGraph()
    G.add_nodes_from(range(len(rects))) # Mỗi node là một panel (chỉ số 0..N-1).
    for i in range(len(rects)):
        for j in range(len(rects)):
            if i == j:
                continue
            if is_there_a_directed_edge(rects[i], rects[j]):
                G.add_edge(i, j, weight=get_distance(rects[i], rects[j]))
            else:
                G.add_edge(j, i, weight=get_distance(rects[i], rects[j]))
    while True:
        cycles = sorted(nx.simple_cycles(G))
        cycles = [cycle for cycle in cycles if len(cycle) > 1]
        if len(cycles) == 0:
            break
        cycle = cycles[0]
        edges = [e for e in zip(cycle, cycle[1:] + cycle[:1])]
        max_cyclic_edge = max(edges, key=lambda x: G.edges[x]["weight"])
        G.remove_edge(*max_cyclic_edge)
    return list(nx.topological_sort(G))

def is_strictly_above(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return y2A < y1B

def is_strictly_below(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return y2B < y1A

def is_strictly_left_of(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return x2A < x1B

def is_strictly_right_of(rectA, rectB):
    x1A, y1A, x2A, y2A = rectA
    x1B, y1B, x2B, y2B = rectB
    return x2B < x1A

def intersects(rectA, rectB):
    """
    Kiểm tra xem hai hình chữ nhật có giao nhau không.
    Args:
        rectA: hình chữ nhật a
        rectB: hình chữ nhật b
    Returns:
        bool: True nếu hai hình chữ nhật giao nhau, False nếu không.
    """
    return box(*rectA).intersects(box(*rectB))

def is_there_a_directed_edge(rectA: list, rectB: list) -> int:
    """
    Kiểm tra xem có cạnh định hướng từ a đến b không.
    Args:
        rectA: hình chữ nhật của panel a
        rectB: hình chữ nhật của panel b
    Returns:
        int: Trả về 1 nghĩa là có cạnh định hướng từ a → b (a đọc trước b), ngược lại trả về 0 (tức b → a).
    """
    minx_A, miny_A, maxx_A, maxy_A = rectA
    minx_B, miny_B, maxx_B, maxy_B = rectB
    centre_of_A = [minx_A + (maxx_A - minx_A) / 2, miny_A + (maxy_A - miny_A) / 2]
    centre_of_B = [minx_B + (maxx_B - minx_B) / 2, miny_B + (maxy_B - miny_B) / 2]
    if np.allclose(np.array(centre_of_A), np.array(centre_of_B)):
        return box(*rectA).area > (box(*rectB)).area # Tính tâm hai panel. Nếu tâm gần trùng nhau, ưu tiên đọc panel lớn hơn (đọc từ ngoài vào trong)
    copy_A = [rectA[0], rectA[1], rectA[2], rectA[3]]
    copy_B = [rectB[0], rectB[1], rectB[2], rectB[3]]
    while True:
        # Nếu panel A nằm trên panel B và không nằm bên trái panel B, thì có cạnh định hướng từ a → b
        if is_strictly_above(copy_A, copy_B) and not is_strictly_left_of(copy_A, copy_B):
            return 1
        # Nếu panel B nằm trên panel A và không nằm bên trái panel A, thì có cạnh định hướng từ b → a
        if is_strictly_above(copy_B, copy_A) and not is_strictly_left_of(copy_B, copy_A):
            return 0
        # Nếu panel A nằm bên phải panel B và không nằm dưới panel B, thì có cạnh định hướng từ a → b
        if is_strictly_right_of(copy_A, copy_B) and not is_strictly_below(copy_A, copy_B):
            return 1
        # Nếu panel B nằm bên phải panel A và không nằm dưới panel A, thì có cạnh định hướng từ b → a
        if is_strictly_right_of(copy_B, copy_A) and not is_strictly_below(copy_B, copy_A):
            return 0
        if is_strictly_below(copy_A, copy_B) and is_strictly_right_of(copy_A, copy_B):
            return use_cuts_to_determine_edge_from_a_to_b(rectA, rectB)
        if is_strictly_below(copy_B, copy_A) and is_strictly_right_of(copy_B, copy_A):
           return use_cuts_to_determine_edge_from_a_to_b(rectA, rectB)
        # Nếu panel A nằm bên trái panel B và không nằm trên panel B, thì có cạnh định hướng từ a → b
        if is_strictly_left_of(copy_A, copy_B) and not is_strictly_above(copy_A, copy_B):
            return 1
        # Nếu panel B nằm bên trái panel A và không nằm trên panel A, thì có cạnh định hướng từ b → a
        if is_strictly_left_of(copy_B, copy_A) and not is_strictly_above(copy_B, copy_A):
            return 0
        # otherwise they intersect
        #Nếu hai hình chữ nhật giao nhau (khó xác định quan hệ), thu nhỏ (erode) mỗi hình 5% rồi lặp lại đến khi xác định được quan hệ.
        copy_A = erode_rectangle(copy_A, 0.05)
        copy_B = erode_rectangle(copy_B, 0.05)
        return is_there_a_directed_edge(copy_A, copy_B)
    
def get_distance(rectA, rectB):
    """
    Tính khoảng cách giữa hai hình chữ nhật bằng phương pháp
    - Nếu hai hình chữ nhật chồng lấn hoặc chỉ chạm biên, kết quả là 0.
    - Nếu tách rời, trả về độ dài đoạn thẳng ngắn nhất nối hai hình.  
    Args:
        rectA: hình chữ nhật a
        rectB: hình chữ nhật b
    Returns:
        float: khoảng cách giữa hai hình chữ nhật
    """
    return box(rectA[0], rectA[1], rectA[2], rectA[3]).distance(box(rectB[0], rectB[1], rectB[2], rectB[3]))

def use_cuts_to_determine_edge_from_a_to_b(rectA, rectB):
    rects = deepcopy([rectA, rectB])
    while True:
        xmin, ymin, xmax, ymax = min(rects[0][0], rects[1][0]), min(rects[0][1], rects[1][1]), max(rects[0][2], rects[1][2]), max(rects[0][3], rects[1][3])
        rect_index = [i for i in range(len(rects)) if intersects(rects[i], [xmin, ymin, xmax, ymax])]
        rects_copy = [rect for rect in rects if intersects(rect, [xmin, ymin, xmax, ymax])]
        
        # try to split the panels using a "horizontal" lines
        overlapping_y_ranges = merge_overlapping_ranges([(y1, y2) for x1, y1, x2, y2 in rects_copy])
        panel_index_to_split = {}
        for split_index, (y1, y2) in enumerate(overlapping_y_ranges):
            for i, index in enumerate(rect_index):
                if y1 <= rects_copy[i][1] <= rects_copy[i][3] <= y2:
                    panel_index_to_split[index] = split_index
        
        if panel_index_to_split[0] != panel_index_to_split[1]:
            return panel_index_to_split[0] < panel_index_to_split[1]
        
        # try to split the panels using a "vertical" lines
        overlapping_x_ranges = merge_overlapping_ranges([(x1, x2) for x1, y1, x2, y2 in rects_copy])
        panel_index_to_split = {}
        for split_index, (x1, x2) in enumerate(overlapping_x_ranges[::-1]):
            for i, index in enumerate(rect_index):
                if x1 <= rects_copy[i][0] <= rects_copy[i][2] <= x2:
                    panel_index_to_split[index] = split_index
        if panel_index_to_split[0] != panel_index_to_split[1]:
            return panel_index_to_split[0] < panel_index_to_split[1]
        
        # otherwise, erode the rectangles and try again
        rects = [erode_rectangle(rect, 0.05) for rect in rects]

def erode_rectangle(bbox: list, erosion_factor: float) -> list:
    """
    co mỗi hình chữ nhật lại erosion_factor % để giảm nhiễu/chéo mép do detect chưa chuẩn, giúp quan hệ định hướng ổn định hơn.
    Args:
        bbox: list (x1, y1, x2, y2)
        erosion_factor: float
    Returns:
        list (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    if w < h:
        aspect_ratio = w / h
        erosion_factor_width = erosion_factor * aspect_ratio
        erosion_factor_height = erosion_factor
    else:
        aspect_ratio = h / w
        erosion_factor_width = erosion_factor
        erosion_factor_height = erosion_factor * aspect_ratio
    w = w - w * erosion_factor_width
    h = h - h * erosion_factor_height
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [x1, y1, x2, y2]

def merge_overlapping_ranges(ranges):
    """
    ranges: list of tuples (x1, x2)
    """
    if len(ranges) == 0:
        return []
    ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = []
    for i, r in enumerate(ranges):
        if i == 0:
            prev_x1, prev_x2 = r
            continue
        x1, x2 = r
        if x1 > prev_x2:
            merged_ranges.append((prev_x1, prev_x2))
            prev_x1, prev_x2 = x1, x2
        else:
            prev_x2 = max(prev_x2, x2)
    merged_ranges.append((prev_x1, prev_x2))
    return merged_ranges

def sort_text_boxes_in_reading_order(text_bboxes, sorted_panel_bboxes):
    text_bboxes = convert_to_list_of_lists(text_bboxes)
    sorted_panel_bboxes = convert_to_list_of_lists(sorted_panel_bboxes)

    if len(text_bboxes) == 0:
        return []

    def indices_of_same_elements(nums):
        groups = groupby(range(len(nums)), key=lambda i: nums[i])
        return [list(indices) for _, indices in groups]

    panel_id_for_text = get_text_to_panel_mapping(text_bboxes, sorted_panel_bboxes)
    indices_of_texts = list(range(len(text_bboxes)))
    indices_of_texts, panel_id_for_text = zip(*sorted(zip(indices_of_texts, panel_id_for_text), key=lambda x: x[1]))
    indices_of_texts = list(indices_of_texts)
    grouped_indices = indices_of_same_elements(panel_id_for_text)
    for group in grouped_indices:
        subset_of_text_indices = [indices_of_texts[i] for i in group]
        text_bboxes_of_subset = [text_bboxes[i] for i in subset_of_text_indices]
        sorted_subset_indices = sort_texts_within_panel(text_bboxes_of_subset)
        indices_of_texts[group[0] : group[-1] + 1] = [subset_of_text_indices[i] for i in sorted_subset_indices]
    return indices_of_texts

def get_text_to_panel_mapping(text_bboxes, sorted_panel_bboxes):
    text_to_panel_mapping = []
    for text_bbox in text_bboxes:
        shapely_text_polygon = box(*text_bbox)
        all_intersections = []
        all_distances = []
        if len(sorted_panel_bboxes) == 0:
            text_to_panel_mapping.append(-1)
            continue
        for j, annotation in enumerate(sorted_panel_bboxes):
            shapely_annotation_polygon = box(*annotation)
            if shapely_text_polygon.intersects(shapely_annotation_polygon):
                all_intersections.append((shapely_text_polygon.intersection(shapely_annotation_polygon).area, j))
            all_distances.append((shapely_text_polygon.distance(shapely_annotation_polygon), j))
        if len(all_intersections) == 0:
            text_to_panel_mapping.append(min(all_distances, key=lambda x: x[0])[1])
        else:
            text_to_panel_mapping.append(max(all_intersections, key=lambda x: x[0])[1])
    return text_to_panel_mapping

def sort_texts_within_panel(rects):
    smallest_y = float("inf")
    greatest_x = float("-inf")
    for i, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        smallest_y = min(smallest_y, y1)
        greatest_x = max(greatest_x, x2)
    
    reference_point = Point(greatest_x, smallest_y)

    polygons_and_index = []
    for i, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        polygons_and_index.append((box(x1,y1,x2,y2), i))
    # sort points by closest to reference point
    polygons_and_index = sorted(polygons_and_index, key=lambda x: reference_point.distance(x[0]))
    indices = [x[1] for x in polygons_and_index]
    return indices

def x1y1wh_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    return [x1, y1, x1 + w, y1 + h]

def x1y1x2y2_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def convert_to_list_of_lists(rects):
    """
    Convert a list of tuples (x1, y1, x2, y2) to a list of lists.
    Args:
        rects: list of tuples (x1, y1, x2, y2)
    Returns:
        list of lists (x1, y1, x2, y2)
    """
    if isinstance(rects, torch.Tensor):
        return rects.tolist()
    if isinstance(rects, np.ndarray):
        return rects.tolist()
    return [[a, b, c, d] for a, b, c, d in rects]