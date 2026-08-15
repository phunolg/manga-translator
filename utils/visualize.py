import os
from typing import List, Optional

import cv2
import numpy as np


def save_polygons_image(
    image: np.ndarray,
    polygons: List[np.ndarray],
    out_path: str,
    edge_color: tuple[int, int, int] = (0, 255, 0),
    fill_color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.25,
    thickness: int = 2,
    labels: Optional[List[str]] = None,
) -> str:
    """
    Lưu ảnh với các đa giác (polygons) được vẽ đè lên để trực quan hóa.

    Args:
        image: ảnh gốc (BGR)
        polygons: danh sách đa giác, mỗi phần tử là mảng (4,2) hoặc (N,2)
        out_path: đường dẫn file để lưu
        edge_color: màu viền BGR
        fill_color: màu fill BGR
        alpha: độ trong suốt khi fill (0..1)
        thickness: độ dày viền
        labels: nhãn hiển thị cho từng polygon (tùy chọn)
    Returns:
        Đường dẫn file đã lưu
    """
    if image is None:
        raise ValueError("image is None")

    overlay = image.copy()
    h, w = image.shape[:2]

    # Vẽ fill trên overlay
    for poly, _ in polygons:
        if poly is None:
            continue
        pts = np.asarray(poly).reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [pts], fill_color)

    # Pha trộn overlay và ảnh gốc
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Vẽ viền và nhãn
    for idx, poly in enumerate(polygons):
        if poly is None:
            continue
        pts = np.asarray(poly).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(blended, [pts], isClosed=True, color=edge_color,
                      thickness=thickness, lineType=cv2.LINE_AA)

        if labels and idx < len(labels):
            poly_arr = np.asarray(poly).reshape(-1, 2)
            cx, cy = np.mean(poly_arr[:, 0]), np.mean(poly_arr[:, 1])
            tx, ty = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            cv2.putText(
                blended,
                labels[idx],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                edge_color,
                1,
                cv2.LINE_AA,
            )

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, blended)
    return out_path

import matplotlib.pyplot as plt
def draw_detections(image, result, save_path=None):
    """
    Vẽ debug với matplotlib - hiển thị đúng kích thước ảnh và các bounding box
    """
    # Tính toán kích thước figure dựa trên kích thước ảnh
    height, width = image.shape[:2]
    fig_width = max(8, width / 100 )  # Tối thiểu 8 inches
    fig_height = max(6, height / 100)  # Tối thiểu 6 inches
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(image)
    
    # Màu sắc
    colors = {
        'panel': 'red',
        'character': 'green', 
        'text': 'lightblue',
        'essential_text': 'darkblue',  # Màu khác cho essential text
        'tail': 'orange',
        'tail_speaking': 'orange',     # Màu cho tail speaking
        'tail_thinking': 'purple'      # Màu cho tail thinking
    }
    
    # Vẽ panels
    for i, bbox in enumerate(result['panels']):
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor=colors['panel'], 
                           facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"panel_{i}", color=colors['panel'], fontsize=20, weight='bold')
    
    # Vẽ characters
    for i, bbox in enumerate(result['characters']):
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=3, edgecolor=colors['character'],
                           facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"character_{i}", color=colors['character'], fontsize=20, weight='bold')
    
    # Vẽ texts - phân biệt essential và non-essential, hiển thị speech type
    is_essential = result.get('is_essential_text', [])
    text_speech_types = result.get('text_speech_type', [])
    
    for i, bbox in enumerate(result['texts']):
        x1, y1, x2, y2 = bbox
        # Chọn màu dựa trên is_essential_text
        if i < len(is_essential) and is_essential[i]:
            color = colors['essential_text']
            label = f"text_{i}*"  # Dấu * cho essential text
        else:
            color = colors['text']
            label = f"text_{i}"
        
        # Thêm speech type vào label nếu có
        if i < len(text_speech_types) and text_speech_types[i] in ['speaking', 'thinking']:
            label += f"_{text_speech_types[i]}"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=3, edgecolor=color,
                           facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, label, color=color, fontsize=20, weight='bold')
    
    # Vẽ tails - phân biệt speaking và thinking
    text_tail_associations = result.get('text_tail_associations', [])
    text_speech_types = result.get('text_speech_type', [])
    
    # Tạo mapping từ tail_idx đến speech_type
    tail_to_speech_type = {}
    for text_idx, tail_idx in text_tail_associations:
        if text_idx < len(text_speech_types):
            tail_to_speech_type[tail_idx] = text_speech_types[text_idx]
    
    for i, bbox in enumerate(result['tails']):
        x1, y1, x2, y2 = bbox
        
        # Xác định speech type cho tail này
        speech_type = tail_to_speech_type.get(i, 'unknown')
        if speech_type == 'thinking':
            color = colors['tail_thinking']
            label = f"tail_{i}_thinking"
        elif speech_type == 'speaking':
            color = colors['tail_speaking'] 
            label = f"tail_{i}_speaking"
        else:
            color = colors['tail']
            label = f"tail_{i}"
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=3, edgecolor=color,
                           facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, label, color=color, fontsize=20, weight='bold')
    
    # Vẽ liên kết text-character
    for text_idx, char_idx in result['text_character_associations']:
        text_bbox = result['texts'][text_idx]
        char_bbox = result['characters'][char_idx]
        
        text_center = ((text_bbox[0] + text_bbox[2])/2, (text_bbox[1] + text_bbox[3])/2)
        char_center = ((char_bbox[0] + char_bbox[2])/2, (char_bbox[1] + char_bbox[3])/2)
        
        ax.plot([text_center[0], char_center[0]], 
                [text_center[1], char_center[1]], 
                color='magenta', linewidth=3, alpha=0.7)
    
    # Thêm padding xung quanh ảnh
    padding = 20  # Khoảng cách padding tính bằng pixel
    
    # Đặt giới hạn với padding
    ax.set_xlim(-padding, width + padding)
    ax.set_ylim(height + padding, -padding)  # Đảo ngược trục y để hiển thị đúng
    
    ax.set_title("Debug: Bounding Boxes Detection\n* = Essential Text | Orange = Speaking Tail | Purple = Thinking Tail", fontsize=10)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        print(f"Đã lưu debug image tại: {save_path}")
    
    plt.show()
    return fig