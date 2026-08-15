import numpy as np
import cv2

def create_oval_mask(width: int, height: int, padding: int = 5) -> np.ndarray:
    """
    Tạo mask hình oval
    
    Args:
        width: Chiều rộng mask
        height: Chiều cao mask
        padding: Khoảng cách lề
        
    Returns:
        mask: Mask hình oval
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(width // 2 - padding, 1), max(height // 2 - padding, 1))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask