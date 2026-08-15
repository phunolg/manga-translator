#!/usr/bin/env python3
"""
Chương trình test nhỏ để kiểm tra module detection
"""

import os
import sys
import argparse
import asyncio
import numpy as np
import cv2
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from module.detection import dispatch, prepare, unload
from module.config import Detector, DetectorConfig
from module.utils import TextBlock


def draw_textlines(image, textlines, color=(0, 255, 0), thickness=2):
    """Vẽ các đường viền textline lên ảnh"""
    img_with_boxes = image.copy()
    for txtln in textlines:
        cv2.polylines(img_with_boxes, [txtln.pts], True, color=color, thickness=thickness)
    return img_with_boxes


async def test_detection(image_path, output_dir, config=None, device='cpu', verbose=True):
    """
    Thực hiện detection trên ảnh và lưu kết quả
    
    Args:
        image_path: Đường dẫn đến ảnh đầu vào
        output_dir: Thư mục đầu ra để lưu kết quả
        config: Cấu hình detector, mặc định là None (sử dụng cấu hình mặc định)
        device: Thiết bị để chạy model ('cpu' hoặc 'cuda')
        verbose: Hiển thị thông tin chi tiết
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Chuyển đổi BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Sử dụng cấu hình mặc định nếu không được cung cấp
    if config is None:
        config = DetectorConfig()
    
    # Chuẩn bị detector
    await prepare(config.detector)
    
    # Thực hiện detection
    print(f"Đang thực hiện detection trên ảnh {image_path}...")
    textlines, raw_mask, mask = await dispatch(
        detector_key=config.detector,
        image=image_rgb,
        detect_size=config.detection_size,
        text_threshold=config.text_threshold,
        box_threshold=config.box_threshold,
        unclip_ratio=config.unclip_ratio,
        invert=config.det_invert,
        gamma_correct=config.det_gamma_correct,
        rotate=config.det_rotate,
        auto_rotate=config.det_auto_rotate,
        device=device,
        verbose=verbose
    )
    
    # Lấy tên file từ đường dẫn
    image_name = Path(image_path).stem
    
    # Vẽ các đường viền textline lên ảnh
    img_with_boxes = draw_textlines(image_rgb, textlines)
    
    # Lưu ảnh với các đường viền
    output_path = os.path.join(output_dir, f"{image_name}_detected.png")
    cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    # Lưu mask nếu có
    if raw_mask is not None:
        mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
        cv2.imwrite(mask_path, raw_mask)
    
    # Giải phóng tài nguyên
    await unload(config.detector)
    
    print(f"Đã phát hiện {len(textlines)} vùng văn bản")
    print(f"Kết quả đã được lưu tại {output_path}")
    
    if raw_mask is not None:
        print(f"Mask đã được lưu tại {mask_path}")
    
    return textlines, raw_mask, mask


def main():
    """Hàm chính để chạy chương trình"""
    parser = argparse.ArgumentParser(description='Test detection module')
    parser.add_argument('--image', '-i', type=str, required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--output', '-o', type=str, default='output', help='Thư mục đầu ra để lưu kết quả')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Thiết bị để chạy model (cpu hoặc cuda)')
    parser.add_argument('--detection-size', type=int, default=2048, help='Kích thước ảnh dùng cho detection')
    parser.add_argument('--text-threshold', type=float, default=0.5, help='Ngưỡng cho text detection')
    parser.add_argument('--box-threshold', type=float, default=0.7, help='Ngưỡng cho bbox generation')
    parser.add_argument('--unclip-ratio', type=float, default=2.3, help='Tỷ lệ mở rộng text skeleton')
    parser.add_argument('--invert', action='store_true', help='Đảo ngược màu ảnh cho detection')
    parser.add_argument('--gamma-correct', action='store_true', help='Áp dụng gamma correction cho detection')
    parser.add_argument('--rotate', action='store_true', help='Xoay ảnh cho detection')
    parser.add_argument('--auto-rotate', action='store_true', help='Tự động xoay ảnh cho detection')
    
    args = parser.parse_args()
    
    # Tạo cấu hình detector
    config = DetectorConfig(
        detector=Detector.default,
        detection_size=args.detection_size,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        unclip_ratio=args.unclip_ratio,
        det_invert=args.invert,
        det_gamma_correct=args.gamma_correct,
        det_rotate=args.rotate,
        det_auto_rotate=args.auto_rotate
    )
    
    # Chạy detection
    asyncio.run(test_detection(args.image, args.output, config, args.device))


if __name__ == '__main__':
    main()
