from abc import ABC, abstractmethod
import os
from typing import List
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch

from processing_magiv2 import Magiv2Processor
from settings import BASE_DIR, DEBUG

class OCR(ABC):
    def __init__(self, processor: Magiv2Processor):
        self.processor = processor
        
    def _save_image_ocr(self, images_batch: List[List[np.ndarray]]):
        folder_path = os.path.join(BASE_DIR, "debug", "ocr")
        os.makedirs(folder_path, exist_ok=True)
        if DEBUG:
            for batch_idx, images in enumerate(images_batch):
                for image_idx, image in enumerate(images):
                    cv2.imwrite(os.path.join(folder_path, f"{batch_idx}_{image_idx}.png"), image)
                    
    def _preprocess_image_for_ocr(self, image, method='enhanced'):
        """
        Cải thiện ảnh trước khi OCR
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        if method == 'enhanced':
            # Tăng độ tương phản
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Tăng độ sắc nét
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Điều chỉnh độ sáng
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
        elif method == 'opencv':
            # Chuyển sang OpenCV
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Chuyển sang grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Áp dụng Gaussian blur để làm mịn
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Tăng độ tương phản bằng CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            
            # Chuyển lại sang PIL
            pil_image = Image.fromarray(enhanced)
            
        elif method == 'denoise':
            # Giảm nhiễu
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            
            # Làm sắc nét
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
        return np.array(pil_image.convert("RGB"))
        
    @abstractmethod
    async def predict(self, images: List, crop_bboxes: List[List[List[int]]], **kwargs: dict) -> List[str]:
        pass
    