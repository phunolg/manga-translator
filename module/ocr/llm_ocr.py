import os
from PIL import Image
import cv2
from safetensors.torch import load_file
import torch
import asyncio
from typing import List, Optional

from module.llm import get_llm
from module.ocr.prompt import ocr_japanese_prompt_system
from module.translator.prompts import ocr_prompt_system
from module.llm.constants import LLM_MODEL
from utils.log import setup_logger
from module.ocr.base import OCR
from processing_magiv2 import Magiv2Processor
logger = setup_logger(__name__)

class LLM_OCR(OCR):
    def __init__(self, processor: Magiv2Processor):
        super().__init__(processor)
        self.llm = get_llm(LLM_MODEL.gemma3)
        
    async def ocr_per_page(self, image, bboxes):
        crops = self.processor.crop_image(image, bboxes)
        crops = [self._preprocess_image_for_ocr(crop, method='enhanced') for crop in crops]
        texts = await asyncio.gather(*[
            self.llm.get_answer(
                question="Hãy trích xuất văn bản trong ảnh này",
                prompt_system=ocr_japanese_prompt_system,
                image=crop,
                temperature=0.1,
                max_tokens=200
            ) for crop in crops
        ])
        self._save_image_ocr([crops])
        return texts

    async def predict(self, images: List, crop_bboxes: List[List[List[int]]], **kwargs):
        return await asyncio.gather(*[self.ocr_per_page(image, bboxes) for image, bboxes in zip(images, crop_bboxes)])

    
    def unload(self):
        pass

