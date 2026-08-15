import os
from PIL import Image
import cv2
from safetensors.torch import load_file
import torch
import asyncio
from typing import List, Optional
import numpy as np
from PIL import ImageEnhance, ImageFilter

from transformers import VisionEncoderDecoderModel
from module.llm import get_llm
from module.ocr.base import OCR
from module.translator.prompts import ocr_prompt_system
from utils.log import setup_logger
from settings import BASE_DIR, DEVICE
from tqdm import tqdm
from module.llm.constants import LLM_MODEL
logger = setup_logger(__name__)

class OCRService(OCR):
    def __init__(self, ocr_model, processor):
        super().__init__(processor)
        self.ocr_model = ocr_model
        self.device = DEVICE

    @classmethod
    def from_safetensors(cls, processor, model_config, weight_path: str):
        model = VisionEncoderDecoderModel(model_config)  # khởi tạo kiến trúc
        state = load_file(weight_path)  # đọc safetensors
        missing, unexpected = model.load_state_dict(state, strict=True)
        model = model.to(DEVICE)
        return cls(ocr_model=model, processor=processor)

    @classmethod
    def from_pretrained(cls, processor, model_id_or_path: str, dtype: str = "auto"):
        torch_dtype = torch.float16 if dtype == "fp16" else None
        model = VisionEncoderDecoderModel.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        return cls(ocr_model=model, processor=processor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    def unload(self):
        try:
            if self.ocr_model is not None:
                del self.ocr_model
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    @torch.no_grad()
    async def predict(self, images: List, crop_bboxes: List[List[List[int]]], move_to_device_fn=None,
                      batch_size=32, max_new_tokens=64, llm_batch=8):
        assert self.ocr_model is not None, "OCR model is required"
        logger.info("=========== OCRService.predict ===========")

        output_dir = os.path.join(BASE_DIR, "debug", "ocr_bbox")
        os.makedirs(output_dir, exist_ok=True)

        crops_info = []
        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]

        for i, (image, bboxes, num_crops) in enumerate(zip(images, crop_bboxes, num_crops_per_batch)):
            crops = self.processor.crop_image(image, bboxes)
            crops = [self._preprocess_image_for_ocr(crop, method='denoise') for crop in crops]
            assert len(crops) == num_crops
            for idx, crop in enumerate(crops):
                area = crop.shape[0] * crop.shape[1]
                use_llm = area < 40000
                crops_info.append({"image_idx": i, "crop_idx": idx, "crop": crop, "area": area, "use_llm": use_llm})
                cv2.imwrite(os.path.join(output_dir, f"crop_{i}_{idx}.png"), crop)

        crops_for_ocr, crops_for_ocr_indices = [], []
        crops_for_llm, crops_for_llm_indices = [], []
        for idx, info in enumerate(crops_info):
            if info["use_llm"]:
                crops_for_llm.append(info["crop"])
                crops_for_llm_indices.append(idx)
            else:
                crops_for_ocr.append(info["crop"])
                crops_for_ocr_indices.append(idx)

        ocr_results = [""] * len(crops_info)

        if crops_for_ocr:
            crops_tensor = self.processor.preprocess_inputs_for_ocr(crops_for_ocr)
            if move_to_device_fn:
                crops_tensor = move_to_device_fn(crops_tensor)
            else:
                crops_tensor = crops_tensor.to(self.device)

            all_texts = []
          
            it = tqdm(range(0, len(crops_tensor), batch_size))
            for i in it:
                batch = crops_tensor[i:i+batch_size]
                generated_ids = self.ocr_model.generate(batch, max_new_tokens=max_new_tokens)
                generated_texts = self.processor.postprocess_ocr_tokens(generated_ids)
                all_texts.extend(generated_texts)
            for idx, text in zip(crops_for_ocr_indices, all_texts):
                ocr_results[idx] = (text or "").replace("\n", "")

        if crops_for_llm:
            llm = get_llm(LLM_MODEL.gemma3)
            enhanced_crops = []
            for crop_idx, crop in enumerate(crops_for_llm):
                idx = crops_for_llm_indices[crop_idx]
                enhanced_crops.append(crop)
                cv2.imwrite(os.path.join(output_dir, f"crop_llm_{idx}.png"), crop)

            logger.info(f"Processing {len(crops_for_llm)} small crops with LLM in batches of {llm_batch}")
            batch_results = []
            for i in range(0, len(enhanced_crops), llm_batch):
                batch_crops = enhanced_crops[i:i+llm_batch]
                batch_questions = ["Hãy trích xuất văn bản trong ảnh này"] * len(batch_crops)
                answers = await asyncio.gather(*[
                    llm.get_answer(
                        question=q,
                        prompt_system=ocr_prompt_system,
                        image=c,
                        temperature=0.1,
                        max_tokens=200
                    ) for q, c in zip(batch_questions, batch_crops)
                ])
                batch_results.extend(answers)

            for i, idx in enumerate(crops_for_llm_indices):
                texts = batch_results[i]
                if isinstance(texts, list) and texts:
                    ocr_results[idx] = " ".join(texts)
                elif isinstance(texts, str):
                    ocr_results[idx] = texts.strip()
                else:
                    ocr_results[idx] = ""

        texts_for_images, start = [], 0
        for num_crops in num_crops_per_batch:
            texts_for_images.append(ocr_results[start:start+num_crops])
            start += num_crops
        return texts_for_images

