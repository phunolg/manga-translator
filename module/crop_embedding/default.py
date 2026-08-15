from typing import List
from safetensors.torch import load_file
import torch
from transformers import ViTMAEModel

from settings import DEVICE
from utils.log import setup_logger

logger = setup_logger(__name__)


class CropEmbeddingService:
    """
    Mô hình được sử dụng để tạo ra các vector đặc trưng (embeddings) cho các vùng ảnh chứa nhân vật. Hiệu quả với dữ liệu hạn chế: Học được các đặc trưng tốt ngay cả khi không có nhiều dữ liệu có nhãn.
    """
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.device = DEVICE
        self.model_config = config
        
    @classmethod
    def from_safetensors(cls, model_config, processor, weight_path: str):
        model = ViTMAEModel(model_config)
        state = load_file(weight_path)
        missing, unexpected = model.load_state_dict(state, strict=True)
        model = model.to(DEVICE)
        return cls(model=model, processor=processor, config=model_config)

    @torch.no_grad()
    def predict(self, images, crop_bboxes, move_to_device_fn=None, mask_ratio=0.0, batch_size=256):
        if self.model is None:
            raise ValueError("Crop embedding model is not loaded")

        assert isinstance(
            crop_bboxes, List), "please provide a list of bboxes for each image to get embeddings for"

        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        # Đoạn code đang tạm thời thay đổi mask_ratio từ giá trị mặc định sang giá trị được chỉ định trong tham số
        old_mask_ratio = self.model.embeddings.config.mask_ratio
        self.model.embeddings.config.mask_ratio = mask_ratio

        crops_per_image = []
        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]
        for image, bboxes, num_crops in zip(images, crop_bboxes, num_crops_per_batch):
            crops = self.processor.crop_image(image, bboxes)
            assert len(crops) == num_crops
            crops_per_image.extend(crops)

        if len(crops_per_image) == 0:
            return [move_to_device_fn(torch.zeros(0, self.model_config.hidden_size)) for _ in crop_bboxes]

        crops_per_image = self.processor.preprocess_inputs_for_crop_embeddings(
            crops_per_image)
        crops_per_image = move_to_device_fn(crops_per_image)

        # process the crops in batches to avoid OOM
        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            crops = crops_per_image[i:i+batch_size]
            embeddings_per_batch = self.model(crops).last_hidden_state[:, 0]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []
        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            embeddings = embeddings[num_crops:]

        # restore the mask ratio to the default
        self.model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    def unload(self):
        try:
            if self.model is not None:
                del self.model
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
