from typing import Optional
from safetensors.torch import load_file
import torch
from transformers import ConditionalDetrModel
from settings import DEVICE
from utils.model import Model

class MicrosoftConditionalDetrResnet50Detector(Model):
    def __init__(self, model, processor):
        super().__init__(model, processor)
        self.model = model
        self.processor = processor
        self.device = DEVICE

    @classmethod
    def from_safetensors(cls, processor, model_config, weight_path: str):
        model = ConditionalDetrModel(model_config)  # khởi tạo kiến trúc
        state = load_file(weight_path)  # đọc safetensors
        missing, unexpected = model.load_state_dict(state, strict=True)
        model = model.to(DEVICE)
        return cls(model=model, processor=processor)   
    
    @torch.no_grad()
    def predict(
            self, 
            pixel_values: torch.FloatTensor,
            pixel_mask: Optional[torch.LongTensor] = None
    ):
        """
        Mô hình này phát hiện các đối tượng (panel, text, character, tail) trong hình ảnh
        """
        return self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
