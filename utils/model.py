from abc import ABC, abstractmethod

from safetensors.torch import load_file
import torch

from settings import DEVICE

class Model(ABC):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = DEVICE
        
    @classmethod
    def from_safetensors(cls, processor, model_config, weight_path: str):
        pass
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    def unload(self):
        try:
            model_ref = getattr(self, "model", None)
            if model_ref is not None:
                del self.model
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @abstractmethod
    @torch.no_grad()
    def predict(self, *args, **kwargs):
        pass
    
    
