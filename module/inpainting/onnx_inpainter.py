import os
import numpy as np
import torch
import cv2
from typing import List, Tuple
import onnxruntime as ort
from settings import BASE_DIR
from module.config import InpainterConfig
from module.utils.generic import resize_keep_aspect
from .common import OfflineInpainter

class ONNXInpainter(OfflineInpainter):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://example.com/dummy-url-not-used.onnx',  # URL giả, không được sử dụng
            'hash': None,
            'file': '.',
        }
    }

    def __init__(self, *args, **kwargs):
        print(f"init ONNXInpainter")
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)
        self.session = None

    async def _load(self, device: str):
        # Tìm file ONNX - ưu tiên file trong thư mục models/inpainting
        onnx_path = os.path.join(BASE_DIR, 'models', 'inpainting', 'inpainting_lama_mpe.onnx')
        if not os.path.exists(onnx_path):
            # Fallback đến vị trí mặc định trong model_dir
            onnx_path = self._get_file_path('inpainting_lama_mpe.onnx')
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"Không tìm thấy file ONNX: {onnx_path}. Hãy chạy convert_inpainting_to_onnx.py trước.")
        
        # Tạo session ONNX
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.device = device
        self.logger.info(f"Đã tải model ONNX từ {onnx_path}")

    async def _unload(self):
        self.session = None

    async def _infer(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        """
        Thực hiện inpainting với model ONNX
        
        Args:
            image: Ảnh đầu vào
            mask: Mask đầu vào
            config: Cấu hình inpainting
            inpainting_size: Kích thước inpainting
            verbose: In thông tin chi tiết
            
        Returns:
            Ảnh đã được inpainting
        """
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]
        
        img_original = image.copy()
        
        height, width, c = image.shape
        if max(image.shape[0: 2]) > inpainting_size:
            image = resize_keep_aspect(image, inpainting_size)
            mask = resize_keep_aspect(mask, inpainting_size)
        
        pad_size = 8
        h, w, c = image.shape
        if h % pad_size != 0:
            new_h = (pad_size - (h % pad_size)) + h
        else:
            new_h = h
        if w % pad_size != 0:
            new_w = (pad_size - (w % pad_size)) + w
        else:
            new_w = w
        
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
        
        self.logger.info(f'Inpainting resolution: {new_w}x{new_h}')
        
        # Chuẩn bị dữ liệu đầu vào cho ONNX
        img_torch = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0  # HWC -> CHW
        img_torch = np.expand_dims(img_torch, axis=0)  # Add batch dimension
        
        mask_torch = np.expand_dims(mask, axis=0).astype(np.float32) / 255.0  # Add batch dimension
        mask_torch = np.expand_dims(mask_torch, axis=0)  # Add channel dimension
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        
        # Kết hợp ảnh và mask
        masked_img = img_torch * (1 - mask_torch)
        
        # Chuẩn bị input cho ONNX
        input_data = {
            'input_image': np.concatenate([masked_img, mask_torch], axis=1),  # Kết hợp ảnh đã mask và mask
            'input_mask': mask_torch
        }
        
        # Thực hiện inference
        outputs = self.session.run(None, input_data)
        img_inpainted = outputs[0][0]  # Lấy kết quả đầu tiên
        
        # Chuyển về định dạng numpy
        img_inpainted = np.transpose(img_inpainted, (1, 2, 0))  # CHW -> HWC
        img_inpainted = (img_inpainted * 255.0).astype(np.uint8)
        
        # Resize về kích thước gốc nếu cần
        if new_h != height or new_w != width:
            img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation = cv2.INTER_LINEAR)
        
        # Kết hợp kết quả với ảnh gốc
        ans = img_inpainted * mask_original + img_original * (1 - mask_original)
        return ans
