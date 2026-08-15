from typing import Optional

import numpy as np

from module.inpainting.common import CommonInpainter, OfflineInpainter
from module.inpainting.inpainting_lama_mpe import LamaMPEInpainter, LamaLargeInpainter
# Thêm import ONNXInpainter
from module.inpainting.onnx_inpainter import ONNXInpainter
from module.config import Inpainter, InpainterConfig
from utils.log import setup_logger

INPAINTERS = {
    Inpainter.lama_large: LamaLargeInpainter,
    Inpainter.lama_mpe: LamaMPEInpainter,
    Inpainter.onnx: ONNXInpainter,  # Thêm ONNXInpainter vào danh sách
}
inpainter_cache = {}
logger = setup_logger(__name__)


def get_inpainter(key: Inpainter, *args, **kwargs) -> CommonInpainter:
    if key not in INPAINTERS:
        raise ValueError(
            f'Could not find inpainter for: "{key}". Choose from the following: %s' % ','.join(INPAINTERS))
    if not inpainter_cache.get(key):
        inpainter = INPAINTERS[key]
        inpainter_cache[key] = inpainter(*args, **kwargs)
    return inpainter_cache[key]


async def prepare(inpainter_key: Inpainter, device: str = 'cpu'):
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.download()
        await inpainter.load(device)


async def dispatch(inpainter_key: Inpainter, image: np.ndarray, mask: np.ndarray, config: Optional[InpainterConfig], inpainting_size: int = 1024, device: str = 'cpu', verbose: bool = False) -> np.ndarray:
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.load(device)
    config = config or InpainterConfig()
    return await inpainter.inpaint(image, mask, config, inpainting_size, verbose)


async def unload(inpainter_key: Inpainter):
    inpainter = inpainter_cache.pop(inpainter_key, None)
    if inpainter and isinstance(inpainter, OfflineInpainter):
        await inpainter.unload()
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.warning(f"Error during inpainting unload", exc_info=True)
        pass
