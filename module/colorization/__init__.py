from PIL import Image

from .common import CommonColorizer, OfflineColorizer
from .manga_colorization_v2 import MangaColorizationV2
from ..config import Colorizer
from utils.log import setup_logger

COLORIZERS = {
    Colorizer.mc2: MangaColorizationV2,
}
colorizer_cache = {}
logger = setup_logger(__name__)


def get_colorizer(key: Colorizer, *args, **kwargs) -> CommonColorizer:
    if key not in COLORIZERS:
        raise ValueError(
            f'Could not find colorizer for: "{key}". Choose from the following: %s' % ','.join(COLORIZERS))
    if not colorizer_cache.get(key):
        upscaler = COLORIZERS[key]
        colorizer_cache[key] = upscaler(*args, **kwargs)
    return colorizer_cache[key]


async def prepare(key: Colorizer):
    upscaler = get_colorizer(key)
    if isinstance(upscaler, OfflineColorizer):
        await upscaler.download()


async def dispatch(key: Colorizer, device: str = 'cpu', **kwargs) -> Image.Image:
    colorizer = get_colorizer(key)
    if isinstance(colorizer, OfflineColorizer):
        await colorizer.load(device)
    return await colorizer.colorize(**kwargs)


async def unload(key: Colorizer):
    colorizer = colorizer_cache.pop(key, None)
    if colorizer and isinstance(colorizer, OfflineColorizer):
        await colorizer.unload()  # gọi _unload() bên trong
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.warning(f"Error during colorization unload", exc_info=True)
        pass
