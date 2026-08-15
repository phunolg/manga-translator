import json
from typing import List
import cv2
import numpy as np
import httpx
def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

async def get_color_name(rgb: List[int]) -> str:
        try:
            # TODO: Maybe replace with offline alternative
            url = f'https://www.thecolorapi.com/id?format=json&rgb={rgb[0]},{rgb[1]},{rgb[2]}'
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('name', {}).get('value', 'Unnamed')
            else:
                return 'Unnamed'
        except Exception:
            return 'Unnamed'


def color_difference(rgb1: List, rgb2: List) -> float:
    # https://en.wikipedia.org/wiki/Color_difference#CIE76
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32) - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff[..., 0] *= 0.392
    diff = np.linalg.norm(diff, axis=2) 
    return diff.item()


def fg_bg_compare(fg, bg):
    """
    So sánh màu sắc foreground và background, nếu chênh lệch ít hơn 30 thì sẽ đổi màu background thành màu trắng hoặc đen tùy thuộc vào màu foreground.
    Args:
        fg: màu sắc foreground
        bg: màu sắc background
    Returns:
        fg: màu sắc foreground
        bg: màu sắc background
    """
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg