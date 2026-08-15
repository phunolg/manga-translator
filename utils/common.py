import os
from typing import List
import numpy as np
import torch

from settings import DEVICE
url = "http://127.0.0.1:8000/predict"  # Đảm bảo port đúng
from pprint import pprint
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

def test_predict(image_path):
    # Mở các files
    with open(image_path, "rb") as f1, \
         open("data/test2/chấp sư đại nhân.png", "rb") as f2, \
         open("data/test2/dư lạc.png", "rb") as f3, \
         open("data/test2/tuyết ly.png", "rb") as f4, \
         open("data/test2/thuộc hạ.png", "rb") as f5, \
         open("data/test2/thuộc hạ 2.png", "rb") as f6, \
         open("data/test2/Shizuka.png", "rb") as f7:
        
        # Tạo files theo đúng định dạng FastAPI mong đợi
        files = [
            ("chapter_pages", ("12.jpg", f1, "image/jpeg")),
        ]
        
        # Tạo files cho character_images riêng biệt
        character_files = [
            ("character_images", ("chấp sư đại nhân.png", f2, "image/png")),
            ("character_images", ("dư lạc.png", f3, "image/png")),
            ("character_images", ("tuyết ly.png", f4, "image/png")),
            ("character_images", ("thuộc hạ.png", f5, "image/png")),
            ("character_images", ("thuộc hạ 2.png", f6, "image/png")),
            ("character_images", ("Shizuka.png", f7, "image/png")),
        ]
        
        # Kết hợp tất cả files
        all_files = files + character_files
        
        # Dữ liệu form
        data = {
            "character_names": ["Chấp Sư Đại Nhân", "Dư Lạc", "Tuyết Ly", "Thuộc Hạ", "Dư Y Ba", "Shizuka"],
            "introduction": """
                Nhân vật & cách xưng hô:
                1. Dư Lạc (nam, trưởng thành, uy nghiêm):
                    - Xưng: "ta"
                    - Gọi Tuyết Ly: "cô"
                    - Gọi Dư Y Ba: "cô nương"
                    - Gọi người khác: "ngươi"
             
                2. Tuyết Ly (nữ, trẻ, kính trọng Dư Lạc):
                    - Xưng: "ta"
                    - Gọi Dư Lạc: "công tử, ngài"
                    - Gọi người khác: "ngươi"
                3. Chấp Sư Đại Nhân (nam, quyền lực cao):
                    - Xưng: "ta"
                    - Gọi mọi người: "ngươi"
                4. Dư Y Ba, Thuộc hạ (dưới quyền Chấp Sư Đại Nhân):
                    - Xưng: "ta"
                    - Gọi Chấp Sư Đại Nhân: "ngài"
                    - Gọi người khác: "ngươi"
                    - Gọi Dư Lạc: "công tử"
                    - Xưng với Dư Lạc: "tiện thiếp"
            """,
            "response_type": "image",
            "name_mapping": "Yu le:Dư Lạc,Red water town:Trấn Hồng Thủy, Yu Yibo: Dư Y Ba, Leader: Chấp Sư Đại Nhân",
        }
        
        # Chuyển đổi data thành danh sách các tuple
        form_data = []
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    form_data.append((key, item))
            else:
                form_data.append((key, value))
        
        # Gửi request
        response = requests.post(url, files=all_files, data=form_data)
        
        if response.status_code == 200:
            translated_image = Image.open(io.BytesIO(response.content))
            original_image = Image.open(image_path)
            manual_translated = Image.open(image_path.replace("Raw", "Team dịch"))
            
            # Lưu ảnh vào biến để sử dụng trong hàm click
            images = {
                'original': original_image,
                'manual': manual_translated,
                'translated': translated_image
            }
            
            # Hiển thị ảnh gốc và ảnh dịch song song
            fig = plt.figure(figsize=(14, 24))  # Tăng kích thước hiển thị
            
            ax1 = plt.subplot(1, 3, 1)
            ax1.set_title("Ảnh gốc (click để phóng to)")
            ax1.imshow(original_image)
            ax1.axis('off')
            
            ax2 = plt.subplot(1, 3, 2)
            ax2.set_title("Ảnh dịch tay (click để phóng to)")
            ax2.imshow(manual_translated)
            ax2.axis('off')

            ax3 = plt.subplot(1, 3, 3)
            ax3.set_title(f"Ảnh đã dịch (click để phóng to)")
            ax3.imshow(translated_image)
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Lỗi: {response.status_code}")
            print(response.text)


def move_to_device(inputs, device = DEVICE):
    """
    Di chuyển tensor, list, tuple, numpy array, hoặc dict sang device đã được định nghĩa trong settings.py
    Args:
        inputs: tensor, list, tuple, numpy array, hoặc dict
        device: device đã được định nghĩa trong settings.py
    Returns:
        inputs: tensor, list, tuple, numpy array, hoặc dict đã được di chuyển sang device đã được định nghĩa trong settings.py
    """
    if hasattr(inputs, "keys"):
        return {k: move_to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, list):
        return [move_to_device(v, device) for v in inputs]
    elif isinstance(inputs, tuple):
        return tuple([move_to_device(v, device) for v in inputs])
    elif isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs).to(device)
    else:
        return inputs.to(device)

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    """
    Phân tích đường dẫn font, nếu đường dẫn không tồn tại thì sẽ sử dụng đường dẫn mặc định.
    Args:
        path: đường dẫn font
        default: đường dẫn mặc định
    Returns:
        parsed: danh sách đường dẫn font
    """
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed