import uuid
from settings import BASE_DIR
import os
import cv2
import glob
output_dir = os.path.join(BASE_DIR, "module", "tail_classification", "cropped_bubbles")

def crop_to_image(folder_path):
    label_paths = glob.glob(os.path.join(folder_path, "labels", "*.txt"))
    for label_path in label_paths:
        with open(label_path, "r") as f:
            lines = f.readlines()
        image_name = label_path.split("/")[-1].rstrip(".txt") + ".jpg"
        image_path = os.path.join(folder_path, "images", image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            label_id, x_norm, y_norm, w_norm, h_norm = parts
            label_id = int(label_id)
            x_norm, y_norm, w_norm, h_norm = map(float, [x_norm, y_norm, w_norm, h_norm])
            
            # Chuyển đổi từ normalized coordinates sang pixel coordinates
            img_h, img_w = image.shape[:2]
            x_center = x_norm * img_w
            y_center = y_norm * img_h
            width = w_norm * img_w
            height = h_norm * img_h
            
            # Chuyển đổi từ center coordinates sang top-left coordinates
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)
            w = int(width)
            h = int(height)
            
            # Đảm bảo tọa độ không vượt quá kích thước ảnh
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            
            match label_id:
                case 0:
                    label = "speaking"
                case 1:
                    label = "thinking"
                case _:
                    label = "unknown"
            os.makedirs(os.path.join(output_dir, label), exist_ok=True)
            unique_id = str(uuid.uuid4())
            save_path = os.path.join(output_dir, label, f"{unique_id}.jpg")
            crop_image = image[y:y+h, x:x+w]
            cv2.imwrite(save_path, crop_image)
    

if __name__ == "__main__":
    crop_to_image(os.path.join(BASE_DIR, "module", "tail_classification", "dataset", "train"))
    crop_to_image(os.path.join(BASE_DIR, "module", "tail_classification", "dataset", "valid"))
    crop_to_image(os.path.join(BASE_DIR, "module", "tail_classification", "dataset", "test"))
