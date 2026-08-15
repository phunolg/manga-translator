from contextlib import ExitStack
from pathlib import Path
import mimetypes
import time
from typing import List
import os
import requests
from pprint import pprint
from evaluation_metrics import evaluate_from_test_file
from settings import BASE_DIR
import glob


URL = "http://localhost:8000/main/get-transcript"

def run_predict(
    CHARACTER_DETECTION_THRESHOLD = 0.2,
    TEXT_CLASSIFICATION_THRESHOLD = 0.01,
    ETA_FOR_CHARACTER_NAMING = 1.0,
    TEXT_CHARACTER_MATCHING_THRESHOLD = 0.35,
    CHARACTER_CHARACTER_MATCHING_THRESHOLD = 0.55,
):
    new_settings = f"""
CHARACTER_DETECTION_THRESHOLD = {CHARACTER_DETECTION_THRESHOLD}
TEXT_CLASSIFICATION_THRESHOLD = {TEXT_CLASSIFICATION_THRESHOLD}
ETA_FOR_CHARACTER_NAMING = {ETA_FOR_CHARACTER_NAMING}
TEXT_CHARACTER_MATCHING_THRESHOLD = {TEXT_CHARACTER_MATCHING_THRESHOLD}
CHARACTER_CHARACTER_MATCHING_THRESHOLD = {CHARACTER_CHARACTER_MATCHING_THRESHOLD}
"""
    with open("settings.py", "r") as f:
        lines = f.readlines()
    # Giữ lại các dòng không phải là các biến trên
    with open("settings.py", "w") as f:
        for line in lines:
            if not line.strip().startswith(("CHARACTER_DETECTION_THRESHOLD", "TEXT_CLASSIFICATION_THRESHOLD", "ETA_FOR_CHARACTER_NAMING", "TEXT_CHARACTER_MATCHING_THRESHOLD", "CHARACTER_CHARACTER_MATCHING_THRESHOLD")):
                f.write(line)
        # Ghi đè lại phần tham số
        f.write(new_settings)
    print(f"New settings: {new_settings}")
    print("Restarting server...")
    os.system("pkill -f 'uvicorn main:app'")   # Dừng server
    os.system("nohup uvicorn main:app &")      # Khởi động lại server
    time.sleep(10)
    print("Server restarted")
    def guess_mime(path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or "application/octet-stream"

    def test_predict(
        image_paths: List[str],
        manga_name: str,
        chapter_name: str,
    ):
        """
        Gửi nhiều trang + ảnh nhân vật + danh sách tên nhân vật.

        Params:
        image_paths: danh sách file ảnh trang truyện.
        character_image_paths: danh sách file ảnh nhân vật (nếu None dùng DEFAULT_CHARACTER_IMAGES).
        character_names: nếu cung cấp sẽ dùng trực tiếp. Nếu None -> suy từ character_image_paths.
        auto_lower_character_names: nếu True sẽ chuyển tên về lowercase khi auto suy ra.
        url: endpoint API.
        """
        if not image_paths:
            raise ValueError("image_paths rỗng.")



        data_base = {
            "story_name": manga_name,
            "chapter_name": chapter_name,
        }

        with ExitStack() as stack:
            # Mở trang truyện
            chapter_file_objs = []
            for p in image_paths:
                f = stack.enter_context(Path(p).open("rb"))
                chapter_file_objs.append(
                    ("chapter_pages", (os.path.basename(p), f, guess_mime(p)))
                )


            files = chapter_file_objs 

            # Chuyển dict data -> list tuple
            form_data = []
            for k, v in data_base.items():
                form_data.append((k, str(v)))

            timeout = 3000 # 
            resp = requests.post(URL, files=files, data=form_data, timeout=timeout)
            try:
                js = resp.json()
                pprint(js)
            except Exception:
                js = {
                    "error": "Response is not JSON",
                    "status_code": resp.status_code,
                    "text": resp.text[:500]
                }
                print(js)
            return js
    
    chapter_name = 133
    folder_path = os.path.join(BASE_DIR, "test_commic/data/Yule/Raw", f"{chapter_name}/*.jpg")
    print(folder_path)
    image_paths = glob.glob(folder_path)
    batch_size = 8
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    print(f"Tổng số trang: {len(image_paths)}, chia thành {total_batches} batch")

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        current_batch = i // batch_size + 1
        print(f"\nĐang xử lý batch {current_batch}/{total_batches} ({len(batch)} trang)")
    
    test_predict(
        image_paths=batch,
        manga_name="Yule",
        chapter_name=f"{chapter_name}",  # Thêm số batch vào tên chương
    )
import itertools

import numpy as np

CHARACTER_DETECTION_THRESHOLD_list = np.arange(0.1, 0.5, 0.1).tolist()
TEXT_CLASSIFICATION_THRESHOLD_list = np.arange(0.01, 0.05, 0.01).tolist()
ETA_FOR_CHARACTER_NAMING_list = np.arange(0.65, 0.95, 0.1).tolist()
TEXT_CHARACTER_MATCHING_THRESHOLD_list = np.arange(0.15, 0.55, 0.1).tolist()
CHARACTER_CHARACTER_MATCHING_THRESHOLD_list = np.arange(0.45, 0.75, 0.1).tolist()

def test_score():
    TEST_FILE = "/home/aorus/workspaces/magiv2/test_data/test_transcripts_133.json"
    results = evaluate_from_test_file(TEST_FILE, BASE_DIR, visualize=False)
    return (3 * results["summary"]["avg_speaker_acc"] + results["summary"]["avg_text_bleu"] + results["summary"]["avg_char_acc"] ) / 5

best_score = None
best_params = None

# Lặp qua tất cả tổ hợp tham số
for CHARACTER_DETECTION_THRESHOLD, TEXT_CLASSIFICATION_THRESHOLD, ETA_FOR_CHARACTER_NAMING, TEXT_CHARACTER_MATCHING_THRESHOLD, CHARACTER_CHARACTER_MATCHING_THRESHOLD in itertools.product(CHARACTER_DETECTION_THRESHOLD_list, TEXT_CLASSIFICATION_THRESHOLD_list, ETA_FOR_CHARACTER_NAMING_list, TEXT_CHARACTER_MATCHING_THRESHOLD_list, CHARACTER_CHARACTER_MATCHING_THRESHOLD_list):
    run_predict(CHARACTER_DETECTION_THRESHOLD, TEXT_CLASSIFICATION_THRESHOLD, ETA_FOR_CHARACTER_NAMING, TEXT_CHARACTER_MATCHING_THRESHOLD, CHARACTER_CHARACTER_MATCHING_THRESHOLD)
    score = test_score()
    print(f"Params: {CHARACTER_DETECTION_THRESHOLD}, {TEXT_CLASSIFICATION_THRESHOLD}, {ETA_FOR_CHARACTER_NAMING}, {TEXT_CHARACTER_MATCHING_THRESHOLD}, {CHARACTER_CHARACTER_MATCHING_THRESHOLD} -> Score: {score}")
    if (best_score is None) or (score > best_score):
        best_score = score
        best_params = (CHARACTER_DETECTION_THRESHOLD, TEXT_CLASSIFICATION_THRESHOLD, ETA_FOR_CHARACTER_NAMING, TEXT_CHARACTER_MATCHING_THRESHOLD, CHARACTER_CHARACTER_MATCHING_THRESHOLD)

    print(f"Best params: {best_params} with score: {best_score}")