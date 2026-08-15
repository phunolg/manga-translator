import asyncio
import os
import json
from pprint import pformat
from typing import List, Dict, Any, Optional, Tuple
import cv2
from fastapi import UploadFile
from PIL import Image
import numpy as np
from module.llm import get_llm
from module.llm.base import LLM
from module.llm.constants import LLM_MODEL
from settings import BASE_DIR
from type import Transcript
from utils.log import setup_logger

logger = setup_logger(__name__)


def enhance_text_contrast(crop):
    # Chuyển sang ảnh xám
    if len(crop.shape) > 2 and crop.shape[2] > 1:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop

    # Áp dụng adaptive threshold để tăng độ tương phản
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Chuyển lại về RGB nếu cần
    if len(crop.shape) > 2 and crop.shape[2] > 1:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return thresh


def read_image_file(file: UploadFile | str) -> np.ndarray:
    # Optional: kiểm tra loại nội dung
    if isinstance(file, str):
        image = Image.open(file)
        return np.array(image.convert("RGB"))
    
    # Kiểm tra nếu đối tượng có các thuộc tính của UploadFile
    if hasattr(file, "file") and hasattr(file, "filename") and hasattr(file, "content_type"):
        if getattr(file, "content_type", None) and not file.content_type.startswith("image/"):
            raise ValueError(
                f"Not an image: {file.filename} ({file.content_type})")
        logger.debug(
            f"Đã kiểm tra loại nội dung file: {file.filename} ({file.content_type})")
        file.file.seek(0)  # đảm bảo đọc từ đầu
        image = Image.open(file.file)
        image.load()       # ép PIL đọc ngay để bắt lỗi sớm
        return np.array(image.convert("RGB"))
    
    raise ValueError(f"Invalid file type: {type(file)}")

def crop_images(
    image: np.ndarray,
    bboxes: List[List[int]],
    folder_dir: str | None = None
) -> List[np.ndarray]:
    """
    Cắt các đối tượng từ hình ảnh dựa trên các bounding box.
    Args:
        image: Hình ảnh để cắt các đối tượng.
        bboxes: Danh sách các bounding box để cắt các đối tượng.
    Returns:
        crops_for_image: Danh sách các đối tượng đã được cắt.
    """

    crops_for_image = []
    img_h, img_w = image.shape[:2]
    output_dir = os.path.join(BASE_DIR, "debug", "crop") if folder_dir is None else os.path.join(
        BASE_DIR, "debug", "crop", folder_dir)
    os.makedirs(output_dir, exist_ok=True)
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox

        # fix the bounding box in case it is out of bounds or too small
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # đảm bảo x1 < x2 và y1 < y2
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

        # đảm bảo bbox không vượt quá kích thước hình ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x1, y1 = min(img_w, x1), min(img_h, y1)
        x2, y2 = max(0, x2), max(0, y2)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # Đảm bảo bounding box có kích thước tối thiểu là 10x10 pixel.
        if x2 - x1 < 10:
            if img_w - x1 > 10:
                x2 = x1 + 10
            else:
                x1 = x2 - 10
        if y2 - y1 < 10:
            if img_h - y1 > 10:
                y2 = y1 + 10
            else:
                y1 = y2 - 10

        crop = image[y1:y2, x1:x2]
        debug_enhanced_path = os.path.join(
            output_dir, f"crop_{idx}_enhanced.png")
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            cv2.imwrite(debug_enhanced_path, cv2.cvtColor(
                crop, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(debug_enhanced_path, crop)
        crops_for_image.append(crop)

    return crops_for_image


async def recognize_characters_from_images(character_images: List[Image.Image], character_names: List[str]) -> Dict[str, Dict]:
    """
    Sử dụng LLM để trích xuất đặc điểm nhận dạng từ các hình ảnh nhân vật

    Args:
        character_images: Danh sách hình ảnh nhân vật
        character_names: Danh sách tên nhân vật tương ứng

    Returns:
        character_features: Dictionary chứa đặc điểm của từng nhân vật
    """
    character_features = {}

    # Prompt để trích xuất đặc điểm nhân vật
    system_prompt = """
    Bạn là một hệ thống phân tích hình ảnh nhân vật. Nhiệm vụ của bạn là mô tả chi tiết đặc điểm nhận dạng của nhân vật trong hình.
    
    Hãy phân tích và mô tả các đặc điểm sau:
    1. Khuôn mặt: hình dáng, đặc điểm nổi bật
    2. Tóc: màu sắc, kiểu tóc, độ dài
    3. Mắt: màu, hình dạng, đặc điểm
    4. Trang phục: loại, màu sắc, phong cách
    5. Phụ kiện: mũ, kính, trang sức, vũ khí, v.v.
    6. Đặc điểm nhận dạng khác: nét đặc trưng, vết sẹo, hình xăm, v.v.
    
    Đầu ra phải có định dạng JSON:
    {
      "face": "mô tả khuôn mặt",
      "hair": "mô tả tóc",
      "eyes": "mô tả mắt",
      "outfit": "mô tả trang phục",
      "accessories": "mô tả phụ kiện",
      "distinctive_features": "đặc điểm nhận dạng khác"
    }
    """
    llm = get_llm(LLM_MODEL.gemma3)
    tasks = []
    for i, (image, name) in enumerate(zip(character_images, character_names)):
        tasks.append(
            llm.get_answer(
                question=f"Đây là nhân vật {name}. Hãy mô tả chi tiết đặc điểm nhận dạng của nhân vật này.",
                prompt_system=system_prompt,
                image=image,
                history=None,
                temperature=0.1,
                top_p=0.1,
            )
        )

    character_features = await asyncio.gather(*tasks)
    character_features_dict = {name: features for name, features in zip(character_names, character_features)}
    return character_features_dict


def map_characters_to_panel(panel_bbox: List[int], character_bboxes: List[List[int]], character_names: List[str]) -> List[str]:
    """
    Xác định nhân vật nào xuất hiện trong panel

    Args:
        panel_bbox: Bounding box của panel
        character_bboxes: Danh sách bounding box của nhân vật
        character_names: Danh sách tên nhân vật

    Returns:
        characters_in_panel: Danh sách tên nhân vật xuất hiện trong panel
    """
    characters_in_panel = []

    for i, char_bbox in enumerate(character_bboxes):
        # Tính tâm của nhân vật
        char_center_x = (char_bbox[0] + char_bbox[2]) / 2
        char_center_y = (char_bbox[1] + char_bbox[3]) / 2

        # Kiểm tra xem nhân vật có nằm trong panel không
        if (panel_bbox[0] <= char_center_x <= panel_bbox[2] and
                panel_bbox[1] <= char_center_y <= panel_bbox[3]):
            characters_in_panel.append(character_names[i])

    return characters_in_panel


async def generate_caption_from_panel(
    original_image: np.ndarray,
    panel_bboxes: List[List[int]],
    character_bboxes: List[List[int]],
    character_names: List[str],
    character_info: Dict[str, Dict[str, str]] = None,
):
    """
    Tạo mô tả cho mỗi panel trong hình ảnh, tích hợp thông tin nhân vật nếu có

    Args:
        original_image: Hình ảnh gốc
        panel_bboxes: Danh sách các bounding box của các panel
        character_info: Thông tin về các nhân vật (tùy chọn)
    Returns:
        captions: Danh sách các mô tả cho mỗi panel
    """
    panels = crop_images(original_image, panel_bboxes)
    characters_in_panels = []
    for panel_bbox in panel_bboxes:
        characters_in_panel = map_characters_to_panel(
            panel_bbox,
            character_bboxes,
            character_names
        )
        characters_in_panels.append(characters_in_panel)
    # Chuẩn bị thông tin nhân vật
    character_contexts = []
    for characters_in_panel in characters_in_panels:
        filter_character_info = {
            char_name: char_info
            for char_name, char_info in character_info.items() if char_name in characters_in_panel
        }
        character_context = f"Mô tả nhân vật đã biết:\n{pformat(filter_character_info)}"
        character_contexts.append(character_context)

    # Sử dụng asyncio.gather để gửi nhiều yêu cầu đồng thời
    system_prompt = """
Bạn là một hệ thống mô tả truyện tranh, nhiệm vụ của bạn là tạo chú thích 
(character-aware caption) cho mỗi panel. 

{character_context}

Hãy quan sát hình ảnh và viết mô tả ngắn gọn, khách quan, chỉ dựa trên chi tiết nhìn thấy.  

Yêu cầu:
- Nếu có nhiều nhân vật, hãy mô tả vị trí tương đối (bên trái, giữa, bên phải).
- Ghi nhận đặc điểm trực quan: tóc, quần áo, biểu cảm, tư thế, hành động.
- Không được suy diễn nội tâm, không giả định mối quan hệ nếu không hiển thị rõ.
- Không gian: chỉ mô tả các vật thể chính và không gian / bối cảnh / môi trường có thể nhìn thấy.
- Bỏ qua văn bản, bóng thoại, hiệu ứng chữ.

Đầu ra phải có dạng:

"Cảnh: ...
Nhân vật:
- "Tên nhân vật": ...
- "Tên nhân vật": ...
Hành động: ...
Không gian: ..."
    """
    llm = get_llm(LLM_MODEL.gemma3)
    tasks = [
        llm.get_answer(
            question="Hãy mô tả hình ảnh này",
            prompt_system=system_prompt.format(
                character_context=character_context),
            image=panel,
            history=None,
            temperature=0.1,
            top_p=0.1,
        )
        for panel, character_context in zip(panels, character_contexts)
    ]

    captions = await asyncio.gather(*tasks)
    return captions


async def generate_prose(captions: List[str], transcripts: List[Transcript]) -> str:
    """
    Tạo văn xuôi từ các caption và transcript

    Args:
        captions: Danh sách các mô tả panel
        transcripts: Danh sách các đoạn hội thoại

    Returns:
        prose: Văn xuôi mô tả trang truyện
    """
    # Kết hợp caption và transcript
    combined_content = ""
    for i, caption in enumerate(captions):
        combined_content += f"Panel {i+1}:\n{caption}\n\n"

    combined_content += "Hội thoại:\n"
    for transcript in transcripts:
        combined_content += f"{str(transcript)}\n"

    # Prompt để tạo văn xuôi
    system_prompt = """
    Bạn là một nhà văn chuyên viết tiểu thuyết từ truyện tranh. Nhiệm vụ của bạn là chuyển đổi các mô tả panel và hội thoại thành văn xuôi mạch lạc.
    
    Yêu cầu:
    1. Tạo văn xuôi tự nhiên, mạch lạc từ các mô tả và hội thoại
    2. Giữ nguyên nội dung và ý nghĩa của các mô tả và hội thoại
    3. Sắp xếp các sự kiện theo thứ tự thời gian hợp lý
    4. Tích hợp hội thoại vào văn xuôi một cách tự nhiên
    5. Sử dụng ngôn ngữ phù hợp với thể loại truyện
    6. Đối với các câu thoại bằng tiếng Anh, hãy giữ nguyên tiếng Anh.
    7. Không mô tả ngoại hình nhân vật, chỉ tập trung vào khung cảnh xung quanh, hành động và cảm xúc của nhân vật.
    
    Đầu ra là một đoạn văn xuôi mô tả các sự kiện trong trang truyện.
    """
    llm = get_llm(LLM_MODEL.gemma3)
    # Gọi LLM để tạo văn xuôi
    prose = await llm.get_answer(
        question=f"Hãy chuyển đổi các mô tả panel và hội thoại sau thành văn xuôi:\n\n{combined_content}",
        prompt_system=system_prompt,
        history=None,
        temperature=0.01,
        top_p=0.4,
    )

    return prose


async def generate_prose_with_speaker_assignment(captions: List[str], transcripts: List[Transcript], character_names: List[str], character_features: Dict[str, Dict]) -> str:
    """
    Tạo văn xuôi và sử dụng LLM để gán lại speaker cho các đoạn hội thoại có label 'Other' hoặc 'unsure'
    
    Args:
        captions: Danh sách các mô tả panel
        transcripts: List[(transcript, metadata)] - danh sách transcript với metadata
        character_names: Danh sách tên nhân vật trong trang này
        character_features: Đặc điểm nhận dạng của các nhân vật
        
    Returns:
        prose: Văn xuôi đã được xử lý với speaker assignment được cải thiện
    """
    # Phân loại transcript theo loại speaker assignment
    normal_transcripts = []
    problem_transcripts = []
    
    for transcript_data in transcripts:
        speaker = transcript_data.speaker
        if speaker in ["Other", "unsure"]:
            problem_transcripts.append(transcript_data)
        else:
            normal_transcripts.append(transcript_data)
    
    # Nếu không có vấn đề với speaker assignment, sử dụng hàm generate_prose thông thường
    if not problem_transcripts:
        return await generate_prose(captions, transcripts), transcripts
    
    # Chuẩn bị context cho LLM để phân tích và gán speaker lại
    panel_context = ""
    for i, caption in enumerate(captions):
        panel_context += f"Panel {i+1}:\n{caption}\n\n"
    
    # Thông tin về nhân vật khả dụng
    available_characters = [name for name in character_names if name not in ["Other", "unsure"]]
    character_info_context = ""
    if available_characters and character_features:
        for char_name in available_characters:
            if char_name in character_features:
                char_features = character_features[char_name]
                character_info_context += f"- {char_name}: {char_features}\n"
    
    transcripts_context = ""
    for transcript_data in problem_transcripts:
        transcripts_context += f"{str(transcript_data)}\n"
    
    # Prompt cho việc phân tích và gán speaker
    speaker_assignment_prompt = """
    Bạn là một hệ thống NLP chuyên phân tích truyện tranh và gán speaker cho các đoạn hội thoại.
    
    Nhiệm vụ:
    1. Phân tích context của panels và available characters
    2. Gán speaker cho các đoạn hội thoại chưa được xác định (có label 'Other' hoặc 'unsure')
    3. Dựa trên nội dung hội thoại, context panel, và thông tin characters
    
    Thông tin panel:
    {panel_context}
    
    Nhân vật khả dụng và đặc điểm:
    {character_info_context}
    
    Các đoạn hội thoại cần gán speaker:
    {transcripts_context}
    
    Xuất kết quả theo format JSON:
    {{
        "speaker_assignments": [
            {{"text": "đoạn hội thoại", "speaker": "tên nhân vật"}}
        ],
        "reasoning": "Giải thích cách bạn đã gán speaker"
    }}
    
    Nếu không thể xác định được speaker, hãy để "unknown". Ưu tiên các character có trong danh sách available.
    """
    llm = get_llm(LLM_MODEL.gemma3)
    # Gọi LLM để phân tích và gán speaker
    assignment_data = await llm.get_answer(
        question=f"- Hiển thị avialable characters: {available_characters}\n- Đặc điểm nhận dạng:\n{character_info_context}",
        prompt_system=speaker_assignment_prompt.format(
            panel_context=panel_context,
            character_info_context=character_info_context,
            transcripts_context=transcripts_context
        ),
        history=None,
        temperature=0.01,
        top_p=0.1,
    )
    
    # Parse kết quả từ LLM
    try:
        # Cập nhật transcripts với speaker mới
        updated_transcripts: List[Transcript] = []
        speaker_assignments = assignment_data.get("speaker_assignments", [])
        
        # Tạo dict tìm kiếm nhanh
        assignment_dict = {item["text"].strip(): item["speaker"] for item in speaker_assignments}
        
        for transcript in transcripts:
            text = transcript.text.strip()
            if text in assignment_dict:
                new_speaker = assignment_dict[text]          # Cập nhật metadata
                transcript.speaker = new_speaker                
                updated_transcripts.append(transcript)
            else:
                updated_transcripts.append(transcript)
    
        prose = await generate_prose(captions, updated_transcripts)
        
        logger.info(f"Speaker assignment hoàn thành: {len(speaker_assignments)} transcripts được xử lý")
        
        return prose, updated_transcripts
        
    except Exception as e:
        logger.error(f"Lỗi khi parse JSON từ LLM assignment result: {e}")   
        return await generate_prose(captions, transcripts), transcripts 
    