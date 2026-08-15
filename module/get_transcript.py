import asyncio
from pprint import pformat
import uuid
import torch
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
from configuration_magiv2 import Magiv2Config
from modelling_magiv2 import Magiv2Model
from module.knowledge.type import Topic
from settings import BASE_DIR, ETA_FOR_CHARACTER_NAMING
from type import Character, MangaMetadata, Transcript
from utils.image import generate_caption_from_panel, generate_prose, generate_prose_with_speaker_assignment, read_image_file, recognize_characters_from_images
from utils.log import setup_logger

# Đặt thư mục cache ngay trong workspace hiện tại để tránh dùng ~/.huggingface
CACHE_DIR = os.path.join(BASE_DIR, ".hf_cache")
logger = setup_logger(__name__)


async def get_transcript(chapter_pages: List[Image.Image], characters: List[Character], metadata: MangaMetadata) -> Tuple[List[List[Transcript]], List[List[List[int]]], List[str]]:
    """
    Trích xuất transcript từ các trang truyện tranh

    Args:
        chapter_pages: Danh sách các trang truyện (PIL.Image)
        character_bank: Danh sách các nhân vật
    Returns:
        transcript_per_page: Danh sách các đoạn hội thoại cho mỗi trang
        transcript_bboxes_per_page: Danh sách các bounding box cho mỗi đoạn hội thoại
        proses_per_page: Danh sách các đoạn văn xuôi mô tả mỗi trang
    """
    config = Magiv2Config.from_pretrained(BASE_DIR, trust_remote_code=True)
    model: Magiv2Model = Magiv2Model.from_pretrained(
        "ragavsachdeva/magiv2",
        config=config,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    ).cuda().eval()

    character_features = {character.name: character.character_features for character in characters}
    logger.info(f"Character features extracted: {list(character_features.keys())}")
    
    character_bank = {
        "names": [character.name for character in characters],
        "images": [read_image_file(character.image_path) for character in characters],
    }

    # Magiv2Model mong đợi đầu vào là ndarray, không phải PIL.Image
    page_arrays: List[np.ndarray] = [np.array(img) for img in chapter_pages]

    with torch.no_grad():
        per_page_results = await model.do_chapter_wide_prediction(
            page_arrays,
            character_bank,
            eta=ETA_FOR_CHARACTER_NAMING,
            source_language=metadata.source_language,
        )
        
    transcript_per_page: List[List[str]] = []
    transcript_bboxes_per_page: List[List[List[int]]] = []
    captions_per_page: List[List[str]] = []

    for i, (image, page_result) in enumerate(zip(page_arrays, per_page_results)):
        panel_captions = await generate_caption_from_panel(
            image,
            page_result["panels"],
            page_result["characters"],
            page_result["character_names"],
            character_info=character_features
        )

        captions_per_page.append(panel_captions)

        # Lưu hình ảnh kết quả
        model.visualise_single_image_prediction(
            image, page_result, os.path.join(BASE_DIR, "debug", "detection_and_assiociation.png"))

        # Lấy thông tin về người nói
        speaker_name = {
            text_idx: page_result["character_names"][char_idx] for text_idx, char_idx in
            page_result["text_character_associations"]
        }

        # Lấy thông tin về người nghe (nếu có)
        target_name = page_result.get("dialogue_targets", {})
        logger.debug(f"target_name: {target_name}")

        transcript_bboxes = []
        transcripts: List[Transcript] = []
        for j in range(len(page_result["ocr"])):
            if not page_result["is_essential_text"][j]:
                continue
 
            transcripts.append(Transcript(
                speaker=speaker_name.get(j, "unsure"),
                target=target_name.get(j, "unsure"),
                text=page_result['ocr'][j],
                text_speech_type=page_result["text_speech_type"][j],
            ))
            transcript_bboxes.append(page_result["texts"][j])

        transcript_per_page.append(transcripts)
        transcript_bboxes_per_page.append(transcript_bboxes)
        logger.debug(f"transcripts: {transcripts}")

    # Tạo văn xuôi từ caption và transcript
    tasks = [
        generate_prose(captions, transcripts) if captions else "Không có caption cho trang này"
        for captions, transcripts in zip(captions_per_page, transcript_per_page)
    ]
    proses_per_page = await asyncio.gather(*tasks)
    
    logger.info(f"Transcript per page after speaker assignment:\n{pformat(transcript_per_page)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Trả về transcript, bounding box và văn xuôi
    return transcript_per_page, transcript_bboxes_per_page, proses_per_page
