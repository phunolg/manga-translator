# Router cho init-manga
from fastapi import APIRouter, FastAPI, File, Form, UploadFile
    
import json
import os
from pprint import pformat
import time
from typing import  List, Optional

from PIL import Image
from fastapi.responses import JSONResponse
import numpy as np

from modelling_magiv2 import Magiv2Model
from module.config import Config
from module.features import Features
from module.get_transcript import get_transcript as get_transcript_module
from module.history_retrival.type import PageInfo
from module.knowledge.type import Language, Topic
from module.rendering.text_render import logger
from module.repository.manga_metadata import MangaMetadataRepository
from module.repository.transcript_repository import TranscriptRepository
from module.translator.translator_v1 import Translator

from module.utils.textblock import TextBlock
from response_handler import ResponseFactory
from type import Character
from settings import BASE_DIR
from type import Transcript
from utils.image import read_image_file
from utils.log import setup_logger
from utils.utils import bbox_to_quad
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

model: Optional[Magiv2Model] = None
logger = setup_logger(__name__)
app = FastAPI()

main_routes = APIRouter(prefix="/main", tags=["main"])

async def _translate(
    page_images: List[UploadFile],
    story_name: str,
    chapter_number: int,
    page_numbers: List[int],
    target_lang: str,
):
    page_names = [image.filename for image in page_images]
    logger.debug(f"page_names: {page_names}")
    start_time = time.time()
    metadata_story = MangaMetadataRepository.get_story_by_name(story_name)

    transcript_per_page: List[List[Transcript]] = []
    transcript_bboxes_per_page: List[List[List[int]]] = []
    page_info_batch: List[PageInfo] = []
    chapter_pages = [read_image_file(f) for f in page_images]

    for page_number in page_numbers:
        page_info = PageInfo(story_name=story_name, chapter_number=chapter_number, page_number=page_number)
        transcripts, transcript_bboxes = TranscriptRepository.get_transcript(page_info)
        transcript_per_page.append(transcripts)
        transcript_bboxes_per_page.append(transcript_bboxes)
        page_info_batch.append(page_info)
       
    

    try:
        translated_transcript_per_page = await Translator(input_lang=metadata_story.source_language, target_lang=target_lang, story_name=story_name, topic=Topic(metadata_story.story_type)).translate_batch(
            transcript_per_page,
            image_batch=chapter_pages,
            page_info_batch=page_info_batch,
        )
        batch_outputs = []
        for page_info, transcripts, translations in zip(page_info_batch, transcript_per_page, translated_transcript_per_page):
            outputs = [
                {
                    "source_text": transcript.text,
                    "translation": translation,
                }
                for transcript, translation in zip(transcripts, translations)
            ]
            batch_outputs.append(outputs)
        return {
            "page_number": page_info.page_number,
            "outputs": batch_outputs,
        }
    except Exception as e:
        logger.warning(f"Lỗi khi dịch: {e}", exc_info=True)
        return [[] for _ in page_numbers]
        
async def _inpaint(
    page_images: List[UploadFile],
    story_name: str,
    chapter_number: int,
    page_numbers: List[int],
    target_lang: str,
):
    page_names = [image.filename for image in page_images]
    logger.debug(f"page_names_5: {page_names}")
    start_time = time.time()

    transcript_per_page: List[List[Transcript]] = []
    transcript_bboxes_per_page: List[List[List[int]]] = []
    chapter_pages = [read_image_file(f) for f in page_images]

    for page_number in page_numbers:
        page_info = PageInfo(story_name=story_name, chapter_number=chapter_number, page_number=page_number)
        transcripts, transcript_bboxes = TranscriptRepository.get_transcript(page_info)
        transcript_per_page.append(transcripts)
        transcript_bboxes_per_page.append(transcript_bboxes)
    
    logger.debug(f"transcript_bboxes_per_page: {pformat(transcript_bboxes_per_page)}")
        
    translated_transcript_per_page = [[transcript.translation for transcript in transcripts] for transcripts in transcript_per_page]
    
    # Inpaint từng trang
    processed_images = []
    for i, (page, translated_transcript) in enumerate(zip(chapter_pages, translated_transcript_per_page)):
        try:
            if not translated_transcript:
                processed_images.append(Image.fromarray(page))
                logger.warning(f"Không có transcript cho trang {i}")
                continue

            bboxes = transcript_bboxes_per_page[i]
            quads = bbox_to_quad(bboxes)
            text_bboxes = []
            for j, quad in enumerate(quads):
                if j >= len(translated_transcript):
                    break
                tb = TextBlock(
                    lines=[quad.tolist()],
                    texts=[""],
                    translation=translated_transcript[j],
                    target_lang=target_lang,
                )
                text_bboxes.append(tb)

            features = Features()
            features.parse_init_params({
                "use_gpu_limited": True,
                "models_ttl": 1,
            })
            out = await features.inpaint(
                image=np.array(page),
                config=Config(),
                text_bboxes=text_bboxes,
            )
            processed_images.append(out.result)

        except Exception as e:
            logger.warning(
                f"Lỗi khi xử lý trang {i}: {e}", exc_info=True)
            processed_images.append(Image.fromarray(page))
    response_strategy = ResponseFactory.get_strategy("zip")
    end_time = time.time()
    logger.debug(f"Time taken: {end_time - start_time} seconds")
    return response_strategy.process_images(processed_images, page_names)

@main_routes.post("/get-transcript")
async def get_transcript(
    chapter_pages: List[UploadFile] = File(...),
    story_name: str = Form(...),
    chapter_name: str = Form(...),
):
    chapter_pages_np = [read_image_file(f) for f in chapter_pages]
    
    metadata = MangaMetadataRepository.get_story_by_name(story_name)
    
    characters = [character for character in metadata.character_blank]
   
    transcript_per_page, transcript_bboxes_per_page, proses_per_page = await get_transcript_module(chapter_pages_np, characters, metadata)

    story_folder = os.path.join(BASE_DIR, "transcript_history", story_name)
    os.makedirs(story_folder, exist_ok=True)
    chapter_folder = os.path.join(story_folder, chapter_name)
    os.makedirs(chapter_folder, exist_ok=True)
    save_paths = []
    for transcripts, chapter_page, transcript_bboxes, prose in zip(transcript_per_page, chapter_pages, transcript_bboxes_per_page, proses_per_page):
        save_name = chapter_page.filename.split(".")[0]
        save_path = os.path.join(
            chapter_folder, f"{save_name}.json")
        save_paths.append(save_path)
        with open(os.path.join(chapter_folder, f"{save_name}.json"), "w", encoding="utf-8") as f:
            json.dump({"transcript": [transcript.__dict__ for transcript in transcripts],
                      "transcript_bboxes": transcript_bboxes,
                       "prose": prose}, f, ensure_ascii=False, indent=4)
    return [{
        "save_path": save_path,
        "transcript": transcripts,
        "prose": prose,
    } for save_path, transcripts, prose in zip(save_paths, transcript_per_page, proses_per_page)]


@main_routes.post("/translate")
async def translate(
    page_images: List[UploadFile] = File(...),
    target_lang: str = Form(Language.VIETNAMESE.value),
    story_name: str = Form(...),
    chapter_number: int = Form(...),
    page_numbers: List[int] = Form(...),
):
    translated_transcript_per_page = await _translate(
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
    )
    return translated_transcript_per_page


@main_routes.post("/inpaint")
async def inpaint(
    page_images: List[UploadFile] = File(...),
    target_lang: str = Form(Language.VIETNAMESE.value),
    story_name: str = Form(...),
    chapter_number: int = Form(...),
    page_numbers: List[int] = Form(...),
):
    return await _inpaint(
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
    )
    

@main_routes.post("/translate-and-inpaint")
async def translate_and_inpaint(
    page_images: List[UploadFile] = File(...),
    target_lang: str = Form(Language.VIETNAMESE.value),
    story_name: str = Form(...),
    chapter_number: int = Form(...),
    page_numbers: List[int] = Form(...),

):
    await _translate(
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
    )
    inpainted_images = await _inpaint(
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
    )
    return inpainted_images