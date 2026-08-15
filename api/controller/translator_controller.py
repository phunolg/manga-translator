# Router cho init-manga
from fastapi import APIRouter, FastAPI, File, Form, UploadFile
import asyncio
    
import json
import os
from pprint import pformat
import time
from typing import  List, Optional

from PIL import Image
from fastapi.responses import JSONResponse
import numpy as np

from modelling_magiv2 import Magiv2Model
from module.config import Config, Renderer
from module.features import Features
from module.get_transcript import get_transcript as get_transcript_module
from module.history_retrival.type import PageInfo
from module.knowledge.type import Language, Topic
from module.rendering.text_render import logger
from module.rendering import dispatch_eng_render
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

logger = setup_logger(__name__)
app = FastAPI()

main_routes = APIRouter(prefix="/main", tags=["main"])

async def _translate(
    page_images: List[UploadFile],
    story_name: str,
    chapter_number: int,
    page_numbers: List[int],
    target_lang: str,
) -> List[List[dict]]:
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
    except Exception as e:
        logger.warning(f"Lỗi khi dịch: {e}", exc_info=True)
        return [[] for _ in page_numbers]
    finally:
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
        return batch_outputs
    
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
    features = Features()
    features.parse_init_params({
        "use_gpu_limited": True,
        "models_ttl": 1,
    })
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

async def _erase_only(
    page_images: List[UploadFile],
    story_name: str,
    chapter_number: int,
    page_numbers: List[int],
    target_lang: str,
) -> List[Image.Image]:
    page_names = [image.filename for image in page_images]
    logger.debug(f"page_names_erase_only: {page_names}")
    start_time = time.time()

    transcript_bboxes_per_page: List[List[List[int]]] = []
    chapter_pages = [read_image_file(f) for f in page_images]

    for page_number in page_numbers:
        page_info = PageInfo(story_name=story_name, chapter_number=chapter_number, page_number=page_number)
        _, transcript_bboxes = TranscriptRepository.get_transcript(page_info)
        transcript_bboxes_per_page.append(transcript_bboxes)
    
    processed_images: List[Image.Image] = []
    for i, page in enumerate(chapter_pages):
        try:
            bboxes = transcript_bboxes_per_page[i]
            quads = bbox_to_quad(bboxes)
            text_bboxes = []
            for quad in quads:
                tb = TextBlock(
                    lines=[quad.tolist()],
                    texts=[""],
                    translation="",  # không render
                    target_lang=target_lang,
                )
                text_bboxes.append(tb)

            features = Features()
            features.parse_init_params({
                "use_gpu_limited": True,
                "models_ttl": 1,
            })
            cfg = Config()
            cfg.render.renderer = Renderer.none  # tắt render, chỉ xóa chữ
            out = await features.inpaint(
                image=np.array(page),
                config=cfg,
                text_bboxes=text_bboxes,
            )
            processed_images.append(out.result)
        except Exception as e:
            logger.warning(f"Lỗi khi erase trang {i}: {e}", exc_info=True)
            processed_images.append(Image.fromarray(page))

    end_time = time.time()
    logger.debug(f"Erase-only time taken: {end_time - start_time} seconds")
    return processed_images

async def _typeset(
    inpainted_images: List[Image.Image],
    page_images: List[UploadFile],
    story_name: str,
    chapter_number: int,
    page_numbers: List[int],
    target_lang: str,
    translated_transcript_per_page: List[List[dict]],
):
    page_names = [image.filename for image in page_images]
    start_time = time.time()

    transcript_bboxes_per_page: List[List[List[int]]] = []
    chapter_pages = [read_image_file(f) for f in page_images]
    for page_number in page_numbers:
        page_info = PageInfo(story_name=story_name, chapter_number=chapter_number, page_number=page_number)
        _, transcript_bboxes = TranscriptRepository.get_transcript(page_info)
        transcript_bboxes_per_page.append(transcript_bboxes)

    processed_images: List[Image.Image] = []
    sem = asyncio.Semaphore(4)

    async def _render_one(i: int, canvas_img: Image.Image, page_translations: List[dict]) -> Image.Image:
        async with sem:
            try:
                bboxes = transcript_bboxes_per_page[i]
                quads = bbox_to_quad(bboxes)
                translations: List[str] = [item.get("translation", "") for item in page_translations]

                text_bboxes = []
                for j, quad in enumerate(quads):
                    if j >= len(translations):
                        break
                    tb = TextBlock(
                        lines=[quad.tolist()],
                        texts=[""],
                        translation=translations[j],
                        target_lang=target_lang,
                    )
                    text_bboxes.append(tb)

                # Render-only: chèn chữ lên ảnh đã xóa
                cfg = Config()
                rendered_np = await asyncio.to_thread(
                    dispatch_eng_render,
                    img_canvas=np.array(canvas_img),
                    original_img=np.array(chapter_pages[i]),
                    text_bboxes=text_bboxes,
                    text_regions=text_bboxes,  # dùng cùng danh sách cho vùng render
                    font_path='',  # để mặc định trong renderer
                    line_spacing=cfg.render.line_spacing or 0,
                    disable_font_border=cfg.render.disable_font_border,
                    lang=target_lang,
                )
                return Image.fromarray(rendered_np)
            except Exception as e:
                logger.warning(f"Lỗi khi typeset trang {i}: {e}", exc_info=True)
                return canvas_img

    tasks = [
        asyncio.create_task(_render_one(i, canvas_img, page_translations))
        for i, (canvas_img, page_translations) in enumerate(zip(inpainted_images, translated_transcript_per_page))
    ]
    processed_images = await asyncio.gather(*tasks)

    response_strategy = ResponseFactory.get_strategy("zip")
    end_time = time.time()
    logger.debug(f"Typeset time taken: {end_time - start_time} seconds")
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
    # Chạy dịch và xóa chữ song song
    translate_task = asyncio.create_task(_translate(
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
    ))
    erase_task = asyncio.create_task(_erase_only(
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
    ))

    translated_transcript_per_page, inpainted_images = await asyncio.gather(translate_task, erase_task)

    # Khi cả hai xong, chèn chữ dịch lên ảnh đã xóa chữ
    return await _typeset(
        inpainted_images=inpainted_images,
        page_images=page_images,
        story_name=story_name,
        chapter_number=chapter_number,
        page_numbers=page_numbers,
        target_lang=target_lang,
        translated_transcript_per_page=translated_transcript_per_page,
    )