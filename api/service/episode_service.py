from datetime import date
from time import time
from api.handler.response_handler import ResponseFactory
from api.schema.api_exception_schema import ApiException
from api.schema.api_response_schema import ApiResponse
from api.schema.episode_schema import (
    EpisodeDetailResponse,
    PageDetailResponse,
    TranscriptLineResponse,
)
from api.utils.path_utils import convert_image_path_to_url
from module.config import Config
from module.database.postgre.repository.episode_repository import EpisodeRepository
from module.database.postgre.repository.metadata_repository import MetadataRepository
from module.features import Features
from module.get_transcript import get_transcript as get_transcript_module
from module.history_retrival.type import PageInfo
from module.knowledge import Knowledge
from module.translator.translator_v1 import Translator
from module.utils.textblock import TextBlock
from type import Character as TypeCharacter, CharacterFeatures, MangaMetadata, Transcript
from module.knowledge.type import Language, Topic
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, cast
from fastapi import UploadFile
import os
import shutil
from PIL import Image
from utils.image import read_image_file
from settings import BASE_DIR
from utils.log import setup_logger
from utils.utils import bbox_to_quad

logger = setup_logger(__name__)  
class EpisodeService:
    
    @classmethod
    async def create_episode(
        cls, 
        chapter_pages: List[UploadFile | str], 
        story_name: str, 
        chapter_number: int):
        """
        Tạo episode mới từ các trang chapter và lưu vào database
        """
        # Lấy story từ database
        story = await MetadataRepository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)

        
        characters: List[TypeCharacter] = []
        for db_char in story.characters:
            # Lấy address_matrix từ database
            address_matrix: Dict[str, str] = {}
            for am in db_char.address_matrixes_as_speaker:
                if am.target_id:
                    # Tìm target character name
                    target_char = next(
                        (c for c in story.characters if c.id == am.target_id),
                        None
                    )
                    target_name = target_char.source_name if target_char else None
                else:
                    target_name = "other"
                
                if target_name:
                    address_matrix[target_name] = am.description
            
            # Tạo CharacterFeatures từ database columns
            character_features = CharacterFeatures(
                face=db_char.face,
                hair=db_char.hair,
                eyes=db_char.eyes,
                outfit=db_char.outfit,
                accessories=db_char.accessories,
                distinctive_features=db_char.distinctive_features,
            )
            
            # Tạo type.Character
            type_char = TypeCharacter(
                name=db_char.source_name,
                description=db_char.description,
                address_matrix=address_matrix,
                character_features=character_features,
                image_path=db_char.image_path or "",  # Lấy từ database
            )
            characters.append(type_char)
        
        # Lấy mapping_name từ database (ở cấp story)
        mapping_name: Dict[str, str] = {}
        for mapping in story.mapping_names:
            key = f"{mapping.language.value}:{mapping.source}"
            mapping_name[key] = mapping.translation
        
        # Lấy translate_dict từ database (lấy dict đầu tiên hoặc empty dict)
        translate_dict: Dict[str, str] = {}
        if story.translate_dicts:
            # Lấy translate dict đầu tiên (hoặc có thể chọn theo ngôn ngữ cụ thể)
            translate_dict = story.translate_dicts[0].dictionary
        
        # Tạo MangaMetadata từ database
        story_type_value = story.story_type if isinstance(story.story_type, Topic) else Topic(story.story_type)
        source_language_value = (
            story.source_language if isinstance(story.source_language, Language) else Language(story.source_language)
        )

        metadata = MangaMetadata(
            story_name=str(story.story_name),
            story_type=story_type_value,
            character_blank=characters,
            mapping_name=mapping_name,
            translate_dict=translate_dict,
            source_language=source_language_value,
        )
        
        # Tạo thư mục để lưu images
        story_images_dir = os.path.join(BASE_DIR, "data", "stories", story_name, "chapters", str(chapter_number))
        os.makedirs(story_images_dir, exist_ok=True)
        
        # Convert uploaded files to images
        chapter_page_images = [
            Image.fromarray(read_image_file(page).astype(np.uint8)) for page in chapter_pages
        ]
        print(f"[EpisodeService] Processing {len(chapter_page_images)} pages for chapter {chapter_number}")
    
        # Get transcript data
        transcript_per_page, transcript_bboxes_per_page, proses_per_page = await get_transcript_module(
            chapter_page_images, characters, metadata
        )
        print(f"[EpisodeService] Got {len(transcript_per_page)} transcript pages, {len(transcript_bboxes_per_page)} bbox pages, {len(proses_per_page)} prose pages")

        # Chuẩn bị dữ liệu để lưu vào database và lưu file images
        pages_data = []
        for page_index, (transcripts, chapter_page, transcript_bboxes, prose) in enumerate(
            zip(transcript_per_page, chapter_pages, transcript_bboxes_per_page, proses_per_page)
        ):
            page_number = page_index + 1  # Page number bắt đầu từ 1
            
            if isinstance(chapter_page, str):
                original_filename = os.path.basename(chapter_page)
            else:
                original_filename = chapter_page.filename or f"page_{page_number}.png"
            # Đảm bảo có extension
            if not os.path.splitext(original_filename)[1]:
                original_filename = f"{original_filename}.png"
            
            # Đường dẫn đầy đủ để lưu file
            image_path = os.path.join(story_images_dir, original_filename)
            
            if isinstance(chapter_page, str):
                shutil.copyfile(chapter_page, image_path)
            else:
                chapter_page.file.seek(0)
                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(chapter_page.file, buffer)
             
            pages_data.append({
                "page_number": page_number,
                "prose": prose,
                "image_path": image_path,  # Lưu đường dẫn đầy đủ vào database
                "transcripts": transcripts,
                "transcript_bboxes": transcript_bboxes
            })
        
        print(f"[EpisodeService] Prepared {len(pages_data)} pages_data to save to database")
        
        # Lưu vào database
        episode = await EpisodeRepository.create_episode_with_pages(
            story_id=int(str(story.id)),
            chapter_number=chapter_number,
            pages_data=pages_data
        )
        print(f"[EpisodeService] Successfully saved episode {episode.id} with {len(pages_data)} pages")
        
        return ApiResponse(
            message=f"Episode {chapter_number} created successfully",
            data={
                "episode_id": episode.id,
                "chapter_number": episode.chapter_number,
                "pages_count": len(pages_data)
            },
            status_code=201
        )

    @classmethod
    async def get_episode_detail(
        cls,
        story_name: str,
        chapter_number: int,
    ) -> ApiResponse:
        # Lấy story cơ bản (không eager load) để tìm episode
        story = await EpisodeRepository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)

        episode = await EpisodeRepository.get_episode_with_details(
            story_id=int(str(story.id)),
            chapter_number=chapter_number,
        )
        if not episode:
            raise ApiException(
                message=f"Episode {chapter_number} not found for story {story_name}",
                status_code=404,
            )

        pages: List[PageDetailResponse] = []
        for page in sorted(episode.pages, key=lambda p: p.page_number):
            transcript_lines = sorted(page.transcript_lines, key=lambda tl: tl.line_index)
            transcripts: List[TranscriptLineResponse] = []
            for tl in transcript_lines:
                transcripts.append(
                    TranscriptLineResponse(
                        line_index=tl.line_index,
                        speaker=tl.speaker_character.source_name if tl.speaker_character else None,
                        text=tl.text,
                        text_speech_type=tl.text_speech_type,
                        target=tl.target_character.source_name if tl.target_character else None,
                        translation=next((tr.translated for tr in tl.translations), None),
                        bbox=tl.bbox,
                    )
                )

            pages.append(
                PageDetailResponse(
                    id=page.id,
                    page_number=page.page_number,
                    prose=page.prose,
                    image_url=convert_image_path_to_url(page.image_path),
                    transcripts=transcripts,
                )
            )

        response = EpisodeDetailResponse(
            id=cast(int, episode.id),
            story_id=cast(int, episode.story_id),
            story_name=str(story.story_name),
            chapter_number=cast(int, episode.chapter_number),
            created_at=cast(Optional[datetime], episode.created_at),
            updated_at=cast(Optional[datetime], episode.updated_at),
            pages=pages,
        )

        return ApiResponse(
            message=f"Episode {chapter_number} detail retrieved successfully",
            data=response,
            status_code=200,
        )
    
    @classmethod
    async def translate_episode(
        cls,
        story_name: str,
        chapter_number: int,
        target_lang: Language,
        page_numbers: Optional[List[int]] = None,
        response_mode: str = "zip",
        force_translate: bool = False,
    ):
        start_time = time()

        story = await EpisodeRepository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)

        episode = await EpisodeRepository.get_episode_with_details(
            story_id=int(str(story.id)),
            chapter_number=chapter_number,
        )
        if not episode:
            raise ApiException(
                message=f"Episode {chapter_number} not found for story {story_name}",
                status_code=404,
            )

        def _normalize_bbox(raw_bbox) -> List[float] | None:
            if not raw_bbox:
                return None
            if isinstance(raw_bbox, dict):
                key_sets = (
                    ("x1", "y1", "x2", "y2"),
                    ("xmin", "ymin", "xmax", "ymax"),
                )
                for keys in key_sets:
                    if all(key in raw_bbox for key in keys):
                        return [float(raw_bbox[key]) for key in keys]
                values = list(raw_bbox.values())
                if len(values) >= 4:
                    return [float(values[i]) for i in range(4)]
                return None
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
                return [float(raw_bbox[i]) for i in range(4)]
            return None

        transcript_per_page: List[List[Transcript]] = []
        transcript_bboxes_per_page: List[List[List[int]]] = []
        page_line_ids_per_page: List[List[int]] = []
        page_arrays: List[np.ndarray] = []
        page_images: List[Image.Image] = []
        page_info_batch: List[PageInfo] = []
        page_names: List[str] = []

        sorted_pages = sorted(episode.pages, key=lambda p: p.page_number)
        if page_numbers:
            requested_pages = set(page_numbers)
            filtered_pages = [page for page in sorted_pages if page.page_number in requested_pages]
            missing = requested_pages - {page.page_number for page in filtered_pages}
            if missing:
                raise ApiException(
                    message=f"Các trang không tồn tại trong chapter: {sorted(missing)}",
                    status_code=404,
                )
            sorted_pages = filtered_pages
            if not sorted_pages:
                raise ApiException(
                    message="Không tìm thấy trang hợp lệ để dịch",
                    status_code=404,
                )
        for page in sorted_pages:
            transcript_lines = sorted(page.transcript_lines, key=lambda tl: tl.line_index)
            transcripts: List[Transcript] = []
            bboxes: List[List[int]] = []
            line_ids: List[int] = []

            for tl in transcript_lines:
                speaker_name = tl.speaker_character.source_name if tl.speaker_character else "unsure"
                target_name = tl.target_character.source_name if tl.target_character else None
                bbox = _normalize_bbox(tl.bbox)
                bbox_int: List[int] | None = None
                if bbox:
                    bbox_int = [int(coord) for coord in bbox]
                    bboxes.append(bbox_int)

                transcripts.append(
                    Transcript(
                        speaker=speaker_name,
                        text=tl.text or "",
                        text_speech_type=tl.text_speech_type or "",
                        target=target_name,
                        translation=None,
                        bbox=bbox_int,
                        line_id=tl.id,
                    )
                )
                if tl.id is not None:
                    line_ids.append(tl.id)

            transcript_per_page.append(transcripts)
            transcript_bboxes_per_page.append(bboxes)
            page_line_ids_per_page.append(line_ids)
            page_info_batch.append(
                PageInfo(
                    page_number=page.page_number,
                    story_name=story_name,
                    chapter_number=chapter_number,
                )
            )

            if not page.image_path:
                raise ApiException(
                    message=f"Page {page.page_number} is missing image_path",
                    status_code=500,
                )

            image_array = read_image_file(page.image_path)
            page_arrays.append(image_array)
            page_images.append(Image.fromarray(image_array.astype(np.uint8)))

            filename = os.path.basename(page.image_path)
            if not os.path.splitext(filename)[1]:
                filename = f"{filename}.png"
            page_names.append(filename or f"page_{page.page_number}.png")

        # Lấy đầy đủ metadata (eager load characters, mapping_names, address_matrix, translate_dicts)
        metadata_story = await MetadataRepository.get_story_by_name(story_name)
        if not metadata_story:
            raise ApiException(message=f"Story {story_name} metadata not found", status_code=404)

        topic_value = metadata_story.story_type if isinstance(metadata_story.story_type, Topic) else Topic(metadata_story.story_type)
        translated_transcript_per_page: List[List[str]] = []

        mapping_name: Dict[str, str] = {}
        for mapping in metadata_story.mapping_names:
            if mapping.language == target_lang:
                mapping_name[mapping.source] = mapping.translation
                
        logger.debug(f"mapping_name: {mapping_name}")

        # translate_dictionary: use first available translate dict for target language if present, else first
        translate_dictionary: Dict[str, str] = {}
        if metadata_story.translate_dicts:
            # ưu tiên theo ngôn ngữ đích nếu có
            preferred = next((td for td in metadata_story.translate_dicts if getattr(td.language, "value", str(td.language)) == target_lang.value), None)
            chosen = preferred or metadata_story.translate_dicts[0]
            translate_dictionary = dict(chosen.dictionary or {})
        logger.debug(f"translate_dictionary: {translate_dictionary}")
        
        address_matrix: Dict[str, Dict[str, str]] = {}
        character_profiles: List[Dict[str, str]] = []
        for speaker in metadata_story.characters:
            character_profiles.append(
                {
                    "name": speaker.source_name,
                    "description": speaker.description or "",
                }
            )
            for am in speaker.address_matrixes_as_speaker:
                if am.target_id:
                    target_char = next((c for c in metadata_story.characters if c.id == am.target_id), None)
                    target_name = target_char.source_name if target_char else None
                else:
                    target_name = "other"
                if target_name:
                    address_matrix.setdefault(speaker.source_name, {})[target_name] = am.description
        logger.debug(f"address_matrix: {address_matrix}")
                    
        knowledge = Knowledge(
            story_name=story_name, 
            topic=topic_value,
            mapping_name=mapping_name,
            translate_dict=translate_dictionary,
            address_matrix=address_matrix,
            characters=character_profiles,
        )
        line_ids_flat = [line_id for ids in page_line_ids_per_page for line_id in ids]
        existing_translations = await EpisodeRepository.get_translations_for_lines(
            line_ids_flat,
            target_lang,
        )
        for transcripts, line_ids in zip(transcript_per_page, page_line_ids_per_page):
            for transcript, line_id in zip(transcripts, line_ids):
                transcript.translation = existing_translations.get(line_id)

        missing_translation_ids = [
            line_id for line_id in line_ids_flat if existing_translations.get(line_id) is None
        ]
        should_translate = force_translate or bool(missing_translation_ids)

        if not should_translate:
            translated_transcript_per_page = [
                [transcript.translation or "" for transcript in transcripts]
                for transcripts in transcript_per_page
            ]
        else:
            try:
                translator = Translator(
                    input_lang=metadata_story.source_language.value if hasattr(metadata_story.source_language, "value") else str(metadata_story.source_language),
                    target_lang=target_lang.value,
                    story_name=story_name,
                    knowledge=knowledge,
                    topic=topic_value,
                )
                image_batch = [np.array(image) for image in page_images]

                batch_results = await translator.translate_batch(
                    transcript_per_page,
                    image_batch=image_batch,  # Pass NumPy arrays instead of PIL.Image.Image objects
                    page_info_batch=page_info_batch,
                )
                if batch_results and isinstance(batch_results[0], tuple):
                    translated_transcript_per_page = [item[0] for item in batch_results]
                else:
                    translated_transcript_per_page = batch_results
                logger.debug(
                    "Translated %d pages for story '%s' chapter %d",
                    len(translated_transcript_per_page),
                    story_name,
                    chapter_number,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Lỗi khi dịch tập truyện: %s", exc, exc_info=True)
                raise ApiException(
                    message="Không thể dịch tập truyện hiện tại",
                    status_code=500,
                ) from exc

            line_translation_map: Dict[int, str] = {}
            for line_ids, translations in zip(page_line_ids_per_page, translated_transcript_per_page):
                for line_id, translated_text in zip(line_ids, translations):
                    if line_id is not None:
                        line_translation_map[line_id] = translated_text
            await EpisodeRepository.upsert_translations(
                target_lang,
                line_translation_map,
            )
        
        features = Features()
        features.parse_init_params({
            "use_gpu_limited": True,
            "models_ttl": 1,
        })

        processed_images: List[Image.Image] = []
        for index, (page_array, translated_transcripts, bboxes) in enumerate(
            zip(page_arrays, translated_transcript_per_page, transcript_bboxes_per_page)
        ):
            try:
                if not translated_transcripts:
                    processed_images.append(Image.fromarray(page_array.astype(np.uint8)))
                    logger.warning("Không có transcript cho trang %d", index)
                    continue

                if not bboxes:
                    processed_images.append(Image.fromarray(page_array.astype(np.uint8)))
                    logger.warning("Không tìm thấy bounding box cho trang %d", index)
                    continue

                quads_input = [[float(coord) for coord in bbox] for bbox in bboxes]
                quads = np.asarray(bbox_to_quad(quads_input))
                text_bboxes = []
                for quad, translation in zip(quads, translated_transcripts):
                    tb = TextBlock(
                        lines=[quad.astype(int).tolist()],
                        texts=[""],
                        translation=translation,
                        target_lang=target_lang.value,
                    )
                    text_bboxes.append(tb)

                out = await features.inpaint(
                    image=np.array(page_array),
                    config=Config(),
                    text_bboxes=text_bboxes,
                )
                result_image = out.result if out.result else Image.fromarray(page_array.astype(np.uint8))
                processed_images.append(result_image)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Lỗi khi xử lý trang %d: %s", index, exc, exc_info=True)
                processed_images.append(Image.fromarray(page_array.astype(np.uint8)))

        response_mode_normalized = response_mode if response_mode in {"zip", "json", "inline"} else "zip"
        response_strategy = ResponseFactory.get_strategy(response_mode_normalized)
        end_time = time()
        logger.debug("Time taken to translate and inpaint: %.2f seconds", end_time - start_time)
        original_images = page_images if response_mode_normalized == "inline" else None
        return response_strategy.process_images(
            processed_images=processed_images,
            page_names=page_names,
            original_images=original_images,
        )