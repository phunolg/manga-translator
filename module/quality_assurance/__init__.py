import asyncio
import json
from pprint import pformat
from typing import Any, Dict, List, Tuple
from module.history_retrival.type import StoryContext
from module.knowledge import Knowledge
from module.quality_assurance.prompts import  error_descriptions
from module.quality_assurance.type import ErrorResult, ErrorTranslation, ErrorType
from utils.log import setup_logger
import PIL.Image
from enum import Enum

logger = setup_logger(__name__)




class QualityAssurance:
    def __init__(self, story_context: StoryContext, knowledge: Knowledge, inside: str, source_lang: str = "ENG", target_lang: str = "VIE"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.story_context = story_context
        self.knowledge = knowledge
        self.inside = inside
        self.system_prompt = error_descriptions["system_prompt"]
        
    async def check_error_by_type(
        self,
        error_type: ErrorType,
        input: str,
        output: str,
        image_chapter: PIL.Image.Image = None,
    ) -> List[ErrorResult]:
        """
        Kiểm tra một loại lỗi cụ thể trong bản dịch
        
        Args:
            error_type: Loại lỗi cần kiểm tra
            input: Văn bản gốc
            output: Bản dịch
        Returns:
            Danh sách các lỗi được phát hiện
        """

        user_prompt = error_descriptions[error_type].format(
            source=input,
            translation=output,
            error_type=error_type.value,
            targetLang=self.target_lang,
            name_dictionary=self.knowledge.get_name_mapping("string"),
            term_dictionary=self.knowledge.get_dictionary("string"),
            genre=self.knowledge.get_topic().value,
            context=str(self.story_context),
            inside=self.inside
        )
        # Gọi LLM
        res = await llm.get_answer(
            question=user_prompt,
            prompt_system=self.system_prompt,
            image=None,
            history=None,
        )
        results = []
        for item in res:
            error_translations = [ErrorTranslation(**error) for error in item["errors"]]
            result = ErrorResult(
                original_segment=item["original_segment"],
                translation_segment=item["translation_segment"],
                errors=error_translations
            )
            results.append(result)
        return results

    async def check_quality(self, input: str, output: str, image_chapter: PIL.Image.Image, history: List[Dict[str, str]] = None) -> Tuple[bool, List[ErrorResult]]:
        return await self.check_quality_by_llm(input, output, image_chapter, history)

    async def check_quality_by_llm(
        self,
        input: str,
        output: str,
        image_chapter: PIL.Image.Image,
        history: List[Dict[str, str]] = None
    ) -> Tuple[bool, List[ErrorResult]]:
        """
        Kiểm tra chất lượng bản dịch bằng cách gọi nhiều LLM riêng biệt cho từng loại lỗi
        và sau đó gộp các đề xuất chỉnh sửa
        """
        # Danh sách các loại lỗi cần kiểm tra
        error_types = [
            ErrorType.MISSING_TRANSLATION,
            # ErrorType.CONTEXT,
            # ErrorType.ACCURACY,
            ErrorType.NAMING_CONVENTION,
            ErrorType.FLUENCY,
            ErrorType.STYLE,
        ]
        
        # Gọi các LLM đồng thời để kiểm tra từng loại lỗi
        error_tasks = [
            self.check_error_by_type(
                error_type,
                input,
                output, 
                image_chapter
            )
            for error_type in error_types
        ]
        
        # Chờ tất cả các task hoàn thành
        all_errors_by_type = await asyncio.gather(*error_tasks)
        
        # Tổng hợp tất cả các lỗi
        all_errors = []
        for errors in all_errors_by_type:
            all_errors.extend(errors)
        is_pass = len(all_errors) == 0
        return is_pass, all_errors
    

