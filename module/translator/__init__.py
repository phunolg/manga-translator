import logging
from pprint import pformat
from typing import Any, List, Optional
import PIL
from typing_extensions import Dict

from module.llm import get_llm
from module.llm.constants import LLM_MODEL
from module.quality_assurance import QualityAssurance
from module.history_retrival import history_retrival
from module.history_retrival.type import PageInfo, StoryContext
from module.knowledge import Knowledge
from module.knowledge.type import Topic
from module.quality_assurance.type import ErrorResult
from module.llm.base import TokenLimitExceeded
from module.translator.prompts import inside_prompt_system, inside_prompt_user, translate_prompt_system_v2, translate_prompt_user, try_translate_again_prompt_user
from settings import LIMIT_TRY_AGAIN
from utils.log import setup_logger
import asyncio
import re
from PIL import Image

logger = setup_logger(__name__)


class Translator:
    def __init__(self, target_lang: str, story_name: str, topic: Topic = Topic.BASIC):
        self.target_lang = target_lang
        self.llm = get_llm(LLM_MODEL.gemma3)
        self.max_tokens_per_chunk = 1024
        self.story_name = story_name
        self.knowledge = Knowledge(story_name=story_name, topic=topic)
        
    def _split_texts(self, text: str, prompt_system: str) -> List[str]:
        """
        Chia danh sách văn bản thành các đoạn nhỏ hơn dựa trên số token.
        Sử dụng phương pháp chia đôi đệ quy cho đến khi mỗi chunk đủ nhỏ.
        """
        # Tách text thành danh sách các dòng
        texts = [line for line in text.splitlines() if line.strip()]

        # Kiểm tra xem toàn bộ văn bản có vượt quá giới hạn token không
        full_text = "\n".join(texts)
        if self.llm.count_tokens(full_text) + self.llm.count_tokens(prompt_system) <= self.max_tokens_per_chunk:
            return [full_text]

        # Nếu chỉ có một dòng văn bản nhưng vẫn vượt quá giới hạn
        if len(texts) == 1:
            return texts

        # Chia đệ quy
        return self._split_recursively(texts, prompt_system)

    def _split_recursively(self, texts: List[str], prompt_system: str) -> List[str]:
        """Hàm đệ quy chia danh sách văn bản thành các chunk nhỏ hơn."""
        # Nếu chỉ có một text hoặc không có text nào
        if len(texts) <= 1:
            return texts

        # Kiểm tra xem chunk hiện tại có vượt quá giới hạn token không
        current_text = "\n".join(texts)
        total_tokens = self.llm.count_tokens(
            current_text) + self.llm.count_tokens(prompt_system)

        if total_tokens <= self.max_tokens_per_chunk:
            return [current_text]

        # Chia đôi danh sách và gọi đệ quy
        mid = len(texts) // 2
        left_half = self._split_recursively(texts[:mid], prompt_system)
        right_half = self._split_recursively(texts[mid:], prompt_system)

        # Kết hợp kết quả
        return left_half + right_half

    def extract_texts_from_llm(self, s: str) -> List[str]:
        """
        Trích xuất văn bản từ kết quả LLM, loại bỏ các định dạng như <|số|> và <tên nhân vật>:
        """
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            # Xử lý định dạng <|số|> nếu có
            m = re.match(r'^\s*<\|\s*(\d+)\s*\|>\s*(.*)$', ln)
            if m:
                text = m.group(2).strip()
            else:
                text = ln

            if text.endswith(","):
                text = text[:-1] + "."

            # Loại bỏ phần "<tên nhân vật>:" hoặc "Tên nhân vật:" ở đầu văn bản
            # Pattern 1: <Name>: Text, [Name]: Text, "Name": Text
            pattern1 = r'^\s*(?:<[^>]+>|\[[^\]]+\]|"[^"]+"):\s*(.*)'
            # Pattern 2: Name: Text (chỉ bắt tên nhân vật viết hoa ở đầu)
            pattern2 = r'^\s*([A-Z][^:]{0,30}):\s*(.*)'
            # Pattern 3: Nhân vật A: Text, Character B: Text
            pattern3 = r'^\s*(?:\s+\w+|Character\s+\w+):\s*(.*)'
            # Pattern 4: <Name> Text  (không có dấu :)
            pattern4 = r'^\s*<[^>]+>\s*(.*)'

            match1 = re.match(pattern1, text)
            match2 = re.match(pattern2, text)
            match3 = re.match(pattern3, text, re.IGNORECASE)
            match4 = re.match(pattern4, text, re.IGNORECASE)

            if match1:
                text = match1.group(1).strip()
            elif match2:
                text = match2.group(2).strip()
            elif match3:
                text = match3.group(1).strip()
            elif match4:
                text = match4.group(1).strip()
            out.append(text.strip("\""))
        return out
    
    
    async def consolidate_suggestions(
        self,
        input_text: str,
        output_text: str,
        all_errors: List[ErrorResult],
        system_prompt: str,
        image_chapter: PIL.Image.Image = None,
        history_translate: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Gộp các ý kiến sửa đổi từ nhiều loại lỗi thành một đề xuất chỉnh sửa duy nhất
        
        Args:
            input_text: Văn bản gốc
            output_text: Bản dịch hiện tại
            all_errors: Danh sách các lỗi được phát hiện
            system_prompt: Prompt system
            image_chapter: Hình ảnh chương (tùy chọn)
            history_translate: History của việc dịch
        Returns:
            Một đề xuất chỉnh sửa duy nhất
        """
        if not all_errors:
            return output_text
        
        error_per_translation = {}
        for error in all_errors:
            key = f"[Bản gốc]: {error.original_segment}\n[Bản viết lại]: {error.translation_segment}"
            if key not in error_per_translation:
                error_per_translation[key] = []
            if error.errors:
                error_per_translation[key].extend(error.errors)
            
        error_per_translation_text = ""
        for key, errors in error_per_translation.items():
            if not errors:
                continue
            error_per_translation_text += f"{key}\n"
            for error in errors:
                error_per_translation_text += f"- Lỗi {error.error_type}/{error.subcategory}. Giải thích: {error.description}. Cách sửa: {error.suggestion}\n"
            error_per_translation_text += "\n"
                
        logger.debug(f"error_per_translation_text: {error_per_translation_text}")
        
        # Gọi LLM để gộp các đề xuất
        correction_prompt = try_translate_again_prompt_user.format(
            error=error_per_translation_text,
            source=input_text,
            answer=output_text
        )
        res = await llm.get_answer(
            question=correction_prompt,
            prompt_system=system_prompt,
            history=None,
        )
        history_translate.extend([
            {"role": "user", "content": correction_prompt},
            {"role": "assistant", "content": res}
        ])
            
        return res     

    async def get_inside_transcript(self, transcript: str, context: StoryContext) -> str:
        """sử dụng LLM để bref chi tiết ý muốn, ý định, suy nghĩ của lời nói kết hợp với context
        Args:
            transcript: str
            context: StoryContext
        Returns:
            str
        """
        inside = await llm.get_answer(
            question=inside_prompt_user.format(transcript=transcript),
            prompt_system=inside_prompt_system.format(context=context.local_window),
        )
        return inside

    async def _translate(self, texts: List[str], image: Image.Image, page_info: PageInfo) -> List[str]:
        """Dịch danh sách văn bản, tự động chia nhỏ nếu cần."""
        logger.info(f"========== Translator ==========")
        logger.info(f"{page_info}")

        if not texts:
            return []
        
        context = await history_retrival.get_context(page_info)
        transcript = ""
        for i, text in enumerate(texts, 1):
            clean_text = text.replace("\n", " ")
            transcript += f"<|{i}|>{clean_text}\n"
        logger.debug(f"transcript: {transcript}")
        inside_transcript = await self.get_inside_transcript(transcript, context)
        prompt_system = await self._build_system_prompt(context, inside_transcript)

        
        translate_history = []
        try:
            # Thử dịch toàn bộ danh sách
            answer = await self.llm.get_answer(
                question=translate_prompt_user.format(source=transcript),
                prompt_system=prompt_system,
                image=image,
                history=None
            )
            translate_history.extend([{"role": "user", "content": translate_prompt_user.format(source=transcript)}, {"role": "assistant", "content": answer}])
            quality_assurance = QualityAssurance(story_context=context, knowledge=self.knowledge, inside=inside_transcript, source_lang="ENG", target_lang=self.target_lang)
            try:
                try_again_count = 0
                while True:
                    # Kiểm tra chất lượng lại
                    is_pass, errors = await quality_assurance.check_quality(transcript, answer, image)
                    if is_pass:
                        break
                    
                    # Gọi LLM để sửa lỗi - sử dụng history luân phiên
                    answer = await self.consolidate_suggestions(transcript, answer, errors, prompt_system, image, translate_history)
                    try_again_count += 1
                    if try_again_count == LIMIT_TRY_AGAIN:
                        break

                if try_again_count == LIMIT_TRY_AGAIN:
                    logger.warning(
                        f"Quality assurance failed after {LIMIT_TRY_AGAIN} tries")
            except Exception as e:
                logger.warning(
                    f"Error in quality assurance: {e}", exc_info=True)

            result = self.extract_texts_from_llm(answer)

        except TokenLimitExceeded:
            logger.warning(
                "Token limit exceeded. Splitting texts into smaller chunks...")
            chunks = self._split_texts(transcript, prompt_system)
            logger.debug(f"chunks:\n{pformat(chunks)}")

            # Không sử dụng history cho batch
            chunk_results = await self.llm.get_batch_answer(
                questions=chunks,
                prompt_system=prompt_system,
                image=image,
                history=None,  # Không dùng history cho batch
            )

            translated_script = "\n".join(chunk_results)
            logger.debug(f"Chunk translated:\n{translated_script}")
            result = self.extract_texts_from_llm(translated_script)

        except Exception as e:
            logger.error(f"Error in translation: {e}", exc_info=True)
            result = []
            
        finally:
            return result

    async def _build_system_prompt(
        self,          
        context: StoryContext,
        inside_transcript: str ,
    ) -> str:
        prompt_system = translate_prompt_system_v2.format(
            to_lang=self.target_lang)
        
        prompt_system += "\n\n" + str(context) + "\n\n" + inside_transcript

        logger.info(f"prompt_system: {prompt_system}")
        return prompt_system

    async def translate_batch(
        self, texts_batch: List[List[str]],
        image_batch: List[Image.Image],
        page_info_batch: List[PageInfo],
    ) -> List[List[str]]:

        image_batch = [Image.fromarray(image) for image in image_batch]
        if not texts_batch:
            return []
        return await asyncio.gather(*[self._translate(texts, image, page_info) for texts, image, page_info in zip(texts_batch, image_batch, page_info_batch)])
