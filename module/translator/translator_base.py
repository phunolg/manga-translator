import json
from pprint import pformat
from typing import Any, List, Optional      

from module.history_retrival.type import PageInfo
from module.knowledge import Knowledge
from module.knowledge.type import Topic
from module.llm import get_llm
from module.llm.base import TokenLimitExceeded
from module.llm.constants import LLM_MODEL
from module.llm import get_llm
from type import Transcript
from utils.log import setup_logger
import asyncio
import re
from PIL import Image
import numpy as np
logger = setup_logger(__name__)
from abc import ABC, abstractmethod

class TranslatorBase(ABC):
    def __init__(self, input_lang: str, target_lang: str, story_name: str, topic: Topic = Topic.BASIC):
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.llm = get_llm(LLM_MODEL.gemma3)
        self.max_tokens_per_chunk = 1024
        self.story_name = story_name
        
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
                text = match1.group(1)
            elif match2:
                text = match2.group(2)
            elif match3:
                text = match3.group(1)
            elif match4:
                text = match4.group(1)
            out.append(text.strip().strip("\"").strip("\'").strip())
        return out
    
    async def translate_step(self, translate_prompt_user: str, system_prompt: str, transcript: str, image: Image.Image = None) -> str:
        try:
            # Thử dịch toàn bộ danh sách
            translated_script = await self.llm.get_answer(
                question=translate_prompt_user,
                prompt_system=system_prompt,
                image=image,
                history=None,
                temperature=0.1,
                top_p=0.1,
            )
        except TokenLimitExceeded:
            logger.warning(
                "Token limit exceeded. Splitting texts into smaller chunks...")
            chunks = self._split_texts(transcript, system_prompt)
            logger.debug(f"chunks:\n{pformat(chunks)}")

            # Không sử dụng history cho batch
            chunk_results = await self.llm.get_batch_answer(
                questions=chunks,
                prompt_system=system_prompt,
                image=image,
                history=None,  # Không dùng history cho batch
            )

            translated_script = "\n".join(chunk_results)
            logger.debug(f"Chunk translated:\n{translated_script}")
        except Exception as e:
            logger.error(f"Error in translation: {e}", exc_info=True)
            translated_script = ""
            
        finally:
            return translated_script

    @abstractmethod
    async def _translate(self, texts: List[Transcript], image: Image.Image, page_info: PageInfo, **kwargs) -> tuple[List[str], List[Transcript]]:
        pass
       
  
    async def translate_batch(
        self, 
        texts_batch: List[List[Transcript]],
        image_batch: List[np.ndarray],
        page_info_batch: List[PageInfo],
    ) -> list[tuple[List[str], List[Transcript]]]:
        return await asyncio.gather(*[self._translate(texts, Image.fromarray(image), page_info) for texts, image, page_info in zip(texts_batch, image_batch, page_info_batch)])
