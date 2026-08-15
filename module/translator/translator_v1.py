import json
from pprint import pformat
from typing import List

from module.history_retrival import history_retrival
from module.history_retrival.type import PageInfo, StoryContext
from module.knowledge import Knowledge
from module.knowledge.type import Topic
from module.translator.prompts import check_correct_translate_prompt_system, correct_address_translate_prompt_system, improve_translate_prompt_system_v2, inside_prompt_system, inside_prompt_user, prompt_translate_with_genre, translate_prompt_system_v2, translate_prompt_user, translate_use_dictionary_prompt_system
from module.translator.translator_base import TranslatorBase
from type import Transcript
from utils.log import setup_logger
from PIL import Image

logger = setup_logger(__name__)


class Translator(TranslatorBase):
    def __init__(self, input_lang: str, target_lang: str, story_name: str, knowledge: Knowledge, topic: Topic = Topic.BASIC):
        super().__init__(input_lang, target_lang, story_name, topic)
        self.knowledge = knowledge
        
    async def get_inside_transcript(self, transcript: str, context: StoryContext) -> str:
        """sử dụng LLM để bref chi tiết ý muốn, ý định, suy nghĩ của lời nói kết hợp với context
        Args:
            transcript: str
            context: StoryContext
        Returns:
            str
        """
        inside = await self.llm.get_answer(
            question=inside_prompt_user.format(transcript=transcript),
            prompt_system=inside_prompt_system.format(context=context.local_window),
        )
        return inside

    async def _translate(self, texts: List[Transcript], image: Image.Image, page_info: PageInfo, **kwargs) -> tuple[List[str], List[Transcript]]:
        """Dịch danh sách văn bản, tự động chia nhỏ nếu cần."""
        logger.info(f"========== Translator ==========")
        logger.info(f"{page_info}")

        if not texts:
            return [], []
        
        context = await history_retrival.get_context(page_info)
        logger.debug(f"context: {context}")
        transcript = ""
        for i, text in enumerate(texts, 1):
            clean_text = text.get_source_text()
            transcript += f"<|{i}|>{clean_text}\n"
        
        user_prompt = translate_prompt_user.format(transcript=transcript)
            
        logger.debug(f"transcript: {transcript}")
        inside_transcript = await self.get_inside_transcript(transcript, context)
        address_matrix = self.knowledge.get_address_matrix(return_type='string')
        logger.debug(f"address_matrix: {address_matrix}")
        
        logger.info("Dịch theo ngữ cảnh")
        # Dịch theo ngữ cảnh
        prompt_system = translate_prompt_system_v2.format(to_lang=self.target_lang, context=context)
        translated_script = await self.translate_step(user_prompt, prompt_system, transcript)
        
        # logger.info("Dịch theo cảm xúc, suy nghĩ")
        # # Chỉnh sửa lại cho đúng với cảm xúc, suy nghĩ
        # prompt_system = translate_use_inside_prompt_system.format(to_lang=self.target_lang, inside=inside_transcript, translation=translated_script)
        # translated_script = await self.translate_step(prompt_system, transcript, image)

        logger.info("Kiểm tra lại dịch đúng")
        prompt_system = check_correct_translate_prompt_system.format(targetLang=self.target_lang, translation=translated_script, inside=inside_transcript)
        translated_script = await self.translate_step(user_prompt, prompt_system, transcript)
        
        logger.info("Dịch theo thể loại")
        prompt_system = prompt_translate_with_genre[self.knowledge.topic].format(targetLang=self.target_lang, translation=translated_script, source=transcript)
        translated_script = await self.translate_step(user_prompt, prompt_system, transcript)
        
        logger.info("Dịch theo từ điển tên riêng, thuật ngữ, ...")
        # Chỉnh sửa lại cho đúng với từ điển tên riêng, thuật ngữ, ...
        dictionary = self.knowledge.get_dictionary(return_type='string') + "\n\n" + self.knowledge.get_name_mapping(return_type='string')
        prompt_system = translate_use_dictionary_prompt_system.format(to_lang=self.target_lang, dictionary=dictionary, translation=translated_script, source=transcript)
        translated_script = await self.translate_step(user_prompt, prompt_system, transcript)
        
        logger.info("Chỉnh sửa lại cho đúng với xưng hô")
        prompt_system = correct_address_translate_prompt_system.format(targetLang=self.target_lang, translation=translated_script, inside=inside_transcript, source=transcript, address_matrix=address_matrix)
        # logger.debug(f"prompt_system: {prompt_system}")
        translated_script = await self.translate_step(user_prompt, prompt_system, transcript)
        
        logger.info("Cải thiện lại cho trôi chảy")
        # Cải thiện lại cho trôi chảy
        prompt_system = improve_translate_prompt_system_v2.format(targetLang=self.target_lang, translation=translated_script, inside=inside_transcript, dictionary=dictionary)
        translated_script = await self.translate_step(user_prompt, prompt_system, transcript)
        
        translations = self.extract_texts_from_llm(translated_script)
        for i, translation in enumerate(translations):
            texts[i].translation = translation
        
        return translations, texts
