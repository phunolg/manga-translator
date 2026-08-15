import os
from typing import List, Optional

from type import Transcript
from utils.log import setup_logger
from settings import BASE_DIR, HISTORY_WINDOW_SIZE
from module.history_retrival.type import LocalWindow, PageInfo, StoryContext
from module.database.postgre.repository.episode_repository import EpisodeRepository
from module.database.postgre.tables import Page, TranscriptLine


logger = setup_logger(__name__)


class HistoryRetrival:
    def __init__(self, window_size: int = 1, summary_length: int = 10):
        """
        window_size: số trang trước/sau để lấy context local
        summary_length: số lượng trang tối đa để lấy cho summary
        """
        self.folder_path = os.path.join(BASE_DIR, "transcript_history")
        os.makedirs(self.folder_path, exist_ok=True)
        self.window_size = window_size
        self.summary_length = summary_length

    async def _get_story_id(self, story_name: str) -> Optional[int]:
        story = await EpisodeRepository.get_story_by_name(story_name)
        if not story:
            return None
        return int(str(story.id))

    async def _get_pages_of_chapter(self, story_id: int, chapter_number: int) -> List[Page]:
        """Lấy danh sách Page trong 1 chapter từ database"""
        if chapter_number <= 0:
            return []
        episode = await EpisodeRepository.get_episode_with_details(story_id, chapter_number)
        if not episode:
            return []
        pages = sorted(episode.pages, key=lambda page: page.page_number or 0)
        return pages

    def _page_to_description(self, page: Optional[Page], chapter_number: int, story_name: str) -> str:
        if not page:
            return "Không tìm thấy dữ liệu để đọc"

        transcripts = []
        sorted_lines = sorted(page.transcript_lines, key=lambda tl: tl.line_index if tl.line_index is not None else 0)
        for tl in sorted_lines:
            transcripts.append(
                    Transcript(
                        speaker=tl.speaker_character.source_name if tl.speaker_character else "unknown",
                        target=tl.target_character.source_name if tl.target_character else None,
                        text=tl.text or "",
                        text_speech_type=tl.text_speech_type or "speaking",
                        translation=next((tr.translated for tr in tl.translations), None),
                        bbox=tl.bbox,
                    ).get_source_text()  
            )

        transcript_text = "\n".join(transcripts) if transcripts else "Không có hội thoại"
        prose = page.prose or ""
        return (
            f"Mô tả trang {page.page_number} tập {chapter_number} của truyện {story_name}:\n"
            f"{prose}\nCuộc hội thoại gồm:\n{transcript_text}"
        )

    async def get_local_window(self, page_info: PageInfo, story_id: int) -> LocalWindow:
        """
        Lấy transcript của các trang quanh trang hiện tại.
        Ví dụ: page=10, window=2 → lấy [8,9,10,11,12].
        """
        pages_current = await self._get_pages_of_chapter(story_id, page_info.chapter_number)
        total_pages = len(pages_current)
        page_lookup = {page.page_number: page for page in pages_current}
        prev_pages_cache: Optional[List[Page]] = None
        next_pages_cache: Optional[List[Page]] = None

        prev_conversation: List[str] = []
        current_conversation: List[str] = []
        next_conversation: List[str] = []

        if total_pages == 0:
            return LocalWindow(prev_window=prev_conversation, current_window=current_conversation, next_window=next_conversation)

        for offset in range(-self.window_size, self.window_size + 1):
            target_page_number = page_info.page_number + offset
            target_chapter = page_info.chapter_number
            page_obj: Optional[Page] = None

            if 1 <= target_page_number <= total_pages:
                page_obj = page_lookup.get(target_page_number)
            elif target_page_number < 1:
                if page_info.chapter_number > 1:
                    if prev_pages_cache is None:
                        prev_pages_cache = await self._get_pages_of_chapter(story_id, page_info.chapter_number - 1)
                    if prev_pages_cache:
                        prev_total = len(prev_pages_cache)
                        adjusted = prev_total + target_page_number
                        if 1 <= adjusted <= prev_total:
                            page_obj = prev_pages_cache[adjusted - 1]
                            target_chapter = page_info.chapter_number - 1
            else:
                if next_pages_cache is None:
                    next_pages_cache = await self._get_pages_of_chapter(story_id, page_info.chapter_number + 1)
                if next_pages_cache:
                    adjusted = target_page_number - total_pages
                    if 1 <= adjusted <= len(next_pages_cache):
                        page_obj = next_pages_cache[adjusted - 1]
                        target_chapter = page_info.chapter_number + 1

            description = self._page_to_description(page_obj, target_chapter, page_info.story_name)

            if offset < 0:
                prev_conversation.append(description)
            elif offset == 0:
                current_conversation.append(description)
            else:
                next_conversation.append(description)

        return LocalWindow(prev_window=prev_conversation, current_window=current_conversation, next_window=next_conversation)

    async def get_summary(self, page_info: PageInfo, story_id: int) -> str:
        """
        Lấy summary từ các trang trước local window.
        Ví dụ: page=10, window=2 → local [8..12], summary [3..7].
        """
        if self.summary_length == 0:
            return ""

        pages = await self._get_pages_of_chapter(story_id, page_info.chapter_number)
        if not pages:
            return ""

        current_idx_zero = max(0, page_info.page_number - 1)
        local_start_zero = max(0, current_idx_zero - self.window_size)
        summary_end_zero = local_start_zero
        summary_start_zero = max(0, summary_end_zero - self.summary_length)

        if summary_start_zero == summary_end_zero:
            return ""

        selected_pages = pages[summary_start_zero:summary_end_zero]
        descriptions = [
            self._page_to_description(page, page_info.chapter_number, page_info.story_name) for page in selected_pages
        ]
        return "\n".join(descriptions)

    def select_page_fit_to_current_page(self, page_info: PageInfo) -> PageInfo:
        """
        TODO: Hàm chọn trang phù hợp với current_page (có thể implement sau).
        """
        raise NotImplementedError("Chưa implement")
        return page_info

    async def get_context(self, page_info: PageInfo) -> StoryContext:
        story_id = await self._get_story_id(page_info.story_name)
        if not story_id:
            return StoryContext(
                local_window=LocalWindow(prev_window=[], current_window=[], next_window=[]),
                summary="",
            )

        local_window = await self.get_local_window(page_info, story_id)
        summary = await self.get_summary(page_info, story_id)
        return StoryContext(local_window=local_window, summary=summary)


history_retrival = HistoryRetrival(window_size=HISTORY_WINDOW_SIZE)

if __name__ == "__main__":
    import asyncio

    async def _debug():
        page_info = PageInfo(story_name="Yule", chapter_number=136, page_number=10)
        context = await history_retrival.get_context(page_info)
        logger.info(context)

    asyncio.run(_debug())
