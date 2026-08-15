from pprint import pformat
import pydantic
import os
from settings import BASE_DIR


class PageInfo(pydantic.BaseModel):
    story_name: str
    chapter_number: int
    page_number: int

    def __str__(self) -> str:
        return f"Story name: {self.story_name}, Chapter number: {self.chapter_number}, Page number: {self.page_number}"

    def get_save_path(self) -> str:
        return os.path.join(BASE_DIR, "transcript_history", self.story_name, str(self.chapter_number), str(self.page_number) + ".json")


class LocalWindow(pydantic.BaseModel):
    prev_window: list[str]
    current_window: list[str]
    next_window: list[str]

    def __str__(self):
        return f"Ngữ cảnh tham chiếu:\n**Cuộc hội thoại của các tập trước:**\n{pformat(self.prev_window)}\n**Cuộc hội thoại của trang hiện tại:**\n{pformat(self.current_window)}\n**Cuộc hội thoại của các tập sau:**\n{pformat(self.next_window)}"


class StoryContext(pydantic.BaseModel):
    local_window: LocalWindow
    summary: str

    def __str__(self):
        return f"{str(self.local_window)}"
        # return f"{self.local_window}\n**Tóm tắt sự kiện:**\n{self.summary}"
