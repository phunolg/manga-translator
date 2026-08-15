from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel


class TranscriptLineResponse(BaseModel):
    line_index: int
    speaker: Optional[str] = None
    text: Optional[str] = None
    text_speech_type: Optional[str] = None
    target: Optional[str] = None
    translation: Optional[str] = None
    bbox: Optional[Any] = None


class PageDetailResponse(BaseModel):
    id: int
    page_number: int
    prose: Optional[str] = None
    image_url: Optional[str] = None
    transcripts: List[TranscriptLineResponse] = []


class EpisodeDetailResponse(BaseModel):
    id: int
    story_id: int
    story_name: str
    chapter_number: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    pages: List[PageDetailResponse] = []
