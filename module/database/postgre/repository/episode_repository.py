from module.database.postgre.database import db
from module.database.postgre.tables import (
    Story,
    Episode,
    Page,
    TranscriptLine,
    Character,
    Translation,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Dict, List, Optional
from module.knowledge.type import Language
from type import Transcript


class EpisodeRepository:
    @classmethod
    async def get_story_by_name(cls, story_name: str) -> Optional[Story]:
        """Lấy story theo tên"""
        result = await db.execute(
            select(Story).where(Story.story_name == story_name)
        )
        return result.unique().scalar_one_or_none()
    
    @classmethod
    async def get_or_create_episode(
        cls,
        story_id: int,
        chapter_number: int
    ) -> Episode:
        """Lấy hoặc tạo episode mới"""
        result = await db.execute(
            select(Episode).where(
                Episode.story_id == story_id,
                Episode.chapter_number == chapter_number
            )
        )
        episode = result.unique().scalar_one_or_none()
        
        if not episode:
            episode = Episode(
                story_id=story_id,
                chapter_number=chapter_number
            )
            await db.add(episode)
            await db.commit()
            await db.refresh(episode)
        
        return episode
    
    @classmethod
    async def get_character_by_name(
        cls,
        story_id: int,
        character_name: str
    ) -> Optional[Character]:
        """Lấy character theo tên trong story"""
        result = await db.execute(
            select(Character).where(
                Character.story_id == story_id,
                Character.source_name == character_name
            )
        )
        return result.scalar_one_or_none()
    
    @classmethod
    async def create_page(
        cls,
        episode_id: int,
        page_number: int,
        prose: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Page:
        """Tạo page mới"""
        page = Page(
            episode_id=episode_id,
            page_number=page_number,
            prose=prose,
            image_path=image_path
        )
        await db.add(page)
        await db.commit()
        await db.refresh(page)
        return page
    
    @classmethod
    async def create_transcript_line(
        cls,
        page_id: int,
        line_index: int,
        speaker_id: Optional[int],
        text: Optional[str],
        text_speech_type: Optional[str],
        target_id: Optional[int],
        bbox: Optional[dict]
    ) -> TranscriptLine:
        """Tạo transcript line mới"""
        transcript_line = TranscriptLine(
            page_id=page_id,
            line_index=line_index,
            speaker_id=speaker_id,
            text=text,
            text_speech_type=text_speech_type,
            target_id=target_id,
            bbox=bbox
        )
        await db.add(transcript_line)
        await db.commit()
        await db.refresh(transcript_line)
        return transcript_line
    
    @classmethod
    async def create_episode_with_pages(
        cls,
        story_id: int,
        chapter_number: int,
        pages_data: List[dict]
    ) -> Episode:
        """
        Tạo episode với pages và transcript lines
        
        pages_data: List[dict] với format:
        {
            "page_number": int,
            "prose": str,
            "image_path": str,
            "transcripts": List[Transcript],
            "transcript_bboxes": List[List[int]]
        }
        """
        # Lấy hoặc tạo episode
        episode = await cls.get_or_create_episode(story_id, chapter_number)
        
        # Xóa các pages cũ nếu có (để replace)
        result = await db.execute(
            select(Page).where(Page.episode_id == episode.id)
        )
        old_pages = result.unique().scalars().all()
        for old_page in old_pages:
            # Xóa transcript lines trước
            transcript_lines_result = await db.execute(
                select(TranscriptLine).where(TranscriptLine.page_id == old_page.id)
            )
            transcript_lines = transcript_lines_result.unique().scalars().all()
            for tl in transcript_lines:
                await db.delete(tl)
            await db.delete(old_page)
        await db.commit()
        
        # Tạo tất cả pages và transcript lines trong một transaction
        print(f"[EpisodeRepository] Creating {len(pages_data)} pages for episode {episode.id}")
        created_pages = []
        for page_data in pages_data:
            # Tạo page object (chưa commit)
            page = Page(
                episode_id=episode.id,
                page_number=page_data["page_number"],
                prose=page_data.get("prose"),
                image_path=page_data.get("image_path")
            )
            await db.add(page)
            created_pages.append((page, page_data))
        
        # Flush để lấy IDs của pages
        await db.flush()
        
        # Tạo tất cả transcript lines
        for page, page_data in created_pages:
            transcripts: List[Transcript] = page_data.get("transcripts", [])
            transcript_bboxes = page_data.get("transcript_bboxes", [])
            
            for line_index, (transcript, bbox) in enumerate(zip(transcripts, transcript_bboxes)):
                # Tìm speaker_id từ character name
                speaker_id = None
                if transcript.speaker and transcript.speaker != "unsure":
                    speaker_char = await cls.get_character_by_name(story_id, transcript.speaker)
                    if speaker_char:
                        speaker_id = speaker_char.id
                
                # Tìm target_id từ character name
                target_id = None
                if transcript.target and transcript.target != "unsure" and transcript.target != "Other":
                    target_char = await cls.get_character_by_name(story_id, transcript.target)
                    if target_char:
                        target_id = target_char.id
                
                # Tạo transcript line object (chưa commit)
                transcript_line = TranscriptLine(
                    page_id=page.id,
                    line_index=line_index,
                    speaker_id=speaker_id,
                    text=transcript.text,
                    text_speech_type=transcript.text_speech_type,
                    target_id=target_id,
                    bbox=bbox if bbox else None
                )
                await db.add(transcript_line)
        
        # Commit tất cả cùng một lúc
        await db.commit()
        print(f"[EpisodeRepository] Successfully committed {len(created_pages)} pages and their transcript lines")
        
        # Query lại episode với unique() để tránh lỗi joined eager loads
        result = await db.execute(
            select(Episode).where(Episode.id == episode.id)
        )
        episode = result.unique().scalar_one_or_none()
        return episode

    @classmethod
    async def get_episode_with_details(
        cls,
        story_id: int,
        chapter_number: int,
    ) -> Optional[Episode]:
        """Lấy episode kèm pages và transcript lines"""
        stmt = (
            select(Episode)
            .where(
                Episode.story_id == story_id,
                Episode.chapter_number == chapter_number,
            )
            .options(
                selectinload(Episode.pages)
                .selectinload(Page.transcript_lines)
                .selectinload(TranscriptLine.speaker_character),
                selectinload(Episode.pages)
                .selectinload(Page.transcript_lines)
                .selectinload(TranscriptLine.target_character),
                selectinload(Episode.pages)
                .selectinload(Page.transcript_lines)
                .selectinload(TranscriptLine.translations),
            )
        )
        result = await db.execute(stmt)
        return result.unique().scalar_one_or_none()

    @classmethod
    async def get_translations_for_lines(
        cls,
        line_ids: List[int],
        language: Language,
    ) -> Dict[int, str]:
        if not line_ids:
            return {}
        stmt = select(Translation).where(
            Translation.transcript_line_id.in_(line_ids),
            Translation.language == language,
        )
        result = await db.execute(stmt)
        translations = result.scalars().all()
        return {item.transcript_line_id: item.translated for item in translations}

    @classmethod
    async def upsert_translations(
        cls,
        language: Language,
        translations: Dict[int, str],
    ) -> None:
        if not translations:
            return

        for line_id, translated_text in translations.items():
            stmt = select(Translation).where(
                Translation.transcript_line_id == line_id,
                Translation.language == language,
            )
            result = await db.execute(stmt)
            translation = result.scalar_one_or_none()
            if translation:
                translation.translated = translated_text
            else:
                translation = Translation(
                    transcript_line_id=line_id,
                    language=language,
                    translated=translated_text,
                )
                await db.add(translation)

        await db.commit()

