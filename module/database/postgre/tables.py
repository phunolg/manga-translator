from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Enum,
    ForeignKey, DateTime, func
)
from sqlalchemy.orm import declarative_base, declared_attr, relationship
from sqlalchemy.dialects.postgresql import JSONB
from module.knowledge.type import Topic, Language

Base = declarative_base()

class TableNameMixin:
    """Mixin để tự động tạo tên bảng từ tên class (snake_case + số nhiều chuẩn).

    Ví dụ:
    - Story -> stories
    - Character -> characters
    - Episode -> episodes
    - Page -> pages
    - TranscriptLine -> transcript_lines
    """

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()

    @declared_attr
    def __tablename__(cls) -> str:  # type: ignore[override]
        base = TableNameMixin._camel_to_snake(cls.__name__)
        # Quy tắc số nhiều đơn giản cho các model hiện tại
        if base.endswith("y"):
            return base[:-1] + "ies"
        return base + "s"

class BareBaseModel(TableNameMixin, Base):
    __abstract__ = True
    __allow_unmapped__ = True  # Cho phép legacy annotations không dùng Mapped[]

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class Character(BareBaseModel):
    story_id = Column(
        Integer,
        ForeignKey("stories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_name = Column(String(255), nullable=False)  # Tên gốc trong source language
    description = Column(Text, nullable=True)

    # đường dẫn file ảnh
    image_path = Column(Text, nullable=True)

    # character_features được tách thành các cột riêng
    face = Column(Text, nullable=True)
    hair = Column(Text, nullable=True)
    eyes = Column(Text, nullable=True)
    outfit = Column(Text, nullable=True)
    accessories = Column(Text, nullable=True)
    distinctive_features = Column(Text, nullable=True)

    story = relationship("Story", back_populates="characters")
    
    # Quan hệ với AddressMatrix (khi nhân vật này là speaker)
    address_matrixes_as_speaker = relationship(
        "AddressMatrix",
        foreign_keys="[AddressMatrix.speaker_id]",
        back_populates="speaker",
        cascade="all, delete-orphan",
    )
    # Quan hệ với AddressMatrix (khi nhân vật này là target)
    address_matrixes_as_target = relationship(
        "AddressMatrix",
        foreign_keys="[AddressMatrix.target_id]",
        back_populates="target",
        cascade="all, delete-orphan",
    )
    # Quan hệ với TranscriptLine (khi nhân vật này là speaker)
    transcript_lines_as_speaker = relationship(
        "TranscriptLine",
        foreign_keys="[TranscriptLine.speaker_id]",
        back_populates="speaker_character",
    )
    # Quan hệ với TranscriptLine (khi nhân vật này là target)
    transcript_lines_as_target = relationship(
        "TranscriptLine",
        foreign_keys="[TranscriptLine.target_id]",
        back_populates="target_character",
    )
  
class Story(BareBaseModel):
    story_name = Column(String(255), unique=True, index=True, nullable=False)
    story_type = Column(Enum(Topic), nullable=False)
    source_language = Column(Enum(Language), nullable=False)

    episodes = relationship(
        "Episode",
        back_populates="story",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    characters = relationship(
        "Character",
        back_populates="story",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    translate_dicts = relationship(
        "TranslateDictStory",
        back_populates="story",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    mapping_names = relationship(
        "MappingName",
        back_populates="story",
        cascade="all, delete-orphan",
        lazy="joined",
    )


class MappingName(BareBaseModel):
    """Bảng mapping tên riêng theo ngôn ngữ ở cấp độ story"""
    language = Column(Enum(Language), nullable=False, index=True)
    story_id = Column(
        Integer,
        ForeignKey("stories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source = Column(Text, nullable=False)
    translation = Column(Text, nullable=False)

    story = relationship(
        "Story",
        back_populates="mapping_names",
    )


class TranslateDictStory(BareBaseModel):
    """Bảng normalize translate_dict từ JSONB thành quan hệ riêng theo ngôn ngữ"""
    language = Column(Enum(Language), nullable=False, index=True)
    story_id = Column(
        Integer,
        ForeignKey("stories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dictionary = Column(JSONB, nullable=False, server_default="{}")

    story = relationship(
        "Story",
        back_populates="translate_dicts",
    )



class AddressMatrix(BareBaseModel):
    """Bảng normalize address_matrix từ JSONB thành quan hệ riêng"""
    speaker_id = Column(
        Integer,
        ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_id = Column(
        Integer,
        ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=True,  # nullable vì có thể là "other"
        index=True,
    )
    description = Column(Text, nullable=False)  # cách xưng hô (vd: "ta-cô", "ta-công tử")

    speaker = relationship(
        "Character",
        foreign_keys=[speaker_id],
        back_populates="address_matrixes_as_speaker",
    )
    target = relationship(
        "Character",
        foreign_keys=[target_id],
        back_populates="address_matrixes_as_target",
    )


# Base, Story, Character đã có sẵn ở trên
class Episode(BareBaseModel):
    story_id = Column(
        Integer,
        ForeignKey("stories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chapter_number = Column(Integer, nullable=False)  # vd: 133

    story = relationship("Story", back_populates="episodes")
    pages = relationship(
        "Page",
        back_populates="episode",
        cascade="all, delete-orphan",
        lazy="joined",
    )


class Page(BareBaseModel):
    episode_id = Column(
        Integer,
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer, nullable=False)      # tên file 1.json, 2.json...
    prose = Column(Text, nullable=True)
    image_path = Column(Text, nullable=True)

    episode = relationship("Episode", back_populates="pages")
    transcript_lines = relationship(
        "TranscriptLine",
        back_populates="page",
        cascade="all, delete-orphan",
        lazy="joined",
    )


class TranscriptLine(BareBaseModel):
    page_id = Column(
        Integer,
        ForeignKey("pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    line_index = Column(Integer, nullable=False)       # index trong mảng transcript

    speaker_id = Column(
        Integer,
        ForeignKey("characters.id", ondelete="SET NULL"),
        nullable=True,  # nullable vì có thể là "unsure" hoặc "Other"
        index=True,
    )
    text = Column(Text, nullable=True)
    text_speech_type = Column(String(50), nullable=True)
    target_id = Column(
        Integer,
        ForeignKey("characters.id", ondelete="SET NULL"),
        nullable=True,  # nullable vì có thể là null hoặc "Other"
        index=True,
    )
    bbox = Column(JSONB, nullable=True)
    
    page = relationship("Page", back_populates="transcript_lines")
    speaker_character = relationship(
        "Character",
        foreign_keys=[speaker_id],
        back_populates="transcript_lines_as_speaker",
    )
    target_character = relationship(
        "Character",
        foreign_keys=[target_id],
        back_populates="transcript_lines_as_target",
    )
    translations = relationship(
        "Translation",
        back_populates="transcript_line",
        cascade="all, delete-orphan",
        lazy="joined",
    )


class Translation(BareBaseModel):
    language = Column(Enum(Language, name="language", native_enum=False), nullable=False, index=True)
    transcript_line_id = Column(
        Integer,
        ForeignKey("transcript_lines.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    translated = Column(Text, nullable=False)

    transcript_line = relationship(
        "TranscriptLine",
        back_populates="translations",
    )