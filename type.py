

# Định nghĩa các model cho API character_blank
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
from pprint import pformat
from pydantic import BaseModel, create_model

from module.knowledge.type import Language, Topic

class ApiException(Exception):
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class CharacterFeatures(BaseModel):
    face: Optional[str] = None
    hair: Optional[str] = None
    eyes: Optional[str] = None
    outfit: Optional[str] = None
    accessories: Optional[str] = None
    distinctive_features: Optional[str] = None

class Character(BaseModel):
    name: str
    description: Optional[str] = None
    address_matrix: Dict[str, str] = {}
    character_features: CharacterFeatures = CharacterFeatures()
    image_path: str     
    
    def __str__(self):
        return pformat(self.model_dump())
    
class MangaMetadata(BaseModel):
    story_name: str
    story_type: Topic
    character_blank: List[Character] = []
    mapping_name: Dict[str, str] = {}
    translate_dict: Dict[str, str] = {}
    source_language: Language

    @classmethod
    def create_partial(cls, include_fields: List[str]) -> Type[BaseModel]:
        """
        Tạo một model mới chỉ bao gồm các trường được chỉ định
        Tương tự như Pick<T, K> trong TypeScript
        """
        fields = {}
        for field_name, field_info in cls.__annotations__.items():
            if field_name in include_fields:
                fields[field_name] = (field_info, ... if field_name ==
                                      'story_name' else cls.model_fields[field_name].default)

        return create_model(f"Partial{cls.__name__}", **fields)

@dataclass
class Transcript:
    speaker: str
    text: str
    text_speech_type: str
    target: str | None = field(default=None)
    translation: Optional[str] = field(default=None)
    bbox: Optional[List[int]] = field(default=None)
    line_id: Optional[int] = field(default=None)
    
    def get_source_text(self):
        return "{} {}: '{}'".format(self.speaker, self.text_speech_type, self.text.replace('\n', ' '))
    
    def get_translation_text(self):
        if not self.translation:
            raise ValueError("Không có translation")
        return f"{self.speaker} {self.text_speech_type}: '{self.translation.replace('\n', ' ')}'"
