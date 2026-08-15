from pydantic import BaseModel, ConfigDict
from module.knowledge.type import Topic, Language
from typing import Dict, List, Optional
from datetime import datetime

from type import Character

class CreateStoryRequest(BaseModel):
    story_name: str
    story_type: Topic
    source_language: Language
    
class CreateStoryResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    story_name: str
    story_type: Topic
    source_language: Language

class CharacterResponse(BaseModel):
    id: int
    source_name: str
    description: Optional[str] = None
    image_path: Optional[str] = None
    face: Optional[str] = None
    hair: Optional[str] = None
    eyes: Optional[str] = None
    outfit: Optional[str] = None
    accessories: Optional[str] = None
    distinctive_features: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class MappingNameDictResponse(BaseModel):
    language: Language
    dictionary: Dict[str, str]


class MappingNameUpsertRequest(BaseModel):
    language: Language
    dictionary: Dict[str, str]


class CharacterDetailResponse(BaseModel):
    character: CharacterResponse
    address_matrix: Dict[str, str]

class EpisodeResponse(BaseModel):
    id: int
    chapter_number: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class TranslateDictResponse(BaseModel):
    language: Language
    dictionary: Dict[str, str]

class CreateTranslateDictRequest(BaseModel):
    language: Language
    dictionary: Dict[str, str]

class StoryDetailResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    id: int
    story_name: str
    story_type: Topic
    source_language: Language
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    characters: List[CharacterResponse] = []
    episodes: List[EpisodeResponse] = []
    translate_dicts: List[TranslateDictResponse] = []
    mapping_names: List[MappingNameDictResponse] = []
    
class CharacterBlankUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    address_matrix: Optional[Dict[str, str]] = None