import os
import json
from typing import Dict, Optional, List
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from api.schema.api_exception_schema import ApiException
from api.schema.metadata_schema import (
    CreateStoryRequest,
    CreateStoryResponse,
    CreateTranslateDictRequest,
    MappingNameUpsertRequest,
    MappingNameDictResponse,
)
from module.database.postgre.repository.metadata_repository import MetadataRepository
from module.database.postgre.tables import Story, Character
from settings import BASE_DIR
from utils.image import read_image_file, recognize_characters_from_images
from module.knowledge.type import Language
from module.database.postgre.tables import AddressMatrix, Character
from module.database.postgre.database import db
from sqlalchemy import select
from api.utils.path_utils import convert_image_path_to_url

class MetadataService:
    metadata_repository: MetadataRepository = MetadataRepository()
    
    @classmethod
    async def get_all_stories(cls) -> List[Story]:
        stories = await cls.metadata_repository.get_all_stories()
        return stories
    
    @classmethod
    async def get_story_by_name(cls, story_name: str):
        from api.schema.metadata_schema import StoryDetailResponse, CharacterResponse, EpisodeResponse, TranslateDictResponse
        
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        # Convert characters
        characters = [
            CharacterResponse(
                id=char.id,
                source_name=char.source_name,
                description=char.description,
                image_path=convert_image_path_to_url(char.image_path),
                face=char.face,
                hair=char.hair,
                eyes=char.eyes,
                outfit=char.outfit,
                accessories=char.accessories,
                distinctive_features=char.distinctive_features,
                created_at=char.created_at,
                updated_at=char.updated_at,
            )
            for char in story.characters
        ]
        
        # Convert episodes
        episodes = [
            EpisodeResponse(
                id=ep.id,
                chapter_number=ep.chapter_number,
                created_at=ep.created_at,
                updated_at=ep.updated_at,
            )
            for ep in story.episodes
        ]
        
        # Convert translate_dicts
        translate_dicts = [
            TranslateDictResponse(
                language=td.language,
                dictionary=td.dictionary,
            )
            for td in story.translate_dicts
        ]

        mapping_group: Dict[Language, Dict[str, str]] = {}
        for mapping in story.mapping_names:
            mapping_group.setdefault(mapping.language, {})[mapping.source] = mapping.translation
        mapping_names = [
            MappingNameDictResponse(language=lang, dictionary=dictionary)
            for lang, dictionary in mapping_group.items()
        ]
        
        return StoryDetailResponse(
            id=story.id,
            story_name=story.story_name,
            story_type=story.story_type,
            source_language=story.source_language,
            created_at=story.created_at,
            updated_at=story.updated_at,
            characters=characters,
            episodes=episodes,
            translate_dicts=translate_dicts,
            mapping_names=mapping_names,
        )
    
    @classmethod
    async def create_story(cls, create_story_req: CreateStoryRequest) -> CreateStoryResponse:
        # Kiểm tra story đã tồn tại chưa
        existing_story = await cls.metadata_repository.get_story_by_name(create_story_req.story_name)
        if existing_story:
            raise ApiException(message=f"Story {create_story_req.story_name} already exists", status_code=400)
        
        story = Story(
            story_name=create_story_req.story_name,
            story_type=create_story_req.story_type,
            source_language=create_story_req.source_language,
        )
        created_story = await cls.metadata_repository.create_story(story)
        return CreateStoryResponse(
            story_name=created_story.story_name,
            story_type=created_story.story_type,
            source_language=created_story.source_language,
        )
    
    @classmethod
    async def create_character(
        cls,
        story_name: str,
        name_character: str,
        description: Optional[str],
        character_image: UploadFile
    ) -> Dict:
        # Kiểm tra story có tồn tại không
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        # Kiểm tra character đã tồn tại chưa
        existing_character = await cls.metadata_repository.get_character_by_story_and_name(
            story.id, name_character
        )
        if existing_character:
            raise ApiException(message=f"Character {name_character} already exists", status_code=400)
        
        # Tạo thư mục Character nếu chưa tồn tại
        story_folder = os.path.join(BASE_DIR, "data", "stories", story_name)
        character_folder = os.path.join(story_folder, "Character")
        os.makedirs(character_folder, exist_ok=True)
        
        # Lưu file ảnh
        image_path = os.path.join(character_folder, f"{name_character.lower()}.png")
        with open(image_path, "wb") as f:
            f.write(await character_image.read())
        
        # Nhận diện character features
        character_image.seek(0)  # Reset file pointer
        character_features = await recognize_characters_from_images(
            [read_image_file(character_image)],
            [name_character]
        )
        features = character_features.get(name_character, {})
        
        # Tạo character mới
        character = Character(
            story_id=story.id,
            source_name=name_character,
            description=description,
            image_path=image_path,
            face=features.get("face"),
            hair=features.get("hair"),
            eyes=features.get("eyes"),
            outfit=features.get("outfit"),
            accessories=features.get("accessories"),
            distinctive_features=features.get("distinctive_features"),
        )
        created_character = await cls.metadata_repository.create_character(character)
        
        return {
            "message": f"Character {name_character} created successfully",
            "character": {
                "id": created_character.id,
                "source_name": created_character.source_name,
                "description": created_character.description,
                "image_path": convert_image_path_to_url(created_character.image_path),
            }
        }
    
    @classmethod
    async def update_metadata(
        cls,
        story_name: str,
        mapping_name: Optional[str] = None,
        translate_dict: Optional[str] = None,
        story_type: Optional[str] = None,
        source_language: Optional[str] = None
    ) -> Dict:
        mapping_name_dict = None
        if mapping_name:
            try:
                raw_mapping = json.loads(mapping_name)
                if isinstance(raw_mapping, dict):
                    mapping_name_dict = {}
                    for lang, entries in raw_mapping.items():
                        if not isinstance(entries, dict):
                            raise ApiException(message="mapping_name entries must be dictionary", status_code=400)
                        mapping_name_dict[lang] = entries
                else:
                    raise ApiException(message="mapping_name must be a JSON object", status_code=400)
            except json.JSONDecodeError:
                raise ApiException(message="Invalid mapping_name format", status_code=400)
        
        translate_dict_obj = None
        if translate_dict:
            try:
                translate_dict_obj = json.loads(translate_dict)
            except json.JSONDecodeError:
                raise ApiException(message="Invalid translate_dict format", status_code=400)
        
        source_language_enum = None
        if source_language:
            try:
                source_language_enum = Language(source_language)
            except ValueError:
                raise ApiException(message=f"Invalid source_language: {source_language}", status_code=400)
        
        updated_story = await cls.metadata_repository.update_story_metadata(
            story_name=story_name,
            mapping_name=mapping_name_dict,
            translate_dict=translate_dict_obj,
            story_type=story_type,
            source_language=source_language_enum
        )
        
        return {
            "data": {
                "story_name": updated_story.story_name,
                "story_type": updated_story.story_type.value,
                "source_language": updated_story.source_language.value,
            },
            "message": f"Metadata for story {story_name} updated successfully"
        }
    
    @classmethod
    async def update_character(
        cls,
        story_name: str,
        character_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        character_image: Optional[UploadFile] = None
    ) -> Dict:
        # Kiểm tra story có tồn tại không
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        # Xử lý ảnh nếu có
        image_path = None
        face = None
        hair = None
        eyes = None
        outfit = None
        accessories = None
        distinctive_features = None
        
        if character_image:
            story_folder = os.path.join(BASE_DIR, "transcript_history", story_name)
            character_folder = os.path.join(story_folder, "Character")
            os.makedirs(character_folder, exist_ok=True)
            
            image_name = name.lower() if name else character_name.lower()
            image_path = os.path.join(character_folder, f"{image_name}.png")
            
            with open(image_path, "wb") as f:
                f.write(await character_image.read())
            
            # Nhận diện character features
            character_image.seek(0)
            character_features = await recognize_characters_from_images(
                [read_image_file(character_image)],
                [name or character_name]
            )
            features = character_features.get(name or character_name, {})
            face = features.get("face")
            hair = features.get("hair")
            eyes = features.get("eyes")
            outfit = features.get("outfit")
            accessories = features.get("accessories")
            distinctive_features = features.get("distinctive_features")
        
        updated_character = await cls.metadata_repository.update_character(
            story_id=story.id,
            character_name=character_name,
            new_name=name,
            description=description,
            image_path=image_path,
            face=face,
            hair=hair,
            eyes=eyes,
            outfit=outfit,
            accessories=accessories,
            distinctive_features=distinctive_features
        )
        
        return {
            "message": f"Character {character_name} updated successfully",
            "character": {
                "id": updated_character.id,
                "source_name": updated_character.source_name,
                "description": updated_character.description,
                "image_path": convert_image_path_to_url(updated_character.image_path),
            }
        }
    
    @classmethod
    async def get_character_detail(
        cls,
        story_name: str,
        character_name: str
    ):
        from api.schema.metadata_schema import CharacterDetailResponse, CharacterResponse
        
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        # Tìm character
        character = await cls.metadata_repository.get_character_by_story_and_name(
            story.id, character_name
        )
        if not character:
            raise ApiException(message=f"Character {character_name} not found", status_code=404)
        
        # Lấy address matrix
        address_matrices = await cls.metadata_repository.get_address_matrices_by_character(
            story.id, character_name
        )
        
        # Convert address matrix to dict
        address_matrix_dict = {}
        for am in address_matrices:
            if am.target_id:
                target_char_result = await db.execute(
                    select(Character).where(Character.id == am.target_id)
                )
                target_char = target_char_result.scalar_one_or_none()
                target_name = target_char.source_name if target_char else None
            else:
                target_name = None
            
            key = target_name if target_name else "other"
            address_matrix_dict[key] = am.description
        
        # Convert character to response
        character_response = CharacterResponse(
            id=character.id,
            source_name=character.source_name,
            description=character.description,
            image_path=convert_image_path_to_url(character.image_path),
            face=character.face,
            hair=character.hair,
            eyes=character.eyes,
            outfit=character.outfit,
            accessories=character.accessories,
            distinctive_features=character.distinctive_features,
            created_at=character.created_at,
            updated_at=character.updated_at,
        )
        
        return CharacterDetailResponse(
            character=character_response,
            address_matrix=address_matrix_dict,
        )
    
    @classmethod
    async def get_address_matrices(
        cls,
        story_name: str,
        character_name: str
    ) -> Dict:
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        address_matrices = await cls.metadata_repository.get_address_matrices_by_character(
            story.id, character_name
        )
        
        # Convert to dict format
        result = {}
        for am in address_matrices:
            if am.target_id:
                # Tìm target character name
                target_char_result = await db.execute(
                    select(Character).where(Character.id == am.target_id)
                )
                target_char = target_char_result.scalar_one_or_none()
                target_name = target_char.source_name if target_char else None
            else:
                target_name = None
            
            key = target_name if target_name else "other"
            result[key] = am.description
        
        return {
            "address_matrix": result,
            "message": f"Address matrices for character {character_name} retrieved successfully"
        }
    
    @classmethod
    async def merge_address_matrices(
        cls,
        story_name: str,
        character_name: str,
        address_matrix: Dict[str, str]
    ) -> Dict:
        """
        Merge address matrix với dữ liệu hiện có (bổ sung hoặc ghi đè nếu đã tồn tại)
        """
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        # Lấy address matrix hiện có
        existing_matrices = await cls.metadata_repository.get_address_matrices_by_character(
            story.id, character_name
        )
        
        # Convert existing to dict
        existing_dict = {}
        for am in existing_matrices:
            if am.target_id:
                target_char_result = await db.execute(
                    select(Character).where(Character.id == am.target_id)
                )
                target_char = target_char_result.scalar_one_or_none()
                target_name = target_char.source_name if target_char else None
            else:
                target_name = None
            
            key = target_name if target_name else "other"
            existing_dict[key] = am.description
        
        # Merge với dữ liệu mới (ghi đè nếu đã tồn tại)
        merged_dict = {**existing_dict, **address_matrix}
        
        # Replace all với merged dict
        new_matrices = await cls.metadata_repository.replace_all_address_matrices(
            story.id, character_name, merged_dict
        )
        
        # Convert to dict format
        result = {}
        for am in new_matrices:
            if am.target_id:
                target_char_result = await db.execute(
                    select(Character).where(Character.id == am.target_id)
                )
                target_char = target_char_result.scalar_one_or_none()
                target_name = target_char.source_name if target_char else None
            else:
                target_name = None
            
            key = target_name if target_name else "other"
            result[key] = am.description
        
        return {
            "address_matrix": result,
            "message": f"Address matrices for character {character_name} merged successfully"
        }
    
    @classmethod
    async def replace_address_matrices(
        cls,
        story_name: str,
        character_name: str,
        address_matrix: Dict[str, str]
    ) -> Dict:
        """
        Replace toàn bộ address matrix (ghi đè tất cả)
        """
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        new_matrices = await cls.metadata_repository.replace_all_address_matrices(
            story.id, character_name, address_matrix
        )
        
        # Convert to dict format
        result = {}
        for am in new_matrices:
            if am.target_id:
                target_char_result = await db.execute(
                    select(Character).where(Character.id == am.target_id)
                )
                target_char = target_char_result.scalar_one_or_none()
                target_name = target_char.source_name if target_char else None
                key = target_name if target_name else "other"
            else:
                # target_id = None nghĩa là "Những người còn lại"
                key = "other"
            
            result[key] = am.description
        
        return {
            "address_matrix": result,
            "message": f"Address matrices for character {character_name} replaced successfully"
        }
    
    @classmethod
    async def delete_address_matrix(
        cls,
        story_name: str,
        character_name: str,
        target_name: str
    ) -> Dict:
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        
        deleted = await cls.metadata_repository.delete_address_matrix(
            story.id, character_name, target_name
        )
        
        if not deleted:
            raise ApiException(
                message=f"Address matrix for target {target_name} not found",
                status_code=404
            )
        
        return {
            "message": f"Address matrix for target {target_name} deleted successfully"
        }

    @classmethod
    async def get_mapping_names(
        cls,
        story_name: str,
        language: Optional[Language] = None,
    ) -> Dict:
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)

        mapping_rows = await cls.metadata_repository.get_mapping_names_by_story(
            story.id, language=language
        )
        grouped: Dict[Language, Dict[str, str]] = {}
        for row in mapping_rows:
            grouped.setdefault(row.language, {})[row.source] = row.translation

        response = [
            MappingNameDictResponse(language=lang, dictionary=dictionary)
            for lang, dictionary in grouped.items()
        ]
        return {
            "mapping_names": response,
            "message": f"Mapping names for story {story_name} retrieved successfully",
        }

    @classmethod
    async def upsert_mapping_names(
        cls,
        story_name: str,
        payload: MappingNameUpsertRequest,
    ) -> Dict:
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)

        if not payload.dictionary:
            raise ApiException(message="dictionary must not be empty", status_code=400)

        await cls.metadata_repository.upsert_mapping_entries(
            story.id,
            language=payload.language,
            entries=payload.dictionary,
        )

        updated_rows = await cls.metadata_repository.get_mapping_names_by_story(
            story.id, language=payload.language
        )
        dictionary = {row.source: row.translation for row in updated_rows}

        return {
            "mapping_name": MappingNameDictResponse(
                language=payload.language,
                dictionary=dictionary,
            ),
            "message": f"Mapping names for story {story_name} ({payload.language.value}) saved successfully",
        }

    @classmethod
    async def delete_mapping_name(
        cls,
        story_name: str,
        language: Language,
        source: str,
    ) -> Dict:
        story = await cls.metadata_repository.get_story_by_name(story_name)
        if not story:
            raise ApiException(message=f"Story {story_name} not found", status_code=404)

        deleted = await cls.metadata_repository.delete_mapping_entry(
            story.id,
            language=language,
            source=source,
        )
        if not deleted:
            raise ApiException(
                message=f"Mapping name '{source}' for language {language.value} not found",
                status_code=404,
            )
        return {
            "message": f"Mapping entry '{source}' ({language.value}) deleted successfully"
        }
    
    @classmethod
    async def create_translate_dict(
        cls,
        story_name: str,
        translate_dict_request: CreateTranslateDictRequest
    ) -> Dict:
        from api.schema.metadata_schema import TranslateDictResponse
        
        translate_dict_obj = await cls.metadata_repository.create_translate_dict(
            story_name=story_name,
            language=translate_dict_request.language,
            dictionary=translate_dict_request.dictionary
        )
        
        return {
            "message": f"Translate dictionary for language {translate_dict_request.language.value} created/updated successfully",
            "data": TranslateDictResponse(
                language=translate_dict_obj.language,
                dictionary=translate_dict_obj.dictionary
            )
        }
         