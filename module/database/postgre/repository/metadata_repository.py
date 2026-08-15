from module.database.postgre.database import db
from module.database.postgre.tables import Story, Character, MappingName, TranslateDictStory, AddressMatrix
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Dict, Optional, List
from module.knowledge.type import Language

class MetadataRepository:
    @classmethod
    async def create_story(cls, story: Story) -> Story:
        await db.add(story)
        await db.commit()
        await db.refresh(story)
        return story
    
    @classmethod
    async def get_all_stories(cls) -> List[Story]:
        result = await db.execute(select(Story).order_by(Story.created_at.desc()))
        return list(result.unique().scalars().all())
    
    @classmethod
    async def get_story_by_name(cls, story_name: str) -> Story | None:
        result = await db.execute(
            select(Story)
            .where(Story.story_name == story_name)
            .options(
                selectinload(Story.characters).selectinload(Character.address_matrixes_as_speaker),
                selectinload(Story.translate_dicts),
                selectinload(Story.mapping_names),
            )
        )
        # unique() để loại bỏ duplicate rows do joined eager loads
        return result.unique().scalar_one_or_none()
    
    @classmethod
    async def update_story_metadata(
        cls,
        story_name: str,
        mapping_name: Optional[Dict[str, Dict[str, str]]] = None,
        translate_dict: Optional[Dict[str, Dict]] = None,
        story_type: Optional[str] = None,
        source_language: Optional[Language] = None
    ) -> Story:
        story = await cls.get_story_by_name(story_name)
        if not story:
            raise ValueError(f"Story {story_name} not found")
        
        if story_type:
            from module.knowledge.type import Topic
            story.story_type = Topic(story_type)
        if source_language:
            story.source_language = source_language
        
        # Cập nhật mapping_name (dict các ngôn ngữ -> {source_name: translation})
        if mapping_name:
            for lang_str, entries in mapping_name.items():
                lang = Language(lang_str)
                await cls.upsert_mapping_entries(
                    story_id=story.id,
                    language=lang,
                    entries=entries,
                )
        
        # Cập nhật translate_dict
        if translate_dict:
            for lang_str, dict_value in translate_dict.items():
                lang = Language(lang_str)
                result = await db.execute(
                    select(TranslateDictStory).where(
                        TranslateDictStory.story_id == story.id,
                        TranslateDictStory.language == lang
                    )
                )
                translate_dict_obj = result.scalar_one_or_none()
                if translate_dict_obj:
                    # Merge dictionary
                    translate_dict_obj.dictionary = {**translate_dict_obj.dictionary, **dict_value}
                else:
                    translate_dict_obj = TranslateDictStory(
                        story_id=story.id,
                        language=lang,
                        dictionary=dict_value
                    )
                    await db.add(translate_dict_obj)
        
        await db.commit()
        await db.refresh(story)
        return story
    
    @classmethod
    async def get_character_by_story_and_name(cls, story_id: int, character_name: str) -> Character | None:
        result = await db.execute(
            select(Character).where(
                Character.story_id == story_id,
                Character.source_name == character_name
            )
        )
        return result.scalar_one_or_none()
    
    @classmethod
    async def create_character(cls, character: Character) -> Character:
        await db.add(character)
        await db.commit()
        await db.refresh(character)
        return character
    
    @classmethod
    async def update_character(
        cls,
        story_id: int,
        character_name: str,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
        image_path: Optional[str] = None,
        face: Optional[str] = None,
        hair: Optional[str] = None,
        eyes: Optional[str] = None,
        outfit: Optional[str] = None,
        accessories: Optional[str] = None,
        distinctive_features: Optional[str] = None,
        address_matrix: Optional[Dict[str, str]] = None
    ) -> Character:
        character = await cls.get_character_by_story_and_name(story_id, character_name)
        if not character:
            raise ValueError(f"Character {character_name} not found in story")
        
        if new_name:
            character.source_name = new_name
        if description is not None:
            character.description = description
        if image_path:
            character.image_path = image_path
        if face is not None:
            character.face = face
        if hair is not None:
            character.hair = hair
        if eyes is not None:
            character.eyes = eyes
        if outfit is not None:
            character.outfit = outfit
        if accessories is not None:
            character.accessories = accessories
        if distinctive_features is not None:
            character.distinctive_features = distinctive_features
        
        await db.commit()
        await db.refresh(character)
        return character
    
    @classmethod
    async def get_address_matrices_by_character(cls, story_id: int, character_name: str) -> List[AddressMatrix]:
        character = await cls.get_character_by_story_and_name(story_id, character_name)
        if not character:
            raise ValueError(f"Character {character_name} not found in story")
        
        result = await db.execute(
            select(AddressMatrix).where(AddressMatrix.speaker_id == character.id)
        )
        return list(result.scalars().all())
    
    @classmethod
    async def replace_all_address_matrices(
        cls,
        story_id: int,
        character_name: str,
        address_matrix: Dict[str, str]
    ) -> List[AddressMatrix]:
        character = await cls.get_character_by_story_and_name(story_id, character_name)
        if not character:
            raise ValueError(f"Character {character_name} not found in story")
        
        # Xóa các address_matrix cũ của character này
        result = await db.execute(
            select(AddressMatrix).where(AddressMatrix.speaker_id == character.id)
        )
        old_matrices = result.scalars().all()
        for old_matrix in old_matrices:
            await db.session.delete(old_matrix)
        
        # Tạo address_matrix mới
        new_matrices = []
        for target_name, description_text in address_matrix.items():
            # Xử lý trường hợp "other" (Những người còn lại)
            if target_name == 'other' or target_name == '__OTHERS__':
                target_id = None
            else:
                # Tìm target character nếu có
                target_character = await cls.get_character_by_story_and_name(story_id, target_name)
                target_id = target_character.id if target_character else None
            
            address_matrix_obj = AddressMatrix(
                speaker_id=character.id,
                target_id=target_id,
                description=description_text
            )
            await db.add(address_matrix_obj)
            new_matrices.append(address_matrix_obj)
        
        await db.commit()
        return new_matrices
    
    @classmethod
    async def delete_address_matrix(
        cls,
        story_id: int,
        character_name: str,
        target_name: str
    ) -> bool:
        character = await cls.get_character_by_story_and_name(story_id, character_name)
        if not character:
            raise ValueError(f"Character {character_name} not found in story")
        
        # Tìm target character nếu có
        target_character = await cls.get_character_by_story_and_name(story_id, target_name)
        target_id = target_character.id if target_character else None
        
        # Tìm address_matrix cần xóa
        if target_id:
            result = await db.execute(
                select(AddressMatrix).where(
                    AddressMatrix.speaker_id == character.id,
                    AddressMatrix.target_id == target_id
                )
            )
        else:
            # Nếu target_name không tồn tại, tìm theo target_id is None
            result = await db.execute(
                select(AddressMatrix).where(
                    AddressMatrix.speaker_id == character.id,
                    AddressMatrix.target_id.is_(None)
                )
            )
        
        address_matrix = result.scalar_one_or_none()
        if address_matrix:
            await db.session.delete(address_matrix)
            await db.commit()
            return True
        return False

    @classmethod
    async def get_mapping_names_by_story(
        cls,
        story_id: int,
        language: Optional[Language] = None,
    ) -> List[MappingName]:
        stmt = select(MappingName).where(MappingName.story_id == story_id)
        if language:
            stmt = stmt.where(MappingName.language == language)
        stmt = stmt.order_by(MappingName.language, MappingName.source)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @classmethod
    async def upsert_mapping_entries(
        cls,
        story_id: int,
        language: Language,
        entries: Dict[str, str],
    ) -> List[MappingName]:
        updated: List[MappingName] = []
        for source, translation in entries.items():
            stmt = select(MappingName).where(
                MappingName.story_id == story_id,
                MappingName.language == language,
                MappingName.source == source,
            )
            result = await db.execute(stmt)
            mapping = result.scalar_one_or_none()
            if mapping:
                mapping.translation = translation
            else:
                mapping = MappingName(
                    story_id=story_id,
                    language=language,
                    source=source,
                    translation=translation,
                )
                await db.add(mapping)
            updated.append(mapping)
        await db.commit()
        for mapping in updated:
            await db.refresh(mapping)
        return updated

    @classmethod
    async def delete_mapping_entry(
        cls,
        story_id: int,
        language: Language,
        source: str,
    ) -> bool:
        stmt = select(MappingName).where(
            MappingName.story_id == story_id,
            MappingName.language == language,
            MappingName.source == source,
        )
        result = await db.execute(stmt)
        mapping = result.scalar_one_or_none()
        if not mapping:
            return False

        await db.session.delete(mapping)
        await db.commit()
        return True
    
    @classmethod
    async def create_translate_dict(
        cls,
        story_name: str,
        language: Language,
        dictionary: Dict[str, str]
    ) -> TranslateDictStory:
        story = await cls.get_story_by_name(story_name)
        if not story:
            raise ValueError(f"Story {story_name} not found")
        
        # Kiểm tra xem translate dict cho ngôn ngữ này đã tồn tại chưa
        result = await db.execute(
            select(TranslateDictStory).where(
                TranslateDictStory.story_id == story.id,
                TranslateDictStory.language == language
            )
        )
        existing_dict = result.scalar_one_or_none()
        
        if existing_dict:
            # Merge với dictionary hiện có
            existing_dict.dictionary = {**existing_dict.dictionary, **dictionary}
            await db.commit()
            await db.refresh(existing_dict)
            return existing_dict
        else:
            # Tạo mới
            translate_dict_obj = TranslateDictStory(
                story_id=story.id,
                language=language,
                dictionary=dictionary
            )
            await db.add(translate_dict_obj)
            await db.commit()
            await db.refresh(translate_dict_obj)
            return translate_dict_obj