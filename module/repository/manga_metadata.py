import json
import os

from fastapi.responses import JSONResponse
from module.knowledge.type import Language, Topic
from settings import BASE_DIR
from type import ApiException, MangaMetadata


class MangaMetadataRepository:
    def __init__(self):
        pass
    
    @staticmethod
    def get_story_by_name(story_name: str) -> MangaMetadata:
        file_path = os.path.join(BASE_DIR, "transcript_history", story_name, "metadata.json")
        if not os.path.exists(file_path):
            raise ApiException(message=f"Story {story_name} not found", status_code=404)
        with open(file_path, "r", encoding="utf-8") as f:
            return MangaMetadata.model_validate_json(f.read())

    @staticmethod
    def save_story(metadata_param: MangaMetadata) -> MangaMetadata:
        # Tạo folder cho truyện
        story_folder = os.path.join(
            BASE_DIR, "transcript_history", metadata_param.story_name)  
        if os.path.exists(story_folder):
            raise ApiException(message=f"Story {metadata_param.story_name} already exists", status_code=400)
        
        if metadata_param.story_type not in Topic.__members__.values():
            raise ApiException(message=f"Story type must be one of {Topic.__members__.values()}", status_code=400)
        
        if metadata_param.source_language not in Language.__members__.values():
            raise ApiException(message=f"Source language must be one of {Language.__members__.values()}", status_code=400)
        
        os.makedirs(story_folder, exist_ok=True)
        
        file_path = os.path.join(story_folder, "metadata.json")
        # Khởi tạo metadata từ request
        metadata = {
            "story_name": metadata_param.story_name,
            "character_blank": [],
            "mapping_name": metadata_param.mapping_name,
            "translate_dict": metadata_param.translate_dict,
            "story_type": metadata_param.story_type.value,
            "source_language": metadata_param.source_language.value
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        # Trả về MangaMetadata object thay vì dict
        return MangaMetadata(
            story_name=metadata_param.story_name,
            character_blank=metadata_param.character_blank,
            mapping_name=metadata["mapping_name"],
            translate_dict=metadata["translate_dict"],
            story_type=metadata_param.story_type,
            source_language=metadata_param.source_language
        )
    
    @staticmethod
    def update_story(metadata_param: MangaMetadata) -> MangaMetadata:
        file_path = os.path.join(BASE_DIR, "transcript_history", metadata_param.story_name, "metadata.json")
        if not os.path.exists(file_path):
            raise ApiException(message=f"Story {metadata_param.story_name} has not been initialized", status_code=404)
        with open(file_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        
         # Cập nhật mapping_name nếu có
        if metadata_param.mapping_name:
            try:
                # Nếu mapping_name là string, parse nó thành dict
                if isinstance(metadata_param.mapping_name, str):
                    mapping_name_dict = json.loads(metadata_param.mapping_name)
                else:
                    mapping_name_dict = metadata_param.mapping_name
                # Cập nhật từng cặp key-value trong mapping_name
                for key, value in mapping_name_dict.items():
                    existing_metadata["mapping_name"][key] = value
            except json.JSONDecodeError:
                raise ApiException(message="Invalid mapping_name format", status_code=400)
            
        if metadata_param.translate_dict:
            try:
                # Nếu translate_dict là string, parse nó thành dict
                if isinstance(metadata_param.translate_dict, str):
                    translate_dict_dict = json.loads(metadata_param.translate_dict)
                else:
                    translate_dict_dict = metadata_param.translate_dict
                # Cập nhật từng cặp key-value trong translate_dict
                for key, value in translate_dict_dict.items():
                    existing_metadata["translate_dict"][key] = value
            except json.JSONDecodeError:
                raise ApiException(message="Invalid translate_dict format", status_code=400)
            
        if metadata_param.story_type:
            if metadata_param.story_type not in Topic.__members__.values():
                raise ApiException(message=f"Story type must be one of {Topic.__members__.values()}", status_code=400)
            existing_metadata["story_type"] = metadata_param.story_type.value
            
        if metadata_param.source_language:
            if metadata_param.source_language not in Language.__members__.values():
                raise ApiException(message=f"Source language must be one of {Language.__members__.values()}", status_code=400)
            existing_metadata["source_language"] = metadata_param.source_language.value
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_metadata, f, indent=4, ensure_ascii=False)
            
        return MangaMetadata.model_validate_json(json.dumps(existing_metadata))