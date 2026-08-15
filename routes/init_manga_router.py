from pprint import pformat
from fastapi import APIRouter, File, Form, UploadFile, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model
from typing import Dict, List, Optional, Any, Type
import os
import json
from module.knowledge.type import Topic
from module.repository.manga_metadata import MangaMetadataRepository
from settings import BASE_DIR
from module.rendering.text_render import logger
from type import ApiException, MangaMetadata
from utils.image import read_image_file, recognize_characters_from_images

class CharacterBlankUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    address_matrix: Optional[Dict[str, str]] = None


# Router cho init-manga
init_manga_router = APIRouter(prefix="/init-manga", tags=["init-manga"])

InitStoryRequest = MangaMetadata.create_partial(
    ['story_name', 'story_type', 'mapping_name', 'translate_dict', 'source_language'])
@init_manga_router.post("/init-story")
async def init_story(
    story_request: InitStoryRequest = Body(...)
):
    """
    Khởi tạo folder truyện mới với metadata.json chỉ với story_name, mapping_name và translate_dict
    """
    try:
        metadata = MangaMetadataRepository.save_story(story_request)
     
        return {
            "message": f"Story {story_request.story_name} initialized successfully",
            "story_path": os.path.join(BASE_DIR, "transcript_history", story_request.story_name),
            "metadata": metadata
        }
    except ApiException as e:
        return JSONResponse(content={"error": e.message}, status_code=e.status_code)
    except Exception as e:
        logger.error(f"Error initializing story:\n{e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@init_manga_router.post("/character")
async def create_character(
    story_name: str = Form(...),
    name_character: str = Form(...),
    description: Optional[str] = Form(None),
    address_matrix: Optional[str] = Form(None),
    character_image: UploadFile = File(...)
):
    """ 
    Tạo nhân vật mới và thêm vào character_blank trong metadata.json
    """
    try:
        # Kiểm tra story_name có tồn tại không
        story_folder = os.path.join(BASE_DIR, "transcript_history", story_name)
        if not os.path.exists(story_folder):
            return JSONResponse(content={"error": f"Story {story_name} not found"}, status_code=404)

        # Đường dẫn đến file metadata.json
        metadata_path = os.path.join(story_folder, "metadata.json")

        # Đọc metadata.json nếu tồn tại, nếu không thì tạo mới
        metadata = {"character_blank": [],
                    "mapping_name": {}, "translate_dict": {}}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Kiểm tra xem nhân vật đã tồn tại chưa
        for i, existing_character in enumerate(metadata["character_blank"]):
            if name_character == existing_character["name"]:
                return JSONResponse(content={"error": f"Character {name_character} already exists"}, status_code=400)

        # Tạo thư mục Character nếu chưa tồn tại
        character_folder = os.path.join(story_folder, "Character")
        os.makedirs(character_folder, exist_ok=True)

        # Lưu file ảnh
        image_path = os.path.join(
            character_folder, f"{name_character.lower()}.png")
        with open(image_path, "wb") as f:
            f.write(await character_image.read())

        if address_matrix:
            try:
                address_matrix_dict = json.loads(address_matrix)
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "Invalid address_matrix format"}, status_code=400)
            
        character_features = await recognize_characters_from_images(
            [read_image_file(character_image)],
            [name_character]
        )

        # Tạo character_blank mới
        new_character = {
            "name": name_character,
            "description": description,
            "address_matrix": address_matrix_dict,
            "image_path": image_path,
            "character_features": character_features[name_character]
        }

        # Thêm nhân vật vào character_blank
        metadata["character_blank"].append(new_character)

        # Lưu metadata.json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        return {"message": f"Character {name_character} created successfully", "character": new_character}

    except Exception as e:
        logger.error(f"Error creating character: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@init_manga_router.patch("/metadata/{story_name}")
async def update_metadata(
    story_name: str = Path(...),
    mapping_name: Optional[str] = Form(None),
    translate_dict: Optional[str] = Form(None),
    story_type: Optional[str] = Form(None)
):
    """
    Cập nhật mapping_name và translate_dict trong metadata.json
    """
    try:
        # Kiểm tra story_name có tồn tại không
        story_folder = os.path.join(BASE_DIR, "transcript_history", story_name)
        if not os.path.exists(story_folder):
            return JSONResponse(content={"error": f"Story {story_name} not found"}, status_code=404)

        # Đường dẫn đến file metadata.json
        metadata_path = os.path.join(story_folder, "metadata.json")
        if not os.path.exists(metadata_path):
            return JSONResponse(content={"error": f"Metadata for story {story_name} not found"}, status_code=404)

        # Đọc metadata.json
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Cập nhật mapping_name nếu có
        if mapping_name:
            try:
                mapping_name_dict = json.loads(mapping_name)
                # Cập nhật từng cặp key-value trong mapping_name
                for key, value in mapping_name_dict.items():
                    metadata["mapping_name"][key] = value
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "Invalid mapping_name format"}, status_code=400)

        # Cập nhật translate_dict nếu có
        if translate_dict:
            try:
                translate_dict_obj = json.loads(translate_dict)
                # Cập nhật từng cặp key-value trong translate_dict
                for key, value in translate_dict_obj.items():
                    metadata["translate_dict"][key] = value
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "Invalid translate_dict format"}, status_code=400)

        # Cập nhật story_type nếu có
        if story_type:
            if story_type not in Topic.__members__.values():
                return JSONResponse(content={"error": f"Story type must be one of {Topic.__members__.values()}"}, status_code=400)
            metadata["story_type"] = story_type

        # Lưu metadata.json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        return {
            "message": f"Metadata for story {story_name} updated successfully",
            "mapping_name": metadata["mapping_name"],
            "translate_dict": metadata["translate_dict"],
            "story_type": metadata["story_type"]
        }

    except Exception as e:
        logger.error(f"Error updating metadata: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@init_manga_router.patch("/character/{story_name}/{character_name}")
async def update_character(
    story_name: str = Path(...),
    character_name: str = Path(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    address_matrix: Optional[str] = Form(None),
    character_image: Optional[UploadFile] = File(None)
):
    """
    Cập nhật thông tin nhân vật trong character_blank
    """
    try:
        # Kiểm tra story_name có tồn tại không
        story_folder = os.path.join(BASE_DIR, "transcript_history", story_name)
        if not os.path.exists(story_folder):
            return JSONResponse(content={"error": f"Story {story_name} not found"}, status_code=404)

        # Đường dẫn đến file metadata.json
        metadata_path = os.path.join(story_folder, "metadata.json")
        if not os.path.exists(metadata_path):
            return JSONResponse(content={"error": f"Metadata for story {story_name} not found"}, status_code=404)

        # Đọc metadata.json
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Tìm nhân vật cần cập nhật
        character_found = False
        for i, character in enumerate(metadata["character_blank"]):
            if character["name"] == character_name:
                character_found = True

                # Cập nhật tên nếu có
                if name:
                    metadata["character_blank"][i]["name"] = name

                # Cập nhật mô tả nếu có
                if description:
                    metadata["character_blank"][i]["description"] = description

                # Cập nhật address_matrix nếu có
                if address_matrix:
                    try:
                        address_matrix_dict = json.loads(address_matrix)
                        # Cập nhật từng trường trong address_matrix
                        if "address_matrix" not in metadata["character_blank"][i]:
                            metadata["character_blank"][i]["address_matrix"] = {}

                        for key, value in address_matrix_dict.items():
                            metadata["character_blank"][i]["address_matrix"][key] = value
                    except json.JSONDecodeError:
                        return JSONResponse(content={"error": "Invalid address_matrix format"}, status_code=400)

                # Cập nhật ảnh nhân vật nếu có
                if character_image:
                    # Tạo thư mục Character nếu chưa tồn tại
                    character_folder = os.path.join(story_folder, "Character")
                    os.makedirs(character_folder, exist_ok=True)

                    # Tên file ảnh sẽ dựa trên tên mới nếu có, nếu không thì dùng tên cũ
                    image_name = name.lower() if name else character_name.lower()
                    image_path = os.path.join(
                        character_folder, f"{image_name}.png")

                    # Lưu file ảnh
                    with open(image_path, "wb") as f:
                        f.write(await character_image.read())

                    # Cập nhật đường dẫn ảnh
                    metadata["character_blank"][i]["image_path"] = image_path

                break

        if not character_found:
            return JSONResponse(content={"error": f"Character {character_name} not found"}, status_code=404)

        # Lưu metadata.json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        return {"message": f"Character {character_name} updated successfully"}

    except Exception as e:
        logger.error(f"Error updating character: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@init_manga_router.patch("/metadata/{story_name}")
async def update_metadata(
    story_name: str = Path(...),
    mapping_name: Optional[str] = Form(None),
    translate_dict: Optional[str] = Form(None),
    story_type: Optional[str] = Form(None),
    source_language: Optional[str] = Form(None)
):
    """
    Cập nhật mapping_name và translate_dict trong metadata.json
    """
    try:

        metadata = MangaMetadata(
            story_name=story_name,
            mapping_name=mapping_name,
            translate_dict=translate_dict,
            story_type=story_type,
            source_language=source_language
        )

        metadata = MangaMetadataRepository.update_story(metadata)

        return {
            "data": metadata,
            "message": f"Metadata for story {story_name} updated successfully"
        }
    except ApiException as e:
        return JSONResponse(content={"error": e.message}, status_code=e.status_code)
    except Exception as e:
        logger.error(f"Server error when updating metadata:\n{e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
