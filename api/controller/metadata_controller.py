
from fastapi import APIRouter, File, Form, UploadFile, Path, Depends, Body, Query
from typing import Optional, List, Dict, Any, cast
from sqlalchemy.ext.asyncio import AsyncSession
from api.schema.metadata_schema import (
    CreateStoryRequest,
    CreateStoryResponse,
    StoryDetailResponse,
    CreateTranslateDictRequest,
    TranslateDictResponse,
    MappingNameUpsertRequest,
)
from api.schema.api_response_schema import ApiResponse
from api.service.metadata_service import MetadataService
from api.dependencies import get_db
from module.knowledge.type import Language, Topic


# Router cho metadata
metadata_router = APIRouter(prefix="/metadata", tags=["metadata"])


@metadata_router.get("/stories")
async def get_all_stories(
    _: AsyncSession = Depends(get_db)
):
    """
    Lấy danh sách tất cả các truyện
    """
    stories = await MetadataService.get_all_stories()
    stories_response = [
        CreateStoryResponse(
            story_name=str(story.story_name),
            story_type=Topic(story.story_type),
            source_language=Language(story.source_language),
        )
        for story in stories
    ]
    return ApiResponse[List[CreateStoryResponse]](
        message="Stories retrieved successfully",
        data=stories_response,
        status_code=200
    )


@metadata_router.get("/stories/{story_name}", response_model=ApiResponse[StoryDetailResponse])
async def get_story_by_name(
    story_name: str = Path(...),
    _: AsyncSession = Depends(get_db)
):
    """
    Lấy thông tin chi tiết của một truyện theo tên
    """
    story = await MetadataService.get_story_by_name(story_name)
    return ApiResponse[StoryDetailResponse](
        message=f"Story {story_name} retrieved successfully",
        data=story,
        status_code=200
    )


@metadata_router.post("/stories", response_model=ApiResponse[CreateStoryResponse])
async def init_story(
    story_request: CreateStoryRequest,
    _: AsyncSession = Depends(get_db)  # Chỉ cần để set context, không dùng trực tiếp
):
    """
    Khởi tạo story mới với story_name, story_type và source_language
    """
    story = await MetadataService.create_story(story_request)
    return ApiResponse[CreateStoryResponse](
        message=f"Story {story_request.story_name} created successfully", 
        data=story,
        status_code=201
    )

@metadata_router.post("/stories/{story_name}/translate-dict", response_model=ApiResponse[TranslateDictResponse])
async def create_translate_dict(
    story_name: str = Path(...),
    translate_dict_request: CreateTranslateDictRequest = Body(...),
    _: AsyncSession = Depends(get_db)
):
    """ 
    Thêm hoặc cập nhật từ điển dịch cho một ngôn ngữ cụ thể
    """
    result = await MetadataService.create_translate_dict(
        story_name=story_name,
        translate_dict_request=translate_dict_request
    )
    return ApiResponse(
        message=result["message"],
        data=result["data"],
        status_code=201
    )


@metadata_router.patch("/metadata/{story_name}")
async def update_metadata(
    story_name: str = Path(...),
    mapping_name: Optional[str] = Form(None),
    translate_dict: Optional[str] = Form(None),
    story_type: Optional[str] = Form(None),
    source_language: Optional[str] = Form(None),
    _: AsyncSession = Depends(get_db)
):
    """
    Cập nhật mapping_name và translate_dict trong database
    """

    result = await MetadataService.update_metadata(
        story_name=story_name,
        mapping_name=mapping_name,
        translate_dict=translate_dict,
        story_type=story_type,
        source_language=source_language
    )
    return result


@metadata_router.get("/stories/{story_name}/mapping-names")
async def list_mapping_names(
    story_name: str = Path(...),
    language: Optional[Language] = Query(None),
    _: AsyncSession = Depends(get_db),
):
    result = await MetadataService.get_mapping_names(
        story_name=story_name,
        language=language,
    )
    return ApiResponse(
        message=result["message"],
        data=result["mapping_names"],
        status_code=200,
    )


@metadata_router.post("/stories/{story_name}/mapping-names")
async def upsert_mapping_names(
    story_name: str = Path(...),
    payload: MappingNameUpsertRequest = Body(...),
    _: AsyncSession = Depends(get_db),
):
    result = await MetadataService.upsert_mapping_names(
        story_name=story_name,
        payload=payload,
    )
    return ApiResponse(
        message=result["message"],
        data=result["mapping_name"],
        status_code=201,
    )


@metadata_router.delete("/stories/{story_name}/mapping-names")
async def delete_mapping_name(
    story_name: str = Path(...),
    language: Language = Query(...),
    source: str = Query(...),
    _: AsyncSession = Depends(get_db),
):
    result = await MetadataService.delete_mapping_name(
        story_name=story_name,
        language=language,
        source=source,
    )
    return ApiResponse(
        message=result["message"],
        status_code=200,
    )

