from fastapi import APIRouter, File, Form, UploadFile, Path, Depends, Body
from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from api.schema.metadata_schema import CharacterDetailResponse
from api.schema.api_response_schema import ApiResponse
from api.service.metadata_service import MetadataService
from api.dependencies import get_db


# Router cho character
character_router = APIRouter(prefix="/character", tags=["character"])


@character_router.post("")
async def create_character(
    story_name: str = Form(...),
    name_character: str = Form(...),
    description: Optional[str] = Form(None),
    character_image: UploadFile = File(...),
    _: AsyncSession = Depends(get_db)
):
    """ 
    Tạo nhân vật mới và thêm vào database
    """
    result = await MetadataService.create_character(
        story_name=story_name,
        name_character=name_character,
        description=description,
        character_image=character_image
    )
    return result


@character_router.get("/{story_name}/{character_name}", response_model=ApiResponse[CharacterDetailResponse])
async def get_character_detail(
    story_name: str = Path(...),
    character_name: str = Path(...),
    _: AsyncSession = Depends(get_db)
):
    """
    Lấy thông tin chi tiết của một nhân vật bao gồm address matrix
    """
    result = await MetadataService.get_character_detail(
        story_name=story_name,
        character_name=character_name
    )
    return ApiResponse[CharacterDetailResponse](
        message=f"Character {character_name} retrieved successfully",
        data=result,
        status_code=200
    )


@character_router.patch("/{story_name}/{character_name}")
async def update_character(
    story_name: str = Path(...),
    character_name: str = Path(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    character_image: Optional[UploadFile] = File(None),
    _: AsyncSession = Depends(get_db)
):
    """
    Cập nhật thông tin nhân vật trong database
    """
    result = await MetadataService.update_character(
        story_name=story_name,
        character_name=character_name,
        name=name,
        description=description,
        character_image=character_image
    )
    return result


@character_router.get("/{story_name}/{character_name}/address-matrix")
async def get_address_matrices(
    story_name: str = Path(...),
    character_name: str = Path(...),
    _: AsyncSession = Depends(get_db)
):
    """
    Lấy tất cả address matrix của một character
    """
    result = await MetadataService.get_address_matrices(
        story_name=story_name,
        character_name=character_name
    )
    return result


@character_router.post("/{story_name}/{character_name}/address-matrix")
async def merge_address_matrices(
    story_name: str = Path(...),
    character_name: str = Path(...),
    address_matrix: Dict[str, str] = Body(...),
    _: AsyncSession = Depends(get_db)
):
    """
    Bổ sung/merge address matrix với dữ liệu hiện có (ghi đè nếu đã tồn tại)
    """
    result = await MetadataService.merge_address_matrices(
        story_name=story_name,
        character_name=character_name,
        address_matrix=address_matrix
    )
    return result


@character_router.put("/{story_name}/{character_name}/address-matrix")
async def replace_address_matrices(
    story_name: str = Path(...),
    character_name: str = Path(...),
    address_matrix: Dict[str, str] = Body(...),
    _: AsyncSession = Depends(get_db)
):
    """
    Thay thế toàn bộ address matrix của một character (replace all)
    """
    result = await MetadataService.replace_address_matrices(
        story_name=story_name,
        character_name=character_name,
        address_matrix=address_matrix
    )
    return result


@character_router.delete("/{story_name}/{character_name}/address-matrix/{target_name}")
async def delete_address_matrix(
    story_name: str = Path(...),
    character_name: str = Path(...),
    target_name: str = Path(...),
    _: AsyncSession = Depends(get_db)
):
    """
    Xóa một address matrix cụ thể của một character
    """
    result = await MetadataService.delete_address_matrix(
        story_name=story_name,
        character_name=character_name,
        target_name=target_name
    )
    return result

