from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from typing import List, Optional
import os
import shutil
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from api.service.episode_service import EpisodeService
from api.service.episode_job_manager import EpisodeJobManager
from api.schema.api_response_schema import ApiResponse
from api.dependencies import get_db
from module.knowledge.type import Language
from settings import BASE_DIR


# Router cho episode
episode_router = APIRouter(prefix="/episode", tags=["episode"])


@episode_router.post("")
async def create_episode(
    chapter_pages: List[UploadFile] = File(...),
    story_name: str = Form(...),
    chapter_number: int = Form(...),
    _: AsyncSession = Depends(get_db),
):
    """
    Tạo episode theo kiểu async đơn luồng, xử lý trực tiếp trong request
    (không dùng job queue/background task).
    """
    response = await EpisodeService.create_episode(
        chapter_pages=chapter_pages,
        story_name=story_name,
        chapter_number=chapter_number,
    )
    return response

@episode_router.get("/{story_name}/chapters/{chapter_number}")
async def get_episode_detail(
    story_name: str,
    chapter_number: int,
    _: AsyncSession = Depends(get_db),
):
    return await EpisodeService.get_episode_detail(story_name, chapter_number)


@episode_router.get("/{story_name}/chapters/{chapter_number}/translated")
async def get_translated_chapters(
    story_name: str,
    chapter_number: int,
    language: Language = Language.VIETNAMESE,
    pages: Optional[List[int]] = Query(default=None),
    mode: str = Query(default="zip"),
    translate: bool = Query(default=False),
    _: AsyncSession = Depends(get_db),
):
    return await EpisodeService.translate_episode(
        story_name=story_name,
        chapter_number=chapter_number,
        target_lang=language,
        page_numbers=pages,
        response_mode=mode,
        force_translate=translate,
    )


@episode_router.get("/jobs/{job_id}")
async def get_episode_job(job_id: str):
    job = EpisodeJobManager.get_job(job_id)
    return ApiResponse(
        message=f"Job {job_id} status",
        data={
            "job_id": job["job_id"],
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
        },
        status_code=200,
    )