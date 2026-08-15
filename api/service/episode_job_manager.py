import asyncio
import os
import shutil
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from api.schema.api_response_schema import ApiResponse
from api.schema.api_exception_schema import ApiException
from api.service.episode_service import EpisodeService
from module.database.postgre.database import AsyncSessionLocal, db_session_context


class EpisodeJobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class EpisodeJobManager:
    jobs: Dict[str, Dict] = {}

    @classmethod
    def create_job(
        cls,
        story_name: str,
        chapter_number: int,
        file_paths: List[str],
        temp_dir: str,
        job_id: Optional[str] = None,
    ) -> Dict:
        job_id = job_id or str(uuid4())
        job_info = {
            "job_id": job_id,
            "story_name": story_name,
            "chapter_number": chapter_number,
            "status": EpisodeJobStatus.QUEUED,
            "result": None,
            "error": None,
        }
        cls.jobs[job_id] = job_info
        asyncio.create_task(
            cls._process_job(job_id, story_name, chapter_number, file_paths, temp_dir)
        )
        return job_info

    @classmethod
    async def _process_job(
        cls,
        job_id: str,
        story_name: str,
        chapter_number: int,
        file_paths: List[str],
        temp_dir: str,
    ) -> None:
        job = cls.jobs[job_id]
        job["status"] = EpisodeJobStatus.PROCESSING
        try:
            # Mỗi job chạy với một DB session riêng, độc lập với request
            async with AsyncSessionLocal() as session:
                token = db_session_context.set(session)
                try:
                    response: ApiResponse = await EpisodeService.create_episode(
                        chapter_pages=file_paths,
                        story_name=story_name,
                        chapter_number=chapter_number,
                    )
                    job["status"] = EpisodeJobStatus.SUCCESS
                    job["result"] = response.dict()
                finally:
                    db_session_context.reset(token)
        except Exception as exc:  # noqa: BLE001
            job["status"] = EpisodeJobStatus.FAILED
            job["error"] = str(exc)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def get_job(cls, job_id: str) -> Dict:
        job = cls.jobs.get(job_id)
        if not job:
            raise ApiException(message=f"Job {job_id} not found", status_code=404)
        return job

