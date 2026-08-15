from fastapi import APIRouter
from api.controller.metadata_controller import metadata_router
from api.controller.character_controller import character_router
from api.controller.file_controller import file_router
from api.controller.episode_controller import episode_router


app_router = APIRouter(prefix="/api/v1")

app_router.include_router(metadata_router)
app_router.include_router(character_router)
app_router.include_router(file_router)
app_router.include_router(episode_router)
