from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from urllib.parse import unquote
from settings import BASE_DIR

file_router = APIRouter(prefix="/files", tags=["files"])


def get_media_type(file_path: str) -> str:
    """Xác định media type dựa trên extension"""
    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "application/octet-stream")


@file_router.get("/images/{file_path:path}")
async def get_image(file_path: str):
    """
    Serve images từ thư mục data/stories hoặc transcript_history
    file_path: đường dẫn tương đối đã được URL encoded, ví dụ: "Dư%20Lạc%20và%20những%20người%20bạn/Character/dư%20lạc.png"
    """
    # Decode URL path
    decoded_path = unquote(file_path)
    
    # Thử tìm trong data/stories trước
    data_stories_dir = os.path.normpath(os.path.join(BASE_DIR, "data", "stories"))
    full_path = os.path.normpath(os.path.join(data_stories_dir, decoded_path))
    
    # Security check: đảm bảo file nằm trong thư mục cho phép
    if not full_path.startswith(data_stories_dir):
        # Thử transcript_history
        transcript_history_dir = os.path.normpath(os.path.join(BASE_DIR, "transcript_history"))
        full_path = os.path.normpath(os.path.join(transcript_history_dir, decoded_path))
        
        if not full_path.startswith(transcript_history_dir):
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Kiểm tra file có phải là file không (không phải directory)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Trả về file với proper content type
    return FileResponse(
        full_path,
        media_type=get_media_type(full_path)
    )

