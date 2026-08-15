import os
from settings import BASE_DIR
from urllib.parse import quote


def convert_image_path_to_url(image_path: str | None) -> str | None:
    """
    Convert absolute image path thành URL endpoint
    Ví dụ: "/home/aorus/workspaces/magiv2/data/stories/Dư Lạc và những người bạn/Character/dư lạc.png"
    -> "/api/v1/files/images/Dư%20Lạc%20và%20những%20người%20bạn/Character/dư%20lạc.png"
    """
    if not image_path:
        return None
    
    # Normalize path
    image_path = os.path.normpath(image_path)
    data_stories_dir = os.path.normpath(os.path.join(BASE_DIR, "data", "stories"))
    
    # Kiểm tra xem path có nằm trong data/stories không
    if image_path.startswith(data_stories_dir):
        # Lấy relative path từ data/stories
        relative_path = os.path.relpath(image_path, data_stories_dir)
        # Convert thành URL endpoint với URL encoding
        # Sử dụng quote với safe='' để encode tất cả ký tự đặc biệt
        encoded_path = "/".join(quote(part, safe="") for part in relative_path.split(os.sep))
        return f"/api/v1/files/images/{encoded_path}"
    
    # Nếu không nằm trong data/stories, kiểm tra transcript_history
    transcript_history_dir = os.path.normpath(os.path.join(BASE_DIR, "transcript_history"))
    if image_path.startswith(transcript_history_dir):
        relative_path = os.path.relpath(image_path, transcript_history_dir)
        encoded_path = "/".join(quote(part, safe="") for part in relative_path.split(os.sep))
        return f"/api/v1/files/images/{encoded_path}"
    
    # Nếu không match, trả về None
    return None

