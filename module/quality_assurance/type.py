from enum import Enum
from typing import List
from pydantic import BaseModel

class ErrorType(Enum):
    ACCURACY = "độ chính xác"
    FLUENCY = "độ trôi chảy"
    CONTEXT = "ngữ cảnh trước/sau/hiện tại"
    NAMING_CONVENTION = "quy ước tên riêng/xưng hô"
    STYLE = "phong cách"
    MISSING_TRANSLATION = "dịch sai từ điển"

class ErrorTranslation(BaseModel):
    error_type: ErrorType | str
    subcategory: str
    description: str
    suggestion: str

class ErrorResult(BaseModel):
    original_segment: str
    translation_segment: str
    errors: List[ErrorTranslation] = []