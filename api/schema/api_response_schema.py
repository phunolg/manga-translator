from typing import Optional, List, TypeVar
from pydantic import BaseModel

type T = TypeVar('T')

class ApiResponse[T](BaseModel):
    message: str
    data: Optional[T | List[T]] = None
    status_code: int