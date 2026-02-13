from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    message: str
    file_name: str
    file_id: str


class CountResponse(BaseModel):
    current_face_count: int
    total_unique_faces: int
    is_active: bool
    last_error: Optional[str] = None
