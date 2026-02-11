from pydantic import BaseModel
from typing import List, Tuple


class FaceBox(BaseModel):
    box: Tuple[int, int, int, int]
    confidence: float


class FaceDetectionResult(BaseModel):
    face_count: int
    faces: List[FaceBox]
    image_base64: str
