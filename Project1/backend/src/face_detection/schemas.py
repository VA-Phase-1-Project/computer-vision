from pydantic import BaseModel

class UploadResponse(BaseModel):
    file_name: str

class CountResponse(BaseModel):
    faces: int
