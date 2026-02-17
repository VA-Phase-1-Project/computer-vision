import uuid
from pathlib import Path
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response
from .schemas import UploadResponse, CountResponse
from .service import DETECTOR, draw_faces
from .video import mjpeg_stream_from_video_path, get_counts

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(file: UploadFile):
    ext = Path(file.filename).suffix
    file_id = uuid.uuid4().hex
    file_name = f"{file_id}{ext}"
    path = UPLOAD_DIR / file_name
    with open(path, "wb") as f:
        f.write(file.file.read())
    return file_name

# IMAGE UPLOAD
@router.post("/upload/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    file_name = save_upload(file)
    return UploadResponse(file_name=file_name)

@router.get("/image/{file_name}")
def get_image(file_name: str):
    image_path = UPLOAD_DIR / file_name
    img = cv2.imread(str(image_path))
    boxes = DETECTOR.detect_boxes(img)
    draw_faces(img, boxes)
    _, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# VIDEO UPLOAD
@router.post("/upload/video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    file_name = save_upload(file)
    return UploadResponse(file_name=file_name)

@router.get("/video/stream/{file_name}")
def stream_video(file_name: str):
    path = UPLOAD_DIR / file_name
    return StreamingResponse(
        mjpeg_stream_from_video_path(str(path)),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/video/count", response_model=CountResponse)
def video_count():
    return CountResponse(**get_counts())
