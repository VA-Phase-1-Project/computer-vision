import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response

from .schemas import UploadResponse, CountResponse
from .service import DETECTOR, draw_faces
from .video import mjpeg_stream_from_video_path, get_counts

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/src
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file: UploadFile) -> tuple[str, str]:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    ext = Path(file.filename).suffix.lower().strip()
    file_id = uuid.uuid4().hex
    saved_name = f"{file_id}{ext}" if ext else file_id

    out_path = UPLOAD_DIR / saved_name
    with open(out_path, "wb") as f:
        f.write(file.file.read())

    return file_id, saved_name


@router.post("/upload/video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    file_id, saved_name = _save_upload(file)
    return UploadResponse(message="Video uploaded", file_name=saved_name, file_id=file_id)


@router.get("/video/stream/{file_name}")
def stream_video(file_name: str):
    video_path = UPLOAD_DIR / file_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return StreamingResponse(
        mjpeg_stream_from_video_path(str(video_path)),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/video/count", response_model=CountResponse)
def video_count():
    c = get_counts()
    return CountResponse(**c)


@router.post("/upload/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    file_id, saved_name = _save_upload(file)
    return UploadResponse(message="Image uploaded", file_name=saved_name, file_id=file_id)


@router.get("/image/annotated/{file_name}")
def annotated_image(file_name: str):
    image_path = UPLOAD_DIR / file_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image")

    boxes = DETECTOR.detect_boxes(img)
    draw_faces(img, boxes)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
