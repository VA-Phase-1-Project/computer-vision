import os
import shutil
import base64
import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from .service import FaceDetector, draw_faces
from .schemas import FaceBox, FaceDetectionResult
from .video import mjpeg_stream_from_video_path, current_face_count

router = APIRouter(prefix="/face", tags=["Face Detection"])

detector = FaceDetector(min_confidence=0.1)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/image", response_model=FaceDetectionResult)
async def detect_faces_in_image(file: UploadFile = File(...)):

    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detections = detector.detect(img)
    img_with_boxes = draw_faces(img, detections)

    _, buffer = cv2.imencode(".jpg", img_with_boxes)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    faces = [
        FaceBox(box=box, confidence=score)
        for (box, score) in detections
    ]

    return FaceDetectionResult(
        face_count=len(faces),
        faces=faces,
        image_base64=image_base64
    )


@router.post("/video")
async def upload_video(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "stream_url": f"/face/video/stream/{file.filename}"
    }


@router.get("/video/stream/{filename}")
def stream_video(filename: str):

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    return StreamingResponse(
        mjpeg_stream_from_video_path(file_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video/count")
def get_current_face_count():
    return {"face_count": current_face_count}
