from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from face_detection.router import router as face_router
import os

app = FastAPI(title="Video Analytics API")

app.include_router(face_router)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
