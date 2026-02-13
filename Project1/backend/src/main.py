from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from face_detection.router import router as face_router

app = FastAPI(title="Face Detection Web App", version="1.0.0")

app.include_router(face_router, prefix="/face", tags=["Face Detection"])

# main.py is: Project1/backend/src/main.py
# parents[2] = Project1/
BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    return FileResponse(index_file)
