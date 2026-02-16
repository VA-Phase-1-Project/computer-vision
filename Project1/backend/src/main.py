from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from face_detection.router import router as face_router

app = FastAPI(title="Face Detection Web App", version="1.0.0")

# NO PREFIX → IMPORTANT
app.include_router(face_router)

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
