from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path

from .analysis_service import analyze_mri

app = FastAPI(title="Radiomics Alzheimer API")

# React CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-mri")
async def analyze_mri_endpoint(
    img_file: UploadFile = File(...),
    hdr_file: UploadFile = File(...)
):
    if not img_file.filename.endswith(".img"):
        raise HTTPException(400, "img_file must be .img")

    if not hdr_file.filename.endswith(".hdr"):
        raise HTTPException(400, "hdr_file must be .hdr")

    session_dir = Path("temp_uploads") / str(uuid.uuid4())
    session_dir.mkdir(parents=True, exist_ok=True)

    img_path = session_dir / img_file.filename
    hdr_path = session_dir / hdr_file.filename

    try:
        with open(img_path, "wb") as f:
            shutil.copyfileobj(img_file.file, f)

        with open(hdr_path, "wb") as f:
            shutil.copyfileobj(hdr_file.file, f)

        result = analyze_mri(str(img_path))
        return result

    finally:
        shutil.rmtree(session_dir, ignore_errors=True)