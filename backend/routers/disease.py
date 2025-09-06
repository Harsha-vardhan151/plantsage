from fastapi import APIRouter, UploadFile, File
import shutil
import os

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/disease")
async def detect_disease(file: UploadFile = File(...)):
    # Save file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 👉 TODO: Run your disease detection model here
    disease_name = "Leaf Rust (dummy prediction)"

    return {"disease": disease_name}
