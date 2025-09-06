# plantsage/server/app.py
import os, time, io, json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from server.inference import load_models, analyze_image

app = FastAPI(title="PlantSage API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)

MODELS = load_models()


class AnalyzeResponse(BaseModel):
    species: list
    issues: list
    boxes: list
    tips: list
    metadata: dict


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    topk: int = Form(5),
    locale: str = Form("en")
):
    t0 = time.time()
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    result = analyze_image(MODELS, img, topk=topk, locale=locale)
    result["metadata"]["latency_ms"] = int((time.time() - t0) * 1000)
    return result


@app.get("/healthz")
def healthz():
    return {"ok": True}
