from fastapi import FastAPI
from backend.routers import identify, disease


app = FastAPI(title="PlantSage API")

app.include_router(identify.router, prefix="/v1")
app.include_router(disease.router, prefix="/v1")
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
