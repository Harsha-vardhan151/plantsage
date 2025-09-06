import io, json
from PIL import Image
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] == True


def test_analyze_minimal():
    img = Image.new("RGB", (256, 256), (120, 180, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"image": ("test.jpg", buf, "image/jpeg")}
    data = {"topk": "3", "locale": "en"}

    r = client.post("/v1/analyze", data=data, files=files)
    assert r.status_code == 200

    js = r.json()
    assert "species" in js and "issues" in js and "boxes" in js and "metadata" in js
