# server/inference.py
import os, time, json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from training.model import MultiHeadPlantModel

# ---- Environment / defaults ----
DEVICE = os.getenv("TORCH_DEVICE", "cpu")
CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS", "server/models/classifier_last.ckpt")
TORCHSCRIPT_PATH = os.getenv("TORCHSCRIPT_WEIGHTS", "server/models/classifier.ts.pt")
ONNX_PATH = os.getenv("ONNX_WEIGHTS", "")  # optional; if empty or fails import, we skip ORT
CALIBRATION_FILE = os.getenv("CALIBRATION_FILE", "server/calibration.json")
LABELMAPS_PATH = os.getenv("LABELMAPS_PATH", "training/labelmaps.json")

# ---- Basic utils ----
def _softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)

def _load_labelmaps() -> Dict[str, List[str]]:
    lm = {"species": [], "issues": []}
    p = Path(LABELMAPS_PATH)
    if p.exists():
        try:
            lm = json.load(open(p, "r"))
        except Exception:
            pass
    return lm

def _load_temperature() -> float:
    p = Path(CALIBRATION_FILE)
    if p.exists():
        try:
            data = json.load(open(p, "r"))
            return float(data.get("temperature", 1.0))
        except Exception:
            return 1.0
    return 1.0

# ---- Optional ORT import (guarded) ----
def _try_import_onnxruntime():
    if not ONNX_PATH or not Path(ONNX_PATH).exists():
        return None
    try:
        import onnxruntime as ort  # noqa
        return ort
    except Exception:
        return None

# ---- Model loading ----
_preproc = T.Compose([T.Resize((224, 224)), T.ToTensor()])

def load_models() -> Dict[str, Any]:
    """Loads models and returns a handle dict used by analyze_image()."""
    labelmaps = _load_labelmaps()
    temperature = _load_temperature()
    ort = _try_import_onnxruntime()

    if ort is not None:
        # ONNX path (only if import & model are OK)
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        mode = "onnx"
        classifier = None
    else:
        # Prefer TorchScript if present
        if Path(TORCHSCRIPT_PATH).exists():
            classifier = torch.jit.load(TORCHSCRIPT_PATH, map_location=DEVICE).eval()
            mode = "torchscript"
        else:
            # Lightning checkpoint fallback
            state = torch.load(CLASSIFIER_WEIGHTS, map_location="cpu")
            h = state.get("hyper_parameters", {})
            n_species = int(h.get("n_species", len(labelmaps.get("species", [])) or 200))
            n_issues = int(h.get("n_issues", len(labelmaps.get("issues", [])) or 0))
            model = MultiHeadPlantModel(n_species=n_species, n_issues=n_issues)
            model.load_state_dict(state["state_dict"], strict=True)
            classifier = model.to(DEVICE).eval()
            mode = "torch"

        sess = None

    return {
        "mode": mode,
        "device": DEVICE,
        "classifier": classifier,
        "onnx_session": sess,
        "labelmaps": labelmaps,
        "temperature": temperature,
        "versions": {
            "classifier_ckpt": Path(CLASSIFIER_WEIGHTS).name if Path(CLASSIFIER_WEIGHTS).exists() else "",
            "torchscript": Path(TORCHSCRIPT_PATH).name if Path(TORCHSCRIPT_PATH).exists() else "",
            "onnx": Path(ONNX_PATH).name if ONNX_PATH else "",
            "labelmaps": Path(LABELMAPS_PATH).name if Path(LABELMAPS_PATH).exists() else "",
        },
    }

# ---- Inference API used by app.py ----
def analyze_image(models: Dict[str, Any], img: Image.Image, topk: int = 5) -> Dict[str, Any]:
    """
    Minimal analysis used by /v1/analyze route:
      - returns Top-K species names + calibrated confidences
      - no detector (whole image)
      - produces metadata with latency and versions
    """
    t0 = time.time()
    x = _preproc(img.convert("RGB")).unsqueeze(0)

    # Forward
    if models["mode"] == "onnx":
        # ONNX path
        ort_inputs = {"input": x.numpy()}
        s_logits, i_logits, sev = models["onnx_session"].run(None, ort_inputs)
        s_logits = torch.from_numpy(s_logits)
    else:
        # Torch/TorchScript
        x = x.to(models["device"])
        with torch.no_grad():
            s_logits, _, _ = models["classifier"](x)

    # Temperature scaling + softmax
    s_logits = s_logits / float(models["temperature"])
    probs = _softmax(s_logits).squeeze(0)  # [C]
    k = min(topk, probs.shape[0])
    top_vals, top_idx = torch.topk(probs, k)

    # Map to names
    species_names = models["labelmaps"].get("species", [])
    out_preds = []
    for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
        name = species_names[idx] if 0 <= idx < len(species_names) else f"class_{idx}"
        out_preds.append({"name": name, "confidence": float(score)})

    latency_ms = (time.time() - t0) * 1000.0
    return {
        "species": out_preds,
        "issues": [],                 # not computed in this minimal path
        "severity": None,             # not computed
        "boxes": [],                  # detector disabled in this CPU minimal setup
        "metadata": {
            "latency_ms": round(latency_ms, 2),
            "mode": models["mode"],
            "temperature": models["temperature"],
            "versions": models["versions"],
            "device": models["device"],
        },
    }
