# tools/quick_infer.py
import sys, json, torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# --- add repo root to sys.path so "training.*" imports work ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.model import MultiHeadPlantModel  # now import works

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/quick_infer.py <image_path>")
        sys.exit(1)
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(2)

    CKPT = REPO_ROOT / "server/models/classifier_last.ckpt"
    LABELS = REPO_ROOT / "training/labelmaps2.json"
    DEVICE = "cpu"

    lm = {"species": []}
    if LABELS.exists():
        lm = json.load(open(LABELS, "r"))
    species_names = lm.get("species", [])

    state = torch.load(CKPT, map_location="cpu")
    h = state.get("hyper_parameters", {})
    n_species = int(h.get("n_species", len(species_names) or 200))
    n_issues = int(h.get("n_issues", 0))

    model = MultiHeadPlantModel(n_species=n_species, n_issues=n_issues)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().to(DEVICE)

    pre = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = pre(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        s, _, _ = model(x)
        p = torch.softmax(s, dim=-1)
        v, i = p.topk(5, dim=-1)

    print(f"\nTop-5 for: {img_path}\n")
    for score, idx in zip(v[0].tolist(), i[0].tolist()):
        name = species_names[idx] if 0 <= idx < len(species_names) else f"class_{idx}"
        print(f"{name:40s} {score*100:5.1f}%")

if __name__ == "__main__":
    main()
