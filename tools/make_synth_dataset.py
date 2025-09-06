import os, json, random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLS_DIR = DATA / "synth_cls"
DET_DIR = DATA / "synth_det"
DET_IMAGES = DET_DIR / "images"
DET_LABELS = DET_DIR / "labels"

SPECIES = [f"species_{i:03d}" for i in range(10)]  # 10 demo species
ISSUES = ["leaf_spot", "mildew", "rust", "nutrient_def", "pest_bite"]


def make_leaf_img(w=512, h=512, spots=3):
    img = Image.new("RGB", (w, h), (230, 240, 230))
    d = ImageDraw.Draw(img)

    # simple leaf: ellipse
    leaf_bbox = (w * 0.2, h * 0.15, w * 0.8, h * 0.85)
    d.ellipse(leaf_bbox, fill=(60, 150, 60))

    boxes = []
    for _ in range(spots):
        x = random.randint(int(w * 0.3), int(w * 0.7))
        y = random.randint(int(h * 0.25), int(h * 0.75))
        r = random.randint(15, 30)
        color = random.choice([(180, 180, 40), (160, 80, 20), (200, 160, 60)])
        d.ellipse((x - r, y - r, x + r, y + r), fill=color)
        boxes.append((max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r)))

    return img, boxes


def main():
    CLS_DIR.mkdir(parents=True, exist_ok=True)
    DET_IMAGES.mkdir(parents=True, exist_ok=True)
    DET_LABELS.mkdir(parents=True, exist_ok=True)

    # labelmaps
    labelmaps = {
        "species": SPECIES,
        "issues": ISSUES
    }
    with open(ROOT / "training" / "labelmaps.json", "w") as f:
        json.dump(labelmaps, f, indent=2)

    # classifier CSV
    csv_lines = ["path,species,issues,severity\n"]
    for i in range(20):
        img, boxes = make_leaf_img(spots=random.randint(1, 4))
        species = random.choice(SPECIES)
        # 0-2 random issues
        n_issues = random.randint(0, 2)
        issues = ",".join(sorted(set(random.choices(ISSUES, k=n_issues))))
        severity = round(random.uniform(0.1, 0.9), 2) if n_issues > 0 else 0.05
        p = CLS_DIR / f"img_{i:03d}.jpg"
        img.save(p, quality=92)
        csv_lines.append(f"{p},{species},{issues},{severity}\n")

    with open(CLS_DIR / "train.csv", "w") as f:
        f.writelines(csv_lines)
    with open(CLS_DIR / "val.csv", "w") as f:
        f.writelines(csv_lines[:10])

    # detection: YOLO txt + images
    for i in range(10):
        img, boxes = make_leaf_img(spots=random.randint(1, 4))
        imp = DET_IMAGES / f"img_{i:03d}.jpg"
        img.save(imp, quality=92)
        w, h = img.size
        # YOLO format: cls cx cy w h normalized; use single class (symptom)
        yolo = []
        for (x1, y1, x2, y2) in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        with open(DET_LABELS / f"img_{i:03d}.txt", "w") as f:
            f.writelines(yolo)

    print("Synthetic data created under data/synth_cls and data/synth_det")


if __name__ == "__main__":
    main()
