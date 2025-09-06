# tools/prepare_disease_csv.py
import csv, re
from pathlib import Path
import random

RAW_DIR = Path("data/disease/raw")      # <— adjust if different
OUT_DIR = Path("data/disease_cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Heuristic: map issue name fragments -> default severity on [0,1]
SEVERITY_MAP = {
    "late_blight": 0.85,
    "early_blight": 0.65,
    "bacterial": 0.60,
    "mosaic": 0.55,
    "septoria": 0.60,
    "leaf_mold": 0.55,
    "target_spot": 0.55,
    "spider_mites": 0.50,
    "yellowleaf": 0.65,
    "curl": 0.60,
    "virus": 0.60,
    "healthy": 0.05,
}

def normalize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = s.replace("___","_").replace("__","_")
    s = re.sub(r"[^A-Za-z0-9_]+","_", s)
    return s.strip("_")

def parse_species_issue(folder_name: str):
    """
    Examples:
      'Tomato_Early_blight' -> species='Tomato', issue='Early_blight'
      'Pepper__bell___Bacterial_spot' -> species='Pepper_bell', issue='Bacterial_spot'
      'Potato_healthy' -> species='Potato', issue=''
    """
    name = normalize(folder_name)
    parts = [p for p in name.split("_") if p]  # collapse empties
    if len(parts) == 1:  # ambiguous, treat as species only
        return parts[0], ""
    # heuristic: first token is species; merge next token if it's lowercase like 'bell'
    species = parts[0]
    # allow multi-token species like Pepper_bell
    if len(parts) >= 3 and parts[1].islower():
        species = f"{parts[0]}_{parts[1]}"
        issue_tokens = parts[2:]
    else:
        issue_tokens = parts[1:]
    issue = "_".join(issue_tokens)
    # healthy special-case
    if issue.lower() in ("healthy", "normal", "control"):
        issue = ""
    return species, issue

rows = []
species_set = set()
issues_set = set()

for cls_dir in sorted(RAW_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    species, issue = parse_species_issue(cls_dir.name)
    species_set.add(species)
    if issue:
        issues_set.add(issue)

    for img_path in cls_dir.rglob("*"):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            # severity via heuristic
            sev_key = issue.lower() if issue else "healthy"
            sev = SEVERITY_MAP.get(sev_key, 0.60 if issue else 0.05)
            rows.append([str(img_path), species, issue, f"{sev:.2f}"])

random.shuffle(rows)
split = int(0.85 * len(rows))
train, val = rows[:split], rows[split:]

for name, part in [("train.csv", train), ("val.csv", val)]:
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","species","issues","severity"])
        w.writerows(part)

print(f"Wrote {len(train)} train and {len(val)} val samples to {OUT_DIR}")
print(f"Detected species: {len(species_set)}, issues: {len(issues_set)}")
