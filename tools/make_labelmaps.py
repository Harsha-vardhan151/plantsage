# tools/make_labelmaps_from_csv.py
import json, csv
from pathlib import Path

CSV_DIR = Path("data/disease_cls")
LM_PATH = Path("training/labelmaps2.json")

species = set()
issues  = set()

for fn in ["train.csv", "val.csv"]:
    with open(CSV_DIR/fn, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            species.add(row["species"])
            if row["issues"].strip():
                issues.add(row["issues"].strip())

labelmaps = {
  "species": sorted(species),
  "issues": sorted(issues)
}

LM_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LM_PATH, "w") as f:
    json.dump(labelmaps, f, indent=2)

print(f"Saved labelmaps with {len(species)} species and {len(issues)} issues to {LM_PATH}")
