# tools/prepare_species_csv.py
import os, csv
from pathlib import Path
import random

DATA_ROOT = Path("data/plantnet_300K/images_test")  # adjust to train folder if you want full training
OUT_DIR = Path("data/plantnet_cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for species_dir in sorted(DATA_ROOT.iterdir()):
    if species_dir.is_dir():
        species = species_dir.name
        for img_file in species_dir.glob("*.jpg"):
            rows.append([str(img_file), species, "", 0.0])  # no issues/severity in PlantNet

random.shuffle(rows)
split = int(0.8 * len(rows))
train, val = rows[:split], rows[split:]

for name, part in [("train.csv", train), ("val.csv", val)]:
    with open(OUT_DIR/name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","species","issues","severity"])
        w.writerows(part)

print(f"Wrote {len(train)} train and {len(val)} val samples into {OUT_DIR}")
