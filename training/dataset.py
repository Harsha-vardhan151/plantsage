# training/dataset.py
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(img_size: int, augment: bool):
    if augment:
        tfm = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        tfm = T.Compose([
            T.Resize(int(img_size*1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return tfm

class PlantSageDataset(Dataset):
    def __init__(self, csv_path: str, labelmaps_path: str, img_size: int = 224, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        lm = json.load(open(labelmaps_path))
        self.species = lm["species"]
        self.issues = lm["issues"]
        self.species_to_idx = {s: i for i, s in enumerate(self.species)}
        self.issues_to_idx = {s: i for i, s in enumerate(self.issues)}
        self.img_size = img_size

        # use no-op lambdas instead of T.Identity() for broad torchvision compatibility
        noaug = (lambda x: x)
        aug_list = [
            T.Resize((img_size, img_size)),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1) if augment else noaug,
            T.RandomHorizontalFlip() if augment else noaug,
            T.ToTensor(),
        ]
        # torchvision.transforms.Compose accepts any callable, not only transforms.*
        self.augs = T.Compose(aug_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["path"]).convert("RGB")
        x = self.augs(img)

        species_idx = self.species_to_idx[str(r["species"])]
        species_target = torch.tensor(species_idx, dtype=torch.long)

        issues = str(r["issues"]) if pd.notna(r["issues"]) else ""
        y_multi = torch.zeros(len(self.issues), dtype=torch.float32)
        if issues.strip():
            for k in issues.split(","):
                if k in self.issues_to_idx:
                    y_multi[self.issues_to_idx[k]] = 1.0

        severity = torch.tensor(float(r["severity"]), dtype=torch.float32)
        return x, species_target, y_multi, severity
