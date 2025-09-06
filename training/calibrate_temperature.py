import argparse, json, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from training.dataset import PlantSageDataset
from training.model import MultiHeadPlantModel, TemperatureScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ece_score(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            gap = abs(accuracies[mask].mean() - confidences[mask].mean())
            ece += mask.mean() * gap
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--val_csv", type=str, default="data/synth_cls/val.csv")
    ap.add_argument("--labelmaps", type=str, default="training/labelmaps.json")
    ap.add_argument("--out", type=str, default="server/calibration.json")
    args = ap.parse_args()

    ds = PlantSageDataset(args.val_csv, args.labelmaps, img_size=224, augment=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    model = MultiHeadPlantModel(len(ds.species), len(ds.issues))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")['state_dict'], strict=False)
    model.eval()

    T = TemperatureScaler(1.0)
    T_opt = torch.nn.Parameter(T.log_T.data.clone(), requires_grad=True)
    opt = torch.optim.LBFGS([T_opt], lr=0.1, max_iter=50)

    logits_all = []
    labels_all = []
    with torch.no_grad():
        for x, y_species, _, _ in dl:
            s, _, _ = model(x)
            logits_all.append(s)
            labels_all.append(y_species)

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    def closure():
        opt.zero_grad()
        scaled = logits / T_opt.exp()
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        p_unc = F.softmax(logits, dim=1).numpy()
        p_cal = F.softmax(logits / T_opt.exp(), dim=1).numpy()

    ece_unc = ece_score(p_unc, labels.numpy())
    ece_cal = ece_score(p_cal, labels.numpy())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"temperature": float(T_opt.exp()), "ece_uncal": ece_unc, "ece_cal": ece_cal},
            f,
            indent=2
        )

    print(f"Saved calibration: T={float(T_opt.exp()):.3f}, ECE_unc={ece_unc:.4f}, ECE_cal={ece_cal:.4f}")


if __name__ == "__main__":
    main()
