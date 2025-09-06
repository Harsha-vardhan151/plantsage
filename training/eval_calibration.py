import argparse, json, torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from training.dataset import PlantSageDataset
from training.model import MultiHeadPlantModel


def reliability(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean())
            ys.append(acc[m].mean())
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_csv", default="data/synth_cls/val.csv")
    ap.add_argument("--labelmaps", default="training/labelmaps.json")
    ap.add_argument("--calibration", default="server/calibration.json")
    ap.add_argument("--out", default="reliability.png")
    args = ap.parse_args()

    ds = PlantSageDataset(args.val_csv, args.labelmaps, 224, False)
    model = MultiHeadPlantModel(len(ds.species), len(ds.issues))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")['state_dict'], strict=False)
    model.eval()

    x = [ds[i][0] for i in range(len(ds))]
    X = torch.stack(x)

    with torch.no_grad():
        logits, _, _ = model(X)
        p_unc = F.softmax(logits, dim=1).numpy()

    cal = json.load(open(args.calibration))
    T = cal.get("temperature", 1.0)
    p_cal = F.softmax(logits / T, dim=1).numpy()

    y = np.array([ds[i][1].item() for i in range(len(ds))])
    xs_u, ys_u = reliability(p_unc, y)
    xs_c, ys_c = reliability(p_cal, y)

    plt.figure()
    plt.plot([0, 1], [0, 1], '--', label='perfect')
    plt.plot(xs_u, ys_u, 'o-', label='uncal')
    plt.plot(xs_c, ys_c, 'o-', label='cal')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out, dpi=150)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
