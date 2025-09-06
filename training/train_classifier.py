# training/train_classifier.py  (verbose checks)
import argparse, os
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from training.dataset import PlantSageDataset
from training.model import MultiHeadPlantModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/synth_cls")
    ap.add_argument("--labelmaps", type=str, default="training/labelmaps.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out", type=str, default="server/models/classifier_last.ckpt")
    args = ap.parse_args()

    print("[INFO] train_classifier entrypoint")
    train_csv = Path(args.data_dir) / "train.csv"
    val_csv   = Path(args.data_dir) / "val.csv"
    print(f"[INFO] data_dir={args.data_dir}")
    print(f"[INFO] train_csv={train_csv} exists={train_csv.exists()}")
    print(f"[INFO] val_csv  ={val_csv} exists={val_csv.exists()}")
    print(f"[INFO] labelmaps={args.labelmaps} exists={Path(args.labelmaps).exists()}")

    if not train_csv.exists() or not val_csv.exists():
        raise SystemExit("[ERROR] train/val CSV not found. Re-create CSVs before training.")

    train_ds = PlantSageDataset(str(train_csv), args.labelmaps, args.img_size, augment=True)
    val_ds   = PlantSageDataset(str(val_csv),   args.labelmaps, args.img_size, augment=False)
    print(f"[INFO] train_samples={len(train_ds)}  val_samples={len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[ERROR] Empty dataset (0 rows). Check your CSV paths and labelmaps.")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiHeadPlantModel(n_species=len(train_ds.species), n_issues=len(train_ds.issues), lr=args.lr)
    print(f"[INFO] n_species={len(train_ds.species)}  n_issues={len(train_ds.issues)}  img_size={args.img_size}")

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices=1, log_every_n_steps=1)
    print("[INFO] starting training ...")
    trainer.fit(model, train_dl, val_dl)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.out)
    print(f"[INFO] Saved checkpoint to {args.out}")

if __name__ == "__main__":
    main()
