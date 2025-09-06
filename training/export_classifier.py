# training/export_classifier.py
import argparse, json, os
from pathlib import Path
import torch
import torch.nn as nn
import timm

def infer_counts_from_state_dict(sd):
    # species head must exist
    n_species = sd["species_head.weight"].shape[0]
    feat_dim  = sd["species_head.weight"].shape[1]
    if "issues_head.weight" in sd:
        n_issues = sd["issues_head.weight"].shape[0]
    else:
        n_issues = 0
    return n_species, n_issues, feat_dim

class PlainCore(nn.Module):
    """
    Plain nn.Module with the same architecture as the Lightning model.
    No Lightning properties -> safe to TorchScript/ONNX export.
    """
    def __init__(self, backbone_name: str, n_species: int, n_issues: int, feat_dim_hint: int = 1280):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.backbone, "num_features", feat_dim_hint)
        self.species_head = nn.Linear(feat_dim, n_species)
        self.has_issues = n_issues > 0
        if self.has_issues:
            self.issues_head = nn.Linear(feat_dim, n_issues)
            self.severity_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
            )
        else:
            self.issues_head = None
            self.severity_head = None

    def forward(self, x):
        f = self.backbone(x)
        s = self.species_head(f)
        i = self.issues_head(f) if self.has_issues else None
        sev = self.severity_head(f).squeeze(1) if self.has_issues else None
        return s, i, sev

class PlainWrapper(nn.Module):
    """Guarantee tensor-only outputs."""
    def __init__(self, core: PlainCore, n_issues: int):
        super().__init__()
        self.core = core
        self.n_issues = int(n_issues)

    def forward(self, x):
        s, i, sev = self.core(x)
        if i is None:
            i = torch.zeros((x.shape[0], max(1, self.n_issues)), dtype=s.dtype, device=s.device)
        if sev is None:
            sev = torch.zeros((x.shape[0],), dtype=s.dtype, device=s.device)
        return s, i, sev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="server/models/classifier_last.ckpt")
    ap.add_argument("--outdir", type=str, default="server/models")
    ap.add_argument("--labelmaps", type=str, default="training/labelmaps.json")  # optional, not required
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ----
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["state_dict"]
    hparams = state.get("hyper_parameters", {})
    backbone_name = hparams.get("backbone", "efficientnet_b0")

    n_species, n_issues, feat_dim = infer_counts_from_state_dict(sd)
    print(f"[INFO] Export with n_species={n_species} n_issues={n_issues} backbone={backbone_name}")

    # ---- Build plain module and load weights (ignore lightning keys) ----
    core = PlainCore(backbone_name, n_species, n_issues, feat_dim_hint=feat_dim)
    # Filter to only the submodules we have
    filtered = {k: v for k, v in sd.items()
                if k.startswith(("backbone.", "species_head.", "issues_head.", "severity_head."))}
    missing, unexpected = core.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys ignored: {unexpected}")
    core.eval()

    wrapper = PlainWrapper(core, n_issues)
    dummy = torch.randn(1, 3, 224, 224)

    # ---- TorchScript ----
    ts_path = outdir / "classifier.ts.pt"
    ts = torch.jit.trace(wrapper, dummy)
    ts.save(str(ts_path))
    print(f"[INFO] Saved TorchScript: {ts_path}")

    # ---- ONNX ----
    onnx_path = outdir / "classifier.onnx"
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["species_logits", "issues_logits", "severity"],
        dynamic_axes={
            "input": {0: "batch"},
            "species_logits": {0: "batch"},
            "issues_logits": {0: "batch"},
            "severity": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"[INFO] Saved ONNX: {onnx_path}")

if __name__ == "__main__":
    main()
# training/export_classifier.py
import argparse, json, os
from pathlib import Path
import torch
import torch.nn as nn
import timm

def infer_counts_from_state_dict(sd):
    # species head must exist
    n_species = sd["species_head.weight"].shape[0]
    feat_dim  = sd["species_head.weight"].shape[1]
    if "issues_head.weight" in sd:
        n_issues = sd["issues_head.weight"].shape[0]
    else:
        n_issues = 0
    return n_species, n_issues, feat_dim

class PlainCore(nn.Module):
    """
    Plain nn.Module with the same architecture as the Lightning model.
    No Lightning properties -> safe to TorchScript/ONNX export.
    """
    def __init__(self, backbone_name: str, n_species: int, n_issues: int, feat_dim_hint: int = 1280):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.backbone, "num_features", feat_dim_hint)
        self.species_head = nn.Linear(feat_dim, n_species)
        self.has_issues = n_issues > 0
        if self.has_issues:
            self.issues_head = nn.Linear(feat_dim, n_issues)
            self.severity_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
            )
        else:
            self.issues_head = None
            self.severity_head = None

    def forward(self, x):
        f = self.backbone(x)
        s = self.species_head(f)
        i = self.issues_head(f) if self.has_issues else None
        sev = self.severity_head(f).squeeze(1) if self.has_issues else None
        return s, i, sev

class PlainWrapper(nn.Module):
    """Guarantee tensor-only outputs."""
    def __init__(self, core: PlainCore, n_issues: int):
        super().__init__()
        self.core = core
        self.n_issues = int(n_issues)

    def forward(self, x):
        s, i, sev = self.core(x)
        if i is None:
            i = torch.zeros((x.shape[0], max(1, self.n_issues)), dtype=s.dtype, device=s.device)
        if sev is None:
            sev = torch.zeros((x.shape[0],), dtype=s.dtype, device=s.device)
        return s, i, sev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="server/models/classifier_last.ckpt")
    ap.add_argument("--outdir", type=str, default="server/models")
    ap.add_argument("--labelmaps", type=str, default="training/labelmaps.json")  # optional, not required
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ----
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["state_dict"]
    hparams = state.get("hyper_parameters", {})
    backbone_name = hparams.get("backbone", "efficientnet_b0")

    n_species, n_issues, feat_dim = infer_counts_from_state_dict(sd)
    print(f"[INFO] Export with n_species={n_species} n_issues={n_issues} backbone={backbone_name}")

    # ---- Build plain module and load weights (ignore lightning keys) ----
    core = PlainCore(backbone_name, n_species, n_issues, feat_dim_hint=feat_dim)
    # Filter to only the submodules we have
    filtered = {k: v for k, v in sd.items()
                if k.startswith(("backbone.", "species_head.", "issues_head.", "severity_head."))}
    missing, unexpected = core.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys ignored: {unexpected}")
    core.eval()

    wrapper = PlainWrapper(core, n_issues)
    dummy = torch.randn(1, 3, 224, 224)

    # ---- TorchScript ----
    ts_path = outdir / "classifier.ts.pt"
    ts = torch.jit.trace(wrapper, dummy)
    ts.save(str(ts_path))
    print(f"[INFO] Saved TorchScript: {ts_path}")

    # ---- ONNX ----
    onnx_path = outdir / "classifier.onnx"
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["species_logits", "issues_logits", "severity"],
        dynamic_axes={
            "input": {0: "batch"},
            "species_logits": {0: "batch"},
            "issues_logits": {0: "batch"},
            "severity": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"[INFO] Saved ONNX: {onnx_path}")

if __name__ == "__main__":
    main()
