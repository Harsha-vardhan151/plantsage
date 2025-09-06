# training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError

class MultiHeadPlantModel(pl.LightningModule):
    def __init__(
        self,
        n_species: int,
        n_issues: int = 0,
        lr: float = 3e-4,
        backbone: str = "tf_efficientnet_b0_ns",
    ):
        super().__init__()
        # keep for checkpoint metadata
        self.save_hyperparameters(ignore=[])

        self.lr = lr
        self.n_species = n_species
        self.n_issues = n_issues

        # backbone (global_pool="avg" makes forward give a feature vector)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        in_feats = self.backbone.num_features

        # heads
        self.species_head  = nn.Linear(in_feats, n_species)
        self.issue_head    = nn.Linear(in_feats, n_issues) if n_issues > 0 else None
        self.severity_head = nn.Linear(in_feats, 1)

        # losses / metrics
        self.loss_species = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.loss_issue   = nn.BCEWithLogitsLoss() if n_issues > 0 else None
        self.loss_sev     = nn.L1Loss()
        self.acc1 = MulticlassAccuracy(num_classes=n_species, top_k=1)
        self.acc5 = MulticlassAccuracy(num_classes=n_species, top_k=5)
        self.mae  = MeanAbsoluteError()

    def forward(self, x):
        f = self.backbone(x)
        s = self.species_head(f)
        i = self.issue_head(f) if self.issue_head is not None else None
        sev = self.severity_head(f)
        return s, i, sev

    def configure_optimizers(self):
        lr = getattr(self, "lr", 3e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def _unpack(self, batch):
        """
        Handles different dataset return formats:
          - (x, y_species)
          - (x, (y_species, y_issues, y_sev))
          - (x, y_species, y_issues, y_sev)
        """
        # Case A: standard (x, y)
        if len(batch) == 2:
            x, y = batch
            if isinstance(y, (tuple, list)):
                # already multiple labels
                if len(y) == 3:
                    y_species, y_issues, y_sev = y
                elif len(y) == 2:
                    y_species, y_issues = y
                    bs = y_species.shape[0]
                    y_sev = torch.zeros(bs, device=y_species.device, dtype=torch.float32)
                else:
                    y_species = y[0]
                    bs = y_species.shape[0]
                    y_issues = None
                    y_sev = torch.zeros(bs, device=y_species.device, dtype=torch.float32)
            else:
                # plain species only
                y_species = y
                bs = y_species.shape[0]
                y_issues = None
                y_sev = torch.zeros(bs, device=y_species.device, dtype=torch.float32)

        # Case B: dataset already returns multiple outputs
        elif len(batch) >= 3:
            x = batch[0]
            y_species = batch[1]
            y_issues = batch[2] if len(batch) > 2 else None
            y_sev = batch[3] if len(batch) > 3 else torch.zeros(
                y_species.shape[0], device=y_species.device, dtype=torch.float32
            )

        else:
            raise ValueError(f"Unexpected batch structure: {batch}")

        return x, y_species, y_issues, y_sev


    def training_step(self, batch, _):
        x, y_species, y_issues, y_sev = self._unpack(batch)
        s, i, sev = self(x)

        loss = self.loss_species(s, y_species)
        if i is not None and y_issues is not None:
            loss = loss + self.loss_issue(i, y_issues.float())
        # severity head is always present; if you don't have labels, it's zeros -> won't dominate
        loss = loss + 0.1 * self.loss_sev(sev.squeeze(-1), y_sev.float())

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc1", self.acc1(s, y_species), prog_bar=True)
        self.log("train/acc5", self.acc5(s, y_species), prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        x, y_species, y_issues, y_sev = self._unpack(batch)
        s, i, sev = self(x)

        loss = self.loss_species(s, y_species)
        if i is not None and y_issues is not None:
            loss = loss + self.loss_issue(i, y_issues.float())
        loss = loss + 0.1 * self.loss_sev(sev.squeeze(-1), y_sev.float())

        self.log("val/loss", loss, prog_bar=True, sync_dist=False)
        self.log("val/acc1", self.acc1(s, y_species), prog_bar=True, sync_dist=False)
        self.log("val/acc5", self.acc5(s, y_species), prog_bar=True, sync_dist=False)
        self.log("val/mae",  self.mae(sev.squeeze(-1), y_sev.float()), prog_bar=False, sync_dist=False)