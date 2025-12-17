"""Training engine for thin-section segmentation models."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from .config import load_config
from .data import get_dataloaders
from .losses import SoftDiceLoss, class_weights_from_mask
from .metrics import batch_metrics
from .models import build_model
from .utils import gpu_report, plot_curves, set_seed


class _NullContext:
    def __enter__(self) -> None:  # noqa: D401
        return None

    def __exit__(self, *args: object) -> bool:
        return False


class Trainer:
    """Wrap training loop, checkpointing, and logging."""

    def __init__(self, cfg: Dict[str, object]):
        self.cfg = cfg
        out_dir = Path(cfg["project"]["outputs_dir"]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
        (out_dir / "logs").mkdir(exist_ok=True, parents=True)
        self.out_dir = out_dir

        set_seed(cfg["project"]["seed"])

        data_bundle = get_dataloaders(cfg, smoketest=False)
        self.ds = data_bundle["dataset"]
        self.train_loader = data_bundle["train_loader"]
        self.val_loader = data_bundle["val_loader"]
        self.test_loader = data_bundle["test_loader"]
        self.num_classes = self.ds.num_classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sample_x, _, _ = self.ds[0]
        in_channels = int(sample_x.shape[0])

        self.model = build_model(
            arch=cfg["model"]["arch"],
            encoder=cfg["model"]["encoder"],
            in_channels=in_channels,
            classes=self.num_classes,
            encoder_weights=cfg["model"]["encoder_weights"],
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"]["weight_decay"],
        )

        self.ce_loss, self.dice_loss = self._build_losses()
        self.ce_w = cfg["optim"]["ce_weight"]
        self.dice_w = cfg["optim"]["dice_weight"]

        if cfg["optim"]["amp"] and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler, autocast

            self.scaler: Optional[torch.cuda.amp.GradScaler] = GradScaler()
            self.autocast = autocast
        else:
            self.scaler = None
            self.autocast = _NullContext()

        self.log_interval = cfg["train"]["log_interval"]
        self.early_pat = cfg["train"]["early_stop_patience"]

    def _build_losses(self) -> tuple[nn.CrossEntropyLoss, SoftDiceLoss]:
        import tifffile as tiff

        label_paths = []
        if self.cfg["data"].get("samples"):
            for sample in self.cfg["data"]["samples"]:
                label_paths.extend(
                    (
                        Path(self.cfg["data"]["root_dir"]) / sample / "label" / p
                        for p in os.listdir(Path(self.cfg["data"]["root_dir"]) / sample / "label")
                    )
                )
        if not label_paths:
            label_paths = list(Path(self.cfg["data"]["root_dir"]).glob("*/label/*.tif*"))

        all_labels = []
        for path in label_paths:
            mask = tiff.imread(str(path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            all_labels.append(mask)

        union = np.max([m.max() for m in all_labels])
        weights = class_weights_from_mask(
            np.concatenate([m.reshape(-1) for m in all_labels]), int(union) + 1
        )
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(self.device))
        dice_loss = SoftDiceLoss()
        return ce_loss, dice_loss

    def _maybe_clip(self) -> None:
        if self.cfg["optim"]["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["optim"]["grad_clip_norm"]
            )

    def train(self) -> Path:
        cfg = self.cfg
        start_time = time.time()
        gpu_info = gpu_report()

        best_val = float("inf")
        bad_epochs = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_miou: list[float] = []
        val_mdice: list[float] = []

        if cfg["sched"]["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=cfg["sched"]["patience"],
                factor=cfg["sched"]["factor"],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg["train"]["max_epochs"]
            )

        for epoch in range(1, cfg["train"]["max_epochs"] + 1):
            epoch_start = time.time()
            self.model.train()
            epoch_loss = 0.0
            batches = 0

            for batch in self.train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device).long()
                self.optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    logits = self.model(x)
                    loss = self.ce_w * self.ce_loss(logits, y) + self.dice_w * self.dice_loss(logits, y)

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self._maybe_clip()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self._maybe_clip()
                    self.optimizer.step()

                epoch_loss += float(loss.item())
                batches += 1

            train_loss = epoch_loss / max(1, batches)
            train_losses.append(train_loss)

            val_loss, miou, mdice = self._validate()
            val_losses.append(val_loss)
            val_miou.append(miou)
            val_mdice.append(mdice)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            epoch_time = time.time() - epoch_start
            peak_mem = (torch.cuda.max_memory_allocated(0) / (1024**3)) if torch.cuda.is_available() else 0.0
            line = (
                f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| mIoU {val_miou[-1]:.4f} | mDice {val_mdice[-1]:.4f} "
                f"| time {epoch_time:.1f}s | peak_mem {peak_mem:.2f} GB"
            )
            print(line, flush=True)
            with open(self.out_dir / "logs" / "train.log", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(
                    {"epoch": epoch, "state_dict": self.model.state_dict()},
                    self.out_dir / "checkpoints" / "best.ckpt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= self.early_pat:
                    print(f"Early stopping at epoch {epoch}; no improvement for {bad_epochs} epochs.")
                    break

        total_time = time.time() - start_time
        plot_curves(self.out_dir, train_losses, val_losses, val_miou, val_mdice)
        meta = {
            "gpu": gpu_info,
            "epochs_trained": len(train_losses),
            "total_train_seconds": total_time,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "final_val_mIoU": val_miou[-1] if val_miou else None,
            "final_val_mDice": val_mdice[-1] if val_mdice else None,
        }
        with open(self.out_dir / "train_summary.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        return self.out_dir

    def _validate(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        ious: list[float] = []
        dices: list[float] = []

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device).long()
                with self.autocast():
                    logits = self.model(x)
                    loss = self.ce_w * self.ce_loss(logits, y) + self.dice_w * self.dice_loss(logits, y)
                total_loss += float(loss.item())
                total_batches += 1
                miou, mdice = batch_metrics(logits, y, self.num_classes)
                ious.append(miou)
                dices.append(mdice)

        val_loss = total_loss / max(1, total_batches)
        return val_loss, float(np.nanmean(ious)), float(np.nanmean(dices))


def main() -> None:
    cfg = load_config("config.yaml")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
