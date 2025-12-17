"""Utility helpers for augmentation, seeding, plotting, and device reporting."""

from __future__ import annotations

import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def build_augmentation_pipeline(cfg: dict, num_channels: Optional[int] = None):
    """Create an Albumentations augmentation pipeline from configuration."""
    if not cfg or not cfg.get("enabled", True):
        return None

    import albumentations as A

    transforms = []

    if cfg.get("hflip", 0) > 0:
        transforms.append(A.HorizontalFlip(p=cfg.get("hflip", 0)))
    if cfg.get("vflip", 0) > 0:
        transforms.append(A.VerticalFlip(p=cfg.get("vflip", 0)))
    if cfg.get("rotate90", 0) > 0:
        transforms.append(A.RandomRotate90(p=cfg.get("rotate90", 0)))

    rot_lim = cfg.get("rotate", 0)
    if rot_lim and rot_lim > 0:
        transforms.append(
            A.Rotate(limit=rot_lim, border_mode=cfg.get("border_mode", 0), p=cfg.get("rotate_p", 0.5))
        )

    if cfg.get("shift_scale_rotate", False):
        ssr = cfg["shift_scale_rotate"]
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=ssr.get("shift", 0.05),
                scale_limit=ssr.get("scale", 0.1),
                rotate_limit=ssr.get("rotate", 10),
                border_mode=ssr.get("border_mode", 0),
                p=ssr.get("p", 0.5),
            )
        )

    if cfg.get("elastic", False):
        el = cfg["elastic"]
        transforms.append(
            A.ElasticTransform(
                alpha=el.get("alpha", 40),
                sigma=el.get("sigma", 6),
                alpha_affine=el.get("alpha_affine", 20),
                interpolation=el.get("interpolation", 1),
                border_mode=el.get("border_mode", 0),
                p=el.get("p", 0.2),
            )
        )

    if cfg.get("color_jitter", False) and (num_channels is None or num_channels in (1, 3)):
        cj = cfg["color_jitter"]
        transforms.append(
            A.ColorJitter(
                brightness=cj.get("brightness", 0.1),
                contrast=cj.get("contrast", 0.1),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.1),
                p=cj.get("p", 0.3),
            )
        )

    if cfg.get("gaussian_noise", False):
        gn = cfg["gaussian_noise"]
        transforms.append(A.GaussNoise(var_limit=gn.get("var_limit", (1e-5, 1e-3)), p=gn.get("p", 0.15)))

    if cfg.get("blur", False):
        bl = cfg["blur"]
        transforms.append(A.GaussianBlur(blur_limit=bl.get("limit", (3, 5)), p=bl.get("p", 0.1)))

    if cfg.get("grid_distortion", False):
        gd = cfg["grid_distortion"]
        transforms.append(
            A.GridDistortion(
                num_steps=gd.get("steps", 5),
                distort_limit=gd.get("limit", 0.3),
                border_mode=gd.get("border_mode", 0),
                p=gd.get("p", 0.2),
            )
        )

    if not transforms:
        return None

    return A.Compose(transforms)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_curves(save_dir, train_losses, val_losses, val_miou, val_mdice):
    _apply_plot_style()

    epochs = np.arange(1, len(train_losses) + 1, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train", color="#1f77b4", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation", color="#d62728", linewidth=2)
    if len(val_losses):
        best_idx = int(np.argmin(val_losses))
        ax.scatter(
            epochs[best_idx],
            val_losses[best_idx],
            color="#d62728",
            marker="o",
            s=55,
            zorder=5,
            label="Best val",
        )
        ax.axvline(epochs[best_idx], color="#d62728", linestyle="--", alpha=0.25)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs. Validation Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, val_miou, label="mIoU", color="#2ca02c", linewidth=2)
    ax.plot(epochs, val_mdice, label="mDice", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Validation Metrics")
    ax.legend()

    fig.suptitle("Thin-Section Segmentation Training Dynamics", fontsize=16)
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, "training_curves.pdf"), dpi=300)
    plt.close(fig)

    legacy_dir = os.path.join(save_dir, "legacy_plots")
    os.makedirs(legacy_dir, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(legacy_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(val_miou, label="mIoU")
    plt.plot(val_mdice, label="mDice")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(legacy_dir, "val_metrics.png"), dpi=200)
    plt.close()


def gpu_report():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"device": device}
    if device == "cuda":
        rep["name"] = torch.cuda.get_device_name(0)
        rep["capability"] = torch.cuda.get_device_capability(0)
        rep["driver"] = torch.version.cuda
    return rep
