from __future__ import annotations
from typing import Any, Dict
import copy
import math
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "ThinSectionSeg",
        "outputs_dir": "./run/plots6",
        "seed": 42,
    },
    "data": {
        "root_dir": "./data",
        "samples": [],
        "mode": "stack",
        "patch_size": 512,
        "patch_stride": 256,
        "normalize_0_1": True,
        "num_workers": 8,
        "batch_size": 32,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "preload_into_ram": True,
        "stack_images_limit": None,
    },
    "split": {
        "mode": "fraction",
        "seed": 42,
        "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sample_holdout": {"train_samples": [], "val_samples": [], "test_samples": []},
        "manifest_path": "",
        "save_manifest": True,
        "output_dir": "outputs/splits",
    },
    "model": {
        "arch": "unetpp",
        "encoder": "timm-efficientnet-b4",
        "encoder_weights": "imagenet",
        "dropout": 0.0,
        "in_channels": 3,
        "num_classes": 5,
    },
    "optim": {
        "lr": 0.0005,
        "weight_decay": 0.0001,
        "ce_weight": 0.5,
        "dice_weight": 0.5,
        "amp": True,
        "grad_clip_norm": 0.0,
    },
    "sched": {
        "type": "ReduceLROnPlateau",
        "patience": 6,
        "factor": 0.5,
    },
    "train": {
        "max_epochs": 150,
        "early_stop_patience": 15,
        "log_interval": 20,
    },
    "eval": {
        "save_confusion_matrix": True,
    },
    "augment": {
        "train": {
            "enabled": True,
            "hflip": 0.5,
            "vflip": 0.2,
            "rotate90": 0.25,
            "rotate": 15,
            "rotate_p": 0.5,
            "shift_scale_rotate": {
                "shift": 0.05,
                "scale": 0.1,
                "rotate": 10,
                "border_mode": 0,
                "p": 0.35,
            },
            "elastic": {"alpha": 25, "sigma": 5, "border_mode": 0, "p": 0.2},
            "color_jitter": {
                "brightness": 0.15,
                "contrast": 0.15,
                "saturation": 0.1,
                "hue": 0.05,
                "p": 0.3,
            },
        }
    },
    "smoketest": {
        "enabled": True,
        "max_train_patches": 64,
        "max_val_patches": 32,
        "num_workers": 0,
        "batches_per_epoch": 5,
    },
    "inference": {
        "checkpoint_path": "models/best.ckpt",
        "patches_dir": "patches",
        "num_examples": 8,
        "save_dir": "outputs/inference",
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    cfg = _deep_update(copy.deepcopy(DEFAULT_CONFIG), user_cfg)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    fractions = cfg.get("split", {}).get("fractions", {})
    total = float(fractions.get("train", 0)) + float(fractions.get("val", 0)) + float(fractions.get("test", 0))
    if cfg.get("split", {}).get("mode") == "fraction" and not math.isclose(total, 1.0, rel_tol=1e-3):
        raise ValueError("Split fractions must sum to 1.0 when split.mode='fraction'.")
