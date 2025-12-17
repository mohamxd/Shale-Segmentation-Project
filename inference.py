from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data import LATE_FUSION_VIEWS, _ensure_hwc3
from src.models import build_model, load_checkpoint_strict
from src.utils import set_seed



def _load_patch_tensor(path: Path) -> torch.Tensor:
    img = tiff.imread(str(path))
    img = _ensure_hwc3(img)
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    return tensor


def _load_patch_rgb01(path: Path) -> np.ndarray:
    img = tiff.imread(str(path))
    img = _ensure_hwc3(img).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img[..., :3], 0.0, 1.0)
    return img


def _discover_separate_groups(patch_dir: Path) -> Dict[str, Dict[str, Path]]:

    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in patch_dir.iterdir():
        if not path.suffix.lower().startswith(".tif"):
            continue
        stem = path.stem
        if stem.endswith("_mask") or stem.endswith("_pred"):
            continue
        for view in LATE_FUSION_VIEWS:
            if stem.endswith(f"_{view}"):
                base = stem[: -(len(view) + 1)]
                groups[base][view] = path
                break
    return groups


def _discover_stack_patches(patch_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in patch_dir.iterdir():
        if not path.suffix.lower().startswith(".tif"):
            continue
        stem = path.stem
        if stem.endswith("_mask") or stem.endswith("_pred"):
            continue
        files.append(path)
    return sorted(files)



def _fixed_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    """
      0 background - black
      1 quartz      - yellow
      2 feldspar    - red
      3 pyrite      - green
      4 clays       - gray

    """
    base = [
        (0, 0, 0),           # 0
        (255, 255, 0),       # 1
        (255, 0, 0),         # 2
        (0, 255, 0),         # 3
        (198, 198, 198),     # 4
    ]
    if num_classes <= len(base):
        return base[:num_classes]
    rng = np.random.default_rng(23)
    extra = rng.integers(60, 220, size=(num_classes - len(base), 3), dtype=np.int32)
    return base + [tuple(map(int, c)) for c in extra]


def _colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask = mask.astype(np.int32)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    pal = _fixed_palette(num_classes)
    for c in range(min(num_classes, len(pal))):
        out[mask == c] = pal[c]
    return out


def _save_color_mask_png(out_png: Path, pred_mask: np.ndarray, num_classes: int) -> None:
    pred_color = _colorize_mask(pred_mask, num_classes)
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(pred_color)
    plt.axis("off")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def _save_two_panel_viz(
    out_png: Path,
    input_rgb01: np.ndarray,
    pred_mask: np.ndarray,
    num_classes: int,
    left_title: str = "Input (PPL)",
    right_title: str = "Prediction",
) -> None:
    pred_color = _colorize_mask(pred_mask, num_classes)

    fig = plt.figure(figsize=(8.8, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.imshow(np.clip(input_rgb01, 0, 1))
    ax0.set_title(left_title, fontsize=12, fontweight="bold")
    ax0.axis("off")

    ax1.imshow(pred_color)
    ax1.set_title(right_title, fontsize=12, fontweight="bold")
    ax1.axis("off")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight inference on stored patches.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]

    in_channels_cfg = model_cfg.get("in_channels")
    num_classes = model_cfg.get("num_classes")
    if num_classes is None:
        raise ValueError("model.num_classes must be set in the config for inference.")

    patch_dir = Path(cfg["inference"]["patches_dir"])
    save_dir = Path(cfg["inference"]["save_dir"])
    if not patch_dir.exists():
        raise FileNotFoundError(patch_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mode = cfg.get("data", {}).get("mode", "stack")
    max_examples = int(cfg["inference"].get("num_examples", 8))

    outputs = []

    selected_bases: List[str] = []
    stack_files: List[Path] = []
    inferred_ch: Optional[int] = None

    if mode == "separate":
        groups = _discover_separate_groups(patch_dir)
        if not groups:
            raise RuntimeError(
                f"No view-specific patches found in {patch_dir} "
                f"(expected *_{LATE_FUSION_VIEWS[0]} / *_XPLxx suffixes)."
            )
        selected_bases = sorted(groups.keys())[:max_examples]
        first_view_path = next(iter(groups[selected_bases[0]].values()))
        inferred_ch = int(_load_patch_tensor(first_view_path).shape[0])
    else:
        stack_files = _discover_stack_patches(patch_dir)
        if not stack_files:
            raise RuntimeError(f"No patches found in {patch_dir}.")
        stack_files = stack_files[:max_examples]
        inferred_ch = int(_load_patch_tensor(stack_files[0]).shape[0])

    in_channels = int(in_channels_cfg or inferred_ch)

    model = build_model(
        arch=model_cfg["arch"],
        encoder=model_cfg["encoder"],
        in_channels=in_channels,
        classes=num_classes,
        encoder_weights=model_cfg.get("encoder_weights", None),
    ).to(device)

    ckpt_path = Path(cfg["inference"]["checkpoint_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    load_checkpoint_strict(
        model,
        str(ckpt_path),
        device,
        arch=model_cfg["arch"],
        encoder=model_cfg["encoder"],
    )
    model.eval()

    with torch.no_grad():
        if mode == "separate":
            groups = _discover_separate_groups(patch_dir)

            for base in selected_bases:
                view_paths = groups[base]

                logits_list = []
                first_path: Optional[Path] = None

                ppl_path = view_paths.get("PPL", None)

                for view in LATE_FUSION_VIEWS:
                    path = view_paths.get(view)
                    if path is None:
                        continue
                    first_path = first_path or path
                    tensor = _load_patch_tensor(path).unsqueeze(0).to(device)
                    logits_list.append(model(tensor))

                if not logits_list:
                    continue


                avg_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
                pred = torch.argmax(avg_logits, dim=1)[0].cpu().numpy().astype(np.uint8)


                color_png = save_dir / f"{base}_pred_color.png"
                viz_png = save_dir / f"{base}_viz.png"
                _save_color_mask_png(color_png, pred, num_classes)

                viz_input_path = ppl_path or first_path
                if viz_input_path is not None:
                    input_rgb = _load_patch_rgb01(viz_input_path)
                    left_title = "Input (PPL Preview)" if viz_input_path == ppl_path else "Input"
                    _save_two_panel_viz(viz_png, input_rgb, pred, num_classes, left_title, "Prediction")

                outputs.append((base, len(logits_list), color_png, viz_png))

        else:
            for path in stack_files:
                tensor = _load_patch_tensor(path).unsqueeze(0).to(device)
                logits = model(tensor)
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)


                color_png = save_dir / f"{path.stem}_pred_color.png"
                viz_png = save_dir / f"{path.stem}_viz.png"
                _save_color_mask_png(color_png, pred, num_classes)

                input_rgb = _load_patch_rgb01(path)
                _save_two_panel_viz(viz_png, input_rgb, pred, num_classes, "Input", "Prediction")

                outputs.append((path.stem, 1, color_png, viz_png))

    summary_lines = [f"Processed {len(outputs)} patch groups on device {device}."]
    for name, nviews, color_png, viz_png in outputs:
        summary_lines.append(f"{name}: views={nviews}, saved {color_png.name} and {viz_png.name}.")
    print("\n".join(summary_lines))



if __name__ == "__main__":
    main()
