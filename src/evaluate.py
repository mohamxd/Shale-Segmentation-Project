import argparse, json, os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .config import load_config
from .data import ThinSectionDataset, base_patch_id
from .models import build_model, load_checkpoint_strict


# ---------------------- plotting style ----------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "DejaVu Sans",
})


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_rgb_preview(x: torch.Tensor) -> np.ndarray:
    """x: (C,H,W) -> (H,W,3) float in [0,1] with per-channel min-max."""
    x = x.detach().cpu().float()
    C, H, W = x.shape
    img = x[:3].clone() if C >= 3 else x[:1].repeat(3, 1, 1)
    for c in range(3):
        v = img[c]
        vmin, vmax = float(v.min()), float(v.max())
        img[c] = (v - vmin) / (vmax - vmin) if vmax > vmin else torch.zeros_like(v)
    return img.permute(1, 2, 0).numpy()

def _fixed_palette(num_classes: int):

    base = [
        (0, 0, 0),           # 0
        (255, 255, 0),       # 1
        (255, 0, 0),         # 2
        (0, 255, 0),         # 3
        (198, 198, 198),     # 4
    ]
    if num_classes <= 5:
        return base[:num_classes]
    # deterministically extend
    rng = np.random.default_rng(23)
    extra = rng.integers(60, 220, size=(num_classes - 5, 3))  # mid tones
    return base + [tuple(map(int, c)) for c in extra]

def _colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """mask (H,W) -> RGB uint8 using fixed palette."""
    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    pal = _fixed_palette(num_classes)
    for c in range(min(num_classes, len(pal))):
        out[mask == c] = pal[c]
    return out

def _legend_handles(class_names, num_classes):
    pal = _fixed_palette(num_classes)
    handles = []
    for i, name in enumerate(class_names):
        r, g, b = pal[i]
        handles.append(Patch(facecolor=(r/255, g/255, b/255), edgecolor="black", label=name))
    return handles

def _percent_confusion(cm_counts: np.ndarray):
    """Row-normalized confusion matrix in percent (0..100)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sum = cm_counts.sum(axis=1, keepdims=True)
        cm_row = cm_counts.astype(float) / np.clip(row_sum, 1e-9, None)
    return cm_row * 100.0

def _plot_confusion_percent(cm_counts: np.ndarray, class_names, out_dir: Path):
    cm_pct = _percent_confusion(cm_counts)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = plt.cm.get_cmap("viridis")
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("True", fontsize=13, fontweight="bold")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(class_names, fontsize=11)
    ax.tick_params(axis="both", length=0)

    for i in range(n):
        for j in range(n):
            val = cm_pct[i, j]
            if np.isnan(val):
                continue
            text = f"{val:.1f}%"
            r, g, b, _ = cmap(val / 100.0)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "black" if luminance > 0.6 else "white"
            ax.text(j, i, text, ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Percent", rotation=-90, va="bottom", fontsize=12, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    png = out_dir / "confusion_matrix_percent.png"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

def _perclass_bars_percent(metrics_dict, class_names, out_dir: Path):
    """
    Plot per-class metrics as horizontal bars in percent.
    metrics_dict keys expected: {"IoU","Dice","Precision","Recall","Support"}.
    Support stays in raw counts; metrics are shown in %.
    """
    if "Support" in metrics_dict:
        vals = np.asarray(metrics_dict["Support"], dtype=float)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, vals, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Pixels")
        ax.set_title("Per-class Support (pixels)")
        for y, v in enumerate(vals):
            ax.text(v, y, f" {int(v):,}", va="center", ha="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_support.png")
        plt.close(fig)

    for key in ["IoU", "Dice", "Precision", "Recall"]:
        if key not in metrics_dict:
            continue
        y = np.array(metrics_dict[key], dtype=float)
        y = np.nan_to_num(y, nan=0.0) * 100.0
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, y, edgecolor="black")
        ax.set_xlim(0, 100)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Percent")
        ax.set_title(f"Per-class {key} (%)")
        for yp, val in enumerate(y):
            ax.text(val, yp, f" {val:.1f}%", va="center", ha="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_{key.lower()}_percent.png")
        plt.close(fig)

def _evenly_spaced_indices(N: int, k: int) -> np.ndarray:
    k = max(0, min(k, N))
    if k == 0:
        return np.array([], dtype=int)
    return np.unique(np.round(np.linspace(0, N - 1, k)).astype(int))

def _overlay_mask_on_rgb(rgb01: np.ndarray, mask: np.ndarray, num_classes: int, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend colorized mask onto rgb01.
    rgb01: (H,W,3) float [0,1]; mask: (H,W) int; returns uint8 RGB."""
    H, W, _ = rgb01.shape
    base = (np.clip(rgb01, 0, 1) * 255.0).astype(np.uint8)
    color = _colorize_mask(mask.astype(np.int32), num_classes).astype(np.uint8)
    out = base.copy()
    blended = (alpha * color + (1.0 - alpha) * base).astype(np.uint8)
    out = blended
    return out

def _canonical_patch_id(group_key: str) -> str:

    parts = group_key.split(":")
    if len(parts) != 4:
        return group_key
    sample, _, y, x = parts
    return f"{sample}:{y}:{x}"

def _percent_by_class(mask: np.ndarray, num_classes: int):
    """Return array of percentages (0..100) per class for given mask."""
    total = mask.size
    if total == 0:
        return np.zeros(num_classes, dtype=float)
    counts = np.bincount(mask.ravel().astype(np.int64), minlength=num_classes)
    return (counts / total) * 100.0

def _class_names_for(num_classes: int):
    """Enforce names 0..4, then generic Ck beyond."""
    base = ["background", "quartz", "feldspar", "pyrite", "clays"]
    if num_classes <= 5:
        return [f"{i} {base[i]}" for i in range(num_classes)]
    rest = [f"{i} C{i}" for i in range(5, num_classes)]
    named = [f"{i} {base[i]}" for i in range(5)]
    return named + rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_vis", type=int, default=12, help="How many patches to visualize (ignored if --idx set)")
    parser.add_argument("--idx", type=int, default=None, help="Visualize ONLY this dataset index (overrides --num_vis)")
    parser.add_argument("--vis_from", type=str, default=None, help="Path to vis_patches.json from another run to reuse same patch positions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = ThinSectionDataset(
        root_dir=cfg["data"]["root_dir"],
        samples=cfg["data"].get("samples"),
        mode=cfg["data"]["mode"],
        patch_size=cfg["data"]["patch_size"],
        patch_stride=cfg["data"]["patch_stride"],
        normalize_0_1=cfg["data"]["normalize_0_1"],
        preload_into_ram=cfg["data"]["preload_into_ram"],
    )
    
    patch_id_to_indices = {}
    for i, rec in enumerate(ds.index):
        pid = base_patch_id(rec)
        patch_id_to_indices.setdefault(pid, []).append(i)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    sample_x, _, _ = ds[0]
    model_cfg = cfg["model"]
    if sample_x.dim() == 3:
        inferred_ch = int(sample_x.shape[0])
    elif sample_x.dim() == 4:
        inferred_ch = int(sample_x.shape[1])
    else:
        raise ValueError(f"Unexpected sample shape from dataset: {tuple(sample_x.shape)}")
    in_ch = int(model_cfg.get("in_channels") or inferred_ch)
    num_classes = int(model_cfg.get("num_classes") or ds.num_classes)
    if ds.num_classes != num_classes:
        print(f"[WARN] Using config num_classes={num_classes} (dataset reports {ds.num_classes}).")

    model = build_model(
        model_cfg["arch"],
        model_cfg["encoder"],
        in_ch,
        num_classes,
        model_cfg["encoder_weights"],
    ).to(device)

    load_checkpoint_strict(
        model,
        args.checkpoint,
        device,
        arch=model_cfg.get("arch"),
        encoder=model_cfg.get("encoder"),
    )
    model.eval()

    base_out = Path(cfg["project"]["outputs_dir"])
    _ensure_dir(base_out)
    out_dir = _ensure_dir(base_out / f"eval_{datetime.now():%Y%m%d_%H%M%S}")
    vis_dir   = _ensure_dir(out_dir / "visualizations")

    with open(out_dir / "config_used.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(out_dir / "checkpoint_path.txt", "w") as fh:
        fh.write(str(Path(args.checkpoint).resolve()))

    loader = None
    if cfg["data"]["mode"] != "separate":
        loader = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"].get("pin_memory", False),
            persistent_workers=cfg["data"].get("persistent_workers", False),
        )

    labels = list(range(num_classes))
    cm_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.inference_mode():
        if cfg["data"]["mode"] == "separate":
            for pid in sorted(patch_id_to_indices.keys()):
                idxs = sorted(patch_id_to_indices[pid], key=lambda i: ds.index[i].angle_idx or 0)
                logits_list = []
                gt = None
                for idx in idxs:
                    x, y, _ = ds[idx]
                    if gt is None:
                        gt = y
                    logits_list.append(model(x.unsqueeze(0).to(device)))
                if not logits_list or gt is None:
                    continue
                avg_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
                p = torch.argmax(avg_logits, dim=1).cpu().numpy().reshape(-1)
                g = gt.cpu().numpy().reshape(-1)
                if g.size and p.size:
                    cm_counts += confusion_matrix(g, p, labels=labels)
        else:
            for batch in loader:
                x, y, _ = batch
                x = x.to(device); y = y.to(device).long()
                logits = model(x)
                p = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().reshape(-1)
                g = y.cpu().numpy().reshape(-1)
                if g.size and p.size:
                    cm_counts += confusion_matrix(g, p, labels=labels)

    per_iou, per_dice, per_prec, per_rec, per_support = [], [], [], [], []
    for c in labels:
        tp = int(cm_counts[c, c])
        fn = int(cm_counts[c, :].sum() - tp)
        fp = int(cm_counts[:, c].sum() - tp)
        union = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        prec_den = tp + fp
        rec_den  = tp + fn
        iou  = tp / union if union > 0 else float("nan")
        dice = (2 * tp) / denom_dice if denom_dice > 0 else float("nan")
        prec = tp / prec_den if prec_den > 0 else float("nan")
        rec  = tp / rec_den  if rec_den  > 0 else float("nan")
        sup  = tp + fn
        per_iou.append(iou); per_dice.append(dice)
        per_prec.append(prec); per_rec.append(rec)
        per_support.append(int(sup))

    mIoU  = float(np.nanmean(per_iou)) if per_iou else float("nan")
    mDice = float(np.nanmean(per_dice)) if per_dice else float("nan")

    class_names_cfg = cfg.get("data", {}).get("classes", None)
    if class_names_cfg and len(class_names_cfg) == num_classes:
        class_names = class_names_cfg
    else:
        class_names = _class_names_for(num_classes)

    summary = {
        "mIoU": mIoU,
        "mDice": mDice,
        "per_class_IuO": [float(x) if not np.isnan(x) else None for x in per_iou],
        "per_class_Dice": [float(x) if not np.isnan(x) else None for x in per_dice],
        "per_class_Precision": [float(x) if not np.isnan(x) else None for x in per_prec],
        "per_class_Recall": [float(x) if not np.isnan(x) else None for x in per_rec],
        "per_class_Support": per_support,
        "class_names": class_names,
        "confusion_matrix_counts": cm_counts.tolist(),
    }
    with open(out_dir / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[INFO] Eval summary saved: {out_dir/'eval_summary.json'}")

    _plot_confusion_percent(cm_counts, class_names, out_dir)
    _perclass_bars_percent(
        {"IoU": per_iou, "Dice": per_dice, "Precision": per_prec, "Recall": per_rec, "Support": per_support},
        class_names, out_dir
    )


    group_ids = sorted(patch_id_to_indices.keys())
    N = len(group_ids)
    if N == 0:
        print("[WARN] Dataset is empty — skipping visualizations.")
        return

    if args.vis_from is not None:
        from pathlib import Path as _Path

        vis_path = _Path(args.vis_from)
        if not vis_path.exists():
            print(f"[WARN] --vis_from path does not exist: {vis_path}. Falling back to automatic selection.")
            idxs = _evenly_spaced_indices(N, args.num_vis)
            selected_ids = [group_ids[i] for i in idxs]
        else:
            with open(vis_path, "r") as fh:
                prev = json.load(fh)

            if isinstance(prev, dict) and "patches" in prev:
                records = prev["patches"]
            else:
                records = prev

            wanted_pids = []
            for rec in records:
                if isinstance(rec, dict) and "patch_id" in rec:
                    wanted_pids.append(rec["patch_id"])

            selected_ids = [pid for pid in wanted_pids if pid in patch_id_to_indices]
            print(f"[INFO] Reusing {len(selected_ids)} patch positions from {vis_path} -> {vis_dir}")
    elif args.idx is not None and 0 <= args.idx < N:
        selected_ids = [group_ids[args.idx]]
        print(f"[INFO] Visualizing single idx={args.idx} -> {vis_dir}")
    elif args.idx is not None:
        print(f"[WARN] --idx {args.idx} out of range 0..{N-1}, skipping visualizations.")
        selected_ids = []
    else:
        idxs = _evenly_spaced_indices(N, args.num_vis)
        selected_ids = [group_ids[i] for i in idxs]
        print(f"[INFO] Visualizing {len(selected_ids)} patches -> {vis_dir}")

    vis_records = []

    with torch.inference_mode():
        for vis_idx, pid in enumerate(selected_ids):
            idxs = sorted(patch_id_to_indices[pid], key=lambda i: ds.index[i].angle_idx or 0)
            logits_list = []
            x_ref = None
            y_ref = None
            for idx in idxs:
                x, y, _ = ds[idx]
                if x_ref is None:
                    x_ref = x
                    y_ref = y
                logits_list.append(model(x.unsqueeze(0).to(device)))

            if not logits_list or x_ref is None or y_ref is None:
                continue

            avg_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
            pred = torch.argmax(avg_logits, dim=1)[0].detach().cpu().numpy().astype(np.int32)

            vis_records.append({
                "dataset_index": int(idxs[0]),
                "patch_id": pid,
            })

            rgb = _to_rgb_preview(x_ref)
            gt_np = y_ref.numpy().astype(np.int32)
            gt_color = _colorize_mask(gt_np, num_classes)
            pr_color = _colorize_mask(pred, num_classes)

            gt_pct = _percent_by_class(gt_np, num_classes)
            pr_pct = _percent_by_class(pred, num_classes)

            fig = plt.figure(figsize=(14.5, 3.8))
            gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.9], wspace=0.15)

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[0, 3])

            ax0.imshow((np.clip(rgb, 0, 1)))
            ax0.set_title("Input")
            ax0.axis("off")

            ax1.imshow(gt_color)
            ax1.set_title("Ground Truth")
            ax1.axis("off")

            ax2.imshow(pr_color)
            ax2.set_title("Prediction")
            ax2.axis("off")


            fig.savefig(vis_dir / f"triptych_{vis_idx:06d}.png")
            plt.close(fig)
    with open(out_dir / "vis_patches.json", "w") as fh:
        json.dump({"patches": vis_records}, fh, indent=2)

    print(f"[INFO] vis_patches.json saved with {len(vis_records)} entries.")
    print(f"[INFO] All artifacts saved under: {out_dir}")



if __name__ == "__main__":
    main()
