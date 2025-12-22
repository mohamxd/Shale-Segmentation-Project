from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .splits import SplitConfig, SplitManager
from .utils import build_augmentation_pipeline

LATE_FUSION_VIEWS = ["PPL", "XPL40", "XPL50", "XPL60", "XPL70"]


def sliding_window_coords(height: int, width: int, patch: int, stride: int) -> List[Tuple[int, int, int, int]]:
    coords: List[Tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        y0 = min(y, max(0, height - patch))
        x = 0
        while x < width:
            x0 = min(x, max(0, width - patch))
            coords.append((y0, x0, patch, patch))
            if x + patch >= width:
                break
            x += stride
        if y + patch >= height:
            break
        y += stride
    uniq: List[Tuple[int, int, int, int]] = []
    seen = set()
    for coord in coords:
        if coord not in seen:
            uniq.append(coord)
            seen.add(coord)
    return uniq


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def ceil_to_stride(v: int, stride: int) -> int:
    return int(math.ceil(v / stride)) * stride if stride else v


def floor_to_stride(v: int, stride: int) -> int:
    return (v // stride) * stride if stride else v


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def make_edges(n: int, size: int) -> List[int]:
    return [round(i * size / n) for i in range(n + 1)]


def region_bounds(edges_y: List[int], edges_x: List[int], r: int, c: int) -> Tuple[int, int, int, int]:
    return edges_y[r], edges_y[r + 1], edges_x[c], edges_x[c + 1]


def nearest_eff_size(size: int, stride: int, parts: int, mode: str) -> int:
    step = (stride * parts) // gcd(stride, parts)
    if mode == "pad":
        s0 = ceil_to_stride(size, stride)
        return ceil_to_stride(s0, step)
    if mode == "crop":
        s0 = floor_to_stride(size, stride)
        candidate = floor_to_stride(s0, step)
        return max(candidate, stride)
    raise ValueError(f"Unsupported eff_mode for nearest_eff_size: {mode}")


def pick_best_grid_plan(
    h: int,
    w: int,
    patch: int,
    stride: int,
    eff_mode: str,
    max_r: int,
    max_c: int,
    max_regions: int,
    min_val_test_each: int,
    min_total_regions: int,
    max_total_regions: int,
) -> Tuple[int, int, int, int, str]:
    min_regions_needed = max(1 + 2 * min_val_test_each, min_total_regions)
    best: Tuple[int, int, int, int, str] | None = None
    best_score: Tuple[int, int] | None = None
    for rr in range(1, max_r + 1):
        for cc in range(1, max_c + 1):
            n_regions = rr * cc
            if n_regions > max_regions or n_regions < min_regions_needed or n_regions > max_total_regions:
                continue
            modes = [eff_mode] if eff_mode in ("pad", "crop") else ["pad", "crop"]
            for mode in modes:
                eff_h = nearest_eff_size(h, stride, rr, mode)
                eff_w = nearest_eff_size(w, stride, cc, mode)
                if eff_h < patch or eff_w < patch:
                    continue
                size_change = abs(eff_h - h) + abs(eff_w - w)
                candidate = (rr, cc, eff_h, eff_w, mode if eff_mode == "both" else eff_mode)
                current_score = (n_regions, size_change)
                if best is None or (best_score is not None and current_score < best_score):
                    best = candidate
                    best_score = current_score
    if best is None:
        rr, cc = 4, 5
        eff_h = nearest_eff_size(h, stride, rr, "pad")
        eff_w = nearest_eff_size(w, stride, cc, "pad")
        return rr, cc, eff_h, eff_w, "pad" if eff_mode != "crop" else "crop"
    return best


def iter_patch_coords_for_region_split(
    h: int,
    w: int,
    rr: int,
    cc: int,
    patch: int,
    stride: int,
    strict: bool,
) -> Tuple[int, int, int, int]:
    y_edges = make_edges(rr, h)
    x_edges = make_edges(cc, w)
    half = patch // 2
    max_y0 = h - patch
    max_x0 = w - patch
    for r in range(rr):
        for c in range(cc):
            y0, y1, x0, x1 = region_bounds(y_edges, x_edges, r, c)
            ey0 = clamp(y0 - half, 0, max_y0)
            ey1 = clamp(y1 - half, 0, max_y0)
            ex0 = clamp(x0 - half, 0, max_x0)
            ex1 = clamp(x1 - half, 0, max_x0)
            gy0 = ceil_to_stride(ey0, stride)
            gy1 = floor_to_stride(ey1, stride)
            gx0 = ceil_to_stride(ex0, stride)
            gx1 = floor_to_stride(ex1, stride)
            if gy0 > gy1 or gx0 > gx1:
                continue
            for ty in range(gy0, gy1 + 1, stride):
                for tx in range(gx0, gx1 + 1, stride):
                    cy = ty + patch / 2.0
                    cx = tx + patch / 2.0
                    if not (y0 <= cy < y1 and x0 <= cx < x1):
                        continue
                    if strict:
                        if not (ty >= y0 and ty + patch <= y1 and tx >= x0 and tx + patch <= x1):
                            continue
                    yield (r, c, ty, tx)


def apply_eff_mode(img: np.ndarray, eff_h: int, eff_w: int, mode: str, pad_value: int | float = 0) -> np.ndarray:
    h, w = img.shape[:2]
    target_h, target_w = eff_h or h, eff_w or w
    if mode == "both":
        mode = "pad" if (h < target_h or w < target_w) else "crop"
    if mode == "pad":
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_config = ((0, pad_h), (0, pad_w))
        if img.ndim == 3:
            pad_config = pad_config + ((0, 0),)
        padded = np.pad(img, pad_config, constant_values=pad_value)
        return padded[:target_h, :target_w, ...] if img.ndim == 3 else padded[:target_h, :target_w]
    if mode == "crop":
        return img[:target_h, :target_w, ...] if img.ndim == 3 else img[:target_h, :target_w]
    raise ValueError(f"Unsupported eff_mode for apply_eff_mode: {mode}")


def _ensure_hwc3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3]
    return img


def canonical_patch_key(rec: "PatchRecord") -> str:
    angle = "stack" if rec.angle_idx is None else f"a{rec.angle_idx}"
    return f"{rec.sample_id}:{angle}:{rec.y0}:{rec.x0}"


def base_patch_id(rec: "PatchRecord") -> str:
    return f"{rec.sample_id}:{rec.y0}:{rec.x0}"


def view_suffix_from_path(path: Path, angle_idx: Optional[int]) -> str:
    stem = path.stem.lower()
    for view in LATE_FUSION_VIEWS:
        if view.lower() in stem:
            return view
    if angle_idx is not None and 0 <= angle_idx < len(LATE_FUSION_VIEWS):
        return LATE_FUSION_VIEWS[angle_idx]
    return f"VIEW{angle_idx if angle_idx is not None else 0}"


def discover_sample(sample_dir: Path) -> Dict[str, Any]:
    img_dir = sample_dir / "image"
    lbl_dir = sample_dir / "label"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing image/ or label/ folder under {sample_dir}")
    images = sorted(img_dir.glob("*.tif*"))
    if not images:
        raise FileNotFoundError(f"No tif images in {img_dir}")
    labels = sorted(lbl_dir.glob("*.tif*"))
    if len(labels) != 1:
        raise RuntimeError(f"Expected 1 label file under {lbl_dir}, found {len(labels)}")
    return {"images": images, "label": labels[0], "name": sample_dir.name}


@dataclass
class PatchRecord:
    sample_id: str
    image_paths: List[Path]
    label_path: Path
    y0: int
    x0: int
    h: int
    w: int
    angle_idx: Optional[int]
    patch_id: str
    region_id: str
    eff_h: Optional[int] = None
    eff_w: Optional[int] = None
    eff_mode: Optional[str] = None
    grid_rr: Optional[int] = None
    grid_cc: Optional[int] = None


class ThinSectionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        samples: Optional[List[str]] = None,
        mode: str = "stack",
        patch_size: int = 512,
        patch_stride: int = 256,
        normalize_0_1: bool = True,
        preload_into_ram: bool = False,
        stack_images_limit: Optional[int] = None,
        index_data: Optional[List[Dict[str, Any]]] = None,
        num_classes: Optional[int] = None,
        split_cfg: Optional[SplitConfig] = None,
    ):
        self.root = Path(root_dir)
        self.samples = samples
        self.mode = mode
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.normalize_0_1 = normalize_0_1
        self.preload = preload_into_ram
        self.stack_images_limit = stack_images_limit
        self.split_cfg = split_cfg

        if index_data is None:
            if split_cfg is None:
                raise ValueError("split_cfg is required when building dataset index.")
            index_data, computed_classes = build_dataset_index(
                root_dir=root_dir,
                samples=samples,
                mode=mode,
                patch_size=patch_size,
                patch_stride=patch_stride,
                stack_images_limit=stack_images_limit,
                split_cfg=split_cfg,
            )
            if num_classes is None:
                num_classes = computed_classes

        self.index: List[PatchRecord] = [PatchRecord(**record) for record in index_data]
        self.num_classes: int = int(num_classes or 1)
        self.group_keys: List[str] = [canonical_patch_key(rec) for rec in self.index]

        self.stacks: Dict[Tuple[str, Tuple[str, ...], str, Optional[int], Optional[int], Optional[str]], Tuple[np.ndarray, np.ndarray]] = {}
        if self.preload:
            for rec in self.index:
                key = (
                    rec.sample_id,
                    tuple(map(str, rec.image_paths)),
                    str(rec.label_path),
                    rec.eff_h,
                    rec.eff_w,
                    rec.eff_mode,
                )
                if key not in self.stacks:
                    stack, mask = self._load_stack(
                        rec.sample_id, rec.image_paths, rec.label_path, rec.eff_h, rec.eff_w, rec.eff_mode
                    )
                    self.stacks[key] = (stack, mask)

    def __len__(self) -> int:
        return len(self.index)

    def _load_stack(
        self,
        sample_name: str,
        image_paths: List[Path],
        label_path: Path,
        eff_h: Optional[int],
        eff_w: Optional[int],
        eff_mode: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (sample_name, tuple(map(str, image_paths)), str(label_path), eff_h, eff_w, eff_mode)
        if self.preload and key in self.stacks:
            return self.stacks[key]
        images: List[np.ndarray] = []
        for path in image_paths:
            im = tiff.imread(str(path))
            im = _ensure_hwc3(im)
            images.append(im)
        stacked = np.concatenate(images, axis=-1)
        label = tiff.imread(str(label_path))
        if label.ndim == 3:
            label = label[..., 0]
        if eff_h is not None and eff_w is not None and eff_mode is not None:
            stacked = apply_eff_mode(stacked, eff_h, eff_w, eff_mode, pad_value=0)
            label = apply_eff_mode(label, eff_h, eff_w, eff_mode, pad_value=0)
        if self.normalize_0_1 and stacked.max() > 1.0:
            stacked = stacked.astype(np.float32) / 255.0
        if self.preload:
            self.stacks[key] = (stacked, label)
        return stacked, label

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        try:
            rec = self.index[index]
        except IndexError as exc:
            raise IndexError(index) from exc

        stack, mask = self._load_stack(rec.sample_id, rec.image_paths, rec.label_path, rec.eff_h, rec.eff_w, rec.eff_mode)
        patch = stack[rec.y0 : rec.y0 + rec.h, rec.x0 : rec.x0 + rec.w, :]
        patch = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()
        mask_patch = mask[rec.y0 : rec.y0 + rec.h, rec.x0 : rec.x0 + rec.w]
        mask_tensor = torch.from_numpy(mask_patch.astype(np.int64))
        return patch, mask_tensor, rec.patch_id


class _TransformingSubset(Dataset):
    def __init__(self, base_ds: Dataset, indices: Sequence[int], transform):
        self.base = base_ds
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        x, y, gid = self.base[base_idx]
        img = np.transpose(x.numpy(), (1, 2, 0))
        mask = y.numpy()
        augmented = self.transform(image=img, mask=mask)
        aug_x = torch.from_numpy(np.transpose(augmented["image"], (2, 0, 1))).float()
        aug_y = torch.from_numpy(augmented["mask"]).long()
        return aug_x, aug_y, gid


def build_dataset_index(
    root_dir: str,
    samples: Optional[List[str]],
    mode: str,
    patch_size: int,
    patch_stride: int,
    stack_images_limit: Optional[int],
    split_cfg: SplitConfig,
) -> Tuple[List[Dict[str, Any]], int]:
    root = Path(root_dir)
    if samples is None or len(samples) == 0:
        samples = [p.name for p in root.iterdir() if (p / "image").exists()]
    if split_cfg.mode not in {"sample_holdout", "region_split"}:
        raise ValueError(f"Unsupported split mode: {split_cfg.mode}")
    index: List[Dict[str, Any]] = []
    all_labels: List[np.ndarray] = []
    for sample in samples:
        meta = discover_sample(root / sample)
        mask = tiff.imread(str(meta["label"]))
        if mask.ndim == 3:
            mask = mask[..., 0]
        height, width = mask.shape[:2]
        if split_cfg.mode == "region_split":
            if split_cfg.region_grid and split_cfg.region_grid.get("rr") and split_cfg.region_grid.get("cc"):
                rr = int(split_cfg.region_grid["rr"])
                cc = int(split_cfg.region_grid["cc"])
                if split_cfg.eff_mode == "both":
                    pad_h = nearest_eff_size(height, patch_stride, rr, "pad")
                    pad_w = nearest_eff_size(width, patch_stride, cc, "pad")
                    crop_h = nearest_eff_size(height, patch_stride, rr, "crop")
                    crop_w = nearest_eff_size(width, patch_stride, cc, "crop")
                    pad_change = abs(pad_h - height) + abs(pad_w - width)
                    crop_change = abs(crop_h - height) + abs(crop_w - width)
                    if crop_h < patch_size or crop_w < patch_size:
                        eff_mode = "pad"
                        eff_h, eff_w = pad_h, pad_w
                    elif pad_change <= crop_change:
                        eff_mode = "pad"
                        eff_h, eff_w = pad_h, pad_w
                    else:
                        eff_mode = "crop"
                        eff_h, eff_w = crop_h, crop_w
                else:
                    eff_mode = split_cfg.eff_mode
                    eff_h = nearest_eff_size(height, patch_stride, rr, eff_mode)
                    eff_w = nearest_eff_size(width, patch_stride, cc, eff_mode)
                if eff_h < patch_size or eff_w < patch_size:
                    rr, cc, eff_h, eff_w, eff_mode = pick_best_grid_plan(
                        h=height,
                        w=width,
                        patch=patch_size,
                        stride=patch_stride,
                        eff_mode=split_cfg.eff_mode,
                        max_r=split_cfg.max_r,
                        max_c=split_cfg.max_c,
                        max_regions=split_cfg.max_regions,
                        min_val_test_each=split_cfg.min_val_test_regions_each,
                        min_total_regions=split_cfg.min_total_regions,
                        max_total_regions=split_cfg.max_total_regions,
                    )
            else:
                rr, cc, eff_h, eff_w, eff_mode = pick_best_grid_plan(
                    h=height,
                    w=width,
                    patch=patch_size,
                    stride=patch_stride,
                    eff_mode=split_cfg.eff_mode,
                    max_r=split_cfg.max_r,
                    max_c=split_cfg.max_c,
                    max_regions=split_cfg.max_regions,
                    min_val_test_each=split_cfg.min_val_test_regions_each,
                    min_total_regions=split_cfg.min_total_regions,
                    max_total_regions=split_cfg.max_total_regions,
                )
            eff_mask = apply_eff_mode(mask, eff_h, eff_w, eff_mode, pad_value=0)
            all_labels.append(eff_mask)
            coords_iter = iter_patch_coords_for_region_split(
                eff_h, eff_w, rr, cc, patch_size, patch_stride, split_cfg.strict_no_pixel_leakage
            )
            patches = list(coords_iter)
            if mode == "stack":
                stack_imgs = meta["images"]
                if stack_images_limit is not None:
                    stack_imgs = stack_imgs[:stack_images_limit]
                if not stack_imgs:
                    raise RuntimeError(f"All images removed for sample {sample} after applying stack_images_limit.")
                for (r, c, y0, x0) in patches:
                    patch_id = f"{sample}|stack|y{y0}x{x0}"
                    region_id = f"{sample}|r{r:02d}c{c:02d}"
                    index.append(
                        {
                            "sample_id": sample,
                            "image_paths": stack_imgs,
                            "label_path": meta["label"],
                            "y0": y0,
                            "x0": x0,
                            "h": patch_size,
                            "w": patch_size,
                            "angle_idx": None,
                            "patch_id": patch_id,
                            "region_id": region_id,
                            "eff_h": eff_h,
                            "eff_w": eff_w,
                            "eff_mode": eff_mode,
                            "grid_rr": rr,
                            "grid_cc": cc,
                        }
                    )
            else:
                img_paths = meta["images"]
                if stack_images_limit is not None:
                    img_paths = img_paths[:stack_images_limit]
                if not img_paths:
                    raise RuntimeError(f"All images removed for sample {sample} after applying stack_images_limit.")
                for ai, img_path in enumerate(img_paths):
                    for (r, c, y0, x0) in patches:
                        patch_id = f"{sample}|a{ai}|y{y0}x{x0}"
                        region_id = f"{sample}|r{r:02d}c{c:02d}"
                        index.append(
                            {
                                "sample_id": sample,
                                "image_paths": [img_path],
                                "label_path": meta["label"],
                                "y0": y0,
                                "x0": x0,
                                "h": patch_size,
                                "w": patch_size,
                                "angle_idx": ai,
                                "patch_id": patch_id,
                                "region_id": region_id,
                                "eff_h": eff_h,
                                "eff_w": eff_w,
                                "eff_mode": eff_mode,
                                "grid_rr": rr,
                                "grid_cc": cc,
                            }
                        )
        else:
            coords = sliding_window_coords(height, width, patch_size, patch_stride)
            all_labels.append(mask)
            if mode == "stack":
                stack_imgs = meta["images"]
                if stack_images_limit is not None:
                    stack_imgs = stack_imgs[:stack_images_limit]
                if not stack_imgs:
                    raise RuntimeError(f"All images removed for sample {sample} after applying stack_images_limit.")
                for (y0, x0, h, w) in coords:
                    patch_id = f"{sample}|stack|y{y0}x{x0}"
                    region_id = f"{sample}|r{y0 // patch_size:02d}c{x0 // patch_size:02d}"
                    index.append(
                        {
                            "sample_id": sample,
                            "image_paths": stack_imgs,
                            "label_path": meta["label"],
                            "y0": y0,
                            "x0": x0,
                            "h": h,
                            "w": w,
                            "angle_idx": None,
                            "patch_id": patch_id,
                            "region_id": region_id,
                            "eff_h": height,
                            "eff_w": width,
                            "eff_mode": "pad",
                        }
                    )
            else:
                img_paths = meta["images"]
                if stack_images_limit is not None:
                    img_paths = img_paths[:stack_images_limit]
                if not img_paths:
                    raise RuntimeError(f"All images removed for sample {sample} after applying stack_images_limit.")
                for ai, img_path in enumerate(img_paths):
                    for (y0, x0, h, w) in coords:
                        patch_id = f"{sample}|a{ai}|y{y0}x{x0}"
                        region_id = f"{sample}|r{y0 // patch_size:02d}c{x0 // patch_size:02d}"
                        index.append(
                            {
                                "sample_id": sample,
                                "image_paths": [img_path],
                                "label_path": meta["label"],
                                "y0": y0,
                                "x0": x0,
                                "h": h,
                                "w": w,
                                "angle_idx": ai,
                                "patch_id": patch_id,
                                "region_id": region_id,
                                "eff_h": height,
                                "eff_w": width,
                                "eff_mode": "pad",
                            }
                        )
    num_classes = int(np.max([m.max() for m in all_labels])) + 1 if all_labels else 1
    return index, num_classes


def get_dataloaders(cfg: Dict[str, Any], smoketest: bool = False):
    split_cfg = SplitConfig.from_dict(cfg.get("split", {}))
    index_data, num_classes = build_dataset_index(
        root_dir=cfg["data"]["root_dir"],
        samples=cfg["data"].get("samples") or None,
        mode=cfg["data"]["mode"],
        patch_size=cfg["data"]["patch_size"],
        patch_stride=cfg["data"]["patch_stride"],
        stack_images_limit=cfg["data"].get("stack_images_limit"),
        split_cfg=split_cfg,
    )

    dataset = ThinSectionDataset(
        root_dir=cfg["data"]["root_dir"],
        samples=cfg["data"].get("samples") or None,
        mode=cfg["data"]["mode"],
        patch_size=cfg["data"]["patch_size"],
        patch_stride=cfg["data"]["patch_stride"],
        normalize_0_1=cfg["data"]["normalize_0_1"],
        preload_into_ram=cfg["data"]["preload_into_ram"],
        stack_images_limit=cfg["data"].get("stack_images_limit"),
        index_data=index_data,
        num_classes=num_classes,
        split_cfg=split_cfg,
    )

    split_manager = SplitManager(split_cfg)
    if split_cfg.manifest_path:
        split_manager.load_manifest(split_cfg.manifest_path)
    else:
        split_manager.generate_manifest(index_data)
        if split_cfg.save_manifest:
            manifest_target = split_cfg.manifest_path or None
            split_manager.save_manifest(manifest_target)

    split_indices = split_manager.apply_manifest_to_patch_list(index_data)

    train_idx = split_indices.get("train", [])
    val_idx = split_indices.get("val", [])
    test_idx = split_indices.get("test", [])

    _validate_disjoint(train_idx, val_idx, test_idx)

    if smoketest and cfg.get("smoketest", {}).get("enabled", False):
        max_train = cfg["smoketest"].get("max_train_patches") or len(train_idx)
        max_val = cfg["smoketest"].get("max_val_patches") or len(val_idx)
        train_idx = train_idx[:max_train]
        val_idx = val_idx[:max_val]
        test_idx = test_idx[: max_val if max_val else len(test_idx)]

    workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"]
    persistent_workers = cfg["data"]["persistent_workers"]
    prefetch_factor = cfg["data"].get("prefetch_factor")
    if smoketest and cfg.get("smoketest", {}).get("enabled", False):
        workers = cfg["smoketest"].get("num_workers", workers)
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None

    train_transform = build_augmentation_pipeline(cfg.get("augment", {}).get("train"), dataset[0][0].shape[0])
    if train_transform is not None:
        train_ds: Dataset = _TransformingSubset(dataset, train_idx, train_transform)
    else:
        train_ds = Subset(dataset, train_idx)
    loader_kwargs = {
        "batch_size": cfg["data"]["batch_size"],
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_ds,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return {
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "manifest_path": split_manager.manifest_path,
        "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
    }


def export_patches(dataset: ThinSectionDataset, out_dir: str | Path) -> List[Path]:
    """Write dataset patches to disk using the late-fusion naming convention.

    Separate mode writes one file per view using suffixes _PPL/_XPL40/_XPL50/_XPL60/_XPL70
    (fallbacks use VIEW{idx} when a suffix cannot be inferred). A single mask per base
    patch id is saved as ``patch_XXXX_mask.tif``.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_keys = sorted({base_patch_id(rec) for rec in dataset.index})
    base_to_name = {bk: f"patch_{i:04d}" for i, bk in enumerate(base_keys)}
    written: List[Path] = []
    saved_masks = set()

    for rec in dataset.index:
        base_key = base_patch_id(rec)
        base_name = base_to_name[base_key]

        stack, mask = dataset._load_stack(
            rec.sample_id, rec.image_paths, rec.label_path, rec.eff_h, rec.eff_w, rec.eff_mode
        )
        patch = stack[rec.y0 : rec.y0 + rec.h, rec.x0 : rec.x0 + rec.w, :]
        mask_patch = mask[rec.y0 : rec.y0 + rec.h, rec.x0 : rec.x0 + rec.w]

        if dataset.mode == "separate":
            suffix = view_suffix_from_path(rec.image_paths[0], rec.angle_idx)
            fname = f"{base_name}_{suffix}.tif"
        else:
            fname = f"{base_name}.tif"

        patch_path = out_path / fname
        tiff.imwrite(patch_path, patch.astype(np.float32))
        written.append(patch_path)

        if base_key not in saved_masks:
            mask_path = out_path / f"{base_name}_mask.tif"
            tiff.imwrite(mask_path, mask_patch.astype(np.uint8))
            saved_masks.add(base_key)

    return written


def _validate_disjoint(train_idx: Sequence[int], val_idx: Sequence[int], test_idx: Sequence[int]) -> None:
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Splits are not mutually exclusive.")
