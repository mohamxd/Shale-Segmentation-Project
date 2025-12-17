from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    ):
        self.root = Path(root_dir)
        self.samples = samples
        self.mode = mode
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.normalize_0_1 = normalize_0_1
        self.preload = preload_into_ram
        self.stack_images_limit = stack_images_limit

        if index_data is None:
            index_data, computed_classes = build_dataset_index(
                root_dir=root_dir,
                samples=samples,
                mode=mode,
                patch_size=patch_size,
                patch_stride=patch_stride,
                stack_images_limit=stack_images_limit,
            )
            if num_classes is None:
                num_classes = computed_classes

        self.index: List[PatchRecord] = [PatchRecord(**record) for record in index_data]
        self.num_classes: int = int(num_classes or 1)
        self.group_keys: List[str] = [canonical_patch_key(rec) for rec in self.index]

        self.stacks: Dict[Tuple[str, Tuple[str, ...], str], Tuple[np.ndarray, np.ndarray]] = {}
        if self.preload:
            for rec in self.index:
                key = (rec.sample_id, tuple(map(str, rec.image_paths)), str(rec.label_path))
                if key not in self.stacks:
                    stack, mask = self._load_stack(rec.sample_id, rec.image_paths, rec.label_path)
                    self.stacks[key] = (stack, mask)

    def __len__(self) -> int:
        return len(self.index)

    def _load_stack(self, sample_name: str, image_paths: List[Path], label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        key = (sample_name, tuple(map(str, image_paths)), str(label_path))
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

        stack, mask = self._load_stack(rec.sample_id, rec.image_paths, rec.label_path)
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
) -> Tuple[List[Dict[str, Any]], int]:
    root = Path(root_dir)
    if samples is None or len(samples) == 0:
        samples = [p.name for p in root.iterdir() if (p / "image").exists()]
    index: List[Dict[str, Any]] = []
    all_labels: List[np.ndarray] = []
    for sample in samples:
        meta = discover_sample(root / sample)
        mask = tiff.imread(str(meta["label"]))
        if mask.ndim == 3:
            mask = mask[..., 0]
        height, width = mask.shape[:2]
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
                        }
                    )
    num_classes = int(np.max([m.max() for m in all_labels])) + 1 if all_labels else 1
    return index, num_classes


def get_dataloaders(cfg: Dict[str, Any], smoketest: bool = False):
    index_data, num_classes = build_dataset_index(
        root_dir=cfg["data"]["root_dir"],
        samples=cfg["data"].get("samples") or None,
        mode=cfg["data"]["mode"],
        patch_size=cfg["data"]["patch_size"],
        patch_stride=cfg["data"]["patch_stride"],
        stack_images_limit=cfg["data"].get("stack_images_limit"),
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
    )

    split_cfg = SplitConfig.from_dict(cfg.get("split", {}))
    split_manager = SplitManager(split_cfg)
    if split_cfg.mode == "fixed_manifest":
        if not split_cfg.manifest_path:
            raise ValueError("split.manifest_path is required when mode='fixed_manifest'.")
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

        stack, mask = dataset._load_stack(rec.sample_id, rec.image_paths, rec.label_path)
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
