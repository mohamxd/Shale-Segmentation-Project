from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json
import random
import yaml


@dataclass
class SplitFractions:
    train: float
    val: float
    test: float


@dataclass
class SplitHoldout:
    train_samples: List[str]
    val_samples: List[str]
    test_samples: List[str]


@dataclass
class SplitConfig:
    mode: str
    seed: int
    fractions: SplitFractions
    strict_no_pixel_leakage: bool
    eff_mode: str
    max_regions: int
    min_total_regions: int
    max_total_regions: int
    max_r: int
    max_c: int
    min_val_test_regions_each: int
    region_grid: Dict[str, int]
    sample_holdout: SplitHoldout
    manifest_path: str
    save_manifest: bool
    output_dir: str

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "SplitConfig":
        fractions = cfg.get("fractions", {})
        holdout = cfg.get("sample_holdout", {})
        region_grid = cfg.get("region_grid") or {}
        return cls(
            mode=cfg.get("mode", "sample_holdout"),
            seed=int(cfg.get("seed", 42)),
            fractions=SplitFractions(
                train=float(fractions.get("train", 0.7)),
                val=float(fractions.get("val", 0.15)),
                test=float(fractions.get("test", 0.15)),
            ),
            strict_no_pixel_leakage=bool(cfg.get("strict_no_pixel_leakage", False)),
            eff_mode=str(cfg.get("eff_mode", "pad")),
            max_regions=int(cfg.get("max_regions", 80)),
            min_total_regions=int(cfg.get("min_total_regions", 1)),
            max_total_regions=int(cfg.get("max_total_regions", cfg.get("max_regions", 80))),
            max_r=int(cfg.get("max_r", 12)),
            max_c=int(cfg.get("max_c", 20)),
            min_val_test_regions_each=int(cfg.get("min_val_test_regions_each", 1)),
            region_grid={"rr": int(region_grid.get("rr", 0)), "cc": int(region_grid.get("cc", 0))}
            if region_grid
            else {},
            sample_holdout=SplitHoldout(
                train_samples=list(holdout.get("train_samples", [])),
                val_samples=list(holdout.get("val_samples", [])),
                test_samples=list(holdout.get("test_samples", [])),
            ),
            manifest_path=str(cfg.get("manifest_path", "")),
            save_manifest=bool(cfg.get("save_manifest", True)),
            output_dir=str(cfg.get("output_dir", "outputs/splits")),
        )


class SplitManager:
    def __init__(self, config: SplitConfig):
        self.config = config
        self.manifest: Dict[str, str] = {}
        self.manifest_path: Path | None = None

    def generate_manifest(self, dataset_index: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        by_region: Dict[str, List[Dict[str, Any]]] = {}
        for entry in dataset_index:
            region_id = entry["region_id"]
            by_region.setdefault(region_id, []).append(entry)

        if self.config.mode == "sample_holdout":
            self.manifest = self._manifest_sample_holdout(dataset_index)
            return self.manifest

        if self.config.mode != "region_split":
            raise ValueError(f"Unsupported split mode: {self.config.mode}")

        region_to_split: Dict[str, str] = {}
        by_sample: Dict[str, List[str]] = {}
        for region in by_region:
            sample = region.split("|")[0]
            by_sample.setdefault(sample, []).append(region)

        for sample, regions in by_sample.items():
            rr = by_region[regions[0]][0].get("grid_rr")
            cc = by_region[regions[0]][0].get("grid_cc")
            assigned = self._assign_regions_by_fraction(
                sample_name=sample,
                regions=regions,
                rr=rr,
                cc=cc,
                min_val_test_regions_each=self.config.min_val_test_regions_each,
            )
            region_to_split.update(assigned)

        manifest: Dict[str, str] = {}
        for region, patches in by_region.items():
            split = region_to_split.get(region, "train")
            for entry in patches:
                manifest[entry["patch_id"]] = split

        self.manifest = manifest
        return manifest

    def _manifest_sample_holdout(self, dataset_index: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        cfg = self.config.sample_holdout
        manifest: Dict[str, str] = {}
        for entry in dataset_index:
            sample = entry["sample_id"]
            if sample in cfg.test_samples:
                split = "test"
            elif sample in cfg.val_samples:
                split = "val"
            elif cfg.train_samples and sample not in cfg.train_samples:
                split = "test"
            else:
                split = "train"
            manifest[entry["patch_id"]] = split
        return manifest

    def _assign_regions_by_fraction(
        self,
        sample_name: str,
        regions: Iterable[str],
        rr: int | None = None,
        cc: int | None = None,
        min_val_test_regions_each: int = 1,
    ) -> Dict[str, str]:
        region_list = list(regions)
        rng = random.Random(hash((sample_name, self.config.seed)) & 0xFFFFFFFF)
        rng.shuffle(region_list)

        n = len(region_list)
        raw = {
            "train": n * self.config.fractions.train,
            "val": n * self.config.fractions.val,
            "test": n * self.config.fractions.test,
        }
        counts = {k: int(raw[k]) for k in raw}
        remainder = n - sum(counts.values())
        parts = [(raw[k] - counts[k], k) for k in raw]
        rng.shuffle(parts)
        parts.sort(reverse=True, key=lambda x: x[0])
        for i in range(remainder):
            counts[parts[i][1]] += 1

        if n >= 1 + 2 * min_val_test_regions_each:
            if counts["val"] < min_val_test_regions_each:
                needed = min_val_test_regions_each - counts["val"]
                counts["val"] += needed
                counts["train"] -= needed
            if counts["test"] < min_val_test_regions_each:
                needed = min_val_test_regions_each - counts["test"]
                counts["test"] += needed
                counts["train"] -= needed
            if counts["train"] < 0:
                counts = {"train": n, "val": 0, "test": 0}

        manifest: Dict[str, str] = {}
        idx = 0
        for _ in range(counts["train"]):
            if idx < len(region_list):
                manifest[region_list[idx]] = "train"
            idx += 1
        for _ in range(counts["val"]):
            if idx < len(region_list):
                manifest[region_list[idx]] = "val"
            idx += 1
        for _ in range(counts["test"]):
            if idx < len(region_list):
                manifest[region_list[idx]] = "test"
            idx += 1
        while idx < len(region_list):
            manifest[region_list[idx]] = "train"
            idx += 1
        return manifest

    def save_manifest(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self._default_manifest_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in (".yaml", ".yml"):
            target.write_text(yaml.safe_dump(self.manifest))
        else:
            target.write_text(json.dumps(self.manifest, indent=2))
        self.manifest_path = target
        return target

    def load_manifest(self, path: str | Path) -> Dict[str, str]:
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        if manifest_path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(manifest_path.read_text())
        else:
            data = json.loads(manifest_path.read_text())
        self.manifest = {str(k): str(v) for k, v in data.items()}
        self.manifest_path = manifest_path
        return self.manifest

    def apply_manifest_to_patch_list(self, dataset_index: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        seen: Dict[str, str] = {}
        for idx, entry in enumerate(dataset_index):
            patch_id = entry["patch_id"]
            split = self.manifest.get(patch_id)
            if split is None:
                continue
            if patch_id in seen and seen[patch_id] != split:
                raise RuntimeError(f"Patch {patch_id} assigned to multiple splits.")
            seen[patch_id] = split
            if split not in splits:
                continue
            splits[split].append(idx)
        return splits

    def _default_manifest_path(self) -> Path:
        base = Path(self.config.output_dir)
        base.mkdir(parents=True, exist_ok=True)
        name = "split_manifest_seed{0}.json".format(self.config.seed)
        return base / name
