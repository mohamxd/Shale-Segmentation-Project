# Shale Segmentation Code

This repo provides a reproducible deep-learning workflow for semantic segmentation of thin-section mineral imagery.
For detailed steps of how to prepare dataset and run the code refer to guide.pdf in this repository

## Installation
```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.9+
- PyTorch (CPU or GPU builds are supported; install the variant that matches your hardware and CUDA setup)
- See `requirements.txt` for supporting libraries such as Albumentations, tifffile, and segmentation-models-pytorch.

## Dataset layout
```
<data root>/
  sample_name_1/
    image/  # multi-angle images (*.tif*)
    label/  # single-channel ground truth mask (*.tif*)
  sample_name_2/
    ...
```
Patches exported for inference should follow the late-fusion naming convention (e.g., `patch_0001_PPL.tif`, `patch_0001_XPL40.tif`) with optional masks named `patch_0001_mask.tif`.

## Configuration
All pipeline settings are defined in `config.yaml`:
- `project`: experiment name, output directory, random seed.
- `data`: dataset root, sample list, loader mode (`stack` or `separate`), patch size and stride, normalization, worker settings, and optional preloading.
- `split`: split mode (`sample_holdout` or `region_split`), fractions or holdout lists, manifest path, and manifest output directory plus region planning options.
- `model`: architecture, encoder backbone, encoder weights, dropout, input channels, and number of classes.
- `optim` and `sched`: optimizer hyperparameters, loss weighting, AMP toggle, gradient clipping, and scheduler choice.
- `train`: epoch count, early stopping patience, and logging interval.
- `eval`: confusion-matrix saving flag.
- `augment`: Albumentations options for training patches.
- `smoketest`: limits for dataset size, workers, and batches per epoch for quick runs.
- `inference`: checkpoint path, patch directory, number of examples, and output directory for predictions.

## Data splits
- Two split modes are supported: `sample_holdout` (unchanged behavior) and `region_split`.
- `region_split` plans a grid (rows x columns) per sample automatically within configured bounds, or uses an explicit `region_grid` override.
- The effective canvas can be padded or cropped (`eff_mode`) to align grid and stride; padding uses background for labels.
- Patch coordinates are taken from the stride grid, keeping patches whose centers land inside a region; `strict_no_pixel_leakage` additionally requires full containment to avoid cross-region pixels.
- Region assignments occur at the region level using deterministic shuffles and requested fractions, ensuring region IDs are angle-agnostic to prevent leakage across views.

## Training
Run full training with the specified configuration:
```bash
python -m src.train --config config.yaml
```
Outputs (checkpoints, logs, curves, and summaries) are stored under `project.outputs_dir`.

## Evaluation
Generate metrics and visualizations against the configured dataset and a trained checkpoint:
```bash
python -m src.evaluate --config config.yaml --checkpoint models/best.ckpt
```
Artifacts (per-class metrics, confusion matrix, and sample triptychs) are written to `project.outputs_dir`.

## Inference
Run lightweight inference on stored patches:
```bash
python inference.py --config config.yaml
```
Predictions are saved to `inference.save_dir` with both colorized masks and comparison visualizations.

## Reproducibility
- Deterministic seeding via `project.seed`.
- Split manifests saved to `split.output_dir` when `split.save_manifest` is enabled.
- Smoke tests (`traintest.py`) validate the end-to-end pipeline with minimal resources:
  ```bash
  python traintest.py --config config.yaml
  ```

## Outputs
- Training: checkpoints under `outputs_dir/checkpoints/`, logs under `outputs_dir/logs/`, and summary JSON/plots under `outputs_dir`.
- Evaluation: metrics JSON, confusion matrix, and visualization images saved to `outputs_dir`.
- Inference: prediction color masks and visualizations saved to `inference.save_dir`.

## Troubleshooting
- Ensure dataset paths in `config.yaml` are correct and contain `.tif` images and labels.
- If `segmentation_models_pytorch` or `timm` are missing install them.
- For CPU-only environments, install the CPU build of PyTorch and reduce `batch_size` and worker counts.

## Citation
```bibtex
@software{shalesseg_2025,
  title  = {Shales Thin-section mineral segmentation pipeline},
  year   = {2025},
  note   = {Research software.}
}

```
