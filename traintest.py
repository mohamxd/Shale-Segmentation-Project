from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from itertools import islice

from src.config import load_config
from src.data import get_dataloaders
from src.losses import SoftDiceLoss, class_weights_from_mask
from src.models import build_model
from src.utils import set_seed


def run_smoketest(cfg: dict) -> None:
    print("[smoketest] starting...", flush=True)
    set_seed(cfg["project"]["seed"])

    bundle = get_dataloaders(cfg, smoketest=True)
    dataset = bundle["dataset"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]

    print(f"[smoketest] dataset_len={len(dataset)}", flush=True)

    try:
        train_len = len(train_loader)
    except TypeError:
        train_len = None
    try:
        val_len = len(val_loader)
    except TypeError:
        val_len = None

    print(f"[smoketest] train_loader_len={train_len}, val_loader_len={val_len}", flush=True)

    if len(dataset) == 0:
        raise RuntimeError("[smoketest] dataset is empty. Check cfg data paths/splits and smoketest settings.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = int(dataset[0][0].shape[0])
    num_classes = dataset.num_classes
    print(f"[smoketest] device={device}, in_channels={in_channels}, num_classes={num_classes}", flush=True)

    model = build_model(
        arch=cfg["model"]["arch"],
        encoder=cfg["model"]["encoder"],
        in_channels=in_channels,
        classes=num_classes,
        encoder_weights=cfg["model"]["encoder_weights"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    labels = []
    for _, mask, _ in islice(dataset, 0, min(len(dataset), 10)):
        labels.append(mask.numpy())
    flat = np.concatenate([m.reshape(-1) for m in labels]) if labels else np.array([0], dtype=np.int64)
    weights = class_weights_from_mask(flat, num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = SoftDiceLoss()

    batches_per_epoch = int(cfg.get("smoketest", {}).get("batches_per_epoch") or 1)
    print(f"[smoketest] batches_per_epoch={batches_per_epoch}", flush=True)

    start = time.time()
    model.train()
    did_train = 0

    for step, batch in enumerate(train_loader, 1):
        if step > batches_per_epoch:
            break
        x, y = batch[0].to(device), batch[1].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = 0.5 * ce_loss(logits, y) + 0.5 * dice_loss(logits, y)
        loss.backward()
        optimizer.step()
        did_train += 1

        if step == 1:
            print(f"[smoketest] train step1 loss={loss.item():.4f} x={tuple(x.shape)} y={tuple(y.shape)}", flush=True)

    train_time = time.time() - start

    if did_train == 0:
        raise RuntimeError("[smoketest] train_loader produced 0 batches. Check your train split / smoketest settings.")

    model.eval()
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader, 1):
            if step > batches_per_epoch:
                break
            x, y = batch[0].to(device), batch[1].to(device).long()
            logits = model(x)
            loss = 0.5 * ce_loss(logits, y) + 0.5 * dice_loss(logits, y)
            val_loss += float(loss.item())
            n_val += 1

            if step == 1:
                print(f"[smoketest] val step1 loss={loss.item():.4f}", flush=True)

    if n_val == 0:
        raise RuntimeError("[smoketest] val_loader produced 0 batches. Check your val split / smoketest settings.")

    val_loss = val_loss / max(1, n_val)

    summary = Path(cfg["project"]["outputs_dir"]) / "smoketest_summary.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        f"train_batches={did_train}, val_batches={n_val}, val_loss={val_loss:.4f}, time={train_time:.2f}s\n"
    )

    print(f"[smoketest] wrote: {summary.resolve()}", flush=True)
    print(
        f"Smoke test completed: train {did_train} batches, val {n_val} batches, "
        f"val_loss={val_loss:.4f}, time={train_time:.2f}s",
        flush=True,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-epoch smoke test.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_smoketest(cfg)


if __name__ == "__main__":
    main()
