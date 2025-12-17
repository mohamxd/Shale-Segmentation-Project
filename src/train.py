"""CLI entrypoint for training."""

import argparse

from .config import load_config
from .engine import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a thin-section segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    out_dir = trainer.train()
    print("All done. Outputs at:", out_dir)


if __name__ == "__main__":
    main()
