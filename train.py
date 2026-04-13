"""
FIT-VTON Training Entry Point.

Usage
-----
  python train.py --config configs/train.yaml
  python train.py --config configs/train.yaml --resume outputs/checkpoints/step_50000
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FIT-VTON: Measurement-Conditioned Virtual Try-On",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--override",
        nargs="+",
        default=[],
        help=(
            "Config key=value overrides, e.g. "
            "--override training.learning_rate=5e-5 training.batch_size=2"
        ),
    )
    return parser.parse_args()


def apply_overrides(cfg, overrides):
    """Applies key=value string overrides to an OmegaConf config."""
    from omegaconf import OmegaConf

    for override in overrides:
        if "=" not in override:
            print(f"WARNING: Skipping malformed override '{override}' (expected key=value)")
            continue
        key, value = override.split("=", 1)
        # Parse the value as a Python literal if possible
        try:
            import ast
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value  # keep as string

        OmegaConf.update(cfg, key, parsed, merge=True)

    return cfg


def main():
    args = parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("ERROR: omegaconf is required. Install with: pip install omegaconf")
        sys.exit(1)

    cfg = OmegaConf.load(str(config_path))

    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # --- Print config summary ---
    print("=" * 60)
    print("FIT-VTON Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # --- Import and launch trainer ---
    try:
        from fit_vton.training.trainer import FitVTONTrainer
    except ImportError as e:
        print(f"ERROR: Could not import FitVTONTrainer: {e}")
        print("Make sure the package is installed: pip install -e .")
        sys.exit(1)

    trainer = FitVTONTrainer(cfg, resume_from_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
