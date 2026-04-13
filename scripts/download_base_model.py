"""
Download IDM-VTON base model from HuggingFace Hub.

Usage
-----
  python scripts/download_base_model.py [--output_dir cache/models]
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download IDM-VTON base model from HuggingFace"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="yisol/IDM-VTON",
        help="HuggingFace model ID to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/models",
        help="Directory to store downloaded model files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (for gated models)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch to download",
    )
    return parser.parse_args()


def download_model(model_id: str, output_dir: str, token: str = None, revision: str = "main"):
    """Downloads a HuggingFace model to a local directory."""
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    local_dir = output_path / model_id.replace("/", "--")

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Model already downloaded at: {local_dir}")
        print("Delete the directory to re-download.")
        return str(local_dir)

    print(f"Downloading {model_id} to {local_dir} ...")
    print("This may take several minutes (~10GB).")

    if token:
        login(token=token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    try:
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            revision=revision,
            ignore_patterns=["*.msgpack", "flax_model*"],  # skip JAX weights
        )
        print(f"\nModel downloaded successfully to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"\nERROR downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection.")
        print("  2. If the model is gated, pass --token <your_hf_token>")
        print("     or set HF_TOKEN environment variable.")
        print("  3. Alternatively, download manually from:")
        print(f"     https://huggingface.co/{model_id}")
        sys.exit(1)


def verify_download(local_dir: str) -> bool:
    """Checks that essential model files are present."""
    required_files = [
        "unet/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    ]
    missing = []
    for f in required_files:
        full_path = Path(local_dir) / f
        if not full_path.exists():
            missing.append(f)

    if missing:
        print(f"\nWARNING: Some expected files are missing: {missing}")
        print("The download may be incomplete.")
        return False

    print("\nModel integrity check passed.")
    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("FIT-VTON: IDM-VTON Base Model Downloader")
    print("=" * 60)
    print(f"Model ID  : {args.model_id}")
    print(f"Output dir: {args.output_dir}")
    print()

    local_dir = download_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        token=args.token,
        revision=args.revision,
    )

    verify_download(local_dir)

    print("\nNext steps:")
    print("  1. Generate synthetic training data:")
    print("     python scripts/download_fit_dataset.py --mini --output_dir data/fit_dataset")
    print("  2. Start training:")
    print("     python train.py --config configs/train.yaml")


if __name__ == "__main__":
    main()
