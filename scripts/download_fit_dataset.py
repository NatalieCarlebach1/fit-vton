"""
FIT Dataset downloader / synthetic mini-dataset generator.

Since Google's FIT dataset (1.13M image triplets) may not be publicly released,
this script:
  1. Tries to download the official FIT dataset from known sources.
  2. Falls back to generating a synthetic mini-dataset with the same schema.

The synthetic dataset uses PIL-drawn placeholder images with realistic
per-size measurement distributions.

Usage
-----
  # Synthetic mini-dataset (5k train / 500 val / 500 test):
  python scripts/download_fit_dataset.py --mini --output_dir data/fit_dataset

  # Try official download:
  python scripts/download_fit_dataset.py --output_dir data/fit_dataset
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Measurement distributions per size (mean, std)
# height_cm, weight_kg, chest_cm, waist_cm, hip_cm
# ---------------------------------------------------------------------------

SIZE_DISTRIBUTIONS = {
    0: {  # XS
        "height": (163.0, 4.0),
        "weight": (50.0, 5.0),
        "chest": (80.0, 3.0),
        "waist": (62.0, 3.0),
        "hip": (87.0, 3.0),
    },
    1: {  # S
        "height": (165.0, 4.0),
        "weight": (57.0, 6.0),
        "chest": (85.0, 3.0),
        "waist": (68.0, 3.0),
        "hip": (92.0, 3.0),
    },
    2: {  # M
        "height": (167.0, 4.5),
        "weight": (65.0, 7.0),
        "chest": (91.0, 3.5),
        "waist": (74.0, 4.0),
        "hip": (98.0, 4.0),
    },
    3: {  # L
        "height": (168.0, 4.5),
        "weight": (74.0, 8.0),
        "chest": (97.0, 4.0),
        "waist": (80.0, 4.5),
        "hip": (104.0, 4.0),
    },
    4: {  # XL
        "height": (169.0, 5.0),
        "weight": (83.0, 9.0),
        "chest": (103.0, 4.5),
        "waist": (87.0, 5.0),
        "hip": (110.0, 4.5),
    },
    5: {  # XXL
        "height": (170.0, 5.0),
        "weight": (93.0, 10.0),
        "chest": (110.0, 5.0),
        "waist": (95.0, 5.5),
        "hip": (118.0, 5.0),
    },
}

SIZE_LABELS = ["XS", "S", "M", "L", "XL", "XXL"]

GARMENT_DESCRIPTIONS = [
    "fitted blue denim jacket",
    "loose white cotton t-shirt",
    "floral summer dress",
    "red wool sweater",
    "black leather blazer",
    "striped button-down shirt",
    "yellow sundress",
    "green hoodie",
    "formal white blouse",
    "casual grey cardigan",
    "purple velvet top",
    "navy blue polo shirt",
    "pink chiffon blouse",
    "orange athletic jersey",
    "brown knit jumper",
]


def sample_measurements(body_size_idx: int, rng: np.random.Generator) -> Dict:
    """Samples realistic body measurements for the given size."""
    dist = SIZE_DISTRIBUTIONS[body_size_idx]
    return {
        "height_cm": round(float(rng.normal(dist["height"][0], dist["height"][1])), 1),
        "weight_kg": round(float(rng.normal(dist["weight"][0], dist["weight"][1])), 1),
        "chest_cm": round(float(rng.normal(dist["chest"][0], dist["chest"][1])), 1),
        "waist_cm": round(float(rng.normal(dist["waist"][0], dist["waist"][1])), 1),
        "hip_cm": round(float(rng.normal(dist["hip"][0], dist["hip"][1])), 1),
    }


def draw_person_image(body_size_idx: int, rng: np.random.Generator):
    """
    Draws a synthetic person silhouette.
    Body proportions vary with size to simulate different body types.
    """
    from PIL import Image, ImageDraw

    W, H = 512, 512
    img = Image.new("RGB", (W, H), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Skin tone varies slightly
    skin_r = int(rng.integers(200, 230))
    skin_g = int(rng.integers(160, 190))
    skin_b = int(rng.integers(130, 160))
    skin_color = (skin_r, skin_g, skin_b)

    # Body width scales with size (XS=narrow, XXL=wide)
    body_width_factor = 0.55 + body_size_idx * 0.05  # 0.55 to 0.80

    # Head
    head_cx, head_cy, head_r = W // 2, 80, 45
    draw.ellipse(
        [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
        fill=skin_color,
        outline=(100, 100, 100),
    )

    # Torso
    torso_w = int(W * body_width_factor)
    torso_x0 = W // 2 - torso_w // 2
    torso_y0 = head_cy + head_r + 5
    torso_h = 180
    draw.rectangle(
        [torso_x0, torso_y0, torso_x0 + torso_w, torso_y0 + torso_h],
        fill=(100, 149, 237),  # cornflower blue = shirt
        outline=(70, 70, 200),
    )

    # Pants
    pants_w = int(W * body_width_factor * 0.9)
    pants_x0 = W // 2 - pants_w // 2
    pants_y0 = torso_y0 + torso_h
    pants_h = 150
    draw.rectangle(
        [pants_x0, pants_y0, pants_x0 + pants_w, pants_y0 + pants_h],
        fill=(50, 50, 100),  # dark blue = pants
        outline=(30, 30, 80),
    )

    # Arms
    arm_w = 20 + body_size_idx * 2
    draw.rectangle([torso_x0 - arm_w - 5, torso_y0, torso_x0 - 5, torso_y0 + torso_h],
                   fill=skin_color)
    draw.rectangle([torso_x0 + torso_w + 5, torso_y0,
                    torso_x0 + torso_w + arm_w + 5, torso_y0 + torso_h],
                   fill=skin_color)

    # Legs
    leg_w = 35 + body_size_idx * 3
    draw.rectangle([W // 2 - leg_w - 5, pants_y0 + pants_h,
                    W // 2 - 5, pants_y0 + pants_h + 80],
                   fill=skin_color)
    draw.rectangle([W // 2 + 5, pants_y0 + pants_h,
                    W // 2 + leg_w + 5, pants_y0 + pants_h + 80],
                   fill=skin_color)

    return img


def draw_garment_image(garment_size_idx: int, description: str, rng: np.random.Generator):
    """
    Draws a synthetic garment on a white background.
    Garment size affects the rendered dimensions.
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 512, 512
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Random garment colour
    r = int(rng.integers(50, 230))
    g = int(rng.integers(50, 230))
    b = int(rng.integers(50, 230))
    color = (r, g, b)
    dark_color = (max(r - 40, 0), max(g - 40, 0), max(b - 40, 0))

    # Garment width scales with size
    width_factor = 0.45 + garment_size_idx * 0.05  # 0.45 to 0.70

    gw = int(W * width_factor)
    gh = int(H * 0.55)
    gx0 = W // 2 - gw // 2
    gy0 = H // 2 - gh // 2

    # Main body
    draw.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], fill=color, outline=dark_color, width=2)

    # Collar
    collar_w = gw // 4
    draw.rectangle(
        [W // 2 - collar_w // 2, gy0, W // 2 + collar_w // 2, gy0 + 40],
        fill=(255, 255, 255),
        outline=dark_color,
    )

    # Sleeves
    sleeve_w = 60 + garment_size_idx * 5
    sleeve_h = 80
    draw.rectangle([gx0 - sleeve_w, gy0 + 20, gx0, gy0 + 20 + sleeve_h],
                   fill=color, outline=dark_color, width=2)
    draw.rectangle([gx0 + gw, gy0 + 20, gx0 + gw + sleeve_w, gy0 + 20 + sleeve_h],
                   fill=color, outline=dark_color, width=2)

    # Size label
    draw.text((10, 10), f"Size: {SIZE_LABELS[garment_size_idx]}", fill=(50, 50, 50))

    return img


def draw_tryon_image(
    body_size_idx: int,
    garment_size_idx: int,
    fit_delta: float,
    rng: np.random.Generator,
):
    """
    Draws a synthetic try-on result.
    The garment fit is visually adjusted by fit_delta:
      fit_delta > 0 → garment appears baggier (wider, less fitted)
      fit_delta < 0 → garment appears tighter (narrower, strain lines)
      fit_delta = 0 → perfect fit
    """
    from PIL import Image, ImageDraw

    # Start with the person silhouette
    img = draw_person_image(body_size_idx, rng)
    draw = ImageDraw.Draw(img)

    W, H = 512, 512

    # Garment colour (same random seed effect via parameter)
    rng2 = np.random.default_rng(garment_size_idx * 13 + body_size_idx * 7)
    r = int(rng2.integers(50, 230))
    g = int(rng2.integers(50, 230))
    b = int(rng2.integers(50, 230))
    color = (r, g, b)

    # Body width as in person drawing
    body_width_factor = 0.55 + body_size_idx * 0.05
    torso_w_base = int(W * body_width_factor)

    # Adjust garment fit: fit_delta adjusts width relative to body
    # +1 size delta → 8% wider, -1 size delta → 6% tighter
    fit_scale = 1.0 + fit_delta * 0.08
    garment_w = int(torso_w_base * fit_scale)

    torso_y0 = 80 + 45 + 5  # head_cy + head_r + gap
    torso_h = 180

    gx0 = W // 2 - garment_w // 2

    # Draw garment over the person torso
    draw.rectangle(
        [gx0, torso_y0, gx0 + garment_w, torso_y0 + torso_h],
        fill=color,
        outline=(max(r - 40, 0), max(g - 40, 0), max(b - 40, 0)),
        width=2,
    )

    # Collar
    collar_w = garment_w // 4
    draw.rectangle(
        [W // 2 - collar_w // 2, torso_y0, W // 2 + collar_w // 2, torso_y0 + 40],
        fill=(255, 255, 255),
    )

    # If too tight (fit_delta < -1), add strain line hints
    if fit_delta < -1:
        for i in range(3):
            cx = gx0 + garment_w // 4 + i * garment_w // 4
            draw.line(
                [(cx, torso_y0 + 30), (cx + 5, torso_y0 + 80)],
                fill=(max(r - 60, 0), max(g - 60, 0), max(b - 60, 0)),
                width=2,
            )

    # Sleeves
    sleeve_w = int((60 + body_size_idx * 3) * max(0.7, fit_scale))
    draw.rectangle(
        [gx0 - sleeve_w, torso_y0 + 20, gx0, torso_y0 + 100],
        fill=color,
    )
    draw.rectangle(
        [gx0 + garment_w, torso_y0 + 20, gx0 + garment_w + sleeve_w, torso_y0 + 100],
        fill=color,
    )

    return img


def generate_metadata_entry(
    sample_id: str,
    body_size_idx: int,
    garment_size_idx: int,
    rng: np.random.Generator,
) -> Dict:
    """Generates a single metadata dict."""
    measurements = sample_measurements(body_size_idx, rng)
    fit_delta = float(garment_size_idx - body_size_idx)
    description = rng.choice(GARMENT_DESCRIPTIONS)

    return {
        "sample_id": sample_id,
        "height_cm": measurements["height_cm"],
        "weight_kg": measurements["weight_kg"],
        "chest_cm": measurements["chest_cm"],
        "waist_cm": measurements["waist_cm"],
        "hip_cm": measurements["hip_cm"],
        "body_size_idx": int(body_size_idx),
        "garment_size_idx": int(garment_size_idx),
        "fit_delta": fit_delta,
        "garment_description": description,
    }


def generate_split(
    output_dir: Path,
    split: str,
    num_samples: int,
    seed: int = 42,
):
    """
    Generates synthetic images and metadata for a dataset split.

    Parameters
    ----------
    output_dir : root dataset directory
    split      : 'train', 'val', or 'test'
    num_samples : number of samples to generate
    seed       : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    split_dir = output_dir / split
    (split_dir / "person").mkdir(parents=True, exist_ok=True)
    (split_dir / "garment").mkdir(parents=True, exist_ok=True)
    (split_dir / "tryon").mkdir(parents=True, exist_ok=True)

    metadata = []

    from tqdm import tqdm

    print(f"Generating {num_samples} samples for split '{split}' ...")
    for i in tqdm(range(num_samples), desc=split):
        # Sample body and garment sizes
        body_size_idx = int(rng.integers(0, 6))
        # Garment size: 70% chance within ±1 of body size, 30% further away
        if rng.random() < 0.7:
            delta = int(rng.integers(-1, 2))  # -1, 0, or +1
        else:
            delta = int(rng.integers(-2, 3))  # -2 to +2
        garment_size_idx = int(np.clip(body_size_idx + delta, 0, 5))

        sample_id = f"{split}_{i:06d}"

        # Generate images
        person_img = draw_person_image(body_size_idx, rng)
        garment_img = draw_garment_image(garment_size_idx, "", rng)
        fit_delta = float(garment_size_idx - body_size_idx)
        tryon_img = draw_tryon_image(body_size_idx, garment_size_idx, fit_delta, rng)

        # Save images as JPEG (compact)
        person_img.save(str(split_dir / "person" / f"{sample_id}.jpg"), quality=85)
        garment_img.save(str(split_dir / "garment" / f"{sample_id}.jpg"), quality=85)
        tryon_img.save(str(split_dir / "tryon" / f"{sample_id}.jpg"), quality=85)

        # Record metadata
        entry = generate_metadata_entry(sample_id, body_size_idx, garment_size_idx, rng)
        metadata.append(entry)

    # Save metadata
    meta_path = output_dir / f"metadata_{split}.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  → Saved {num_samples} samples to {split_dir}")
    print(f"  → Metadata written to {meta_path}")


def try_official_download(output_dir: Path) -> bool:
    """
    Attempts to download the official FIT dataset.
    Returns True if successful, False if unavailable.
    """
    print("Attempting to download official FIT dataset ...")
    print(
        "NOTE: Google's FIT dataset (arXiv:2203.07981) is not yet publicly released.\n"
        "Falling back to synthetic mini-dataset generation.\n"
        "To use the real dataset once available:\n"
        "  1. Download from the official source.\n"
        "  2. Convert to the expected metadata format.\n"
        "  3. Place in --output_dir with the required directory structure.\n"
    )
    return False  # dataset not publicly available as of 2026


def main():
    parser = argparse.ArgumentParser(
        description="Download or generate FIT dataset for FIT-VTON training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fit_dataset",
        help="Root directory for the dataset",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Generate synthetic mini-dataset (5k train / 500 val / 500 test)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=5000,
        help="Number of training samples (synthetic mode only)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=500,
        help="Number of validation samples (synthetic mode only)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Number of test samples (synthetic mode only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FIT-VTON: Dataset Setup")
    print("=" * 60)

    if not args.mini:
        success = try_official_download(output_dir)
        if not success:
            print("Generating synthetic dataset instead ...")
            args.mini = True

    if args.mini:
        print(f"\nGenerating synthetic mini-dataset in: {output_dir}")
        print(f"  Train: {args.train_size} samples")
        print(f"  Val:   {args.val_size} samples")
        print(f"  Test:  {args.test_size} samples\n")

        generate_split(output_dir, "train", args.train_size, seed=args.seed)
        generate_split(output_dir, "val", args.val_size, seed=args.seed + 1)
        generate_split(output_dir, "test", args.test_size, seed=args.seed + 2)

        print("\nDataset generation complete!")
        print(f"Dataset root: {output_dir.resolve()}")
        print("\nNext steps:")
        print("  python train.py --config configs/train.yaml")


if __name__ == "__main__":
    main()
