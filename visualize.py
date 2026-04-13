"""
FIT-VTON Visualization Script.

Creates a grid image: each row shows [person | garment | IDM-VTON baseline | FIT-VTON (ours)]
across different fit deltas.

Usage
-----
  python visualize.py \\
      --checkpoint outputs/checkpoints/step_50000 \\
      --data_dir data/fit_dataset/test \\
      --n_samples 16 \\
      --output outputs/grid.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise FIT-VTON results as a comparison grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory. If None, uses untrained adapter.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/fit_dataset",
        help="Root of the FIT dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to visualise.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of samples to include in the grid.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/grid.png",
        help="Path to save the output grid image.",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="yisol/IDM-VTON",
        help="HuggingFace model ID for IDM-VTON base.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Denoising steps for each inference call.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of each cell in the grid (pixels).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def add_label(
    img: Image.Image,
    label: str,
    font_size: int = 14,
    bg_color: Tuple = (30, 30, 30),
    text_color: Tuple = (255, 255, 255),
) -> Image.Image:
    """Adds a text label bar at the bottom of an image."""
    W, H = img.size
    label_h = font_size + 8
    composite = Image.new("RGB", (W, H + label_h), bg_color)
    composite.paste(img, (0, 0))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Centre the text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = (W - tw) // 2
    draw.text((tx, H + 4), label, fill=text_color, font=font)

    return composite


def build_grid(
    rows: List[List[Image.Image]],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cell_size: int = 256,
    border: int = 3,
    header_height: int = 30,
) -> Image.Image:
    """
    Assembles a list of lists of PIL images into a single grid image.

    Parameters
    ----------
    rows         : list of row lists (each row is a list of PIL images)
    row_labels   : optional label for each row (shown on the left)
    col_labels   : optional column header labels
    cell_size    : resize each cell to this square size
    border       : pixel gap between cells
    header_height: height of column label header row

    Returns
    -------
    PIL Image
    """
    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0

    row_label_w = 80 if row_labels else 0
    col_header_h = header_height if col_labels else 0

    grid_w = row_label_w + num_cols * (cell_size + border) + border
    grid_h = col_header_h + num_rows * (cell_size + border) + border

    canvas = Image.new("RGB", (grid_w, grid_h), (50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_sm = font

    # Column headers
    if col_labels:
        for j, label in enumerate(col_labels):
            x = row_label_w + j * (cell_size + border) + border + cell_size // 2
            y = col_header_h // 2
            draw.text((x - 40, y - 8), label, fill=(220, 220, 220), font=font_sm)

    # Rows
    for i, row in enumerate(rows):
        y_offset = col_header_h + i * (cell_size + border) + border

        # Row label
        if row_labels and i < len(row_labels):
            draw.text((3, y_offset + cell_size // 2 - 8), row_labels[i],
                      fill=(200, 200, 200), font=font_sm)

        for j, img in enumerate(row):
            x_offset = row_label_w + j * (cell_size + border) + border
            cell = img.resize((cell_size, cell_size), Image.LANCZOS)
            canvas.paste(cell, (x_offset, y_offset))

    return canvas


@torch.no_grad()
def generate_row(
    pipeline,
    person_img: Image.Image,
    garment_img: Image.Image,
    body_height: float,
    body_weight: float,
    body_chest: float,
    body_waist: float,
    body_hip: float,
    body_size: str,
    garment_size: str,
    num_steps: int,
    seed: int,
) -> Tuple[Image.Image, Image.Image]:
    """Generates baseline and FIT-VTON results for one sample."""
    baseline = pipeline(
        person_image=person_img,
        garment_image=garment_img,
        body_height=body_height,
        body_weight=body_weight,
        body_chest=body_chest,
        body_waist=body_waist,
        body_hip=body_hip,
        body_size=body_size,
        garment_size=garment_size,
        num_inference_steps=num_steps,
        seed=seed,
        adapter_scale=0.0,  # baseline: no fit conditioning
    )
    fit_vton = pipeline(
        person_image=person_img,
        garment_image=garment_img,
        body_height=body_height,
        body_weight=body_weight,
        body_chest=body_chest,
        body_waist=body_waist,
        body_hip=body_hip,
        body_size=body_size,
        garment_size=garment_size,
        num_inference_steps=num_steps,
        seed=seed,
        adapter_scale=1.0,  # full fit conditioning
    )
    return baseline, fit_vton


def main():
    args = parse_args()

    import torch
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("FIT-VTON Visualisation")
    print("=" * 60)
    print(f"Data dir   : {args.data_dir}")
    print(f"Split      : {args.split}")
    print(f"N samples  : {args.n_samples}")
    print(f"Output     : {args.output}")
    print()

    # Load dataset
    try:
        from fit_vton.data.fit_dataset import FITDataset, FITDatasetConfig
        from fit_vton.data.transforms import denormalize
        from fit_vton.data.fit_dataset import MEASUREMENT_MEAN, MEASUREMENT_STD, SIZE_LABELS
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    cfg = FITDatasetConfig(
        data_dir=args.data_dir,
        split=args.split,
        image_size=512,
        use_augmentation=False,
        max_samples=args.n_samples,
    )
    dataset = FITDataset(cfg)
    n = min(args.n_samples, len(dataset))
    print(f"Loaded {n} samples from dataset.")

    # Load pipeline
    try:
        from fit_vton.models.pipeline import FitVTONPipeline
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    adapter_ckpt = None
    encoder_ckpt = None
    if args.checkpoint is not None:
        ckpt_dir = Path(args.checkpoint)
        if (ckpt_dir / "fit_adapter.pt").exists():
            adapter_ckpt = str(ckpt_dir / "fit_adapter.pt")
        if (ckpt_dir / "measurement_encoder.pt").exists():
            encoder_ckpt = str(ckpt_dir / "measurement_encoder.pt")

    pipeline = FitVTONPipeline.from_pretrained(
        base_model_id=args.base_model_id,
        adapter_checkpoint=adapter_ckpt,
        encoder_checkpoint=encoder_ckpt,
        device=device,
    )

    # Generate grid rows
    rows = []
    row_labels = []

    from tqdm import tqdm
    for idx in tqdm(range(n), desc="Generating visualisations"):
        sample = dataset[idx]

        # Convert tensors to PIL
        def t_to_pil(t):
            arr = (denormalize(t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        person_pil = t_to_pil(sample["person_image"])
        garment_pil = t_to_pil(sample["garment_image"])
        tryon_pil = t_to_pil(sample["tryon_image"])

        # Decode measurements
        body_meas_np = sample["body_measurements"].numpy()
        raw = body_meas_np * MEASUREMENT_STD.numpy() + MEASUREMENT_MEAN.numpy()
        height, weight, chest, waist, hip = raw

        garment_size_idx = int(sample["garment_size_idx"].item())
        fit_delta = float(sample["fit_delta"].item())
        body_size_idx = int(np.clip(garment_size_idx - round(fit_delta), 0, 5))

        body_size = SIZE_LABELS[body_size_idx]
        garment_size = SIZE_LABELS[garment_size_idx]

        baseline, fit_vton_result = generate_row(
            pipeline=pipeline,
            person_img=person_pil,
            garment_img=garment_pil,
            body_height=float(height),
            body_weight=float(weight),
            body_chest=float(chest),
            body_waist=float(waist),
            body_hip=float(hip),
            body_size=body_size,
            garment_size=garment_size,
            num_steps=args.num_inference_steps,
            seed=args.seed + idx,
        )

        rows.append([person_pil, garment_pil, tryon_pil, baseline, fit_vton_result])

        delta_str = f"Δ={fit_delta:+.0f} ({body_size}→{garment_size})"
        row_labels.append(delta_str)

    # Build grid
    col_labels = ["Person", "Garment", "Ground Truth", "IDM-VTON", "FIT-VTON (ours)"]

    print("\nBuilding grid image ...")
    grid = build_grid(
        rows=rows,
        row_labels=row_labels,
        col_labels=col_labels,
        cell_size=args.image_size,
        border=4,
        header_height=35,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))

    print(f"Grid saved to: {output_path.resolve()}")
    print(f"Grid size: {grid.size[0]}×{grid.size[1]} px, {n} rows × 5 columns")


if __name__ == "__main__":
    main()
