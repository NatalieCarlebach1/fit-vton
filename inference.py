"""
FIT-VTON Single-Sample Inference.

Usage
-----
  python inference.py \\
      --person_image path/to/person.jpg \\
      --garment_image path/to/garment.jpg \\
      --body_height 170 \\
      --body_weight 65 \\
      --body_chest 92 \\
      --body_waist 74 \\
      --body_hip 99 \\
      --body_size M \\
      --garment_size L \\
      --checkpoint outputs/checkpoints/step_50000 \\
      --output result.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="FIT-VTON Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input images
    parser.add_argument(
        "--person_image",
        type=str,
        required=True,
        help="Path to the person image (JPEG or PNG).",
    )
    parser.add_argument(
        "--garment_image",
        type=str,
        required=True,
        help="Path to the garment image on white background.",
    )

    # Body measurements
    parser.add_argument("--body_height", type=float, default=170.0, help="Height in cm.")
    parser.add_argument("--body_weight", type=float, default=70.0, help="Weight in kg.")
    parser.add_argument("--body_chest", type=float, default=92.0, help="Chest circumference in cm.")
    parser.add_argument("--body_waist", type=float, default=78.0, help="Waist circumference in cm.")
    parser.add_argument("--body_hip", type=float, default=98.0, help="Hip circumference in cm.")

    # Size labels
    parser.add_argument(
        "--body_size",
        type=str,
        default="M",
        choices=["XS", "S", "M", "L", "XL", "XXL"],
        help="Body clothing size.",
    )
    parser.add_argument(
        "--garment_size",
        type=str,
        default="M",
        choices=["XS", "S", "M", "L", "XL", "XXL"],
        help="Garment size label.",
    )

    # Model
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="yisol/IDM-VTON",
        help="HuggingFace model ID for the base IDM-VTON model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to FitAdapter checkpoint directory. If None, uses random adapter weights.",
    )

    # Diffusion settings
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--adapter_scale",
        type=float,
        default=1.0,
        help="Adapter conditioning strength (0=baseline IDM-VTON, 1=full FIT conditioning).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Output
    parser.add_argument("--output", type=str, default="result.png", help="Output image path.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu'). Auto-detected if not specified.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Validate inputs ---
    person_path = Path(args.person_image)
    garment_path = Path(args.garment_image)

    if not person_path.exists():
        print(f"ERROR: Person image not found: {person_path}")
        sys.exit(1)
    if not garment_path.exists():
        print(f"ERROR: Garment image not found: {garment_path}")
        sys.exit(1)

    # --- Load images ---
    person_image = Image.open(str(person_path)).convert("RGB")
    garment_image = Image.open(str(garment_path)).convert("RGB")

    # --- Determine checkpoint paths ---
    adapter_checkpoint = None
    encoder_checkpoint = None
    if args.checkpoint is not None:
        ckpt_dir = Path(args.checkpoint)
        adapter_ckpt = ckpt_dir / "fit_adapter.pt"
        encoder_ckpt = ckpt_dir / "measurement_encoder.pt"
        if adapter_ckpt.exists():
            adapter_checkpoint = str(adapter_ckpt)
        if encoder_ckpt.exists():
            encoder_checkpoint = str(encoder_ckpt)

    # --- Determine device ---
    import torch
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("FIT-VTON Inference")
    print("=" * 60)
    print(f"Person image  : {person_path}")
    print(f"Garment image : {garment_path}")
    print(f"Body size     : {args.body_size} ({args.body_height}cm, {args.body_weight}kg)")
    print(f"  Chest/Waist/Hip: {args.body_chest}/{args.body_waist}/{args.body_hip} cm")
    print(f"Garment size  : {args.garment_size}")
    fit_delta = (
        ["XS", "S", "M", "L", "XL", "XXL"].index(args.garment_size)
        - ["XS", "S", "M", "L", "XL", "XXL"].index(args.body_size)
    )
    print(f"Fit delta     : {fit_delta:+d} sizes ({'baggy' if fit_delta > 0 else 'tight' if fit_delta < 0 else 'perfect fit'})")
    print(f"Device        : {device}")
    print(f"Output        : {args.output}")
    print()

    # --- Load pipeline ---
    try:
        from fit_vton.models.pipeline import FitVTONPipeline
    except ImportError as e:
        print(f"ERROR: Could not import FitVTONPipeline: {e}")
        print("Make sure the package is installed: pip install -e .")
        sys.exit(1)

    pipeline = FitVTONPipeline.from_pretrained(
        base_model_id=args.base_model_id,
        adapter_checkpoint=adapter_checkpoint,
        encoder_checkpoint=encoder_checkpoint,
        device=device,
        adapter_scale=args.adapter_scale,
    )

    print(f"Trainable parameters: {pipeline.num_trainable_parameters:,}")
    print()

    # --- Run inference ---
    print("Running inference ...")
    result_image = pipeline(
        person_image=person_image,
        garment_image=garment_image,
        body_height=args.body_height,
        body_weight=args.body_weight,
        body_chest=args.body_chest,
        body_waist=args.body_waist,
        body_hip=args.body_hip,
        body_size=args.body_size,
        garment_size=args.garment_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        adapter_scale=args.adapter_scale,
    )

    # --- Save result ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(str(output_path))
    print(f"\nResult saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
