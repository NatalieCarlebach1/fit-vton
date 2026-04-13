"""
FIT-VTON Evaluation Script.

Evaluates both IDM-VTON baseline and FIT-VTON on a test set,
then prints a comparison table.

Usage
-----
  python evaluate.py \\
      --checkpoint outputs/checkpoints/step_50000 \\
      --data_dir data/vitonhd \\
      --output_dir outputs/eval
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FIT-VTON vs IDM-VTON baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FitAdapter checkpoint directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/vitonhd",
        help="Test dataset directory (VITON-HD format or FIT dataset format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="yisol/IDM-VTON",
        help="HuggingFace model ID for IDM-VTON base.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Denoising steps per sample.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_fit_dataset(
    data_dir: str,
    split: str,
    num_samples: Optional[int] = None,
) -> "torch.utils.data.DataLoader":
    """Loads the evaluation dataset."""
    from fit_vton.data.fit_dataset import FITDataset, FITDatasetConfig, build_dataloader

    cfg = FITDatasetConfig(
        data_dir=data_dir,
        split=split,
        image_size=512,
        use_augmentation=False,
        max_samples=num_samples,
    )
    return build_dataloader(cfg, batch_size=1, num_workers=2, shuffle=False, pin_memory=False)


@torch.no_grad()
def run_inference_batch(
    pipeline,
    batch: Dict,
    num_steps: int,
    device: torch.device,
    use_fit_conditioning: bool = True,
) -> torch.Tensor:
    """
    Runs inference for one batch and returns predicted images in [0, 1].

    Returns
    -------
    FloatTensor (B, 3, 512, 512)
    """
    from fit_vton.data.transforms import denormalize

    B = batch["person_image"].shape[0]
    results = []

    for i in range(B):
        person_img_t = batch["person_image"][i]
        garment_img_t = batch["garment_image"][i]

        # Convert normalised tensors to PIL
        person_pil = Image.fromarray(
            (denormalize(person_img_t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        garment_pil = Image.fromarray(
            (denormalize(garment_img_t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        body_meas = batch["body_measurements"][i].cpu().numpy()
        garment_size_idx = int(batch["garment_size_idx"][i].item())
        fit_delta = float(batch["fit_delta"][i].item())

        size_labels = ["XS", "S", "M", "L", "XL", "XXL"]
        garment_size = size_labels[garment_size_idx]
        # Body size = garment_size - fit_delta (approx.)
        body_size_idx = int(np.clip(garment_size_idx - round(fit_delta), 0, 5))
        body_size = size_labels[body_size_idx]

        # Denormalize measurements
        from fit_vton.data.fit_dataset import MEASUREMENT_MEAN, MEASUREMENT_STD
        raw = body_meas * MEASUREMENT_STD.numpy() + MEASUREMENT_MEAN.numpy()
        height, weight, chest, waist, hip = raw

        adapter_scale = 1.0 if use_fit_conditioning else 0.0

        result_pil = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            body_height=float(height),
            body_weight=float(weight),
            body_chest=float(chest),
            body_waist=float(waist),
            body_hip=float(hip),
            body_size=body_size,
            garment_size=garment_size,
            num_inference_steps=num_steps,
            seed=42,
            adapter_scale=adapter_scale,
        )

        # Convert PIL back to tensor
        result_t = torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0)
        result_t = result_t.permute(2, 0, 1)  # (3, H, W)
        results.append(result_t)

    return torch.stack(results, dim=0)


def save_comparison_images(
    person_images: torch.Tensor,
    garment_images: torch.Tensor,
    ground_truth: torch.Tensor,
    baseline_preds: torch.Tensor,
    fit_vton_preds: torch.Tensor,
    output_dir: Path,
    start_idx: int,
):
    """Saves side-by-side comparison images."""
    from fit_vton.data.transforms import denormalize

    os.makedirs(str(output_dir), exist_ok=True)

    B = person_images.shape[0]
    for i in range(B):
        idx = start_idx + i

        # Convert all to PIL
        def to_pil(t):
            arr = (denormalize(t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        def pred_to_pil(t):
            arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        imgs = [
            to_pil(person_images[i]),
            to_pil(garment_images[i]),
            to_pil(ground_truth[i]),
            pred_to_pil(baseline_preds[i]),
            pred_to_pil(fit_vton_preds[i]),
        ]

        W, H = imgs[0].size
        labels = ["Person", "Garment", "Ground Truth", "IDM-VTON", "FIT-VTON (ours)"]
        from PIL import ImageDraw
        composite = Image.new("RGB", (W * 5 + 10 * 4, H + 25), (255, 255, 255))
        for j, (img, label) in enumerate(zip(imgs, labels)):
            composite.paste(img, (j * (W + 10), 25))
            draw = ImageDraw.Draw(composite)
            draw.text((j * (W + 10) + 5, 5), label, fill=(0, 0, 0))

        composite.save(str(output_dir / f"comparison_{idx:05d}.jpg"), quality=85)


def main():
    args = parse_args()

    import torch
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    real_dir = output_dir / "real"
    baseline_fake_dir = output_dir / "baseline_fake"
    fitvton_fake_dir = output_dir / "fitvton_fake"

    for d in [images_dir, real_dir, baseline_fake_dir, fitvton_fake_dir]:
        d.mkdir(exist_ok=True)

    print("=" * 60)
    print("FIT-VTON Evaluation")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}")
    print()

    # Load FIT-VTON pipeline
    try:
        from fit_vton.models.pipeline import FitVTONPipeline
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    ckpt_dir = Path(args.checkpoint)
    adapter_ckpt = str(ckpt_dir / "fit_adapter.pt") if (ckpt_dir / "fit_adapter.pt").exists() else None
    encoder_ckpt = str(ckpt_dir / "measurement_encoder.pt") if (ckpt_dir / "measurement_encoder.pt").exists() else None

    pipeline = FitVTONPipeline.from_pretrained(
        base_model_id=args.base_model_id,
        adapter_checkpoint=adapter_ckpt,
        encoder_checkpoint=encoder_ckpt,
        device=device,
    )

    # Load dataset
    dataloader = load_fit_dataset(args.data_dir, args.split, num_samples=args.num_samples)
    print(f"Evaluating on {len(dataloader.dataset)} samples ...")

    # Metrics
    from fit_vton.utils.metrics import MetricsBundle
    baseline_metrics = MetricsBundle()
    fitvton_metrics = MetricsBundle()

    from fit_vton.data.transforms import denormalize

    global_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        ground_truth = denormalize(batch["tryon_image"])  # (B, 3, H, W) in [0,1]
        fit_deltas = batch["fit_delta"]

        # Baseline inference (adapter_scale=0)
        baseline_preds = run_inference_batch(
            pipeline, batch, args.num_inference_steps, device, use_fit_conditioning=False
        )

        # FIT-VTON inference (adapter_scale=1)
        fitvton_preds = run_inference_batch(
            pipeline, batch, args.num_inference_steps, device, use_fit_conditioning=True
        )

        baseline_metrics.update(baseline_preds, ground_truth, fit_deltas)
        fitvton_metrics.update(fitvton_preds, ground_truth, fit_deltas)

        # Save images for FID computation
        fitvton_metrics.save_images_to_dir(
            ground_truth, str(real_dir)
        )
        baseline_metrics.save_images_to_dir(
            baseline_preds, str(baseline_fake_dir)
        )
        fitvton_metrics.save_images_to_dir(
            fitvton_preds, str(fitvton_fake_dir)
        )

        save_comparison_images(
            person_images=batch["person_image"],
            garment_images=batch["garment_image"],
            ground_truth=batch["tryon_image"],
            baseline_preds=baseline_preds,
            fit_vton_preds=fitvton_preds,
            output_dir=images_dir,
            start_idx=global_idx,
        )
        global_idx += batch["person_image"].shape[0]

    # Compute final metrics (including FID)
    print("\nComputing final metrics ...")
    baseline_results = baseline_metrics.compute(
        real_dir=str(real_dir), fake_dir=str(baseline_fake_dir)
    )
    fitvton_results = fitvton_metrics.compute(
        real_dir=str(real_dir), fake_dir=str(fitvton_fake_dir)
    )

    # Print comparison table
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    import json
    print(f"\n{'Metric':<20} {'IDM-VTON (baseline)':>20} {'FIT-VTON (ours)':>20} {'Delta':>10}")
    print("-" * 75)

    metric_labels = {
        "fid": ("FID ↓", True),         # lower is better
        "lpips": ("LPIPS ↓", True),
        "ssim": ("SSIM ↑", False),      # higher is better
        "fit_accuracy": ("Fit Acc ↑", False),
    }

    for key, (label, lower_better) in metric_labels.items():
        b_val = baseline_results.get(key, float("nan"))
        f_val = fitvton_results.get(key, float("nan"))
        if not (np.isnan(b_val) or np.isnan(f_val)):
            delta = f_val - b_val
            # Positive delta for lower-is-better means our model is worse
            better = (delta < 0) if lower_better else (delta > 0)
            delta_str = f"{'↑' if not lower_better and delta > 0 else '↓' if lower_better and delta < 0 else '~'} {abs(delta):.4f}"
        else:
            delta_str = "N/A"
        print(f"{label:<20} {b_val:>20.4f} {f_val:>20.4f} {delta_str:>10}")

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(str(results_path), "w") as f:
        json.dump(
            {
                "baseline": {k: (v if not np.isnan(v) else None) for k, v in baseline_results.items()},
                "fit_vton": {k: (v if not np.isnan(v) else None) for k, v in fitvton_results.items()},
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")
    print(f"Comparison images: {images_dir}")


if __name__ == "__main__":
    main()
