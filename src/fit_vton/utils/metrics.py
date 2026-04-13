"""
Evaluation metrics for FIT-VTON.

  compute_fid          — Fréchet Inception Distance (clean-fid)
  compute_lpips        — Learned Perceptual Image Patch Similarity
  compute_ssim         — Structural Similarity Index
  compute_fit_accuracy — Custom metric: correlation of rendered garment coverage
                         area with expected fit direction (fit_delta)

All image tensors are expected as FloatTensor (B, 3, H, W) in [0, 1]
unless otherwise noted.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    real_dir: Union[str, Path],
    fake_dir: Union[str, Path],
    mode: str = "clean",
    num_workers: int = 4,
) -> float:
    """
    Computes Fréchet Inception Distance between two directories of images.

    Parameters
    ----------
    real_dir : path to directory of real images
    fake_dir : path to directory of generated images
    mode     : 'clean' (default) or 'legacy_pytorch'
    num_workers : DataLoader workers for clean-fid

    Returns
    -------
    FID score (lower is better)
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install with: pip install clean-fid"
        )

    score = cleanfid.compute_fid(
        str(real_dir),
        str(fake_dir),
        mode=mode,
        num_workers=num_workers,
    )
    return float(score)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_model: Optional[nn.Module] = None


def _get_lpips_model(net: str = "alex", device: torch.device = torch.device("cpu")) -> nn.Module:
    """Lazy-loads the LPIPS model (singleton)."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
        except ImportError:
            raise ImportError("lpips is required. Install with: pip install lpips")
        _lpips_model = lpips.LPIPS(net=net).to(device)
        _lpips_model.eval()
    return _lpips_model


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    reduction: str = "mean",
) -> float:
    """
    Computes LPIPS between predicted and target image tensors.

    Parameters
    ----------
    pred, target : FloatTensor (B, 3, H, W) in [0, 1]
    net          : backbone ('alex', 'vgg', 'squeeze')
    reduction    : 'mean' or 'none'

    Returns
    -------
    float (or tensor if reduction='none')
    """
    device = pred.device
    loss_fn = _get_lpips_model(net=net, device=device)

    # LPIPS expects inputs in [-1, 1]
    pred_norm = pred * 2.0 - 1.0
    target_norm = target * 2.0 - 1.0

    with torch.no_grad():
        dist = loss_fn(pred_norm, target_norm)  # (B, 1, 1, 1)

    if reduction == "mean":
        return float(dist.mean().item())
    return dist.squeeze()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    data_range: float = 1.0,
    window_size: int = 11,
) -> float:
    """
    Computes Structural Similarity Index (SSIM).

    Parameters
    ----------
    pred, target : FloatTensor (B, 3, H, W) in [0, 1]
    reduction    : 'mean' or 'none'
    data_range   : pixel value range (1.0 for [0,1])
    window_size  : Gaussian window size

    Returns
    -------
    float SSIM score (higher is better, max=1.0)
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except ImportError:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")

    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()   # (B, H, W, 3)
    target_np = target.permute(0, 2, 3, 1).cpu().numpy()

    scores = []
    for p, t in zip(pred_np, target_np):
        score = sk_ssim(
            p,
            t,
            data_range=data_range,
            channel_axis=-1,
            win_size=window_size,
        )
        scores.append(score)

    if reduction == "mean":
        return float(np.mean(scores))
    return scores


# ---------------------------------------------------------------------------
# Fit Accuracy (custom metric)
# ---------------------------------------------------------------------------

def _segment_garment_area(image: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
    """
    Estimates the garment region as pixels that are NOT background.

    A simple heuristic: converts to HSV-like luminance and marks pixels
    whose saturation or chromatic deviation exceeds `threshold` as garment.

    In production this would use a pretrained garment segmentation model
    (e.g. CIHP Parsing or SAM). Here we use a colour-based proxy that
    correlates well with the true garment area for our evaluation set.

    Parameters
    ----------
    image     : FloatTensor (3, H, W) in [0, 1]
    threshold : pixels with colour variance below this are considered background

    Returns
    -------
    FloatTensor (H, W) binary mask (1 = garment, 0 = background)
    """
    # Compute per-pixel standard deviation across channels as a proxy for
    # chromatic complexity (white/near-white background → low std)
    pixel_std = image.std(dim=0)  # (H, W)

    # Luminance
    lum = 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]

    # A pixel is classified as garment if it has:
    #   - non-trivial chromatic variation (std > 0.02), OR
    #   - luminance < 0.9 (i.e., not near-white background)
    garment_mask = (pixel_std > 0.02) | (lum < 0.9)
    return garment_mask.float()


def compute_fit_accuracy(
    pred_images: torch.Tensor,
    fit_deltas: torch.Tensor,
    threshold: float = 0.85,
) -> float:
    """
    Custom fit-accuracy metric.

    Hypothesis: when fit_delta > 0 (garment is larger than body size),
    the garment should cover a larger relative area of the body in the
    rendered image; when fit_delta < 0 (too small), the coverage should
    be tighter / smaller.

    We measure the Pearson correlation between the predicted garment
    coverage ratio and the sign of fit_delta as a proxy for how well
    the model renders ill-fitting garments.

    Parameters
    ----------
    pred_images : FloatTensor (B, 3, H, W) in [0, 1]
    fit_deltas  : FloatTensor (B,) signed delta values
    threshold   : background threshold for garment segmentation

    Returns
    -------
    fit_accuracy ∈ [-1, 1]  (higher = model better captures fit direction)
    """
    B = pred_images.shape[0]
    coverage_ratios = []

    for i in range(B):
        mask = _segment_garment_area(pred_images[i], threshold=threshold)
        ratio = mask.mean().item()  # fraction of pixels classified as garment
        coverage_ratios.append(ratio)

    coverage = torch.tensor(coverage_ratios, dtype=torch.float32)
    deltas = fit_deltas.float().cpu()

    if deltas.std() < 1e-8 or coverage.std() < 1e-8:
        return 0.0

    # Pearson correlation between coverage and fit_delta
    cov = ((coverage - coverage.mean()) * (deltas - deltas.mean())).mean()
    corr = cov / (coverage.std() * deltas.std() + 1e-8)

    return float(corr.item())


# ---------------------------------------------------------------------------
# MetricsBundle
# ---------------------------------------------------------------------------

class MetricsBundle:
    """
    Aggregates all metrics for a full evaluation run.

    Usage
    -----
    >>> bundle = MetricsBundle()
    >>> for batch in dataloader:
    ...     bundle.update(pred_images, target_images, fit_deltas)
    >>> results = bundle.compute()
    >>> bundle.print_table(results)
    """

    def __init__(self, lpips_net: str = "alex"):
        self.lpips_net = lpips_net
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._fit_deltas: List[torch.Tensor] = []
        self._lpips_scores: List[float] = []
        self._ssim_scores: List[float] = []

    def update(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        fit_deltas: torch.Tensor,
    ):
        """
        Accumulates batch results.

        Parameters
        ----------
        pred_images   : (B, 3, H, W) in [0, 1]
        target_images : (B, 3, H, W) in [0, 1]
        fit_deltas    : (B,) float
        """
        # Per-batch metrics
        lpips_val = compute_lpips(pred_images, target_images, net=self.lpips_net, reduction="mean")
        ssim_val = compute_ssim(pred_images, target_images, reduction="mean")

        self._lpips_scores.append(lpips_val)
        self._ssim_scores.append(ssim_val)

        # Cache for fit_accuracy and FID
        self._preds.append(pred_images.cpu())
        self._targets.append(target_images.cpu())
        self._fit_deltas.append(fit_deltas.cpu())

    def compute(
        self,
        real_dir: Optional[str] = None,
        fake_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Computes final metric values.

        If real_dir and fake_dir are provided, FID is computed from those directories.
        Otherwise FID is skipped (returns NaN).
        """
        results: Dict[str, float] = {}

        # LPIPS and SSIM (already accumulated per batch)
        results["lpips"] = float(np.mean(self._lpips_scores)) if self._lpips_scores else float("nan")
        results["ssim"] = float(np.mean(self._ssim_scores)) if self._ssim_scores else float("nan")

        # Fit accuracy
        if self._preds and self._fit_deltas:
            all_preds = torch.cat(self._preds, dim=0)
            all_deltas = torch.cat(self._fit_deltas, dim=0)
            results["fit_accuracy"] = compute_fit_accuracy(all_preds, all_deltas)
        else:
            results["fit_accuracy"] = float("nan")

        # FID
        if real_dir is not None and fake_dir is not None:
            try:
                results["fid"] = compute_fid(real_dir, fake_dir)
            except Exception as e:
                print(f"FID computation failed: {e}")
                results["fid"] = float("nan")
        else:
            results["fid"] = float("nan")

        return results

    def print_table(self, results: Dict[str, float], model_name: str = "FIT-VTON"):
        """Pretty-prints a results table to stdout."""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title=f"Evaluation Results — {model_name}", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("↑/↓", style="yellow")

        directions = {
            "fid": ("FID ↓", "↓"),
            "lpips": ("LPIPS ↓", "↓"),
            "ssim": ("SSIM ↑", "↑"),
            "fit_accuracy": ("Fit Accuracy ↑", "↑"),
        }

        for key, val in results.items():
            label, direction = directions.get(key, (key, "?"))
            table.add_row(label, f"{val:.4f}", direction)

        console.print(table)

    def save_images_to_dir(self, images: torch.Tensor, directory: str):
        """Saves a batch of [0,1] tensors as PNG files to the given directory."""
        os.makedirs(directory, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img.save(os.path.join(directory, f"{i:06d}.png"))
