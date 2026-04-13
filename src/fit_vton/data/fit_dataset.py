"""
PyTorch Dataset for FIT triplets + body/garment measurements.

The FIT dataset provides image triplets (person, garment, tryon) alongside
per-sample body measurements and garment size metadata.

Directory layout expected:
  data_dir/
    {split}/
      person/      *.jpg or *.png
      garment/     *.jpg or *.png
      tryon/       *.jpg or *.png
    metadata_{split}.json   OR   {split}/metadata.json

metadata.json schema (list of dicts):
  {
    "sample_id":          str,
    "height_cm":          float,
    "weight_kg":          float,
    "chest_cm":           float,
    "waist_cm":           float,
    "hip_cm":             float,
    "body_size_idx":      int (0=XS, 1=S, 2=M, 3=L, 4=XL, 5=XXL),
    "garment_size_idx":   int (0=XS, …5=XXL),
    "fit_delta":          float  (garment_size_idx - body_size_idx),
    "garment_description": str
  }
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fit_vton.data.transforms import Compose, get_train_transforms, get_val_transforms


# Size label helpers
SIZE_LABELS = ["XS", "S", "M", "L", "XL", "XXL"]
SIZE_LABEL_TO_IDX = {s: i for i, s in enumerate(SIZE_LABELS)}

# Body measurement normalisation statistics (approximate real-world values)
MEASUREMENT_MEAN = torch.tensor([170.0, 70.0, 92.0, 78.0, 98.0])  # cm, kg, cm, cm, cm
MEASUREMENT_STD = torch.tensor([10.0, 15.0, 8.0, 10.0, 8.0])


@dataclass
class FITDatasetConfig:
    data_dir: str = "data/fit_dataset"
    split: str = "train"
    image_size: int = 512
    normalize_measurements: bool = True
    use_augmentation: bool = True  # only applied when split == "train"
    max_samples: Optional[int] = None  # cap dataset size for debugging


class FITDataset(Dataset):
    """
    PyTorch Dataset for FIT virtual try-on triplets with measurement conditioning.

    Each item contains:
      - person_image:      FloatTensor (3, H, W) in [-1, 1]
      - garment_image:     FloatTensor (3, H, W) in [-1, 1]
      - tryon_image:       FloatTensor (3, H, W) in [-1, 1]  (ground-truth result)
      - body_measurements: FloatTensor (5,)  [height, weight, chest, waist, hip]
                           normalised if config.normalize_measurements=True
      - garment_size_idx:  LongTensor scalar  0–5
      - fit_delta:         FloatTensor scalar  (garment - body size, signed)
      - sample_id:         str
    """

    def __init__(self, config: FITDatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.split = config.split

        self.metadata = self._load_metadata()

        if config.max_samples is not None:
            self.metadata = self.metadata[: config.max_samples]

        if config.use_augmentation and config.split == "train":
            self.transform = get_train_transforms(config.image_size)
        else:
            self.transform = get_val_transforms(config.image_size)

        self._check_image_dirs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> List[Dict]:
        """Tries several candidate paths for the metadata JSON."""
        candidates = [
            self.data_dir / f"metadata_{self.split}.json",
            self.data_dir / self.split / "metadata.json",
            self.data_dir / f"{self.split}.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "samples" in data:
                    return data["samples"]
                raise ValueError(f"Unexpected metadata format in {path}")
        raise FileNotFoundError(
            f"Could not find metadata for split '{self.split}' in {self.data_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _check_image_dirs(self):
        """Validates that image subdirectories exist."""
        split_dir = self.data_dir / self.split
        for sub in ("person", "garment", "tryon"):
            d = split_dir / sub
            if not d.exists():
                raise FileNotFoundError(
                    f"Image directory not found: {d}. "
                    "Run scripts/download_fit_dataset.py to generate the dataset."
                )

    def _load_image(self, path: Path) -> Image.Image:
        """Loads an image as RGB PIL."""
        if not path.exists():
            # Try common extensions
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                alt = path.with_suffix(ext)
                if alt.exists():
                    return Image.open(alt).convert("RGB")
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def _resolve_image_path(self, sample_id: str, subdir: str) -> Path:
        """Resolves the full path for a sample image."""
        base = self.data_dir / self.split / subdir / sample_id
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = base.with_suffix(ext) if not base.suffix else base
            if ext and not str(p).endswith(ext):
                p = Path(str(p) + ext) if not p.suffix else p.with_suffix(ext)
            p_direct = self.data_dir / self.split / subdir / f"{sample_id}{ext}"
            if p_direct.exists():
                return p_direct
        # Return without extension; _load_image will try alternatives
        return self.data_dir / self.split / subdir / sample_id

    @staticmethod
    def _encode_measurements(
        meta: Dict,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes measurements from a metadata dict.

        Returns:
          body_measurements: (5,) float tensor
          garment_size_idx:  scalar long tensor
          fit_delta:         scalar float tensor
        """
        raw = torch.tensor(
            [
                float(meta["height_cm"]),
                float(meta["weight_kg"]),
                float(meta["chest_cm"]),
                float(meta["waist_cm"]),
                float(meta["hip_cm"]),
            ],
            dtype=torch.float32,
        )
        if normalize:
            raw = (raw - MEASUREMENT_MEAN) / MEASUREMENT_STD

        garment_size_idx = torch.tensor(int(meta["garment_size_idx"]), dtype=torch.long)
        fit_delta = torch.tensor(float(meta["fit_delta"]), dtype=torch.float32)

        return raw, garment_size_idx, fit_delta

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.metadata[idx]
        sample_id = meta["sample_id"]

        # Load images
        person_img = self._load_image(self._resolve_image_path(sample_id, "person"))
        garment_img = self._load_image(self._resolve_image_path(sample_id, "garment"))
        tryon_img = self._load_image(self._resolve_image_path(sample_id, "tryon"))

        sample = {
            "person_image": person_img,
            "garment_image": garment_img,
            "tryon_image": tryon_img,
        }

        # Apply transforms (resize, normalise, augment)
        sample = self.transform(sample)

        # Encode measurements
        body_meas, garment_size_idx, fit_delta = self._encode_measurements(
            meta, normalize=self.config.normalize_measurements
        )

        sample["body_measurements"] = body_meas
        sample["garment_size_idx"] = garment_size_idx
        sample["fit_delta"] = fit_delta
        sample["sample_id"] = sample_id
        sample["garment_description"] = meta.get("garment_description", "")

        return sample

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate that handles string fields."""
        keys = batch[0].keys()
        collated: Dict = {}
        for key in keys:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                collated[key] = torch.stack(vals)
            else:
                collated[key] = vals  # strings
        return collated


def build_dataloader(
    config: FITDatasetConfig,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
) -> torch.utils.data.DataLoader:
    """Convenience factory for creating a DataLoader from a FITDatasetConfig."""
    dataset = FITDataset(config)
    _shuffle = (config.split == "train") if shuffle is None else shuffle
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=FITDataset.collate_fn,
        drop_last=(config.split == "train"),
    )
