"""
Image augmentation transforms for FIT-VTON training and validation.
"""

import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class PairedRandomHorizontalFlip:
    """Applies the same random horizontal flip to person, garment, and tryon images."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            for key in ["person_image", "garment_image", "tryon_image"]:
                if key in sample and sample[key] is not None:
                    sample[key] = TF.hflip(sample[key])
        return sample


class PairedRandomCrop:
    """Applies the same random crop to all images in the sample."""

    def __init__(self, size: int, pad: int = 20):
        self.size = size
        self.pad = pad

    def __call__(self, sample: Dict) -> Dict:
        # Pad first
        padding = self.pad
        padded_size = self.size + 2 * padding

        # Get crop parameters from the person image
        person = sample.get("person_image")
        if person is None:
            return sample

        w, h = person.size if hasattr(person, "size") else (person.shape[-1], person.shape[-2])

        # Resize to padded_size first
        for key in ["person_image", "garment_image", "tryon_image"]:
            if key in sample and sample[key] is not None:
                img = sample[key]
                if isinstance(img, Image.Image):
                    sample[key] = img.resize((padded_size, padded_size), Image.BILINEAR)
                else:
                    sample[key] = TF.resize(img, [padded_size, padded_size])

        # Random crop params
        i = random.randint(0, 2 * padding)
        j = random.randint(0, 2 * padding)

        for key in ["person_image", "garment_image", "tryon_image"]:
            if key in sample and sample[key] is not None:
                img = sample[key]
                if isinstance(img, Image.Image):
                    sample[key] = TF.crop(img, i, j, self.size, self.size)
                else:
                    sample[key] = TF.crop(img, i, j, self.size, self.size)

        return sample


class PairedColorJitter:
    """Applies the same color jitter to person and tryon images (not garment)."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        p: float = 0.5,
    ):
        self.p = p
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            fn = self.jitter
            for key in ["person_image", "tryon_image"]:
                if key in sample and sample[key] is not None:
                    sample[key] = fn(sample[key])
        return sample


class ToTensor:
    """Converts PIL images to normalized tensors in [-1, 1]."""

    def __init__(self, image_size: int = 512):
        self.resize = T.Resize((image_size, image_size), antialias=True)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, sample: Dict) -> Dict:
        for key in ["person_image", "garment_image", "tryon_image"]:
            if key in sample and sample[key] is not None:
                img = sample[key]
                if isinstance(img, Image.Image):
                    img = img.convert("RGB")
                    img = self.resize(img)
                    img = self.to_tensor(img)
                    img = self.normalize(img)
                    sample[key] = img
        return sample


class Compose:
    """Composes multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transforms(image_size: int = 512) -> Compose:
    """Returns the full augmentation pipeline for training."""
    return Compose([
        PairedRandomHorizontalFlip(p=0.5),
        PairedColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.3,
        ),
        ToTensor(image_size=image_size),
    ])


def get_val_transforms(image_size: int = 512) -> Compose:
    """Returns deterministic transforms for validation/test."""
    return Compose([
        ToTensor(image_size=image_size),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a [-1, 1] tensor back to [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a CHW tensor in [-1, 1] to a PIL Image."""
    img = denormalize(tensor).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)
