"""
FitVTONPipeline: wraps IDM-VTON (frozen SDXL-based virtual try-on) together
with the MeasurementEncoder and FitAdapter, providing a single inference API.

The pipeline:
  1. Encodes body + garment measurements into fit tokens via MeasurementEncoder.
  2. Injects fit tokens into the UNet cross-attention via FitAdapter.
  3. Runs the standard IDM-VTON denoising loop to produce the try-on image.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from fit_vton.models.measurement_encoder import MeasurementEncoder
from fit_vton.models.fit_adapter import FitAdapter
from fit_vton.data.transforms import denormalize, tensor_to_pil
from fit_vton.data.fit_dataset import SIZE_LABEL_TO_IDX, SIZE_LABELS, MEASUREMENT_MEAN, MEASUREMENT_STD


class FitVTONPipeline(nn.Module):
    """
    Full FIT-VTON inference pipeline.

    Parameters
    ----------
    idm_vton_pipeline : the loaded IDM-VTON diffusers pipeline object
    measurement_encoder : MeasurementEncoder instance
    fit_adapter : FitAdapter instance (already installed into the UNet)
    device : torch.device
    """

    def __init__(
        self,
        idm_vton_pipeline,
        measurement_encoder: MeasurementEncoder,
        fit_adapter: FitAdapter,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.pipe = idm_vton_pipeline
        self.measurement_encoder = measurement_encoder
        self.fit_adapter = fit_adapter
        self.device = device

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        base_model_id: str = "yisol/IDM-VTON",
        adapter_checkpoint: Optional[str] = None,
        encoder_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.float16,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_tokens: int = 4,
        adapter_scale: float = 1.0,
    ) -> "FitVTONPipeline":
        """
        Loads IDM-VTON from HuggingFace and attaches a FitAdapter.

        Parameters
        ----------
        base_model_id : HuggingFace model ID for IDM-VTON
        adapter_checkpoint : optional path to saved FitAdapter weights
        encoder_checkpoint : optional path to saved MeasurementEncoder weights
        device : target device (auto-detected if None)
        torch_dtype : dtype for the base model (fp16 recommended)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from diffusers import AutoPipelineForInpainting
        except ImportError:
            raise ImportError("diffusers is required. Install with: pip install diffusers")

        print(f"[FitVTONPipeline] Loading IDM-VTON from {base_model_id} ...")
        pipe = AutoPipelineForInpainting.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
        )
        pipe = pipe.to(device)
        pipe.unet.eval()

        # Determine cross_attention_dim from the UNet config
        cross_attention_dim = getattr(
            pipe.unet.config, "cross_attention_dim", 2048
        )
        # For SDXL it can be a list; take the last (largest)
        if isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = max(cross_attention_dim)

        print(f"[FitVTONPipeline] cross_attention_dim = {cross_attention_dim}")

        # Build MeasurementEncoder
        # embed_dim must match cross_attention_dim for the IP-Adapter injection
        encoder = MeasurementEncoder(
            embed_dim=cross_attention_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
        ).to(device)

        # Build FitAdapter and install into frozen UNet
        adapter = FitAdapter(
            unet=pipe.unet,
            cross_attention_dim=cross_attention_dim,
            adapter_scale=adapter_scale,
        )
        adapter.install()

        # Load checkpoints if provided
        if adapter_checkpoint is not None:
            print(f"[FitVTONPipeline] Loading adapter weights from {adapter_checkpoint}")
            adapter.load_adapter(adapter_checkpoint)

        if encoder_checkpoint is not None:
            print(f"[FitVTONPipeline] Loading encoder weights from {encoder_checkpoint}")
            ckpt = torch.load(encoder_checkpoint, map_location="cpu")
            encoder.load_state_dict(ckpt)
        encoder = encoder.to(device)

        return cls(
            idm_vton_pipeline=pipe,
            measurement_encoder=encoder,
            fit_adapter=adapter,
            device=device,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_size(size: Union[str, int]) -> int:
        """Converts 'M', 'L', etc. to integer index 0–5."""
        if isinstance(size, str):
            size = size.strip().upper()
            if size not in SIZE_LABEL_TO_IDX:
                raise ValueError(f"Unknown size '{size}'. Valid: {SIZE_LABELS}")
            return SIZE_LABEL_TO_IDX[size]
        return int(size)

    @staticmethod
    def _normalize_measurements(
        height_cm: float,
        weight_kg: float,
        chest_cm: float,
        waist_cm: float,
        hip_cm: float,
    ) -> torch.Tensor:
        """Returns a normalised (5,) float tensor."""
        raw = torch.tensor(
            [height_cm, weight_kg, chest_cm, waist_cm, hip_cm],
            dtype=torch.float32,
        )
        return (raw - MEASUREMENT_MEAN) / MEASUREMENT_STD

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        # Body measurements
        body_height: float = 170.0,
        body_weight: float = 70.0,
        body_chest: float = 92.0,
        body_waist: float = 78.0,
        body_hip: float = 98.0,
        body_size: Union[str, int] = "M",
        garment_size: Union[str, int] = "M",
        # Diffusion parameters
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        # Adapter control
        adapter_scale: Optional[float] = None,
    ) -> Image.Image:
        """
        Runs FIT-VTON inference for a single person + garment pair.

        Parameters
        ----------
        person_image : PIL Image of the person
        garment_image : PIL Image of the garment (on white background)
        body_* : body measurements
        body_size : body size label ('XS' … 'XXL') or int 0–5
        garment_size : garment size label or int 0–5
        num_inference_steps : denoising steps
        guidance_scale : classifier-free guidance scale
        seed : random seed for reproducibility
        adapter_scale : overrides self.fit_adapter.adapter_scale if given

        Returns
        -------
        PIL Image of the person wearing the garment
        """
        if adapter_scale is not None:
            self.fit_adapter.set_scale(adapter_scale)

        # ── Encode measurements ───────────────────────────────────────────────
        body_meas = self._normalize_measurements(
            body_height, body_weight, body_chest, body_waist, body_hip
        ).unsqueeze(0).to(self.device)

        body_size_idx = self._parse_size(body_size)
        garment_size_idx_val = self._parse_size(garment_size)
        fit_delta_val = float(garment_size_idx_val - body_size_idx)

        garment_size_tensor = torch.tensor([garment_size_idx_val], dtype=torch.long).to(self.device)
        fit_delta_tensor = torch.tensor([[fit_delta_val]], dtype=torch.float32).to(self.device)

        fit_tokens = self.measurement_encoder(body_meas, garment_size_tensor, fit_delta_tensor)
        # fit_tokens: (1, num_tokens, embed_dim)

        # ── Inject fit tokens ─────────────────────────────────────────────────
        self.fit_adapter.set_fit_tokens(fit_tokens)

        # ── Set up generator ─────────────────────────────────────────────────
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # ── Run IDM-VTON pipeline ─────────────────────────────────────────────
        # IDM-VTON expects: image (person), mask_image, ip_adapter_image (garment)
        # We create a dummy full-body mask for simplicity; in practice DensePose
        # would provide a precise mask.
        mask_image = Image.new("L", person_image.size, 255)  # full mask

        result = self.pipe(
            prompt="a person wearing the garment, high quality, detailed",
            negative_prompt="deformed, ugly, blurry, low quality",
            image=person_image.resize((512, 512)),
            mask_image=mask_image.resize((512, 512)),
            ip_adapter_image=garment_image.resize((512, 512)),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # ── Clear fit tokens ──────────────────────────────────────────────────
        self.fit_adapter.clear_fit_tokens()

        return result.images[0]

    # ------------------------------------------------------------------
    def save_pretrained(self, output_dir: str):
        """
        Saves the adapter + encoder weights to output_dir.
        The frozen IDM-VTON backbone is NOT saved (too large).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.fit_adapter.save_adapter(str(output_dir / "fit_adapter.pt"))
        torch.save(
            self.measurement_encoder.state_dict(),
            str(output_dir / "measurement_encoder.pt"),
        )
        print(f"[FitVTONPipeline] Saved adapter to {output_dir}")

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only the trainable (adapter + encoder) parameters."""
        return (
            list(self.measurement_encoder.parameters())
            + list(self.fit_adapter.attn_processors.parameters())
        )

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters)
