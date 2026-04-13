"""
MeasurementEncoder: maps body + garment measurements to a sequence of
cross-attention tokens that can be injected into the UNet via FitAdapter.

Architecture
------------
  body_encoder:    Linear(5) → SiLU → Linear(hidden_dim) → SiLU → Linear(embed_dim)
  garment_embed:   nn.Embedding(num_sizes, embed_dim)
  delta_encoder:   Linear(1) → SiLU → Linear(hidden_dim//2) → SiLU → Linear(embed_dim)
  fusion:          Linear(embed_dim*3, embed_dim*num_tokens) → reshape → (B, num_tokens, embed_dim)
  layer_norm:      LayerNorm(embed_dim)  applied per token

Inputs
------
  body_measurements : (B, 5)  [height_cm, weight_kg, chest_cm, waist_cm, hip_cm]
                      Assumed to be normalised (zero mean, unit std) before passing in.
  garment_size_idx  : (B,)    int64, 0..num_sizes-1
  fit_delta         : (B, 1)  float, garment_size_idx - body_size_idx  (signed)

Output
------
  (B, num_tokens, embed_dim)   sequence of measurement tokens
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLUMLP(nn.Sequential):
    """A simple fully connected block: Linear → SiLU → Linear (→ SiLU → Linear …)."""

    def __init__(self, *dims: int):
        if len(dims) < 2:
            raise ValueError("SiLUMLP requires at least two dimensions.")
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        super().__init__(*layers)


class MeasurementEncoder(nn.Module):
    """
    Encodes body and garment measurement signals into a sequence of tokens
    suitable for cross-attention injection (IP-Adapter style).

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each output token (should match UNet cross-attn dim).
    hidden_dim : int
        Hidden size for the body and delta MLPs.
    num_tokens : int
        Number of output tokens (sequence length L).
    num_garment_sizes : int
        Number of discrete garment size categories (default 6 for XS..XXL).
    dropout : float
        Dropout applied to the fused representation before projection.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_tokens: int = 4,
        num_garment_sizes: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_garment_sizes = num_garment_sizes

        # ── 1. Body measurement encoder ──────────────────────────────────────
        # Input: (B, 5) normalised [height, weight, chest, waist, hip]
        self.body_encoder = SiLUMLP(5, hidden_dim, hidden_dim, embed_dim)

        # ── 2. Garment size embedding ─────────────────────────────────────────
        # Input: (B,) int64 size index
        self.garment_embed = nn.Embedding(num_garment_sizes, embed_dim)
        nn.init.normal_(self.garment_embed.weight, std=0.02)

        # ── 3. Fit-delta encoder ──────────────────────────────────────────────
        # Input: (B, 1) signed difference
        half_hidden = max(hidden_dim // 2, 64)
        self.delta_encoder = SiLUMLP(1, half_hidden, half_hidden, embed_dim)

        # ── 4. Fusion ──────────────────────────────────────────────────────────
        # Concat the three embed_dim vectors → project to (num_tokens * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Linear(embed_dim * 3, embed_dim * num_tokens)

        # ── 5. Per-token layer norm ────────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in [self.body_encoder, self.delta_encoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        body_measurements: torch.Tensor,
        garment_size_idx: torch.Tensor,
        fit_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        body_measurements : (B, 5)
        garment_size_idx  : (B,)
        fit_delta         : (B,) or (B, 1)

        Returns
        -------
        tokens : (B, num_tokens, embed_dim)
        """
        B = body_measurements.shape[0]

        # Ensure fit_delta is (B, 1)
        if fit_delta.dim() == 1:
            fit_delta = fit_delta.unsqueeze(1)

        # (B, embed_dim) per branch
        body_feat = self.body_encoder(body_measurements)            # (B, E)
        garment_feat = self.garment_embed(garment_size_idx)         # (B, E)
        delta_feat = self.delta_encoder(fit_delta.float())          # (B, E)

        # Fuse
        fused = torch.cat([body_feat, garment_feat, delta_feat], dim=-1)  # (B, 3E)
        fused = self.dropout(fused)

        tokens = self.fusion(fused)                                 # (B, num_tokens * E)
        tokens = tokens.view(B, self.num_tokens, self.embed_dim)    # (B, L, E)
        tokens = self.layer_norm(tokens)                            # (B, L, E)

        return tokens

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, "
            f"num_tokens={self.num_tokens}, num_garment_sizes={self.num_garment_sizes}"
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
