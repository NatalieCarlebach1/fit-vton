"""
FitAdapter: IP-Adapter-style injection of measurement tokens into each
cross-attention layer of the frozen IDM-VTON UNet.

For every cross-attention processor we add:
  - to_k_fit : Linear(cross_attention_dim → hidden_size, bias=False)
  - to_v_fit : Linear(cross_attention_dim → hidden_size, bias=False)

During the forward pass:
  out = softmax(Q K^T / sqrt(d)) V              # standard text-conditioned attn
      + scale * softmax(Q K_fit^T / sqrt(d)) V_fit  # measurement-conditioned attn

The adapter_scale can be adjusted at inference to control the strength of
the fit conditioning (0 = pure IDM-VTON baseline, 1 = full conditioning).
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.models.attention_processor import Attention
except ImportError:
    # Fallback for older diffusers versions
    from diffusers.models.cross_attention import CrossAttention as Attention


class FitAttnProcessor(nn.Module):
    """
    Drop-in replacement for a single cross-attention processor in the UNet.

    Implements:
      standard_attn(hidden_states, encoder_hidden_states)
      + scale * fit_attn(hidden_states, fit_hidden_states)

    Parameters
    ----------
    hidden_size : int
        Output dimensionality of the attention (number of heads × head_dim).
    cross_attention_dim : int
        Dimensionality of the encoder hidden states (text tokens).
    scale : float
        Initial value for adapter_scale; can be overridden at forward time.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        # Measurement-conditioned K and V projections
        self.to_k_fit = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_fit = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.to_k_fit.weight, std=1 / math.sqrt(self.cross_attention_dim))
        nn.init.zeros_(self.to_v_fit.weight)  # zero-init V so adapter starts as identity

    # ------------------------------------------------------------------
    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes scaled dot-product attention."""
        scale_factor = 1.0 / math.sqrt(query.shape[-1])
        attn_weight = torch.bmm(query, key.transpose(-2, -1)) * scale_factor
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = F.softmax(attn_weight, dim=-1)
        return torch.bmm(attn_weight, value)

    # ------------------------------------------------------------------
    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        fit_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        attn                : the Attention module (provides Q/K/V weight matrices)
        hidden_states       : (B, seq_len, hidden_size)   — spatial tokens
        encoder_hidden_states : (B, text_len, cross_attn_dim) — text tokens
        fit_hidden_states   : (B, num_tokens, cross_attn_dim) — measurement tokens
        attention_mask      : optional mask for text attention
        temb                : optional time embedding (not used here)
        """
        residual = hidden_states

        # ── Pre-norm / spatial norm ───────────────────────────────────────────
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, B)
            attention_mask = attention_mask.view(B * attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # ── Query ─────────────────────────────────────────────────────────────
        query = attn.to_q(hidden_states)

        # ── Standard cross-attention ──────────────────────────────────────────
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention: (B, seq, dim) → (B*heads, seq, head_dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Standard attention output
        attn_output = self._scaled_dot_product_attention(query, key, value, attention_mask)

        # ── Fit-conditioned cross-attention ───────────────────────────────────
        if fit_hidden_states is not None and self.scale != 0.0:
            # Project measurement tokens to K and V spaces
            key_fit = self.to_k_fit(fit_hidden_states)     # (B, L, hidden_size)
            value_fit = self.to_v_fit(fit_hidden_states)   # (B, L, hidden_size)

            key_fit = attn.head_to_batch_dim(key_fit)      # (B*heads, L, head_dim)
            value_fit = attn.head_to_batch_dim(value_fit)

            fit_attn_output = self._scaled_dot_product_attention(query, key_fit, value_fit)
            attn_output = attn_output + self.scale * fit_attn_output

        # ── Output projection ─────────────────────────────────────────────────
        attn_output = attn.batch_to_head_dim(attn_output)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)  # dropout

        if input_ndim == 4:
            attn_output = attn_output.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            attn_output = attn_output + residual

        attn_output = attn_output / attn.rescale_output_factor

        return attn_output


class FitAdapter(nn.Module):
    """
    Holds all FitAttnProcessors and the MeasurementEncoder.
    Installs processors into a frozen UNet and provides a context manager
    for passing fit_hidden_states during the forward pass.

    Usage
    -----
    >>> adapter = FitAdapter(unet, cross_attention_dim=2048, scale=1.0)
    >>> adapter.install()
    >>> # During training:
    >>> fit_tokens = encoder(body_meas, garment_size, fit_delta)  # (B, L, D)
    >>> adapter.set_fit_tokens(fit_tokens)
    >>> noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)
    >>> adapter.clear_fit_tokens()
    """

    def __init__(
        self,
        unet: nn.Module,
        cross_attention_dim: int = 2048,
        adapter_scale: float = 1.0,
    ):
        super().__init__()
        self.unet = unet
        self.cross_attention_dim = cross_attention_dim
        self.adapter_scale = adapter_scale

        self.attn_processors: nn.ModuleDict = nn.ModuleDict()
        self._fit_hidden_states: Optional[torch.Tensor] = None

        self._build_processors()

    # ------------------------------------------------------------------
    def _build_processors(self):
        """
        Iterates over all cross-attention layers in the UNet and creates
        a FitAttnProcessor for each one that uses cross-attention (not self-attn).
        """
        unet_attn_procs = {}
        for name, module in self.unet.attn_processors.items():
            # Only inject into cross-attention layers (not self-attention)
            # Cross-attention layers have names ending with "attn2.processor"
            if "attn2" in name:
                # Determine hidden_size from the layer name
                hidden_size = self._get_hidden_size(name)
                proc = FitAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=self.cross_attention_dim,
                    scale=self.adapter_scale,
                )
                self.attn_processors[name.replace(".", "_")] = proc
                unet_attn_procs[name] = proc
            else:
                # Keep original processor for self-attention
                unet_attn_procs[name] = module

        self.unet.set_attn_processor(unet_attn_procs)

    def _get_hidden_size(self, attn_name: str) -> int:
        """
        Infers the hidden size for a given attention processor name.
        Walks the module hierarchy to retrieve the to_q weight shape.
        """
        # Navigate to the parent Attention module
        parts = attn_name.split(".")
        # Remove trailing ".processor"
        if parts[-1] == "processor":
            parts = parts[:-1]
        module = self.unet
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                # Fall back to a reasonable default
                return 320
        if hasattr(module, "to_q"):
            return module.to_q.out_features
        return 320

    # ------------------------------------------------------------------
    def install(self):
        """
        Convenience method: freezes the UNet backbone and makes the
        adapter parameters the only trainable ones.
        """
        for param in self.unet.parameters():
            param.requires_grad_(False)
        for param in self.attn_processors.parameters():
            param.requires_grad_(True)

    def set_fit_tokens(self, fit_hidden_states: torch.Tensor):
        """Stores measurement tokens to be used in the next UNet forward pass."""
        self._fit_hidden_states = fit_hidden_states
        # Inject into all FitAttnProcessors
        for proc in self.attn_processors.values():
            proc._fit_hidden_states = fit_hidden_states

    def clear_fit_tokens(self):
        """Clears stored measurement tokens after the forward pass."""
        self._fit_hidden_states = None
        for proc in self.attn_processors.values():
            proc._fit_hidden_states = None

    def set_scale(self, scale: float):
        """Adjusts the adapter scale for all processors."""
        self.adapter_scale = scale
        for proc in self.attn_processors.values():
            proc.scale = scale

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.attn_processors.parameters())

    def save_adapter(self, path: str):
        """Saves only the adapter weights (not the frozen UNet)."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "attn_processors": self.attn_processors.state_dict(),
                "cross_attention_dim": self.cross_attention_dim,
                "adapter_scale": self.adapter_scale,
            },
            path,
        )

    def load_adapter(self, path: str, strict: bool = True):
        """Loads adapter weights from a checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        self.attn_processors.load_state_dict(ckpt["attn_processors"], strict=strict)
        self.adapter_scale = ckpt.get("adapter_scale", self.adapter_scale)
        self.set_scale(self.adapter_scale)
