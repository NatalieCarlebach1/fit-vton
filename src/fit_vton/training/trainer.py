"""
Accelerate-based training loop for FIT-VTON.

Only the MeasurementEncoder + FitAdapter parameters are trained;
the IDM-VTON UNet backbone remains frozen throughout.

Usage
-----
  from fit_vton.training.trainer import FitVTONTrainer
  trainer = FitVTONTrainer(config)
  trainer.train()

Or via CLI:
  python train.py --config configs/train.yaml
"""

import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration, set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    import logging
    def get_logger(name):
        return logging.getLogger(name)

try:
    from diffusers import DDPMScheduler
    from diffusers.training_utils import compute_snr
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from fit_vton.data.fit_dataset import FITDataset, FITDatasetConfig, build_dataloader
from fit_vton.models.measurement_encoder import MeasurementEncoder
from fit_vton.models.fit_adapter import FitAdapter
from fit_vton.data.transforms import tensor_to_pil


logger = get_logger(__name__, log_level="INFO")


class FitVTONTrainer:
    """
    Manages the full training lifecycle for FIT-VTON.

    Parameters
    ----------
    config : OmegaConf DictConfig loaded from configs/train.yaml
    resume_from_checkpoint : optional path to a checkpoint directory to resume from
    """

    def __init__(self, config: DictConfig, resume_from_checkpoint: Optional[str] = None):
        self.config = config
        self.resume_from_checkpoint = resume_from_checkpoint

        # Set up directories
        self.output_dir = Path(config.training.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _setup_accelerator(self) -> "Accelerator":
        """Initialises HuggingFace Accelerate."""
        if not ACCELERATE_AVAILABLE:
            raise ImportError("accelerate is required. Install with: pip install accelerate")

        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
        )

        log_with = "wandb" if (
            self.config.logging.use_wandb and WANDB_AVAILABLE
        ) else None

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.training.mixed_precision,
            log_with=log_with,
            project_config=project_config,
        )
        return accelerator

    # ------------------------------------------------------------------
    def _load_models(self, accelerator: "Accelerator"):
        """Loads IDM-VTON, builds MeasurementEncoder and FitAdapter."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required. Install with: pip install diffusers")

        from diffusers import AutoPipelineForInpainting

        model_id = self.config.model.base_model_id
        dtype = (
            torch.float16
            if self.config.training.mixed_precision == "fp16"
            else torch.bfloat16
            if self.config.training.mixed_precision == "bf16"
            else torch.float32
        )

        logger.info(f"Loading IDM-VTON from {model_id} ...")

        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        unet = pipe.unet
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.noise_scheduler.num_train_timesteps,
            beta_start=self.config.noise_scheduler.beta_start,
            beta_end=self.config.noise_scheduler.beta_end,
            beta_schedule=self.config.noise_scheduler.beta_schedule,
            clip_sample=self.config.noise_scheduler.clip_sample,
            prediction_type=self.config.noise_scheduler.prediction_type,
        )

        # Freeze backbone
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.eval()
        vae.eval()
        text_encoder.eval()

        # Determine cross_attention_dim
        cross_attention_dim = getattr(unet.config, "cross_attention_dim", 2048)
        if isinstance(cross_attention_dim, (list, tuple)):
            cross_attention_dim = max(cross_attention_dim)

        # Build MeasurementEncoder
        measurement_encoder = MeasurementEncoder(
            embed_dim=cross_attention_dim,
            hidden_dim=self.config.model.measurement_hidden_dim,
            num_tokens=self.config.model.num_fit_tokens,
            num_garment_sizes=self.config.model.num_garment_sizes,
        )

        # Build and install FitAdapter
        fit_adapter = FitAdapter(
            unet=unet,
            cross_attention_dim=cross_attention_dim,
            adapter_scale=self.config.model.adapter_scale,
        )
        fit_adapter.install()

        logger.info(
            f"Trainable parameters: "
            f"MeasurementEncoder={measurement_encoder.num_parameters:,} | "
            f"FitAdapter={fit_adapter.num_parameters:,}"
        )

        return (
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            measurement_encoder,
            fit_adapter,
        )

    # ------------------------------------------------------------------
    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        cfg = self.config
        train_ds_cfg = FITDatasetConfig(
            data_dir=cfg.data.data_dir,
            split=cfg.data.train_split,
            image_size=cfg.data.image_size,
            use_augmentation=True,
        )
        val_ds_cfg = FITDatasetConfig(
            data_dir=cfg.data.data_dir,
            split=cfg.data.val_split,
            image_size=cfg.data.image_size,
            use_augmentation=False,
        )
        train_loader = build_dataloader(
            train_ds_cfg,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            shuffle=True,
        )
        val_loader = build_dataloader(
            val_ds_cfg,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            shuffle=False,
        )
        return {"train": train_loader, "val": val_loader}

    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        batch: Dict,
        unet: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer,
        noise_scheduler,
        measurement_encoder: nn.Module,
        fit_adapter: FitAdapter,
        device: torch.device,
        weight_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Computes the diffusion MSE loss for a single batch.

        Steps
        -----
        1. Encode tryon images to latent space via VAE.
        2. Sample noise and timesteps.
        3. Encode measurements → fit tokens.
        4. Inject fit tokens into UNet via FitAdapter.
        5. Predict noise via UNet.
        6. Compute MSE loss against the sampled noise.
        """
        tryon_images = batch["tryon_image"].to(device, dtype=weight_dtype)
        garment_images = batch["garment_image"].to(device, dtype=weight_dtype)

        body_measurements = batch["body_measurements"].to(device, dtype=weight_dtype)
        garment_size_idx = batch["garment_size_idx"].to(device)
        fit_delta = batch["fit_delta"].to(device, dtype=weight_dtype)

        B = tryon_images.shape[0]

        with torch.no_grad():
            # Encode tryon images to latents
            latents = vae.encode(tryon_images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            ).long()

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode garment images separately (used as image conditioning)
            garment_latents = vae.encode(garment_images).latent_dist.sample()
            garment_latents = garment_latents * vae.config.scaling_factor

            # Dummy text conditioning
            dummy_tokens = tokenizer(
                [""] * B,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
            encoder_hidden_states = text_encoder(dummy_tokens)[0]

        # Encode measurements → fit tokens (trainable!)
        fit_tokens = measurement_encoder(
            body_measurements, garment_size_idx, fit_delta
        )

        # Inject fit tokens
        fit_adapter.set_fit_tokens(fit_tokens)

        # UNet forward pass
        # IDM-VTON expects concatenated garment latents on the channel dim
        unet_input = torch.cat([noisy_latents, garment_latents], dim=1)

        noise_pred = unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        fit_adapter.clear_fit_tokens()

        # MSE loss against the target noise
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        return loss

    # ------------------------------------------------------------------
    def train(self):
        """Main training loop."""
        cfg = self.config
        set_seed(cfg.training.seed)

        accelerator = self._setup_accelerator()

        if accelerator.is_main_process:
            if cfg.logging.use_wandb and WANDB_AVAILABLE:
                accelerator.init_trackers(
                    cfg.logging.project_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    init_kwargs={"wandb": {"name": cfg.logging.run_name}},
                )

        # Load models
        (
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            measurement_encoder,
            fit_adapter,
        ) = self._load_models(accelerator)

        # Build optimizer (only trainable params)
        trainable_params = (
            list(measurement_encoder.parameters())
            + list(fit_adapter.attn_processors.parameters())
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Build dataloaders
        dataloaders = self._build_dataloaders()
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        # LR scheduler
        num_update_steps = cfg.training.num_train_steps
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_update_steps,
            eta_min=cfg.training.learning_rate * 0.01,
        )

        # Warmup wrapper
        warmup_steps = cfg.training.warmup_steps

        # Prepare with accelerator
        (
            measurement_encoder,
            fit_adapter.attn_processors,
            optimizer,
            train_loader,
            val_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            measurement_encoder,
            fit_adapter.attn_processors,
            optimizer,
            train_loader,
            val_loader,
            lr_scheduler,
        )

        unet = unet.to(accelerator.device)
        vae = vae.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        unet = unet.to(dtype=weight_dtype)
        vae = vae.to(dtype=weight_dtype)
        text_encoder = text_encoder.to(dtype=weight_dtype)

        # Resume from checkpoint
        global_step = 0
        if self.resume_from_checkpoint is not None:
            ckpt_path = Path(self.resume_from_checkpoint)
            if ckpt_path.exists():
                logger.info(f"Resuming from checkpoint: {ckpt_path}")
                fit_adapter.load_adapter(str(ckpt_path / "fit_adapter.pt"), strict=False)
                encoder_ckpt = ckpt_path / "measurement_encoder.pt"
                if encoder_ckpt.exists():
                    state = torch.load(str(encoder_ckpt), map_location="cpu")
                    accelerator.unwrap_model(measurement_encoder).load_state_dict(state)
                step_ckpt = ckpt_path / "step.txt"
                if step_ckpt.exists():
                    global_step = int(step_ckpt.read_text().strip())
                logger.info(f"Resumed at step {global_step}")

        # Training loop
        progress_bar = tqdm(
            total=cfg.training.num_train_steps,
            initial=global_step,
            disable=not accelerator.is_main_process,
            desc="Training FIT-VTON",
        )

        train_iter = iter(train_loader)
        running_loss = 0.0

        while global_step < cfg.training.num_train_steps:
            measurement_encoder.train()
            fit_adapter.attn_processors.train()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            with accelerator.accumulate(measurement_encoder):
                # Apply warmup manually
                if global_step < warmup_steps:
                    lr_scale = float(global_step + 1) / float(max(1, warmup_steps))
                    for pg in optimizer.param_groups:
                        pg["lr"] = cfg.training.learning_rate * lr_scale

                loss = self._compute_loss(
                    batch=batch,
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    measurement_encoder=measurement_encoder,
                    fit_adapter=fit_adapter,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)

                optimizer.step()
                if global_step >= warmup_steps:
                    lr_scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().item()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % cfg.training.log_every_n_steps == 0:
                    avg_loss = running_loss / cfg.training.log_every_n_steps
                    running_loss = 0.0
                    lr = optimizer.param_groups[0]["lr"]
                    logs = {"train/loss": avg_loss, "train/lr": lr, "step": global_step}
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    accelerator.log(logs, step=global_step)

                # Validation samples
                if (
                    global_step % cfg.training.sample_every_n_steps == 0
                    and accelerator.is_main_process
                    and cfg.logging.log_images
                ):
                    self._save_validation_samples(
                        val_loader=val_loader,
                        measurement_encoder=accelerator.unwrap_model(measurement_encoder),
                        fit_adapter=fit_adapter,
                        unet=unet,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        global_step=global_step,
                        device=accelerator.device,
                        weight_dtype=weight_dtype,
                        num_samples=cfg.training.num_val_samples,
                    )

                # Checkpointing
                if global_step % cfg.training.save_every_n_steps == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = self.checkpoint_dir / f"step_{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)

                        fit_adapter.save_adapter(str(ckpt_dir / "fit_adapter.pt"))
                        torch.save(
                            accelerator.unwrap_model(measurement_encoder).state_dict(),
                            str(ckpt_dir / "measurement_encoder.pt"),
                        )
                        (ckpt_dir / "step.txt").write_text(str(global_step))
                        logger.info(f"Saved checkpoint to {ckpt_dir}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_dir = self.checkpoint_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            fit_adapter.save_adapter(str(final_dir / "fit_adapter.pt"))
            torch.save(
                accelerator.unwrap_model(measurement_encoder).state_dict(),
                str(final_dir / "measurement_encoder.pt"),
            )
            logger.info(f"Training complete. Final model saved to {final_dir}")

        accelerator.end_training()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _save_validation_samples(
        self,
        val_loader: DataLoader,
        measurement_encoder: MeasurementEncoder,
        fit_adapter: FitAdapter,
        unet: nn.Module,
        vae: nn.Module,
        noise_scheduler,
        global_step: int,
        device: torch.device,
        weight_dtype: torch.dtype,
        num_samples: int = 4,
    ):
        """Generates and saves a grid of validation samples for visual inspection."""
        import torchvision.utils as vutils

        measurement_encoder.eval()
        fit_adapter.attn_processors.eval()

        batch = next(iter(val_loader))
        B = min(num_samples, batch["person_image"].shape[0])

        person_images = batch["person_image"][:B].to(device, dtype=weight_dtype)
        garment_images = batch["garment_image"][:B].to(device, dtype=weight_dtype)
        tryon_images = batch["tryon_image"][:B].to(device, dtype=weight_dtype)
        body_meas = batch["body_measurements"][:B].to(device, dtype=weight_dtype)
        garment_size_idx = batch["garment_size_idx"][:B].to(device)
        fit_delta = batch["fit_delta"][:B].to(device, dtype=weight_dtype)

        fit_tokens = measurement_encoder(body_meas, garment_size_idx, fit_delta)
        fit_adapter.set_fit_tokens(fit_tokens)

        # Quick DDIM-style decode: add noise and denoise for a few steps
        latents = vae.encode(tryon_images).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        t = torch.tensor([noise_scheduler.config.num_train_timesteps // 2] * B, device=device)
        noisy = noise_scheduler.add_noise(latents, noise, t)

        garment_latents = (
            vae.encode(garment_images).latent_dist.sample() * vae.config.scaling_factor
        )
        unet_input = torch.cat([noisy, garment_latents], dim=1)
        noise_pred = unet(unet_input, t, encoder_hidden_states=None).sample
        fit_adapter.clear_fit_tokens()

        # Decode
        pred_latents = latents - noise_scheduler.alphas_cumprod[t[0]].sqrt() * noise_pred
        pred_images = vae.decode(pred_latents / vae.config.scaling_factor).sample

        # Clamp and save grid
        grid = torch.cat([person_images, garment_images, tryon_images, pred_images], dim=0)
        grid = (grid * 0.5 + 0.5).clamp(0, 1)

        save_path = self.sample_dir / f"step_{global_step:07d}.png"
        vutils.save_image(grid, str(save_path), nrow=B)
        logger.info(f"Saved validation samples to {save_path}")


def main():
    """CLI entry point: python train.py --config configs/train.yaml [--resume path]"""
    import argparse
    parser = argparse.ArgumentParser(description="Train FIT-VTON")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    trainer = FitVTONTrainer(cfg, resume_from_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
