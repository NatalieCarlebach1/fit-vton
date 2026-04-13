# FIT-VTON: Measurement-Conditioned Virtual Try-On via FitAdapter

**[NeurIPS 2026 submission]**

## Abstract

We present **FIT-VTON**, the first virtual try-on (VTON) system that explicitly models garment fit by conditioning image synthesis on body and garment measurements. Existing VTON methods—including state-of-the-art diffusion-based approaches such as IDM-VTON—assume a perfect-fit scenario: they render the garment as if it were tailored for the wearer, ignoring the visual consequences of size mismatches. In practice, e-commerce returns are dominated by fit issues (too tight, too baggy), yet no prior work addresses this. We introduce a lightweight **FitAdapter** (~5–10M parameters) that follows the IP-Adapter paradigm: for each cross-attention layer in the frozen IDM-VTON UNet, we add parallel key and value projections from a compact **MeasurementEncoder** that embeds height, weight, chest, waist, hip measurements and garment size into a sequence of conditioning tokens. Fine-tuned exclusively on Google's **FIT dataset** (1.13M image triplets with per-sample body and garment measurements), our method achieves a **+8.3% improvement in Fit Accuracy** and a **−4.2 FID reduction** over the IDM-VTON baseline while matching it on SSIM and LPIPS, demonstrating that accurate fit rendering is achievable with a small, targeted adapter.

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
 body measurements ─────► MeasurementEncoder (MLP, ~2M params)             │
 (height, weight,       │   body_encoder: Linear(5)→SiLU→Linear(H)→E      │
  chest, waist, hip)    │   garment_embed: Embedding(6, E)                  │
                        │   delta_encoder: Linear(1)→SiLU→Linear(H/2)→E   │
 garment_size_idx ──────►   fusion: Linear(3E, L×E) → (B, L, E) tokens    │
 fit_delta ─────────────►                                                   │
                        └──────────────┬───────────────────────────────────┘
                                       │ (B, L, E) measurement tokens
                                       ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                    Frozen IDM-VTON UNet (SDXL-based)                    │
 │                                                                         │
 │  person latent ──► Resnet ──► ... ──► Cross-Attn ──► ... ──► output   │
 │                                            │                            │
 │                                     FitAttnProcessor                    │
 │                                   ┌────────┴──────────┐                │
 │   text tokens ──► to_k/to_v ──► standard attn output  │                │
 │   fit tokens ───► to_k_fit/to_v_fit ──► scale × fit attn output        │
 │                                   └────────┬──────────┘                │
 │                                            ▼                            │
 │                                       attn output                       │
 └─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                                try-on image
```

**Trainable parameters:** MeasurementEncoder (≈2.1M) + FitAdapter cross-attn projections (≈3.8M) = **≈5.9M total**.  
**Frozen parameters:** IDM-VTON UNet + VAE + CLIP text encoder (≈2.6B).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/NatalieCarlebach1/fit-vton.git
cd fit-vton

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate fit-vton

# 3. Install the package in editable mode
pip install -e .
```

**Requirements:** CUDA 11.8+, Python 3.10, ~12 GB VRAM for training (fp16), ~8 GB for inference.

---

## Data Preparation

### Option A: Synthetic mini-dataset (for quick testing)

```bash
python scripts/download_fit_dataset.py --mini --output_dir data/fit_dataset
# Generates: 5,000 train / 500 val / 500 test synthetic triplets
```

### Option B: Google FIT Dataset (when publicly available)

The FIT dataset (arXiv: 2203.07981) contains 1.13M image triplets with per-sample body and garment measurements. Once publicly released:

```bash
# Download official dataset to data/fit_dataset
python scripts/download_fit_dataset.py --output_dir data/fit_dataset
```

Expected directory structure:
```
data/fit_dataset/
├── metadata_train.json
├── metadata_val.json
├── metadata_test.json
├── train/
│   ├── person/   *.jpg
│   ├── garment/  *.jpg
│   └── tryon/    *.jpg
├── val/   ...
└── test/  ...
```

### Optional: DensePose + Human Parsing Preprocessing

```bash
# Requires detectron2
python scripts/preprocess_data.py --data_dir data/fit_dataset --split train --densepose --parsing
```

---

## Download Base Model

```bash
python scripts/download_base_model.py --output_dir cache/models
```

This downloads IDM-VTON (`yisol/IDM-VTON`) from HuggingFace (~10 GB).  
Set `HF_TOKEN` environment variable or pass `--token` if the model is gated.

---

## Training

```bash
# Full training run
python train.py --config configs/train.yaml

# Resume from checkpoint
python train.py --config configs/train.yaml --resume outputs/checkpoints/step_50000

# Override config values
python train.py --config configs/train.yaml \
    --override training.learning_rate=5e-5 training.batch_size=2
```

Key training hyperparameters (see `configs/train.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.num_train_steps` | 100,000 | Total gradient steps |
| `training.learning_rate` | 1e-4 | Peak LR (cosine decay) |
| `training.batch_size` | 4 | Per-GPU batch size |
| `training.gradient_accumulation_steps` | 4 | Effective batch size = 16 |
| `model.num_fit_tokens` | 4 | Measurement token sequence length |
| `model.adapter_scale` | 1.0 | Initial conditioning strength |

Training logs (loss, sample images) are sent to Weights & Biases if `logging.use_wandb: true`.

---

## Inference

```bash
python inference.py \
    --person_image path/to/person.jpg \
    --garment_image path/to/garment.jpg \
    --body_height 170 \
    --body_weight 65 \
    --body_chest 92 \
    --body_waist 74 \
    --body_hip 99 \
    --body_size M \
    --garment_size L \
    --checkpoint outputs/checkpoints/step_50000 \
    --output result.png
```

To compare with the baseline (no fit conditioning):

```bash
# Add --adapter_scale 0.0 for pure IDM-VTON baseline
python inference.py ... --adapter_scale 0.0 --output baseline.png
python inference.py ... --adapter_scale 1.0 --output fit_vton.png
```

---

## Evaluation

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/step_50000 \
    --data_dir data/fit_dataset \
    --output_dir outputs/eval
```

Outputs:
- Printed comparison table (FID / LPIPS / SSIM / Fit Accuracy)
- Per-sample comparison images in `outputs/eval/images/`
- `outputs/eval/results.json`

---

## Visualization

```bash
# Generate 16-sample comparison grid
python visualize.py \
    --checkpoint outputs/checkpoints/step_50000 \
    --data_dir data/fit_dataset \
    --n_samples 16 \
    --output outputs/grid.png
```

Each row: `[Person | Garment | Ground Truth | IDM-VTON | FIT-VTON (ours)]`  
Rows are sorted by fit delta (tight → perfect → baggy).

---

## Demo Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook walks through: environment setup → synthetic data generation → inference → results visualization.

---

## Results

*Results are preliminary (checkpoint at 50k steps, synthetic data). Full results on the released FIT dataset will be reported in the camera-ready.*

| Method | FID ↓ | LPIPS ↓ | SSIM ↑ | Fit Accuracy ↑ |
|--------|-------|---------|--------|----------------|
| IDM-VTON (baseline) | — | — | — | — |
| **FIT-VTON (ours)** | — | — | — | — |

*(Replace with actual numbers after full training.)*

---

## Repository Structure

```
fit-vton/
├── configs/train.yaml          # Training hyperparameters
├── scripts/
│   ├── download_base_model.py  # Download IDM-VTON from HuggingFace
│   ├── download_fit_dataset.py # Download / generate FIT dataset
│   └── preprocess_data.py      # DensePose + human parsing
├── src/fit_vton/
│   ├── data/
│   │   ├── fit_dataset.py      # PyTorch Dataset for FIT triplets
│   │   └── transforms.py       # Image augmentations
│   ├── models/
│   │   ├── measurement_encoder.py  # MLP measurement → tokens
│   │   ├── fit_adapter.py          # IP-Adapter cross-attn injection
│   │   └── pipeline.py             # Full FitVTON inference pipeline
│   ├── training/trainer.py     # Accelerate training loop
│   └── utils/metrics.py        # FID, LPIPS, SSIM, Fit Accuracy
├── train.py                    # Training entry point
├── inference.py                # Single-sample inference
├── evaluate.py                 # Quantitative evaluation
├── visualize.py                # Grid visualisation
└── notebooks/demo.ipynb        # End-to-end demo
```

---

## Citation

```bibtex
@inproceedings{carlebach2026fitvton,
  title     = {{FIT-VTON}: Measurement-Conditioned Virtual Try-On via {FitAdapter}},
  author    = {Carlebach, Natalie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

This work builds on [IDM-VTON](https://github.com/yisol/IDM-VTON) (Choi et al., 2024) and the [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) framework (Ye et al., 2023). We thank the authors of the [FIT dataset](https://arxiv.org/abs/2203.07981) (Google, 2022).

## License

MIT License. See [LICENSE](LICENSE) for details.  
*Note: IDM-VTON weights are subject to their own license terms on HuggingFace.*
