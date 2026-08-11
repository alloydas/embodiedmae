# EmbodiedMAE for Sorghum Plant Reconstruction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Multi-modal Masked Autoencoder for RGB, Depth, and Point Cloud reconstruction of Sorghum plants.**

## Overview

This repository implements **EmbodiedMAE** (Embodied Multi-modal Masked Autoencoder) adapted for agricultural plant reconstruction. The model jointly learns from RGB images, depth maps, and point clouds to reconstruct masked portions of each modality, enabling robust 3D plant understanding.

### Key Features

- ✅ **Multi-modal Learning** - Joint reconstruction of RGB, Depth, and Point Cloud
- ✅ **10,000 Point Clouds** - High-resolution 3D reconstruction
- ✅ **Depth Normalization** - Global depth normalization for stable training
- ✅ **YAML Configuration** - Easy experiment management
- ✅ **Multi-GPU Training** - Distributed Data Parallel (DDP) support
- ✅ **Comprehensive Metrics** - RGB/Depth MSE, Chamfer Distance, Earth Mover's Distance
- ✅ **Resume Training** - Checkpoint support with backward compatibility
- ✅ **Validation Tools** - Complete validation and data export scripts
- ✅ **Wandb Integration** - Track experiments with Weights & Biases

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/alloydas/embodiedmae
cd embodiedmae

# Create conda environment
conda create -n embodiedmae python=3.8
conda activate embodiedmae

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow matplotlib tqdm pyyaml scipy wandb open3d
```


### Training

```bash
# Train with default configuration
python train_sorghum_multi.py

# Train with custom config
python train_sorghum_multi.py --config my_config.yaml

# Multi-GPU training (4 GPUs)
python train_sorghum_multi.py --world_size 4

# Resume training
python train_sorghum_multi.py --resume ./outputs_sorghum/checkpoints/checkpoint_epoch_50.pth
```

### Validation

```bash
# Validate model
python validate.py \
    --checkpoint ./outputs_sorghum/checkpoints/best_checkpoint.pth \
    --output_dir ./validation_results


## Model Architecture

### EmbodiedMAE

The model consists of:

1. **Multi-modal Encoder**
   - Vision Transformer for RGB images
   - Vision Transformer for depth maps
   - Point Cloud Transformer with group-based tokenization

2. **Shared Transformer Decoder**
   - Unified decoder for all modalities
   - Separate reconstruction heads

3. **Loss Functions**
   - RGB: MSE with per-patch normalization
   - Depth: MSE with global normalization
   - Point Cloud: Weighted Chamfer distance

### Model Sizes

| Model | Parameters | Embed Dim | Depth | Heads |
|-------|-----------|-----------|-------|-------|
| **Small** | 22M | 384 | 12 | 6 |
| **Base** | 87M | 768 | 12 | 12 |
| **Large** | 304M | 1024 | 24 | 16 |
| **Giant** | 1B | 1280 | 40 | 16 |

## Configuration

### config.yaml

```yaml
# Data
data:
  data_root: "./data"
  img_size: 224
  num_points: 10000
  train_split: 0.9

# Model
model:
  model_size: "base"
  patch_size: 16
  mask_ratio: 0.75
  pc_loss_weight: 50.0
  normalize_depth_global: true
  depth_norm_type: "minmax"

# Training
training:
  batch_size: 8
  epochs: 100
  lr: 1.5e-4
  weight_decay: 0.05
  warmup_epochs: 10
  val_freq: 10

# Checkpointing
checkpointing:
  output_dir: "./outputs_sorghum"
  save_freq: 10

# Wandb
wandb:
  use_wandb: true
  project: "embodied-mae-sorghum"
```

### Key Parameters

**Depth Normalization:**
- `normalize_depth_global: true` - Enable global depth normalization
- `depth_norm_type: "minmax"` - Normalization type (minmax or standard)

**Validation:**
- `val_freq: 10` - Validate every N epochs (reduces overhead)

**Multi-GPU:**
- `world_size: 4` - Number of GPUs to use

## Training Details

### Single GPU

```bash
python train_sorghum_multi.py --world_size 1
```

### Multi-GPU (DDP)

```bash
# 4 GPUs
python train_sorghum_multi.py --world_size 4

# 8 GPUs
python train_sorghum_multi.py --world_size 8
```

### Resume Training

```bash
python train_sorghum_multi.py \
    --resume ./outputs_sorghum/checkpoints/checkpoint_epoch_50.pth
```

### Override Config

```bash
python train_sorghum_multi.py \
    --batch_size 16 \
    --lr 2e-4 \
    --epochs 200 \
    --depth_norm_type standard
```

## Validation & Evaluation

### Quick Validation

```bash
python validate.py \
    --checkpoint ./model.pth \
    --output_dir ./results
```

**Outputs:**
- `validation_metrics_TIMESTAMP.json` - All metrics
- `validation_summary_TIMESTAMP.json` - Complete summary
- `visualizations_TIMESTAMP/` - 3×3 grid visualizations

### Export All Data

```bash
python validate.py \
    --checkpoint ./model.pth \
    --output_dir ./export \
    --num_samples 6
```

**Outputs for each sample:**
```
sample_01_plant001/
├── inputs/
│   ├── rgb.png, rgb.pt
│   ├── depth.png, depth.npy
│   └── pointcloud.ply, pointcloud.npy
├── outputs/
│   ├── rgb.png, rgb.pt
│   ├── depth.png, depth.npy
│   └── pointcloud.ply, pointcloud.npy
├── visualization.png
├── metrics.json
└── README.md
```

## Metrics

### Reconstruction Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **RGB MSE** | Mean Squared Error for RGB | < 0.003 |
| **Depth MSE** | Mean Squared Error for Depth | < 0.002 |
| **Chamfer Distance** | Point-to-point distance (fast) | < 0.001 |
| **Earth Mover's Distance** | Optimal matching distance (rigorous) | < 0.03 |

### During Training

Console output every epoch:
```
Epoch 10/100
Train - Loss: 0.0700 | RGB: 0.0540 | Depth: 0.0145 | PC: 0.0016
🔍 Running validation...
Val   - Loss: 0.0725 | RGB: 0.0555 | Depth: 0.0150 | PC: 0.0015
Metrics - RGB MSE: 0.003456 | Depth MSE: 0.002234 | PC Chamfer: 0.001123 | PC EMD: 0.045678
```

## Features in Detail

### 1. Depth Normalization

Global depth normalization before loss calculation:

```yaml
model:
  normalize_depth_global: true
  depth_norm_type: "minmax"  # Options: minmax, standard
```

**Benefits:**
- Handles varying depth scales across samples
- More stable training
- Better convergence



### 2. Validation Frequency

Control how often validation runs:

```yaml
training:
  val_freq: 10  # Validate every 10 epochs
```

**Benefits:**
- Faster training (less validation overhead)
- Still tracks progress regularly



### 3. Earth Mover's Distance

Rigorous point cloud evaluation:

```python
# Automatically calculated during validation
emd = earth_movers_distance(pred_pc, target_pc, num_samples=1000)
```


### 4. Multi-GPU Training

Distributed Data Parallel for faster training:

```bash
# Automatically uses torch.distributed
python train_sorghum_multi.py --world_size 4
```

**Benefits:**
- Linear scaling with number of GPUs
- Automatic gradient synchronization
- Checkpoint compatibility

### 5. YAML Configuration

Organize experiments with YAML configs:

```bash
# Create experiment configs
python train_sorghum_multi.py --config experiments/exp1.yaml
python train_sorghum_multi.py --config experiments/exp2.yaml
```

**See:** `YAML_CONFIG_GUIDE.md`

## Four-Modality Latent Alignment

Use `analyze_modality_alignment_4m.py` to compare RGB, depth, point-cloud, and
numeric spline-parameter representations. The probe reports Linear CKA,
paired-versus-negative cosine similarity, exact-view retrieval, plant-aware
retrieval, and 2-D latent-space visualizations.

```bash
/work/mech-ai-scratch/yongyun/envs/myenv/bin/python \
  analyze_modality_alignment_4m.py \
  --checkpoint outputs/4m_run_v4_qal_sinkhorn_no_duplicate_sk_loss_1.0/checkpoints/checkpoint_epoch_800.pth \
  --split val \
  --num_samples 1000 \
  --output_dir outputs/4m_run_v4_qal_sinkhorn_no_duplicate_sk_loss_1.0/latent_alignment_ep800 \
  --device cuda \
  --projection pca \
  --amp
```

By default, capped analyses sample two views per plant. The views share identical
spline parameters, so use plant/group retrieval as the primary retrieval result;
exact-view retrieval is retained as a stricter, tie-aware secondary measurement.
For a larger independent-sample analysis, sample one view from each of up to
5,000 training plants:

```bash
/work/mech-ai-scratch/yongyun/envs/myenv/bin/python \
  analyze_modality_alignment_4m.py \
  --checkpoint /path/to/checkpoint.pth \
  --config config_4m.yaml \
  --split train \
  --num_samples 5000 \
  --one_per_group \
  --output_dir latent_alignment_train
```

The output directory contains:

- `metrics.json`: all directional and bidirectional metrics plus random baselines.
- `metrics.csv`: compact pair-by-pair summary.
- `embeddings.npz`: paired pre-encoder and post-encoder embeddings.
- `pairwise_metrics_*.png`: CKA, cosine-gap, and retrieval heatmaps.
- `latent_space_*.png`: L2-normalized and modality-centered 2-D projections;
  gray lines connect the four embeddings belonging to the same sample.

PCA is dependency-free beyond NumPy. `--projection tsne` uses scikit-learn;
`--projection umap` additionally requires `umap-learn`.

When `--config` is omitted, the analyzer uses the immutable `config.json` saved
beside the checkpoint. Pass `--config` or `--data_root` only when an intentional
cross-dataset/OOD comparison is desired; that mismatch is recorded in metadata.

## Output Structure

```
outputs_sorghum/
├── checkpoints/
│   ├── checkpoint_epoch_010.pth
│   ├── checkpoint_epoch_020.pth
│   ├── ...
│   └── best_checkpoint.pth
├── visualizations/
│   ├── epoch_001/
│   ├── epoch_010/
│   └── ...
├── training_history.json
└── config_used.yaml
```

## Checkpoint Format

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'history': training_history,
    'config': config
}
```



## File Structure

```
.
├── embodied_mae.py              # Model implementation
├── train_sorghum.py             # Training script
├── validate.py                  # Validation script
├── sorghum_dataset.py           # Dataset loader
├── config.yaml                  # Default configuration
├── README.md                    # This file
└── data/                        # Data directory

```

## Requirements

### Core Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.5.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
```

### Optional Dependencies
```
scipy>=1.10.0        # For EMD calculation
wandb>=0.15.0        # For experiment tracking
open3d>=0.17.0       # For point cloud visualization
```
