"""
Enhanced Validation Script for EmbodiedMAE
Evaluates model metrics and saves all inputs/outputs separately
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from PIL import Image

from embodied_mae import embodied_mae_small, embodied_mae_base, chamfer_distance, earth_movers_distance
from sorghum_dataset import SorghumDataset
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_rgb_image(tensor, path, denormalize=True):
    """Save RGB image tensor as PNG"""
    if denormalize:
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
    
    # Clamp and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(img_np)
    img.save(path)


def save_depth_map(tensor, path, format='png'):
    """Save depth map as PNG or NPY"""
    depth_np = tensor.squeeze().numpy()
    
    if format == 'png':
        # Normalize to 0-255 for visualization
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = (depth_np * 255).astype(np.uint8)
        
        # Save as grayscale PNG
        img = Image.fromarray(depth_normalized, mode='L')
        img.save(path)
    else:
        # Save raw values as NPY
        np.save(path, depth_np)


def save_point_cloud(tensor, path, format='ply'):
    """Save point cloud as PLY or NPY"""
    pc_np = tensor.numpy()
    
    if format == 'ply':
        # Save as PLY (ASCII format)
        with open(path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {pc_np.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Data
            for point in pc_np:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    else:
        # Save as NPY
        np.save(path, pc_np)


def evaluate_model(model, dataloader, device, num_samples=6):
    """
    Evaluate model and collect samples
    
    Returns:
        metrics: dict with all metric values
        samples: list of samples with all data
    """
    model.eval()
    
    # Accumulators
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    total_pc_loss = 0
    total_rgb_mse = 0
    total_depth_mse = 0
    total_pc_chamfer = 0
    total_pc_emd = 0
    
    # Store samples
    all_samples = []
    
    print("\n🔍 Evaluating model and collecting samples...")
    
    with torch.no_grad():
        for rgb, depth, pc, names in tqdm(dataloader, desc='Evaluating'):
            rgb = rgb.to(device)
            depth = depth.to(device)
            pc = pc.to(device)
            
            # Forward pass
            loss, (loss_rgb, loss_depth, loss_pc), (pred_rgb_patches, pred_depth_patches, pred_pc), (mask_rgb, mask_depth, mask_pc) = model(rgb, depth, pc)
            
            # Accumulate losses
            total_loss += loss.item()
            total_rgb_loss += loss_rgb.item()
            total_depth_loss += loss_depth.item()
            total_pc_loss += loss_pc.item()
            
            # Unpatchify predictions
            B = rgb.shape[0]
            patch_size = model.patch_size
            img_size = rgb.shape[2]
            
            # Unpatchify RGB
            p = patch_size
            h = w = img_size // p
            pred_rgb_img = pred_rgb_patches.reshape(B, h, w, p, p, 3)
            pred_rgb_img = torch.einsum('nhwpqc->nchpwq', pred_rgb_img)
            pred_rgb_img = pred_rgb_img.reshape(B, 3, img_size, img_size)
            
            # Unpatchify Depth
            pred_depth_img = pred_depth_patches.reshape(B, h, w, p, p, 1)
            pred_depth_img = torch.einsum('nhwpqc->nchpwq', pred_depth_img)
            pred_depth_img = pred_depth_img.reshape(B, 1, img_size, img_size)
            
            # Calculate metrics
            rgb_mse = torch.mean((pred_rgb_img - rgb) ** 2).item()
            depth_mse = torch.mean((pred_depth_img - depth) ** 2).item()
            pc_chamfer = chamfer_distance(pred_pc, pc).item()
            pc_emd = earth_movers_distance(pred_pc, pc, num_samples=1000).item()
            
            total_rgb_mse += rgb_mse
            total_depth_mse += depth_mse
            total_pc_chamfer += pc_chamfer
            total_pc_emd += pc_emd
            
            # Store samples (collect all, will select later)
            for i in range(B):
                all_samples.append({
                    'rgb': rgb[i].cpu(),
                    'depth': depth[i].cpu(),
                    'pc': pc[i].cpu(),
                    'pred_rgb': pred_rgb_img[i].cpu(),
                    'pred_depth': pred_depth_img[i].cpu(),
                    'pred_pc': pred_pc[i].cpu(),
                    'mask_rgb': mask_rgb[i].cpu(),
                    'mask_depth': mask_depth[i].cpu(),
                    'mask_pc': mask_pc[i].cpu(),
                    'name': names[i],
                    'metrics': {
                        'rgb_mse': torch.mean((pred_rgb_img[i] - rgb[i]) ** 2).item(),
                        'depth_mse': torch.mean((pred_depth_img[i] - depth[i]) ** 2).item(),
                        'pc_chamfer': chamfer_distance(pred_pc[i:i+1], pc[i:i+1]).item(),
                    }
                })
            
            # Stop collecting if we have enough
            if len(all_samples) >= num_samples:
                break
    
    n = len(dataloader)
    metrics = {
        'loss': total_loss / n,
        'rgb_loss': total_rgb_loss / n,
        'depth_loss': total_depth_loss / n,
        'pc_loss': total_pc_loss / n,
        'rgb_mse': total_rgb_mse / n,
        'depth_mse': total_depth_mse / n,
        'pc_chamfer': total_pc_chamfer / n,
        'pc_emd': total_pc_emd / n,
    }
    
    return metrics, all_samples[:num_samples]


def save_sample_data(sample, sample_dir, idx):
    """
    Save all input and output data for a sample
    
    Directory structure:
    sample_{idx}/
        inputs/
            rgb.png
            rgb.npy (raw tensor)
            depth.png (visualized)
            depth.npy (raw values)
            pointcloud.ply
            pointcloud.npy
        outputs/
            rgb.png
            rgb.npy
            depth.png
            depth.npy
            pointcloud.ply
            pointcloud.npy
        metrics.json
    """
    # Create directories
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = sample_dir / "inputs"
    outputs_dir = sample_dir / "outputs"
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    name = sample['name']
    
    # Save inputs
    # RGB
    save_rgb_image(sample['rgb'], inputs_dir / "rgb.png", denormalize=True)
    torch.save(sample['rgb'], inputs_dir / "rgb.pt")  # Raw tensor
    
    # Depth
    save_depth_map(sample['depth'], inputs_dir / "depth.png", format='png')
    save_depth_map(sample['depth'], inputs_dir / "depth.npy", format='npy')
    
    # Point Cloud
    save_point_cloud(sample['pc'], inputs_dir / "pointcloud.ply", format='ply')
    save_point_cloud(sample['pc'], inputs_dir / "pointcloud.npy", format='npy')
    
    # Save outputs
    # RGB
    save_rgb_image(sample['pred_rgb'], outputs_dir / "rgb.png", denormalize=True)
    torch.save(sample['pred_rgb'], outputs_dir / "rgb.pt")  # Raw tensor
    
    # Depth
    save_depth_map(sample['pred_depth'], outputs_dir / "depth.png", format='png')
    save_depth_map(sample['pred_depth'], outputs_dir / "depth.npy", format='npy')
    
    # Point Cloud
    save_point_cloud(sample['pred_pc'], outputs_dir / "pointcloud.ply", format='ply')
    save_point_cloud(sample['pred_pc'], outputs_dir / "pointcloud.npy", format='npy')
    
    # Save metrics for this sample
    sample_metrics = {
        'name': name,
        'rgb_mse': sample['metrics']['rgb_mse'],
        'depth_mse': sample['metrics']['depth_mse'],
        'pc_chamfer': sample['metrics']['pc_chamfer'],
        'mask_ratios': {
            'rgb': sample['mask_rgb'].mean().item(),
            'depth': sample['mask_depth'].mean().item(),
            'pc': sample['mask_pc'].mean().item()
        }
    }
    
    with open(sample_dir / "metrics.json", 'w') as f:
        json.dump(sample_metrics, f, indent=2)
    
    # Save README
    readme = f"""# Sample {idx}: {name}

## Files Structure

### Inputs (Original Data)
- `rgb.png` - Visualized RGB image (denormalized)
- `rgb.pt` - Raw RGB tensor (PyTorch format)
- `depth.png` - Visualized depth map (8-bit grayscale)
- `depth.npy` - Raw depth values (NumPy array)
- `pointcloud.ply` - Point cloud (PLY ASCII format, viewable in MeshLab/CloudCompare)
- `pointcloud.npy` - Point cloud (NumPy array, shape: [N, 3])

### Outputs (Reconstructed Data)
- `rgb.png` - Reconstructed RGB image
- `rgb.pt` - Raw reconstructed tensor
- `depth.png` - Reconstructed depth map
- `depth.npy` - Raw reconstructed depth
- `pointcloud.ply` - Reconstructed point cloud
- `pointcloud.npy` - Reconstructed point cloud array

### Metrics
- `metrics.json` - Per-sample metrics (RGB MSE, Depth MSE, Chamfer Distance)

## How to Load Data

### Python
```python
import torch
import numpy as np

# Load RGB
rgb = torch.load('inputs/rgb.pt')
pred_rgb = torch.load('outputs/rgb.pt')

# Load Depth
depth = np.load('inputs/depth.npy')
pred_depth = np.load('outputs/depth.npy')

# Load Point Cloud
pc = np.load('inputs/pointcloud.npy')  # Shape: [N, 3]
pred_pc = np.load('outputs/pointcloud.npy')
```

### Visualization Tools
- **RGB/Depth PNG**: Any image viewer
- **PLY files**: MeshLab, CloudCompare, Open3D
- **NPY files**: Python (NumPy)

## Sample Metrics
- RGB MSE: {sample_metrics['rgb_mse']:.6f}
- Depth MSE: {sample_metrics['depth_mse']:.6f}
- Chamfer Distance: {sample_metrics['pc_chamfer']:.6f}
- Mask Ratios: RGB={sample_metrics['mask_ratios']['rgb']:.1%}, Depth={sample_metrics['mask_ratios']['depth']:.1%}, PC={sample_metrics['mask_ratios']['pc']:.1%}
"""
    
    with open(sample_dir / "README.md", 'w') as f:
        f.write(readme)
    
    return str(sample_dir)


def visualize_sample(sample, save_path):
    """Create visualization grid (same as before)"""
    fig = plt.figure(figsize=(15, 15))
    
    rgb = sample['rgb']
    depth = sample['depth']
    pc = sample['pc']
    pred_rgb = sample['pred_rgb']
    pred_depth = sample['pred_depth']
    pred_pc = sample['pred_pc']
    name = sample['name']
    
    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_vis = torch.clamp(rgb * std + mean, 0, 1)
    pred_rgb_vis = torch.clamp(pred_rgb * std + mean, 0, 1)
    
    # Row 1: RGB
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(rgb_vis.permute(1, 2, 0).numpy())
    ax1.set_title('Original RGB', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(rgb_vis.permute(1, 2, 0).numpy())
    ax2.set_title('Input RGB (for reference)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(pred_rgb_vis.permute(1, 2, 0).numpy())
    ax3.set_title(f'Reconstructed RGB\nMSE: {sample["metrics"]["rgb_mse"]:.6f}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Depth
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(depth[0].numpy(), cmap='viridis')
    ax4.set_title('Original Depth', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(depth[0].numpy(), cmap='viridis')
    ax5.set_title('Input Depth (for reference)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(pred_depth[0].numpy(), cmap='viridis')
    ax6.set_title(f'Reconstructed Depth\nMSE: {sample["metrics"]["depth_mse"]:.6f}', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Row 3: Point Cloud
    pc_np = pc.numpy()
    pred_pc_np = pred_pc.numpy()
    
    # Subsample for visualization
    if pc_np.shape[0] > 1000:
        idx = np.random.choice(pc_np.shape[0], 1000, replace=False)
        pc_vis = pc_np[idx]
    else:
        pc_vis = pc_np
    
    if pred_pc_np.shape[0] > 1000:
        idx = np.random.choice(pred_pc_np.shape[0], 1000, replace=False)
        pred_pc_vis = pred_pc_np[idx]
    else:
        pred_pc_vis = pred_pc_np
    
    ax7 = plt.subplot(3, 3, 7, projection='3d')
    ax7.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
               c=pc_vis[:, 2], cmap='viridis', s=2, alpha=0.8)
    ax7.set_title(f'Original Point Cloud\n{pc_np.shape[0]} points', fontsize=10, fontweight='bold')
    ax7.view_init(elev=20, azim=45)
    
    ax8 = plt.subplot(3, 3, 8, projection='3d')
    ax8.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2],
               c=pc_vis[:, 2], cmap='viridis', s=2, alpha=0.8)
    ax8.set_title(f'Input Point Cloud\n(for reference)', fontsize=10, fontweight='bold')
    ax8.view_init(elev=20, azim=45)
    
    ax9 = plt.subplot(3, 3, 9, projection='3d')
    ax9.scatter(pred_pc_vis[:, 0], pred_pc_vis[:, 1], pred_pc_vis[:, 2],
               c=pred_pc_vis[:, 2], cmap='viridis', s=2, alpha=0.8)
    ax9.set_title(f'Reconstructed PC\n{pred_pc_np.shape[0]} points\nChamfer: {sample["metrics"]["pc_chamfer"]:.6f}', 
                 fontsize=10, fontweight='bold')
    ax9.view_init(elev=20, azim=45)
    
    plt.suptitle(f'Sample: {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_report(metrics, save_path):
    """Save and print metrics"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("📊 VALIDATION METRICS SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<25} {'Value':<15} {'Quality':<15}")
    print("-"*55)
    
    print(f"{'Total Loss':<25} {metrics['loss']:<15.6f}")
    print(f"{'RGB Loss':<25} {metrics['rgb_loss']:<15.6f}")
    print(f"{'Depth Loss':<25} {metrics['depth_loss']:<15.6f}")
    print(f"{'PC Loss':<25} {metrics['pc_loss']:<15.6f}")
    print()
    
    rgb_quality = "Excellent ✅" if metrics['rgb_mse'] < 0.003 else "Good ✅" if metrics['rgb_mse'] < 0.005 else "Moderate ⚠️"
    depth_quality = "Excellent ✅" if metrics['depth_mse'] < 0.002 else "Good ✅" if metrics['depth_mse'] < 0.003 else "Moderate ⚠️"
    
    print(f"{'RGB MSE':<25} {metrics['rgb_mse']:<15.6f} {rgb_quality:<15}")
    print(f"{'Depth MSE':<25} {metrics['depth_mse']:<15.6f} {depth_quality:<15}")
    print()
    
    chamfer_quality = "Excellent ✅" if metrics['pc_chamfer'] < 0.001 else "Good ✅" if metrics['pc_chamfer'] < 0.002 else "Moderate ⚠️"
    emd_quality = "Excellent ✅" if metrics['pc_emd'] < 0.03 else "Good ✅" if metrics['pc_emd'] < 0.05 else "Moderate ⚠️"
    
    print(f"{'PC Chamfer Distance':<25} {metrics['pc_chamfer']:<15.6f} {chamfer_quality:<15}")
    print(f"{'PC Earth Mover\'s Dist':<25} {metrics['pc_emd']:<15.6f} {emd_quality:<15}")
    
    print("="*80)
    print(f"\n✅ Metrics saved to: {save_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Validate EmbodiedMAE Model - Save All Data')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load config
    print(f"\n{'='*80}")
    print("🔧 LOADING CONFIGURATION")
    print(f"{'='*80}")
    
    if Path(args.config).exists():
        print(f"📋 Config: {args.config}")
        config = load_config(args.config)
    else:
        print(f"⚠️  Using default config")
        config = {
            'data': {'data_root': './data', 'img_size': 224, 'num_points': 10000},
            'model': {'model_size': 'base'},
            'system': {'num_workers': 4}
        }
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = output_dir / f"samples_{timestamp}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output: {output_dir}")
    print(f"📁 Samples: {samples_dir}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load dataset
    print(f"\n{'='*80}")
    print("📦 LOADING DATASET")
    print(f"{'='*80}")
    
    val_dataset = SorghumDataset(
        data_root=config['data']['data_root'],
        img_size=config['data']['img_size'],
        num_points=config['data']['num_points'],
        train=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True
    )
    
    print(f"✅ Dataset: {len(val_dataset)} samples")
    
    # Load model
    print(f"\n{'='*80}")
    print("🤖 LOADING MODEL")
    print(f"{'='*80}")
    
    if config['model']['model_size'] == 'small':
        model = embodied_mae_small(
            img_size=config['data']['img_size'],
            num_pc_tokens=196,
            target_points=config['data']['num_points']
        )
    else:
        model = embodied_mae_base(
            img_size=config['data']['img_size'],
            num_pc_tokens=196,
            target_points=config['data']['num_points']
        )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    print(f"\n{'='*80}")
    print("🔍 EVALUATING MODEL")
    print(f"{'='*80}")
    
    metrics, samples = evaluate_model(model, val_loader, device, args.num_samples)
    
    # Save metrics
    metrics_path = output_dir / f"validation_metrics_{timestamp}.json"
    save_metrics_report(metrics, metrics_path)
    
    # Save samples
    print(f"\n{'='*80}")
    print(f"💾 SAVING SAMPLES ({len(samples)} samples)")
    print(f"{'='*80}\n")
    
    saved_samples = []
    for i, sample in enumerate(samples):
        sample_dir = samples_dir / f"sample_{i+1:02d}_{sample['name']}"
        print(f"  [{i+1}/{len(samples)}] Saving {sample['name']}...")
        save_sample_data(sample, sample_dir, i+1)
        
        # Also create visualization
        viz_path = sample_dir / "visualization.png"
        visualize_sample(sample, viz_path)
        
        saved_samples.append(str(sample_dir))
    
    print(f"\n✅ Saved all samples to: {samples_dir}")
    
    # Create master README
    readme = f"""# Validation Results - {timestamp}

## Overview
- **Checkpoint**: {args.checkpoint}
- **Epoch**: {checkpoint.get('epoch', 'unknown')}
- **Number of samples**: {len(samples)}

## Metrics
- **RGB MSE**: {metrics['rgb_mse']:.6f}
- **Depth MSE**: {metrics['depth_mse']:.6f}
- **PC Chamfer**: {metrics['pc_chamfer']:.6f}
- **PC EMD**: {metrics['pc_emd']:.6f}

## Directory Structure
```
samples_{timestamp}/
├── sample_01_NAME/
│   ├── inputs/
│   │   ├── rgb.png
│   │   ├── rgb.pt
│   │   ├── depth.png
│   │   ├── depth.npy
│   │   ├── pointcloud.ply
│   │   └── pointcloud.npy
│   ├── outputs/
│   │   ├── rgb.png
│   │   ├── rgb.pt
│   │   ├── depth.png
│   │   ├── depth.npy
│   │   ├── pointcloud.ply
│   │   └── pointcloud.npy
│   ├── visualization.png
│   ├── metrics.json
│   └── README.md
├── sample_02_NAME/
│   └── ...
└── ...
```

## File Formats
- **PNG**: Images (RGB, Depth visualization)
- **PT**: PyTorch tensors (raw RGB data)
- **NPY**: NumPy arrays (Depth values, Point clouds)
- **PLY**: Point cloud format (viewable in MeshLab, CloudCompare)
- **JSON**: Metrics

## Samples
{chr(10).join([f"- {Path(s).name}" for s in saved_samples])}

## How to Use

### Load in Python
```python
import torch
import numpy as np

# Sample 1
rgb = torch.load('sample_01_NAME/inputs/rgb.pt')
depth = np.load('sample_01_NAME/inputs/depth.npy')
pc = np.load('sample_01_NAME/inputs/pointcloud.npy')

pred_rgb = torch.load('sample_01_NAME/outputs/rgb.pt')
pred_depth = np.load('sample_01_NAME/outputs/depth.npy')
pred_pc = np.load('sample_01_NAME/outputs/pointcloud.npy')
```

### View Point Clouds
- Open `.ply` files in MeshLab or CloudCompare
- Or use Open3D in Python

### View Images
- PNG files can be opened with any image viewer
"""
    
    with open(samples_dir / "README.md", 'w') as f:
        f.write(readme)
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Metrics: {metrics_path}")
    print(f"Samples: {samples_dir}")
    print(f"  - {len(samples)} samples saved")
    print(f"  - Each has inputs/ and outputs/ directories")
    print(f"  - Formats: PNG, PT, NPY, PLY, JSON")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
