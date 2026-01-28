"""
Training Script for Sorghum Data with Visualization
Automatically visualizes reconstruction results every 10 epochs
"""

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

from embodied_mae import embodied_mae_small, embodied_mae_base
from sorghum_dataset import SorghumDataset


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """
    Merge YAML config with command-line arguments.
    Command-line arguments override config file values.
    """
    # Data parameters
    if args.data_root is not None:
        config['data']['data_root'] = args.data_root
    if args.img_size is not None:
        config['data']['img_size'] = args.img_size
    if args.num_points is not None:
        config['data']['num_points'] = args.num_points
    
    # Model parameters
    if args.model_size is not None:
        config['model']['model_size'] = args.model_size
    if args.mask_ratio is not None:
        config['model']['mask_ratio'] = args.mask_ratio
    if args.pc_loss_weight is not None:
        config['model']['pc_loss_weight'] = args.pc_loss_weight
    if args.normalize_depth_global is not None:
        config['model']['normalize_depth_global'] = args.normalize_depth_global
    if args.depth_norm_type is not None:
        config['model']['depth_norm_type'] = args.depth_norm_type
    
    # Training parameters
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if args.warmup_epochs is not None:
        config['training']['warmup_epochs'] = args.warmup_epochs
    if args.val_freq is not None:
        config['training']['val_freq'] = args.val_freq
    
    # Checkpointing
    if args.output_dir is not None:
        config['checkpointing']['output_dir'] = args.output_dir
    if args.save_freq is not None:
        config['checkpointing']['save_freq'] = args.save_freq
    if args.resume is not None:
        config['checkpointing']['resume'] = args.resume
    
    # Visualization
    if args.viz_freq is not None:
        config['visualization']['viz_freq'] = args.viz_freq
    if args.num_viz_samples is not None:
        config['visualization']['num_viz_samples'] = args.num_viz_samples
    
    # Distributed
    if args.world_size is not None:
        config['distributed']['world_size'] = args.world_size
    if args.dist_backend is not None:
        config['distributed']['dist_backend'] = args.dist_backend
    if args.dist_url is not None:
        config['distributed']['dist_url'] = args.dist_url
    
    # System
    if args.num_workers is not None:
        config['system']['num_workers'] = args.num_workers
    if args.device is not None:
        config['system']['device'] = args.device
    
    # Wandb
    if hasattr(args, 'use_wandb') and args.use_wandb is not None:
        config['wandb']['use_wandb'] = args.use_wandb
    if args.wandb_project is not None:
        config['wandb']['project'] = args.wandb_project
    if args.wandb_entity is not None:
        config['wandb']['entity'] = args.wandb_entity
    if args.wandb_name is not None:
        config['wandb']['name'] = args.wandb_name
    
    return config


def config_to_flat_dict(config):
    """Convert nested config dict to flat dict for logging"""
    flat = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}/{key}"] = value
        else:
            flat[section] = values
    return flat


def setup_distributed(rank, world_size, dist_backend='nccl', dist_url='env://'):
    """Initialize distributed training"""
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def unpatchify(x, patch_size, channels, img_size):
    """
    Reconstruct image from patches
    """
    p = patch_size
    h = w = img_size // p
    
    x = x.reshape(x.shape[0], h, w, p, p, channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], channels, h * p, w * p)
    return imgs


def visualize_reconstruction_sorghum(model, dataloader, device, epoch, save_dir, num_samples=4):
    """
    Visualize reconstruction results for Sorghum data
    
    Args:
        model: EmbodiedMAE model
        dataloader: DataLoader
        device: torch device
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        
    Returns:
        saved_paths: List of saved visualization file paths
    """
    model.eval()
    
    saved_paths = []
    
    # Get a batch
    rgb_batch, depth_batch, pc_batch, names = next(iter(dataloader))
    
    # Limit to num_samples
    rgb_batch = rgb_batch[:num_samples].to(device)
    depth_batch = depth_batch[:num_samples].to(device)
    pc_batch = pc_batch[:num_samples].to(device)
    names = names[:num_samples]
    
    with torch.no_grad():
        # Forward pass
        loss, (loss_rgb, loss_depth, loss_pc), (pred_rgb, pred_depth, pred_pc), (mask_rgb, mask_depth, mask_pc) = model(
            rgb_batch, depth_batch, pc_batch
        )
    
    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    rgb_denorm = rgb_batch * std + mean
    
    # Unpatchify predictions
    pred_rgb_img = unpatchify(pred_rgb, model.patch_size, 3, model.img_size)
    pred_depth_img = unpatchify(pred_depth, model.patch_size, 1, model.img_size)
    
    # Denormalize predicted RGB
    pred_rgb_denorm = pred_rgb_img * std + mean
    
    # Create background mask for depth (where depth is near 0)
    depth_bg_mask = (depth_batch < 0.01).float()  # Background is near 0
    
    # Create visualization for each sample
    for idx in range(num_samples):
        fig = plt.figure(figsize=(16, 12))  # Square-ish for 3x3 grid
        
        # ========== ROW 1: RGB ==========
        # Original RGB
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(rgb_denorm[idx].cpu().permute(1, 2, 0).clamp(0, 1))
        ax1.set_title(f'Original RGB\n{names[idx]}', fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Masked RGB
        ax2 = plt.subplot(3, 3, 2)
        mask_rgb_img = mask_rgb[idx].reshape(int(mask_rgb.shape[1]**.5), int(mask_rgb.shape[1]**.5))
        mask_rgb_img = mask_rgb_img.unsqueeze(0).unsqueeze(0).float()
        mask_rgb_img = torch.nn.functional.interpolate(
            mask_rgb_img, size=(model.img_size, model.img_size), mode='nearest'
        )
        rgb_masked = rgb_denorm[idx] * (1 - mask_rgb_img[0])
        ax2.imshow(rgb_masked.cpu().permute(1, 2, 0).clamp(0, 1))
        ax2.set_title(f'Masked RGB\n({mask_rgb[idx].mean().item():.1%} masked)', fontsize=10)
        ax2.axis('off')
        
        # Reconstructed RGB
        ax3 = plt.subplot(3, 3, 3)
        ax3.imshow(pred_rgb_denorm[idx].cpu().permute(1, 2, 0).clamp(0, 1))
        ax3.set_title(f'Reconstructed RGB\nLoss: {loss_rgb.item():.4f}', fontsize=10)
        ax3.axis('off')
        
        # ========== ROW 2: Depth ==========
        # Depth - Original
        ax4 = plt.subplot(3, 3, 4)
        depth_data = depth_batch[idx, 0].cpu().numpy()
        # Set background pixels to NaN for complete transparency
        depth_display = depth_data.copy()
        depth_display[depth_data < 0.01] = np.nan
        im4 = ax4.imshow(depth_display, cmap='viridis')
        ax4.set_title('Original Depth\n(no background)', fontsize=10, fontweight='bold')
        ax4.axis('off')
        ax4.set_facecolor('white')  # White background for clarity
        
        # Depth - Masked
        ax5 = plt.subplot(3, 3, 5)
        mask_depth_img = mask_depth[idx].reshape(int(mask_depth.shape[1]**.5), int(mask_depth.shape[1]**.5))
        mask_depth_img = mask_depth_img.unsqueeze(0).unsqueeze(0).float()
        mask_depth_img = torch.nn.functional.interpolate(
            mask_depth_img, size=(model.img_size, model.img_size), mode='nearest'
        )
        depth_masked = depth_batch[idx] * (1 - mask_depth_img[0])
        depth_masked_data = depth_masked[0].cpu().numpy()
        # Set background AND masked pixels to NaN
        depth_masked_display = depth_masked_data.copy()
        depth_masked_display[depth_data < 0.01] = np.nan  # Background
        depth_masked_display[depth_masked_data < 0.01] = np.nan  # Masked regions
        im5 = ax5.imshow(depth_masked_display, cmap='viridis')
        ax5.set_title(f'Masked Depth\n({mask_depth[idx].mean().item():.1%} masked)', fontsize=10)
        ax5.axis('off')
        ax5.set_facecolor('white')
        
        # Depth - Reconstructed
        ax6 = plt.subplot(3, 3, 6)
        pred_depth_data = pred_depth_img[idx, 0].cpu().numpy()
        # Apply background mask to reconstruction (set background to NaN)
        pred_depth_display = pred_depth_data.copy()
        pred_depth_display[depth_data < 0.01] = np.nan
        im6 = ax6.imshow(pred_depth_display, cmap='viridis')
        ax6.set_title(f'Reconstructed Depth\nLoss: {loss_depth.item():.4f}', fontsize=10)
        ax6.axis('off')
        ax6.set_facecolor('white')
        
        # ========== ROW 3: Point Cloud ==========
        # Point Cloud - Original
        ax7 = plt.subplot(3, 3, 7, projection='3d')
        pc_np = pc_batch[idx].cpu().numpy()
        # Subsample for visualization
        if pc_np.shape[0] > 1000:
            vis_indices_orig = np.random.choice(pc_np.shape[0], 1000, replace=False)
            pc_vis = pc_np[vis_indices_orig]
        else:
            pc_vis = pc_np
        
        # Scatter with color based on Z coordinate
        scatter = ax7.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2], 
                             c=pc_vis[:, 2], cmap='viridis', s=2, alpha=0.8)
        ax7.set_title(f'Original Point Cloud\n{pc_np.shape[0]} points', 
                     fontsize=10, fontweight='bold')
        ax7.set_xlabel('X', fontsize=8)
        ax7.set_ylabel('Y', fontsize=8)
        ax7.set_zlabel('Z', fontsize=8)
        ax7.view_init(elev=20, azim=45)
        
        # Point Cloud - Masked
        ax8 = plt.subplot(3, 3, 8, projection='3d')
        # Get mask for this sample
        mask_pc_sample = mask_pc[idx].cpu().numpy()
        # Mask indicates which tokens are MASKED (1 = masked, 0 = visible)
        # We want to show visible tokens
        visible_mask = (1 - mask_pc_sample).astype(bool)
        
        # Since we don't have direct token-to-point mapping in visualization,
        # we'll show a portion of points based on mask ratio to illustrate masking
        num_visible = int(pc_np.shape[0] * (1 - mask_pc_sample.mean()))
        if num_visible > 0:
            visible_indices = np.random.choice(pc_np.shape[0], min(num_visible, 1000), replace=False)
            pc_masked_vis = pc_np[visible_indices]
        else:
            pc_masked_vis = pc_np[:10]  # Show at least a few points
        
        scatter2 = ax8.scatter(pc_masked_vis[:, 0], pc_masked_vis[:, 1], pc_masked_vis[:, 2],
                              c=pc_masked_vis[:, 2], cmap='viridis', s=2, alpha=0.8)
        ax8.set_title(f'Masked Point Cloud\n({mask_pc_sample.mean():.1%} masked, ~{num_visible} visible)', 
                     fontsize=10, fontweight='bold')
        ax8.set_xlabel('X', fontsize=8)
        ax8.set_ylabel('Y', fontsize=8)
        ax8.set_zlabel('Z', fontsize=8)
        ax8.view_init(elev=20, azim=45)
        
        # Point Cloud - Reconstructed
        ax9 = plt.subplot(3, 3, 9, projection='3d')
        pred_pc_np = pred_pc[idx].detach().cpu().numpy()
        if pred_pc_np.shape[0] > 1000:
            vis_indices = np.random.choice(pred_pc_np.shape[0], 1000, replace=False)
            pred_pc_vis = pred_pc_np[vis_indices]
        else:
            pred_pc_vis = pred_pc_np
        
        # Scatter with color based on Z coordinate
        scatter3 = ax9.scatter(pred_pc_vis[:, 0], pred_pc_vis[:, 1], pred_pc_vis[:, 2],
                              c=pred_pc_vis[:, 2], cmap='viridis', s=2, alpha=0.8)
        ax9.set_title(f'Reconstructed Point Cloud\n{pred_pc_np.shape[0]} points | Loss: {loss_pc.item():.4f}', 
                     fontsize=10, fontweight='bold')
        ax9.set_xlabel('X', fontsize=8)
        ax9.set_ylabel('Y', fontsize=8)
        ax9.set_zlabel('Z', fontsize=8)
        ax9.view_init(elev=20, azim=45)  # Same viewing angle for comparison
        
        plt.suptitle(f'Epoch {epoch} - Sample {idx+1} - Total Loss: {loss.item():.4f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = save_dir / f'epoch_{epoch:03d}_sample_{idx+1}_{names[idx]}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(str(save_path))
        print(f"  Saved: {save_path.name}")
    
    model.train()
    return saved_paths


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    total_pc_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (rgb, depth, pc, names) in enumerate(pbar):
        rgb = rgb.to(device)
        depth = depth.to(device)
        pc = pc.to(device)
        
        # Forward pass
        loss, (loss_rgb, loss_depth, loss_pc), _, _ = model(rgb, depth, pc)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_rgb_loss += loss_rgb.item()
        total_depth_loss += loss_depth.item()
        total_pc_loss += loss_pc.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rgb': f'{loss_rgb.item():.4f}',
            'depth': f'{loss_depth.item():.4f}',
            'pc': f'{loss_pc.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss/n, total_rgb_loss/n, total_depth_loss/n, total_pc_loss/n


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and calculate reconstruction metrics"""
    model.eval()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    total_pc_loss = 0
    
    # Metrics
    total_rgb_mse = 0
    total_depth_mse = 0
    total_pc_chamfer = 0
    total_pc_emd = 0  # NEW: Initialize EMD accumulator
    
    with torch.no_grad():
        for rgb, depth, pc, names in tqdm(dataloader, desc='Evaluating'):
            rgb = rgb.to(device)
            depth = depth.to(device)
            pc = pc.to(device)
            
            # Get loss and predictions (predictions are in patch format)
            loss, (loss_rgb, loss_depth, loss_pc), (pred_rgb_patches, pred_depth_patches, pred_pc), _ = model(rgb, depth, pc)
            
            # Accumulate losses
            total_loss += loss.item()
            total_rgb_loss += loss_rgb.item()
            total_depth_loss += loss_depth.item()
            total_pc_loss += loss_pc.item()
            
            # Unpatchify predictions to get back to image format for metric calculation
            # pred_rgb_patches: (B, L, patch_size^2 * 3)
            # pred_depth_patches: (B, L, patch_size^2 * 1)
            B = rgb.shape[0]
            patch_size = model.module.patch_size if hasattr(model, 'module') else model.patch_size
            img_size = rgb.shape[2]
            
            # Unpatchify RGB
            p = patch_size
            h = w = img_size // p
            # pred_rgb_patches: (B, h*w, p*p*3)
            pred_rgb_img = pred_rgb_patches.reshape(B, h, w, p, p, 3)
            pred_rgb_img = torch.einsum('nhwpqc->nchpwq', pred_rgb_img)
            pred_rgb_img = pred_rgb_img.reshape(B, 3, img_size, img_size)
            
            # Unpatchify Depth
            # pred_depth_patches: (B, h*w, p*p*1)
            pred_depth_img = pred_depth_patches.reshape(B, h, w, p, p, 1)
            pred_depth_img = torch.einsum('nhwpqc->nchpwq', pred_depth_img)
            pred_depth_img = pred_depth_img.reshape(B, 1, img_size, img_size)
            
            # Calculate reconstruction metrics on full images
            # RGB MSE (mean over all dimensions)
            rgb_mse = torch.mean((pred_rgb_img - rgb) ** 2).item()
            total_rgb_mse += rgb_mse
            
            # Depth MSE (mean over all dimensions)
            depth_mse = torch.mean((pred_depth_img - depth) ** 2).item()
            total_depth_mse += depth_mse
            
            # Point Cloud Metrics (already in right format)
            from embodied_mae import chamfer_distance, earth_movers_distance
            
            # Chamfer Distance
            pc_chamfer = chamfer_distance(pred_pc, pc).item()
            total_pc_chamfer += pc_chamfer
            
            # Earth Mover's Distance (EMD)
            pc_emd = earth_movers_distance(pred_pc, pc, num_samples=1000).item()
            total_pc_emd += pc_emd
    
    n = len(dataloader)
    metrics = {
        'loss': total_loss / n,
        'rgb_loss': total_rgb_loss / n,
        'depth_loss': total_depth_loss / n,
        'pc_loss': total_pc_loss / n,
        'rgb_mse': total_rgb_mse / n,
        'depth_mse': total_depth_mse / n,
        'pc_chamfer': total_pc_chamfer / n,
        'pc_emd': total_pc_emd / n,  # NEW: Add EMD to metrics
    }
    
    # Return old format for backward compatibility + metrics dict
    return (total_loss/n, total_rgb_loss/n, total_depth_loss/n, total_pc_loss/n), metrics


def train_worker(rank, world_size, args):
    """
    Training worker function for each GPU
    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        args: Parsed arguments
    """
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size, args.dist_backend, args.dist_url)
    
    # Determine if this is the main process (for logging and saving)
    is_main_process = (rank == 0)
    
    # Set device for this process
    if world_size > 1:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print(f"Using {'Multi-GPU' if world_size > 1 else 'Single-GPU'} training")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    # Create output directories (only main process)
    if is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        checkpoint_dir = output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        output_dir = Path(args.output_dir)
        viz_dir = output_dir / 'visualizations'
        checkpoint_dir = output_dir / 'checkpoints'
    
    # Dataset and DataLoader
    if is_main_process:
        print(f"\nLoading data from: {args.data_root}")
    
    train_dataset = SorghumDataset(
        args.data_root, img_size=args.img_size, 
        num_points=args.num_points, train=True
    )
    val_dataset = SorghumDataset(
        args.data_root, img_size=args.img_size,
        num_points=args.num_points, train=False
    )
    
    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle_train = False  # DistributedSampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Model
    if is_main_process:
        print(f"\nInitializing EmbodiedMAE-{args.model_size.capitalize()}...")
    
    if args.model_size == 'small':
        model = embodied_mae_small(
            img_size=args.img_size, 
            num_pc_tokens=196,
            pc_loss_weight=args.pc_loss_weight,
            target_points=args.num_points,
            normalize_depth_global=args.normalize_depth_global,
            depth_norm_type=args.depth_norm_type
        )
    else:
        model = embodied_mae_base(
            img_size=args.img_size, 
            num_pc_tokens=196,
            pc_loss_weight=args.pc_loss_weight,
            target_points=args.num_points,
            normalize_depth_global=args.normalize_depth_global,
            depth_norm_type=args.depth_norm_type
        )
    
    model = model.to(device)
    
    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"Total parameters: {total_params:,}")
    
    # Initialize Weights & Biases (only on main process)
    if args.use_wandb and WANDB_AVAILABLE and is_main_process:
        # Auto-generate run name if not provided
        if args.wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_name = f"sorghum_{args.model_size}_mr{args.mask_ratio}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                'model_size': args.model_size,
                'img_size': args.img_size,
                'num_points': args.num_points,
                'mask_ratio': args.mask_ratio,
                'pc_loss_weight': args.pc_loss_weight,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'warmup_epochs': args.warmup_epochs,
                'total_params': total_params,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
            }
        )
        # Watch model
        wandb.watch(model, log='all', log_freq=100)
        print(f"✅ Weights & Biases initialized: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  wandb requested but not available. Install with: pip install wandb")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            if is_main_process:
                print(f"\n{'='*80}")
                print(f"📂 Loading checkpoint from: {args.resume}")
            
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Handle DDP model state dict (remove 'module.' prefix if needed)
            state_dict = checkpoint['model_state_dict']
            if world_size > 1:
                # Loading into DDP model
                if not list(state_dict.keys())[0].startswith('module.'):
                    # Checkpoint from non-DDP, add 'module.' prefix
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
            else:
                # Loading into non-DDP model  
                if list(state_dict.keys())[0].startswith('module.'):
                    # Checkpoint from DDP, remove 'module.' prefix
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler if available (for backward compatibility with old checkpoints)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if is_main_process:
                    print(f"   ✓ Scheduler state restored")
            else:
                if is_main_process:
                    print(f"   ⚠️  Old checkpoint format - scheduler will restart from current epoch")
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            history = checkpoint.get('history', {
                'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
                'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': [],
                'val_rgb_mse': [], 'val_depth_mse': [], 'val_pc_chamfer': [], 'val_pc_emd': []
            })
            
            # Ensure val_pc_emd exists (backward compatibility with old checkpoints)
            if 'val_pc_emd' not in history:
                history['val_pc_emd'] = []
            
            if is_main_process:
                print(f"✅ Resumed from epoch {checkpoint['epoch']}")
                print(f"   Starting from epoch {start_epoch}")
                print(f"   Best validation loss so far: {best_val_loss:.4f}")
                print(f"{'='*80}\n")
        else:
            if is_main_process:
                print(f"\n⚠️  Checkpoint not found: {args.resume}")
                print(f"   Starting from scratch instead.\n")
            best_val_loss = float('inf')
            history = {
                'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
                'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': [],
                'val_rgb_mse': [], 'val_depth_mse': [], 'val_pc_chamfer': [], 'val_pc_emd': []
            }
    else:
        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
            'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': [],
            'val_rgb_mse': [], 'val_depth_mse': [], 'val_pc_chamfer': [], 'val_pc_emd': []
        }
    
    # Training loop
    
    print(f"\nStarting training for {args.epochs} epochs...")
    if args.resume and start_epoch > 1:
        print(f"Resuming from epoch {start_epoch} (already completed {start_epoch - 1} epochs)")
    print(f"Visualizations will be saved every {args.viz_freq} epochs to: {viz_dir}")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for DistributedSampler (for proper shuffling)
        if world_size > 1:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*80}")
        
        # Train
        train_loss, train_rgb, train_depth, train_pc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        history['train_loss'].append(train_loss)
        history['train_rgb'].append(train_rgb)
        history['train_depth'].append(train_depth)
        history['train_pc'].append(train_pc)
        
        if is_main_process:
            print(f"\nTrain - Loss: {train_loss:.4f} | RGB: {train_rgb:.4f} | Depth: {train_depth:.4f} | PC: {train_pc:.4f}")
        
        # Validate every val_freq epochs (or last epoch)
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            if is_main_process:
                print(f"\n🔍 Running validation...")
            
            # Validate and get metrics
            (val_loss, val_rgb, val_depth, val_pc), val_metrics = evaluate(model, val_loader, device)
            history['val_loss'].append(val_loss)
            history['val_rgb'].append(val_rgb)
            history['val_depth'].append(val_depth)
            history['val_pc'].append(val_pc)
            history['val_rgb_mse'].append(val_metrics['rgb_mse'])
            history['val_depth_mse'].append(val_metrics['depth_mse'])
            history['val_pc_chamfer'].append(val_metrics['pc_chamfer'])
            history['val_pc_emd'].append(val_metrics['pc_emd'])
            
            if is_main_process:
                print(f"Val   - Loss: {val_loss:.4f} | RGB: {val_rgb:.4f} | Depth: {val_depth:.4f} | PC: {val_pc:.4f}")
                print(f"Metrics - RGB MSE: {val_metrics['rgb_mse']:.6f} | Depth MSE: {val_metrics['depth_mse']:.6f} | PC Chamfer: {val_metrics['pc_chamfer']:.6f} | PC EMD: {val_metrics['pc_emd']:.6f}")
        else:
            # Skip validation for this epoch
            val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
            val_rgb = history['val_rgb'][-1] if history['val_rgb'] else 0.0
            val_depth = history['val_depth'][-1] if history['val_depth'] else 0.0
            val_pc = history['val_pc'][-1] if history['val_pc'] else 0.0
            val_metrics = {
                'rgb_mse': history['val_rgb_mse'][-1] if history['val_rgb_mse'] else 0.0,
                'depth_mse': history['val_depth_mse'][-1] if history['val_depth_mse'] else 0.0,
                'pc_chamfer': history['val_pc_chamfer'][-1] if history['val_pc_chamfer'] else 0.0,
                'pc_emd': history['val_pc_emd'][-1] if history['val_pc_emd'] else 0.0
            }
        
        # Log to Weights & Biases (only on main process)
        if is_main_process and args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/rgb_loss': train_rgb,
                'train/depth_loss': train_depth,
                'train/pc_loss': train_pc,
                'val/loss': val_loss,
                'val/rgb_loss': val_rgb,
                'val/depth_loss': val_depth,
                'val/pc_loss': val_pc,
                'metrics/rgb_mse': val_metrics['rgb_mse'],
                'metrics/depth_mse': val_metrics['depth_mse'],
                'metrics/pc_chamfer': val_metrics['pc_chamfer'],
                'metrics/pc_emd': val_metrics['pc_emd'],
                'learning_rate': scheduler.get_last_lr()[0],
            })
        
        # Visualize every N epochs (only on main process)
        if is_main_process and (epoch % args.viz_freq == 0 or epoch == 1):
            print(f"\n📊 Generating visualizations for epoch {epoch}...")
            # Get the actual model (unwrap DDP if needed)
            model_to_visualize = model.module if world_size > 1 else model
            saved_paths = visualize_reconstruction_sorghum(
                model_to_visualize, val_loader, device, epoch, viz_dir, args.num_viz_samples
            )
            
            # Log visualizations to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                images = []
                for path in saved_paths:
                    images.append(wandb.Image(path, caption=Path(path).name))
                wandb.log({'visualizations': images, 'epoch': epoch})
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint (only on main process)
        if is_main_process and epoch % args.save_freq == 0:
            # Get model state dict (remove 'module.' prefix if DDP)
            if world_size > 1:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
                
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'history': history,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path.name}")
        
        # Save best model (only on main process)
        if is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Get model state dict (remove 'module.' prefix if DDP)
            if world_size > 1:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
                
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'history': history,
            }, best_model_path)
            print(f"⭐ New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save history (only on main process)
        if is_main_process:
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=4)
    
    # Cleanup distributed training
    if world_size > 1:
        cleanup_distributed()
    
    # Finish wandb run (only on main process)
    if is_main_process and args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    if is_main_process:
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Outputs saved to: {output_dir}")
        print(f"Visualizations saved to: {viz_dir}")
        print(f"{'='*80}")


def main():
    """Main function - parses arguments and spawns training workers"""
    parser = argparse.ArgumentParser(description='Train EmbodiedMAE on Sorghum Data')
    
    # Config file
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config YAML file (default: config.yaml)')
    
    # Data parameters (override config)
    parser.add_argument('--data_root', type=str, default=None,
                       help='Path to data directory containing Sorghum folders')
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--num_points', type=int, default=None,
                       help='Number of points in point cloud')
    
    # Model parameters (override config)
    parser.add_argument('--model_size', type=str, default=None,
                       choices=['small', 'base'])
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--pc_loss_weight', type=float, default=None,
                       help='Weight for point cloud loss')
    parser.add_argument('--normalize_depth_global', type=lambda x: x.lower() == 'true', default=None,
                       help='Apply global depth normalization before loss (true/false)')
    parser.add_argument('--depth_norm_type', type=str, default=None,
                       choices=['minmax', 'standard'],
                       help='Type of depth normalization: minmax or standard')
    
    # Training parameters (override config)
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size PER GPU')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--val_freq', type=int, default=None,
                       help='Run validation every N epochs')
    
    # Visualization (override config)
    parser.add_argument('--viz_freq', type=int, default=None,
                       help='Visualize every N epochs')
    parser.add_argument('--num_viz_samples', type=int, default=None,
                       help='Number of samples to visualize')
    
    # System (override config)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Multi-GPU / Distributed Training (override config)
    parser.add_argument('--world_size', type=int, default=None,
                       help='Number of GPUs to use')
    parser.add_argument('--dist_backend', type=str, default=None,
                       help='Distributed backend')
    parser.add_argument('--dist_url', type=str, default=None,
                       help='URL for distributed training setup')
    
    # Weights & Biases (wandb) arguments (override config)
    parser.add_argument('--use_wandb', action='store_true', default=None,
                       help='Use Weights & Biases for logging')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                       help='Disable Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    if os.path.exists(args.config):
        print(f"📋 Loading configuration from: {args.config}")
        config = load_config(args.config)
        config = merge_config_with_args(config, args)
    else:
        print(f"⚠️  Config file not found: {args.config}")
        print(f"   Please create a config.yaml file or specify all parameters via command line.")
        return
    
    # Extract values from config for easy access
    cfg = argparse.Namespace()
    cfg.data_root = config['data']['data_root']
    cfg.img_size = config['data']['img_size']
    cfg.num_points = config['data']['num_points']
    cfg.model_size = config['model']['model_size']
    cfg.mask_ratio = config['model']['mask_ratio']
    cfg.pc_loss_weight = config['model']['pc_loss_weight']
    cfg.normalize_depth_global = config['model']['normalize_depth_global']
    cfg.depth_norm_type = config['model']['depth_norm_type']
    cfg.batch_size = config['training']['batch_size']
    cfg.epochs = config['training']['epochs']
    cfg.lr = config['training']['lr']
    cfg.weight_decay = config['training']['weight_decay']
    cfg.warmup_epochs = config['training']['warmup_epochs']
    cfg.val_freq = config['training']['val_freq']
    cfg.output_dir = config['checkpointing']['output_dir']
    cfg.save_freq = config['checkpointing']['save_freq']
    cfg.resume = config['checkpointing']['resume']
    cfg.viz_freq = config['visualization']['viz_freq']
    cfg.num_viz_samples = config['visualization']['num_viz_samples']
    cfg.world_size = config['distributed']['world_size']
    cfg.dist_backend = config['distributed']['dist_backend']
    cfg.dist_url = config['distributed']['dist_url']
    cfg.num_workers = config['system']['num_workers']
    cfg.device = config['system']['device']
    cfg.use_wandb = config['wandb']['use_wandb']
    cfg.wandb_project = config['wandb']['project']
    cfg.wandb_entity = config['wandb']['entity']
    cfg.wandb_name = config['wandb']['name']
    
    # Print multi-GPU info
    if cfg.world_size > 1:
        print(f"\n{'='*80}")
        print(f"🚀 Multi-GPU Training Enabled!")
        print(f"   Number of GPUs: {cfg.world_size}")
        print(f"   Batch size per GPU: {cfg.batch_size}")
        print(f"   Total batch size: {cfg.batch_size * cfg.world_size}")
        print(f"   Backend: {cfg.dist_backend}")
        print(f"{'='*80}\n")
        
        # Spawn processes for multi-GPU training
        mp.spawn(
            train_worker,
            args=(cfg.world_size, cfg),
            nprocs=cfg.world_size,
            join=True
        )
    else:
        # Single GPU training
        train_worker(0, 1, cfg)


if __name__ == '__main__':
    main()