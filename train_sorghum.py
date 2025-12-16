"""
Training Script for Sorghum Data with Visualization
Automatically visualizes reconstruction results every 10 epochs
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

from embodied_mae import embodied_mae_small, embodied_mae_base
from sorghum_dataset import SorghumDataset


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
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    total_pc_loss = 0
    
    for rgb, depth, pc, names in tqdm(dataloader, desc='Evaluating'):
        rgb = rgb.to(device)
        depth = depth.to(device)
        pc = pc.to(device)
        
        loss, (loss_rgb, loss_depth, loss_pc), _, _ = model(rgb, depth, pc)
        
        total_loss += loss.item()
        total_rgb_loss += loss_rgb.item()
        total_depth_loss += loss_depth.item()
        total_pc_loss += loss_pc.item()
    
    n = len(dataloader)
    return total_loss/n, total_rgb_loss/n, total_depth_loss/n, total_pc_loss/n


def main():
    parser = argparse.ArgumentParser(description='Train EmbodiedMAE on Sorghum Data')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to data directory containing Sorghum folders')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_points', type=int, default=2048)
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base'])
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--pc_loss_weight', type=float, default=10.0,
                       help='Weight for point cloud loss (Chamfer distance is naturally small, so we scale it up)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1.5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    # Visualization
    parser.add_argument('--viz_freq', type=int, default=100,
                       help='Visualize every N epochs')
    parser.add_argument('--num_viz_samples', type=int, default=6,
                       help='Number of samples to visualize')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./outputs_sorghum_lr-3mr-15_pc_w_0')
    parser.add_argument('--save_freq', type=int, default=10)
    
    # Weights & Biases (wandb) arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases for logging (default: True)')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                       help='Disable Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='embodied-mae-sorghum_lr-3mr-15_pc_w_0',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name (auto-generated if not specified)')

                       
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    print(f"\nLoading data from: {args.data_root}")
    train_dataset = SorghumDataset(
        args.data_root, img_size=args.img_size, 
        num_points=args.num_points, train=True
    )
    val_dataset = SorghumDataset(
        args.data_root, img_size=args.img_size,
        num_points=args.num_points, train=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    print(f"\nInitializing EmbodiedMAE-{args.model_size.capitalize()}...")
    if args.model_size == 'small':
        model = embodied_mae_small(
            img_size=args.img_size, 
            num_pc_tokens=196,
            pc_loss_weight=args.pc_loss_weight
        )
    else:
        model = embodied_mae_base(
            img_size=args.img_size, 
            num_pc_tokens=196,
            pc_loss_weight=args.pc_loss_weight
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize Weights & Biases
    if args.use_wandb and WANDB_AVAILABLE:
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
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
        'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': []
    }
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Visualizations will be saved every {args.viz_freq} epochs to: {viz_dir}")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
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
        
        print(f"\nTrain - Loss: {train_loss:.4f} | RGB: {train_rgb:.4f} | Depth: {train_depth:.4f} | PC: {train_pc:.4f}")
        
        # Validate
        val_loss, val_rgb, val_depth, val_pc = evaluate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_rgb'].append(val_rgb)
        history['val_depth'].append(val_depth)
        history['val_pc'].append(val_pc)
        
        print(f"Val   - Loss: {val_loss:.4f} | RGB: {val_rgb:.4f} | Depth: {val_depth:.4f} | PC: {val_pc:.4f}")
        
        # Log to Weights & Biases
        if args.use_wandb and WANDB_AVAILABLE:
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
                'learning_rate': scheduler.get_last_lr()[0],
            })
        
        # Visualize every N epochs
        if epoch % args.viz_freq == 0 or epoch == 1:
            print(f"\n📊 Generating visualizations for epoch {epoch}...")
            saved_paths = visualize_reconstruction_sorghum(
                model, val_loader, device, epoch, viz_dir, args.num_viz_samples
            )
            
            # Log visualizations to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                images = []
                for path in saved_paths:
                    images.append(wandb.Image(path, caption=Path(path).name))
                wandb.log({'visualizations': images, 'epoch': epoch})
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path.name}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            print(f"⭐ New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save history
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=4)
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
