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


def _pc_token_membership(xyz: torch.Tensor, fps_idx: torch.Tensor, group_size: int) -> np.ndarray:
    """Return (num_tokens, group_size) array of point indices per FPS token."""
    centers = xyz[0, fps_idx[0]]                       # (S, 3)
    dist    = torch.cdist(centers.unsqueeze(0), xyz)   # (1, S, N)
    _, idx  = torch.topk(dist[0], group_size, dim=1, largest=False)  # (S, k)
    return idx.cpu().numpy()


def _scatter3d(ax, pts, c, s=2, alpha=0.7, **kw):
    """3-D scatter with data-Z mapped to the plot Y-axis for upright plant view."""
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=c, s=s, alpha=alpha, **kw)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Z', fontsize=7)
    ax.set_zlabel('Y', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=20, azim=45)


def visualize_reconstruction_sorghum(model, dataloader, device, epoch, save_dir, num_samples=4):
    """
    Visualize reconstruction results for Sorghum data.

    Layout per sample (4 rows × 3 cols):
      Row 1 — RGB:        Original | Masked | Reconstructed
      Row 2 — Depth:      Original | Masked | Reconstructed
      Row 3 — PC (token): Original | FPS centres (green=visible, red=masked) | Reconstructed
      Row 4 — PC (pts):   Visible points | Masked points | (loss summary)
    """
    model.eval()
    saved_paths = []

    rgb_batch, depth_batch, pc_batch, names = next(iter(dataloader))
    rgb_batch   = rgb_batch[:num_samples].to(device)
    depth_batch = depth_batch[:num_samples].to(device)
    pc_batch    = pc_batch[:num_samples].to(device)
    names       = names[:num_samples]

    with torch.no_grad():
        loss, (loss_rgb, loss_depth, loss_pc), \
            (pred_rgb, pred_depth, pred_pc), \
            (mask_rgb, mask_depth, mask_pc) = model(rgb_batch, depth_batch, pc_batch)

        # FPS centres and token membership (computed once for the whole batch)
        fps_indices  = model.pc_embed.fps(pc_batch, model.num_pc_tokens)   # (B, S)
        member_idx_b = _pc_token_membership(pc_batch, fps_indices,
                                            model.pc_embed.group_size)     # (S, k) for batch item 0

    # Denormalize RGB
    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    rgb_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    rgb_denorm = rgb_batch * rgb_std + rgb_mean

    pred_rgb_img   = unpatchify(pred_rgb,   model.patch_size, 3, model.img_size)
    pred_depth_img = unpatchify(pred_depth, model.patch_size, 1, model.img_size)
    pred_rgb_denorm = pred_rgb_img * rgb_std + rgb_mean

    for idx in range(num_samples):
        fig = plt.figure(figsize=(16, 20))

        # ── Row 1: RGB ────────────────────────────────────────────────────────
        ax = plt.subplot(4, 3, 1)
        ax.imshow(rgb_denorm[idx].cpu().permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f'Original RGB\n{names[idx]}', fontsize=9, fontweight='bold')
        ax.axis('off')

        ax = plt.subplot(4, 3, 2)
        m = mask_rgb[idx].reshape(int(mask_rgb.shape[1]**.5), -1)
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0).unsqueeze(0).float(), size=(model.img_size, model.img_size), mode='nearest')
        ax.imshow((rgb_denorm[idx] * (1 - m[0])).cpu().permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f'Masked RGB\n({mask_rgb[idx].mean().item():.1%} masked)', fontsize=9)
        ax.axis('off')

        ax = plt.subplot(4, 3, 3)
        ax.imshow(pred_rgb_denorm[idx].cpu().permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f'Reconstructed RGB\nLoss: {loss_rgb.item():.4f}', fontsize=9)
        ax.axis('off')

        # ── Row 2: Depth ──────────────────────────────────────────────────────
        depth_data = depth_batch[idx, 0].cpu().numpy()
        bg = depth_data < 0.01

        ax = plt.subplot(4, 3, 4)
        d = depth_data.copy(); d[bg] = np.nan
        ax.imshow(d, cmap='viridis'); ax.set_title('Original Depth', fontsize=9, fontweight='bold')
        ax.axis('off'); ax.set_facecolor('white')

        ax = plt.subplot(4, 3, 5)
        md = mask_depth[idx].reshape(int(mask_depth.shape[1]**.5), -1)
        md = torch.nn.functional.interpolate(
            md.unsqueeze(0).unsqueeze(0).float(), size=(model.img_size, model.img_size), mode='nearest')
        dm = (depth_batch[idx] * (1 - md[0]))[0].cpu().numpy()
        dm_disp = dm.copy(); dm_disp[bg] = np.nan; dm_disp[dm < 0.01] = np.nan
        ax.imshow(dm_disp, cmap='viridis')
        ax.set_title(f'Masked Depth\n({mask_depth[idx].mean().item():.1%} masked)', fontsize=9)
        ax.axis('off'); ax.set_facecolor('white')

        ax = plt.subplot(4, 3, 6)
        pd_disp = pred_depth_img[idx, 0].cpu().numpy().copy(); pd_disp[bg] = np.nan
        ax.imshow(pd_disp, cmap='viridis')
        ax.set_title(f'Reconstructed Depth\nLoss: {loss_depth.item():.4f}', fontsize=9)
        ax.axis('off'); ax.set_facecolor('white')

        # ── Row 3: Point cloud — original / FPS masking / reconstructed ───────
        pc_np = pc_batch[idx].cpu().numpy()
        mask_pc_s = mask_pc[idx].cpu().numpy()          # (S,)  1=masked 0=visible
        n_tok     = model.num_pc_tokens
        n_masked  = int(mask_pc_s.sum())
        n_visible = n_tok - n_masked

        # FPS centres for this sample
        centers = pc_batch[idx, fps_indices[idx]].cpu().numpy()   # (S, 3)
        vis_cen = centers[mask_pc_s == 0]
        msk_cen = centers[mask_pc_s == 1]

        # Original PC
        ax = plt.subplot(4, 3, 7, projection='3d')
        sub = pc_np[np.random.choice(len(pc_np), min(2000, len(pc_np)), replace=False)]
        _scatter3d(ax, sub, c=sub[:, 2], cmap='viridis', s=2)
        ax.set_title(f'Original PC\n{len(pc_np)} pts', fontsize=9, fontweight='bold')

        # FPS token centres coloured by visibility
        ax = plt.subplot(4, 3, 8, projection='3d')
        if len(vis_cen):
            _scatter3d(ax, vis_cen, c='#2ecc71', s=16, alpha=0.9, label='visible')
        if len(msk_cen):
            _scatter3d(ax, msk_cen, c='#e74c3c', s=16, alpha=0.9, label='masked')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_title(f'FPS token centres\nvisible={n_visible}  masked={n_masked}'
                     f'  ({100*n_masked/n_tok:.0f}%)', fontsize=9)

        # Reconstructed PC
        ax = plt.subplot(4, 3, 9, projection='3d')
        pred_np = pred_pc[idx].detach().cpu().numpy()
        pred_sub = pred_np[np.random.choice(len(pred_np), min(2000, len(pred_np)), replace=False)]
        _scatter3d(ax, pred_sub, c=pred_sub[:, 2], cmap='plasma', s=2)
        ax.set_title(f'Reconstructed PC\nLoss: {loss_pc.item():.4f}', fontsize=9)

        # ── Row 4: Visible / masked raw points ────────────────────────────────
        # Use the per-sample member_idx (recompute if sample != 0)
        if idx == 0:
            m_idx = member_idx_b
        else:
            with torch.no_grad():
                fps_i_s = model.pc_embed.fps(pc_batch[idx:idx+1], model.num_pc_tokens)
                m_idx   = _pc_token_membership(pc_batch[idx:idx+1], fps_i_s,
                                               model.pc_embed.group_size)

        vis_pts_idx = np.unique(m_idx[mask_pc_s == 0].flatten()) if n_visible > 0 else np.array([], dtype=int)
        msk_pts_idx = np.unique(m_idx[mask_pc_s == 1].flatten()) if n_masked  > 0 else np.array([], dtype=int)

        ax = plt.subplot(4, 3, 10, projection='3d')
        if len(vis_pts_idx):
            v = pc_np[vis_pts_idx[np.random.choice(len(vis_pts_idx), min(2000, len(vis_pts_idx)), replace=False)]]
            _scatter3d(ax, v, c='#2ecc71', s=3)
        ax.set_title(f'Visible pts\n({len(vis_pts_idx)} pts)', fontsize=9)

        ax = plt.subplot(4, 3, 11, projection='3d')
        if len(msk_pts_idx):
            m = pc_np[msk_pts_idx[np.random.choice(len(msk_pts_idx), min(2000, len(msk_pts_idx)), replace=False)]]
            _scatter3d(ax, m, c='#e74c3c', s=3)
        ax.set_title(f'Masked pts\n({len(msk_pts_idx)} pts)', fontsize=9)

        # Loss summary panel
        ax = plt.subplot(4, 3, 12)
        ax.axis('off')
        summary = (
            f"Epoch {epoch}  —  Sample {idx+1}\n\n"
            f"Total loss : {loss.item():.4f}\n"
            f"RGB loss   : {loss_rgb.item():.4f}\n"
            f"Depth loss : {loss_depth.item():.4f}\n"
            f"PC loss    : {loss_pc.item():.4f}\n\n"
            f"PC tokens masked : {n_masked}/{n_tok}  ({100*n_masked/n_tok:.0f}%)\n"
            f"PC pts visible   : {len(vis_pts_idx)}\n"
            f"PC pts masked    : {len(msk_pts_idx)}"
        )
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        plt.suptitle(f'Epoch {epoch}  |  {names[idx]}  |  Total Loss: {loss.item():.4f}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / f'epoch_{epoch:03d}_sample_{idx+1}_{names[idx]}.png'
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
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
    parser.add_argument('--data_root', type=str, default='./Dataset/SorghumData/',
                       help='Path to data directory containing Sorghum folders')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_points', type=int, default=2048)
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base'])
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--pc_loss_weight', type=float, default=1.0,
                       help='Weight for point cloud loss (Chamfer distance is naturally small, so we scale it up)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./outputs/outputs_sorghum_lr-3mr-15_new_data_bs_64')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., ./outputs_sorghum/checkpoints/checkpoint_epoch_50.pth)')
    

    
    # Weights & Biases (wandb) arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases for logging (default: True)')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                       help='Disable Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='embodied-mae-sorghum_lr-3mr-15_new_data_bs_64',
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
        num_points=args.num_points, split='train'
    )
    val_dataset = SorghumDataset(
        args.data_root, img_size=args.img_size,
        num_points=args.num_points, split='val'
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
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n{'='*80}")
            print(f"📂 Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            history = checkpoint.get('history', {
                'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
                'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': []
            })
            
            print(f"✅ Resumed from epoch {checkpoint['epoch']}")
            print(f"   Starting from epoch {start_epoch}")
            print(f"   Best validation loss so far: {best_val_loss:.4f}")
            print(f"{'='*80}\n")
        else:
            print(f"\n⚠️  Checkpoint not found: {args.resume}")
            print(f"   Starting from scratch instead.\n")
            best_val_loss = float('inf')
            history = {
                'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
                'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': []
            }
    else:
        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'train_rgb': [], 'train_depth': [], 'train_pc': [],
            'val_loss': [], 'val_rgb': [], 'val_depth': [], 'val_pc': []
        }
    
    # Training loop
    
    print(f"\nStarting training for {args.epochs} epochs...")
    if args.resume and start_epoch > 1:
        print(f"Resuming from epoch {start_epoch} (already completed {start_epoch - 1} epochs)")
    print(f"Visualizations will be saved every {args.viz_freq} epochs to: {viz_dir}")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs + 1):
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
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'history': history,
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
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'history': history,
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