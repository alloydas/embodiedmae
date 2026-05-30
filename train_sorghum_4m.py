"""
Training script for EmbodiedMAE-4M (RGB + Depth + PointCloud + Text parameters).
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")

from embodied_mae_4m import (
    EmbodiedMAE4M,
    embodied_mae_4m_small,
    embodied_mae_4m_base,
    _params_to_plant_text,
    _params_to_leaf_text,
    N_PARAMS,
)
from embodied_mae import chamfer_distance, earth_movers_distance
from sorghum_dataset_4m import SorghumDataset4M


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config_with_args(config, args):
    mapping = {
        'data': ['data_root', 'img_size', 'num_points'],
        'model': ['model_size', 'mask_ratio', 'pc_loss_weight',
                  'depth_norm_type', 'spline_loss_weight', 'max_leaves'],
        'training': ['batch_size', 'epochs', 'lr', 'weight_decay',
                     'warmup_epochs', 'val_freq'],
        'checkpointing': ['output_dir', 'save_freq', 'resume'],
        'visualization': ['viz_freq', 'num_viz_samples'],
        'distributed': ['world_size', 'dist_backend', 'dist_url'],
        'system': ['num_workers', 'device'],
        'wandb': ['use_wandb', 'wandb_project', 'wandb_entity', 'wandb_name'],
    }
    for section, keys in mapping.items():
        if section not in config:
            config[section] = {}
        for k in keys:
            v = getattr(args, k, None)
            if v is not None:
                config[section][k] = v
    return config


def config_to_namespace(config):
    ns = argparse.Namespace()
    ns.data_root          = config['data']['data_root']
    ns.img_size           = config['data'].get('img_size', 224)
    ns.num_points         = config['data'].get('num_points', 8196)
    ns.model_size         = config['model'].get('model_size', 'base')
    ns.mask_ratio         = config['model'].get('mask_ratio', 0.15)
    ns.pc_loss_weight     = config['model'].get('pc_loss_weight', 10.0)
    ns.depth_norm_type    = config['model'].get('depth_norm_type', 'minmax')
    ns.spline_loss_weight = config['model'].get('spline_loss_weight', 1.0)
    ns.max_leaves         = config['model'].get('max_leaves', 24)
    ns.batch_size         = config['training'].get('batch_size', 16)
    ns.epochs             = config['training'].get('epochs', 2400)
    ns.lr                 = config['training'].get('lr', 1.5e-4)
    ns.weight_decay       = config['training'].get('weight_decay', 0.05)
    ns.warmup_epochs      = config['training'].get('warmup_epochs', 10)
    ns.val_freq           = config['training'].get('val_freq', 20)
    ns.output_dir         = config['checkpointing'].get('output_dir', './outputs/4m_run')
    ns.save_freq          = config['checkpointing'].get('save_freq', 100)
    ns.resume             = config['checkpointing'].get('resume', None)
    ns.viz_freq           = config['visualization'].get('viz_freq', 50)
    ns.num_viz_samples    = config['visualization'].get('num_viz_samples', 6)
    ns.world_size         = config['distributed'].get('world_size', 1)
    ns.dist_backend       = config['distributed'].get('dist_backend', 'nccl')
    ns.dist_url           = config['distributed'].get('dist_url', 'env://')
    ns.num_workers        = config['system'].get('num_workers', 8)
    ns.device             = config['system'].get('device', 'cuda')
    ns.use_wandb          = config['wandb'].get('use_wandb', True)
    ns.wandb_project      = config['wandb'].get('wandb_project', 'embodied-mae-4m')
    ns.wandb_entity       = config['wandb'].get('wandb_entity', None)
    ns.wandb_name         = config['wandb'].get('wandb_name', None)
    return ns


# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_distributed(rank, world_size, backend, url):
    if int(os.environ.get('LOCAL_RANK', -1)) >= 0:
        dist.init_process_group(backend=backend, init_method='env://',
                                world_size=world_size, rank=rank)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group(backend=backend, init_method='tcp://localhost:12356',
                                world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Image / PC helpers ────────────────────────────────────────────────────────

def unpatchify(x, patch_size, channels, img_size):
    p = patch_size
    h = w = img_size // p
    x = x.reshape(x.shape[0], h, w, p, p, channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(x.shape[0], channels, h * p, w * p)


def _pc_token_membership(xyz, fps_idx, group_size):
    xyz_cpu  = xyz[0].cpu()
    fps_cpu  = fps_idx[0].cpu()
    centers  = xyz_cpu[fps_cpu]
    dist_mat = torch.cdist(centers.unsqueeze(0), xyz_cpu.unsqueeze(0))
    _, idx   = torch.topk(dist_mat[0], group_size, dim=1, largest=False)
    return idx.numpy()


def _scatter3d(ax, pts, c, s=2, alpha=0.7, **kw):
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=c, s=s, alpha=alpha, **kw)
    ax.set_xlabel('X', fontsize=6); ax.set_ylabel('Z', fontsize=6)
    ax.set_zlabel('Y', fontsize=6); ax.tick_params(labelsize=5)
    ax.view_init(elev=20, azim=45)


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_reconstruction_4m(model, dataloader, device, epoch, save_dir,
                                  num_samples=4):
    """5-row × 3-col grid per sample.
    Row 5 shows target vs predicted text for masked tokens.
    """
    model.eval()
    saved_paths = []

    batch = next(iter(dataloader))
    rgb_b, depth_b, pc_b, param_floats_b, text_valid_b, names = batch
    rgb_b         = rgb_b[:num_samples].to(device)
    depth_b       = depth_b[:num_samples].to(device)
    pc_b          = pc_b[:num_samples].to(device)
    param_floats_b = param_floats_b[:num_samples].to(device)
    text_valid_b  = text_valid_b[:num_samples].to(device)
    names         = list(names[:num_samples])

    with torch.no_grad():
        total, (lr, ld, lp, lt), \
            (pred_rgb, pred_depth, pred_pc, pred_params), \
            (m_rgb, m_depth, m_pc, m_text) = model(
                rgb_b, depth_b, pc_b, param_floats_b, text_valid_b
        )

        fps_indices = model.pc_embed.fps(pc_b, model.num_pc_tokens)

        rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        rgb_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

        rgb_dn_np   = (rgb_b * rgb_std + rgb_mean).clamp(0,1).cpu().numpy()
        depth_np    = depth_b.cpu().numpy()
        pc_np_all   = pc_b.cpu().numpy()
        fps_np      = fps_indices.cpu().numpy()
        m_rgb_np    = m_rgb.cpu().numpy()
        m_depth_np  = m_depth.cpu().numpy()
        m_pc_np     = m_pc.cpu().numpy()
        m_text_np   = m_text.cpu().numpy()          # (B, n_text_tokens)
        text_valid_np = text_valid_b.cpu().numpy()
        # Decode predicted params → text strings
        pred_text_strs = model.decode_params_to_text(pred_params)  # list[list[str]]
        tgt_text_strs  = model.decode_params_to_text(param_floats_b)  # ground truth

        pred_rgb_img   = unpatchify(pred_rgb,   model.patch_size, 3, model.img_size)
        pred_depth_img = unpatchify(pred_depth, model.patch_size, 1, model.img_size)
        pred_rgb_np    = (pred_rgb_img * rgb_std + rgb_mean).clamp(0,1).cpu().numpy()
        pred_depth_np  = pred_depth_img.cpu().numpy()
        pred_pc_np     = pred_pc.cpu().numpy()

        loss_val = total.item()
        lr_val, ld_val, lp_val, lt_val = lr.item(), ld.item(), lp.item(), lt.item()

        n_actual = pc_np_all.shape[0]   # batch may be smaller than num_samples under DDP

        member_idx_all = []
        for i in range(n_actual):
            fps_t = torch.from_numpy(fps_np[i:i+1])
            pc_t  = torch.from_numpy(pc_np_all[i:i+1])
            member_idx_all.append(
                _pc_token_membership(pc_t, fps_t, model.pc_embed.group_size))

    del rgb_b, depth_b, pc_b, param_floats_b, text_valid_b
    del pred_rgb, pred_depth, pred_pc, pred_params
    del m_rgb, m_depth, m_pc, m_text, fps_indices
    del pred_rgb_img, pred_depth_img, rgb_mean, rgb_std
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for idx in range(n_actual):
        fig = plt.figure(figsize=(16, 25))

        rgb_i      = rgb_dn_np[idx].transpose(1,2,0)
        depth_data = depth_np[idx, 0]
        bg         = depth_data < 0.01
        pc_np      = pc_np_all[idx]
        mask_pc_s  = m_pc_np[idx]
        n_tok      = model.num_pc_tokens
        n_masked   = int(mask_pc_s.sum())
        n_visible  = n_tok - n_masked
        centers    = pc_np[fps_np[idx]]
        m_idx      = member_idx_all[idx]

        # ── Row 1: RGB ────────────────────────────────────────────────────────
        ax = plt.subplot(5, 3, 1)
        ax.imshow(rgb_i.clip(0,1))
        ax.set_title(f'Original RGB\n{names[idx]}', fontsize=9, fontweight='bold')
        ax.axis('off')

        ax = plt.subplot(5, 3, 2)
        g = int(round(m_rgb_np[idx].shape[0] ** 0.5))
        m_up = np.kron(m_rgb_np[idx].reshape(g, g),
                       np.ones((model.img_size//g, model.img_size//g)))
        ax.imshow((rgb_i * (1 - m_up[:,:,None])).clip(0,1))
        ax.set_title(f'Masked RGB\n({m_rgb_np[idx].mean():.1%} masked)', fontsize=9)
        ax.axis('off')

        ax = plt.subplot(5, 3, 3)
        ax.imshow(pred_rgb_np[idx].transpose(1,2,0).clip(0,1))
        ax.set_title(f'Reconstructed RGB\nLoss: {lr_val:.4f}', fontsize=9)
        ax.axis('off')

        # ── Row 2: Depth ──────────────────────────────────────────────────────
        ax = plt.subplot(5, 3, 4)
        d = depth_data.copy(); d[bg] = np.nan
        ax.imshow(d, cmap='viridis')
        ax.set_title('Original Depth', fontsize=9, fontweight='bold'); ax.axis('off')

        ax = plt.subplot(5, 3, 5)
        gd = int(round(m_depth_np[idx].shape[0] ** 0.5))
        md_up = np.kron(m_depth_np[idx].reshape(gd,gd),
                        np.ones((model.img_size//gd, model.img_size//gd)))
        dm = depth_data * (1 - md_up); dm_d = dm.copy()
        dm_d[bg] = np.nan; dm_d[dm < 0.01] = np.nan
        ax.imshow(dm_d, cmap='viridis')
        ax.set_title(f'Masked Depth\n({m_depth_np[idx].mean():.1%} masked)', fontsize=9)
        ax.axis('off')

        ax = plt.subplot(5, 3, 6)
        pd_d = pred_depth_np[idx,0].copy(); pd_d[bg] = np.nan
        ax.imshow(pd_d, cmap='viridis')
        ax.set_title(f'Reconstructed Depth\nLoss: {ld_val:.4f}', fontsize=9); ax.axis('off')

        # ── Row 3: Point cloud centres ────────────────────────────────────────
        vis_cen = centers[mask_pc_s == 0]
        msk_cen = centers[mask_pc_s == 1]

        ax = plt.subplot(5, 3, 7, projection='3d')
        sub = pc_np[np.random.choice(len(pc_np), min(2000, len(pc_np)), replace=False)]
        _scatter3d(ax, sub, c=sub[:,2], cmap='viridis', s=2)
        ax.set_title(f'Original PC\n{len(pc_np)} pts', fontsize=9, fontweight='bold')

        ax = plt.subplot(5, 3, 8, projection='3d')
        if len(vis_cen): _scatter3d(ax, vis_cen, c='#2ecc71', s=16, alpha=0.9, label='visible')
        if len(msk_cen): _scatter3d(ax, msk_cen, c='#e74c3c', s=16, alpha=0.9, label='masked')
        ax.legend(fontsize=6, loc='upper left')
        ax.set_title(f'FPS centres\nvis={n_visible} msk={n_masked} ({100*n_masked/n_tok:.0f}%)',
                     fontsize=9)

        ax = plt.subplot(5, 3, 9, projection='3d')
        ppc = pred_pc_np[idx]
        ppc_sub = ppc[np.random.choice(len(ppc), min(2000, len(ppc)), replace=False)]
        _scatter3d(ax, ppc_sub, c=ppc_sub[:,2], cmap='plasma', s=2)
        ax.set_title(f'Reconstructed PC\nLoss: {lp_val:.6f}', fontsize=9)

        # ── Row 4: Visible / masked raw points + summary ──────────────────────
        vis_pts = np.unique(m_idx[mask_pc_s==0].flatten()) if n_visible > 0 else np.array([], dtype=int)
        msk_pts = np.unique(m_idx[mask_pc_s==1].flatten()) if n_masked  > 0 else np.array([], dtype=int)

        ax = plt.subplot(5, 3, 10, projection='3d')
        if len(vis_pts):
            vp = pc_np[vis_pts[np.random.choice(len(vis_pts), min(2000,len(vis_pts)), replace=False)]]
            _scatter3d(ax, vp, c='#2ecc71', s=3)
        ax.set_title(f'Visible pts ({len(vis_pts)})', fontsize=9)

        ax = plt.subplot(5, 3, 11, projection='3d')
        if len(msk_pts):
            mp2 = pc_np[msk_pts[np.random.choice(len(msk_pts), min(2000,len(msk_pts)), replace=False)]]
            _scatter3d(ax, mp2, c='#e74c3c', s=3)
        ax.set_title(f'Masked pts ({len(msk_pts)})', fontsize=9)

        ax = plt.subplot(5, 3, 12)
        ax.axis('off')
        n_text_masked  = int(m_text_np[idx].sum())
        n_text_real    = int(text_valid_np[idx].sum())
        summary = (
            f"Epoch {epoch}  —  {names[idx]}\n\n"
            f"Total   : {loss_val:.4f}\n"
            f"RGB     : {lr_val:.4f}\n"
            f"Depth   : {ld_val:.4f}\n"
            f"PC      : {lp_val:.6f}\n"
            f"Text    : {lt_val:.4f}\n\n"
            f"PC tok masked  : {n_masked}/{n_tok}\n"
            f"Text masked    : {n_text_masked}/{n_text_real} real tokens"
        )
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        # ── Row 5: Text — target vs predicted for masked tokens ───────────────
        mask_t  = m_text_np[idx]           # (n_text_tokens,)
        valid_t = text_valid_np[idx]      # (n_text_tokens,)

        token_labels = ['plant'] + [f'leaf{i}' for i in range(model.max_leaves)]
        lines_tgt  = []
        lines_pred = []
        for ti in range(len(valid_t)):
            if valid_t[ti] < 0.5:
                continue
            status   = 'MASKED' if mask_t[ti] > 0.5 else 'visible'
            tgt_str  = tgt_text_strs[idx][ti]   # formatted from ground-truth params
            pred_str = pred_text_strs[idx][ti]   # formatted from predicted params
            lines_tgt.append(f"[{token_labels[ti]}|{status}]\n  {tgt_str}")
            lines_pred.append(f"[{token_labels[ti]}|{status}]\n  {pred_str}")

        def _text_panel(ax, lines, title, bg_col):
            ax.axis('off')
            txt = '\n'.join(lines[:12]) if lines else '(none)'
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=6,
                    va='top', fontfamily='monospace', wrap=True,
                    bbox=dict(boxstyle='round', facecolor=bg_col, alpha=0.6))
            ax.set_title(title, fontsize=8, fontweight='bold')

        ax = plt.subplot(5, 3, 13)
        _text_panel(ax, lines_tgt,  f'Text target ({n_text_real} tokens)', '#d5f5e3')

        ax = plt.subplot(5, 3, 14)
        _text_panel(ax, lines_pred, f'Text predicted (masked={n_text_masked})', '#fde8d8')

        # Col 3: diff — highlight mismatches
        diff_lines = []
        for tgt, pred in zip(lines_tgt, lines_pred):
            match = '✓' if tgt.split('\n')[1].strip() == pred.split('\n')[1].strip() else '✗'
            diff_lines.append(f"{match} {tgt.split(chr(10))[0]}")
        ax = plt.subplot(5, 3, 15)
        _text_panel(ax, diff_lines, f'Match summary\nLoss: {lt_val:.4f}', '#eaf2ff')

        plt.suptitle(
            f'Epoch {epoch}  |  {names[idx]}  |  '
            f'Total: {loss_val:.4f}  RGB: {lr_val:.4f}  '
            f'Depth: {ld_val:.4f}  PC: {lp_val:.4f}  Text: {lt_val:.4f}',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()

        path = save_dir / f'epoch_{epoch:03d}_sample_{idx+1}_{names[idx]}.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_paths.append(str(path))
        print(f"  Saved: {path.name}")

    model.train()
    return saved_paths


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    tot = tot_rgb = tot_depth = tot_pc = tot_txt = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for rgb, depth, pc, param_floats, text_valid, _ in pbar:
        rgb          = rgb.to(device)
        depth        = depth.to(device)
        pc           = pc.to(device)
        param_floats = param_floats.to(device)
        text_valid   = text_valid.to(device)

        loss, (lr, ld, lp, lt), _, _ = model(rgb, depth, pc, param_floats, text_valid)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot     += loss.item()
        tot_rgb += lr.item()
        tot_depth += ld.item()
        tot_pc  += lp.item()
        tot_txt += lt.item()

        pbar.set_postfix({
            'loss':  f'{loss.item():.4f}',
            'rgb':   f'{lr.item():.4f}',
            'depth': f'{ld.item():.4f}',
            'pc':    f'{lp.item():.4f}',
            'text':  f'{lt.item():.4f}',
        })

    n = len(dataloader)
    return tot/n, tot_rgb/n, tot_depth/n, tot_pc/n, tot_txt/n


@torch.no_grad()
def evaluate(model, dataloader, device, compute_emd=False):
    model.eval()
    tot = tot_rgb = tot_depth = tot_pc = tot_txt = 0.0
    tot_rgb_mse = tot_depth_mse = tot_chamfer = tot_emd = 0.0
    tot_param_mse = tot_param_mae = tot_param_mae_masked = tot_param_acc05 = 0.0

    m = model.module if hasattr(model, 'module') else model

    for rgb, depth, pc, param_floats, text_valid, _ in tqdm(dataloader, desc='Evaluating'):
        rgb          = rgb.to(device)
        depth        = depth.to(device)
        pc           = pc.to(device)
        param_floats = param_floats.to(device)
        text_valid   = text_valid.to(device)

        loss, (lr, ld, lp, lt), \
            (pred_rgb_p, pred_depth_p, pred_pc, pred_params), \
            (_, _, _, mask_text) = model(
                rgb, depth, pc, param_floats, text_valid
        )

        tot       += loss.item()
        tot_rgb   += lr.item()
        tot_depth += ld.item()
        tot_pc    += lp.item()
        tot_txt   += lt.item()

        B, _, H, W = rgb.shape
        p = m.patch_size; h = w = H // p
        pred_rgb_img   = pred_rgb_p.reshape(B,h,w,p,p,3)
        pred_rgb_img   = torch.einsum('nhwpqc->nchpwq', pred_rgb_img).reshape(B,3,H,W)
        pred_depth_img = pred_depth_p.reshape(B,h,w,p,p,1)
        pred_depth_img = torch.einsum('nhwpqc->nchpwq', pred_depth_img).reshape(B,1,H,W)

        tot_rgb_mse   += torch.mean((pred_rgb_img   - rgb)   ** 2).item()
        tot_depth_mse += torch.mean((pred_depth_img - depth) ** 2).item()
        del pred_rgb_img, pred_depth_img

        tot_chamfer += chamfer_distance(pred_pc, pc).item()
        if compute_emd:
            tot_emd += earth_movers_distance(pred_pc, pc, num_samples=500).item()

        # Param-modality metrics, all on normalised params in [0, 1].
        # Reduce over the 9 slots per token (matches the Smooth-L1 loss reduction),
        # then average over the requested set of tokens.
        diff_abs = (pred_params - param_floats).abs().mean(-1)   # (B, L)
        diff_sq  = ((pred_params - param_floats) ** 2).mean(-1)  # (B, L)

        real_n   = text_valid.sum().clamp(min=1)
        masked_n = (text_valid * mask_text).sum().clamp(min=1)

        tot_param_mse        += ((diff_sq  * text_valid).sum() / real_n).item()
        tot_param_mae        += ((diff_abs * text_valid).sum() / real_n).item()
        tot_param_mae_masked += ((diff_abs * text_valid * mask_text).sum() / masked_n).item()
        tot_param_acc05      += (((diff_abs < 0.05).float() * text_valid * mask_text)
                                 .sum() / masked_n).item()

        del pred_pc, pred_params, mask_text
        del rgb, depth, pc, param_floats, text_valid

    n = len(dataloader)
    metrics = {
        'loss':              tot / n,
        'rgb_loss':          tot_rgb / n,
        'depth_loss':        tot_depth / n,
        'pc_loss':           tot_pc / n,
        'text_loss':         tot_txt / n,
        'rgb_mse':           tot_rgb_mse / n,
        'depth_mse':         tot_depth_mse / n,
        'pc_chamfer':        tot_chamfer / n,
        'pc_emd':            tot_emd / n,
        'param_mse':         tot_param_mse / n,
        'param_mae':         tot_param_mae / n,
        'param_mae_masked':  tot_param_mae_masked / n,
        'param_acc@0.05':    tot_param_acc05 / n,
    }
    return (tot/n, tot_rgb/n, tot_depth/n, tot_pc/n, tot_txt/n), metrics


# ── Worker ────────────────────────────────────────────────────────────────────

def train_worker(rank, world_size, args):
    if world_size > 1:
        setup_distributed(rank, world_size, args.dist_backend, args.dist_url)

    is_main = (rank == 0)
    device = (torch.device(f'cuda:{rank}') if world_size > 1
              else torch.device(args.device if torch.cuda.is_available() else 'cpu'))

    output_dir     = Path(args.output_dir)
    viz_dir        = output_dir / 'visualizations'
    checkpoint_dir = output_dir / 'checkpoints'
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    if is_main: print(f"\nLoading data from: {args.data_root}")

    train_ds = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                 num_points=args.num_points, split='train',
                                 max_leaves=args.max_leaves)
    val_ds   = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                 num_points=args.num_points, split='val',
                                 max_leaves=args.max_leaves)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=shuffle_train, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    if is_main: print(f"\nInitializing EmbodiedMAE-4M-{args.model_size.capitalize()}...")

    build_fn = embodied_mae_4m_small if args.model_size == 'small' else embodied_mae_4m_base
    model = build_fn(
        img_size=args.img_size,
        num_pc_tokens=196,
        target_points=args.num_points,
        pc_loss_weight=args.pc_loss_weight,
        max_leaves=args.max_leaves,
        spline_loss_weight=args.spline_loss_weight,
        depth_norm_type=args.depth_norm_type,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main: print(f"Total parameters: {total_params:,}")

    if is_main and args.use_wandb and WANDB_AVAILABLE:
        if args.wandb_name is None:
            args.wandb_name = f"4m_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run_id = None
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location='cpu')
            wandb_run_id = ckpt.get('wandb_run_id')
        if wandb_run_id:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       id=wandb_run_id, resume='must')
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_name, config={
                           'model_size': args.model_size, 'num_points': args.num_points,
                           'mask_ratio': args.mask_ratio, 'pc_loss_weight': args.pc_loss_weight,
                           'text_loss_weight': args.spline_loss_weight,
                           'max_leaves': args.max_leaves,
                           'batch_size': args.batch_size, 'epochs': args.epochs,
                           'lr': args.lr, 'total_params': total_params,
                           'train_samples': len(train_ds), 'val_samples': len(val_ds),
                       })
        print(f"✅ W&B: {wandb.run.url}")
        wandb.watch(model, log='gradients', log_freq=500)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (ep - args.warmup_epochs)
                                  / (args.epochs - args.warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch   = 1
    best_val_loss = float('inf')

    def empty_history():
        return {
            'train_loss':[], 'train_rgb':[], 'train_depth':[], 'train_pc':[], 'train_text':[],
            'val_loss':[],   'val_rgb':[],   'val_depth':[],   'val_pc':[],   'val_text':[],
            'val_rgb_mse':[], 'val_depth_mse':[], 'val_pc_chamfer':[], 'val_pc_emd':[],
            'val_param_mse':[], 'val_param_mae':[], 'val_param_mae_masked':[],
            'val_param_acc05':[],
        }
    history = empty_history()

    if args.resume and os.path.exists(args.resume):
        if is_main: print(f"\n📂 Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        sd   = ckpt['model_state_dict']
        if world_size > 1 and not list(sd.keys())[0].startswith('module.'):
            sd = {'module.' + k: v for k, v in sd.items()}
        elif world_size == 1 and list(sd.keys())[0].startswith('module.'):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history       = ckpt.get('history', empty_history())
        for k in empty_history():
            history.setdefault(k, [])
        if is_main:
            print(f"✅ Resumed from epoch {ckpt['epoch']} | best_val_loss={best_val_loss:.4f}")
    elif args.resume:
        if is_main: print(f"⚠️  Checkpoint not found: {args.resume} — starting from scratch")

    if is_main:
        print(f"\nStarting training for {args.epochs} epochs...")
        print(f"Visualizations every {args.viz_freq} epochs → {viz_dir}")
        print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        if is_main:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*80}")

        tr_loss, tr_rgb, tr_depth, tr_pc, tr_txt = train_one_epoch(
            model, train_loader, optimizer, device, epoch)
        for k, v in zip(['train_loss','train_rgb','train_depth','train_pc','train_text'],
                        [tr_loss, tr_rgb, tr_depth, tr_pc, tr_txt]):
            history[k].append(v)

        if is_main:
            print(f"\nTrain — Loss: {tr_loss:.4f}  RGB: {tr_rgb:.4f}  "
                  f"Depth: {tr_depth:.4f}  PC: {tr_pc:.4f}  Text: {tr_txt:.4f}")

        do_val = (epoch % args.val_freq == 0 or epoch == args.epochs or epoch == 1)
        if do_val:
            if is_main: print("\n🔍 Running validation…")
            compute_emd = (epoch == args.epochs)
            (vl, vr, vd, vp, vt), vm = evaluate(
                model, val_loader, device, compute_emd=compute_emd)
            for k, v in zip(['val_loss','val_rgb','val_depth','val_pc','val_text',
                              'val_rgb_mse','val_depth_mse','val_pc_chamfer','val_pc_emd',
                              'val_param_mse','val_param_mae','val_param_mae_masked',
                              'val_param_acc05'],
                            [vl, vr, vd, vp, vt,
                             vm['rgb_mse'], vm['depth_mse'], vm['pc_chamfer'], vm['pc_emd'],
                             vm['param_mse'], vm['param_mae'], vm['param_mae_masked'],
                             vm['param_acc@0.05']]):
                history[k].append(v)
            if is_main:
                print(f"Val   — Loss: {vl:.4f}  RGB: {vr:.4f}  Depth: {vd:.4f}  "
                      f"PC: {vp:.4f}  Text: {vt:.4f}")
                print(f"Metrics — RGB MSE: {vm['rgb_mse']:.6f}  Depth MSE: {vm['depth_mse']:.6f}  "
                      f"PC Chamfer: {vm['pc_chamfer']:.6f}  PC EMD: {vm['pc_emd']:.6f}")
                print(f"Param   — MSE: {vm['param_mse']:.6f}  MAE: {vm['param_mae']:.6f}  "
                      f"MAE(masked): {vm['param_mae_masked']:.6f}  "
                      f"acc@0.05: {vm['param_acc@0.05']:.4f}")
        else:
            vl = history['val_loss'][-1] if history['val_loss'] else float('inf')
            vm = {k: history[v][-1] if history[v] else 0.0 for k, v in {
                'rgb_mse':'val_rgb_mse','depth_mse':'val_depth_mse',
                'pc_chamfer':'val_pc_chamfer','pc_emd':'val_pc_emd',
                'param_mse':'val_param_mse','param_mae':'val_param_mae',
                'param_mae_masked':'val_param_mae_masked',
                'param_acc@0.05':'val_param_acc05'}.items()}

        if is_main and args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': tr_loss, 'train/rgb': tr_rgb, 'train/depth': tr_depth,
                'train/pc': tr_pc, 'train/text': tr_txt,
                'val/loss': vl,
                'metrics/rgb_mse': vm['rgb_mse'], 'metrics/depth_mse': vm['depth_mse'],
                'metrics/pc_chamfer': vm['pc_chamfer'], 'metrics/pc_emd': vm['pc_emd'],
                'metrics/param_mse': vm['param_mse'],
                'metrics/param_mae': vm['param_mae'],
                'metrics/param_mae_masked': vm['param_mae_masked'],
                'metrics/param_acc@0.05': vm['param_acc@0.05'],
                'learning_rate': scheduler.get_last_lr()[0],
            })

        if is_main and (epoch % args.viz_freq == 0 or epoch == 1):
            print(f"\n📊 Generating visualizations for epoch {epoch}…")
            mv = model.module if world_size > 1 else model
            paths = visualize_reconstruction_4m(
                mv, val_loader, device, epoch, viz_dir, args.num_viz_samples)
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({'visualizations': [wandb.Image(p, caption=Path(p).name)
                                               for p in paths], 'epoch': epoch})

        scheduler.step()

        if is_main and epoch % args.save_freq == 0:
            ms = (model.module if world_size > 1 else model).state_dict()
            torch.save({
                'epoch': epoch, 'model_state_dict': ms,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss, 'history': history,
                'wandb_run_id': (wandb.run.id
                                 if args.use_wandb and WANDB_AVAILABLE
                                 and wandb.run else None),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved: checkpoint_epoch_{epoch}.pth")

        if is_main and do_val and vl < best_val_loss:
            best_val_loss = vl
            ms = (model.module if world_size > 1 else model).state_dict()
            torch.save({
                'epoch': epoch, 'model_state_dict': ms,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': vl, 'best_val_loss': best_val_loss, 'history': history,
                'wandb_run_id': (wandb.run.id
                                 if args.use_wandb and WANDB_AVAILABLE
                                 and wandb.run else None),
            }, output_dir / 'best_model.pth')
            print(f"⭐ New best model! Val Loss: {vl:.4f}")

        if is_main:
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=4)

    if world_size > 1:
        cleanup_distributed()
    if is_main and args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    if is_main:
        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Outputs: {output_dir}")
        print(f"{'='*80}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train EmbodiedMAE-4M')

    parser.add_argument('--config',             type=str,   default='config_4m.yaml')
    parser.add_argument('--data_root',          type=str,   default=None)
    parser.add_argument('--img_size',           type=int,   default=None)
    parser.add_argument('--num_points',         type=int,   default=None)
    parser.add_argument('--model_size',         type=str,   default=None, choices=['small','base'])
    parser.add_argument('--mask_ratio',         type=float, default=None)
    parser.add_argument('--pc_loss_weight',     type=float, default=None)
    parser.add_argument('--depth_norm_type',    type=str,   default=None, choices=['minmax','standard'])
    parser.add_argument('--spline_loss_weight', type=float, default=None)
    parser.add_argument('--max_leaves',         type=int,   default=None)
    parser.add_argument('--batch_size',         type=int,   default=None)
    parser.add_argument('--epochs',             type=int,   default=None)
    parser.add_argument('--lr',                 type=float, default=None)
    parser.add_argument('--weight_decay',       type=float, default=None)
    parser.add_argument('--warmup_epochs',      type=int,   default=None)
    parser.add_argument('--val_freq',           type=int,   default=None)
    parser.add_argument('--viz_freq',           type=int,   default=None)
    parser.add_argument('--num_viz_samples',    type=int,   default=None)
    parser.add_argument('--output_dir',         type=str,   default=None)
    parser.add_argument('--save_freq',          type=int,   default=None)
    parser.add_argument('--resume',             type=str,   default=None)
    parser.add_argument('--world_size',         type=int,   default=None)
    parser.add_argument('--dist_backend',       type=str,   default=None)
    parser.add_argument('--dist_url',           type=str,   default=None)
    parser.add_argument('--num_workers',        type=int,   default=None)
    parser.add_argument('--device',             type=str,   default=None)
    parser.add_argument('--use_wandb',          action='store_true', default=None)
    parser.add_argument('--no_wandb',           action='store_false', dest='use_wandb')
    parser.add_argument('--wandb_project',      type=str,   default=None)
    parser.add_argument('--wandb_entity',       type=str,   default=None)
    parser.add_argument('--wandb_name',         type=str,   default=None)

    args = parser.parse_args()

    if os.path.exists(args.config):
        print(f"📋 Loading config: {args.config}")
        cfg = config_to_namespace(merge_config_with_args(load_config(args.config), args))
    else:
        print(f"⚠️  Config not found: {args.config} — using defaults + CLI args")
        default_cfg = {
            'data':          {'data_root': './Dataset/new_data', 'img_size': 224, 'num_points': 8196},
            'model':         {'model_size': 'base', 'mask_ratio': 0.15, 'pc_loss_weight': 10.0,
                              'depth_norm_type': 'minmax', 'spline_loss_weight': 1.0, 'max_leaves': 24},
            'training':      {'batch_size': 16, 'epochs': 2400, 'lr': 1.5e-4,
                              'weight_decay': 0.05, 'warmup_epochs': 10, 'val_freq': 20},
            'checkpointing': {'output_dir': './outputs/4m_run', 'save_freq': 100, 'resume': None},
            'visualization': {'viz_freq': 50, 'num_viz_samples': 6},
            'distributed':   {'world_size': 4, 'dist_backend': 'nccl', 'dist_url': 'env://'},
            'system':        {'num_workers': 8, 'device': 'cuda'},
            'wandb':         {'use_wandb': True, 'wandb_project': 'embodied-mae-4m',
                              'wandb_entity': None, 'wandb_name': None},
        }
        cfg = config_to_namespace(merge_config_with_args(default_cfg, args))

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        cfg.world_size = world_size
        if rank == 0:
            print(f"\n🚀 Multi-GPU (torchrun)  GPUs={world_size}  "
                  f"batch/GPU={cfg.batch_size}  total={cfg.batch_size*world_size}")
        train_worker(rank, world_size, cfg)
    elif cfg.world_size > 1:
        print(f"\n🚀 Multi-GPU (mp.spawn)  GPUs={cfg.world_size}  "
              f"batch/GPU={cfg.batch_size}  total={cfg.batch_size*cfg.world_size}")
        mp.spawn(train_worker, args=(cfg.world_size, cfg),
                 nprocs=cfg.world_size, join=True)
    else:
        train_worker(0, 1, cfg)


if __name__ == '__main__':
    main()
