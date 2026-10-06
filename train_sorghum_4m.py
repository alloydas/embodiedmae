"""
Training script for EmbodiedMAE-4M (RGB + Depth + PointCloud + Text parameters).
"""

import os
import json
import random
import argparse
from dataclasses import replace
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
    embodied_mae_4m_large,
    _params_to_plant_text,
    _params_to_leaf_text,
    N_PARAMS,
)
from embodied_mae import earth_movers_distance
from sorghum_dataset_4m import SorghumDataset4M
from structured_mask import NeighbourMaskConfig, foreground
from occlusion_scene import SceneConfig, compose as compose_scene
from leaf_mask import LeafMaskConfig, mask_weights as leaf_mask_weights


# ── Config helpers ────────────────────────────────────────────────────────────

MODALITY_ORDER = ('rgb', 'depth', 'pc', 'text')

# Metrics evaluate() may return, as (metric key, val-history key, test-history key,
# format). A key is ABSENT -- not zero -- when its modality is inactive, so that a
# reduced E2 arm never records 0.0 for a modality it does not model (a 0.0 there
# reads as a perfect reconstruction). Every consumer must therefore be driven by
# key PRESENCE, never by indexing: `vm['rgb_mse']` KeyErrors on the PC-only arm.
METRIC_KEYS = [
    ('rgb_mse',          'val_rgb_mse',          'test_rgb_mse',          '.6f'),
    ('depth_mse',        'val_depth_mse',        'test_depth_mse',        '.6f'),
    ('pc_chamfer',       'val_pc_chamfer',       'test_pc_chamfer',       '.6f'),
    ('pc_emd',           'val_pc_emd',           None,                    '.6f'),
    ('param_mse',        'val_param_mse',        'test_param_mse',        '.6f'),
    ('param_mae',        'val_param_mae',        'test_param_mae',        '.6f'),
    ('param_mae_masked', 'val_param_mae_masked', 'test_param_mae_masked', '.6f'),
    ('param_acc@0.05',   'val_param_acc05',      'test_param_acc05',      '.4f'),
]

# Point-cloud F-score thresholds, Euclidean on the unit-sphere-normalised cloud
# (the same three eval/eval_occlusion.py reports). PC is active in every arm, so
# these keys are always present.
PC_THRESHOLDS = (0.01, 0.02, 0.03)
PC_FSCORE_KEYS = [f'{m}@{t}' for t in PC_THRESHOLDS for m in ('f1', 'precision', 'recall')]
METRIC_KEYS += [(k, f"val_pc_{k.replace('@0.', '_0')}", f"test_pc_{k.replace('@0.', '_0')}", '.4f')
                for k in PC_FSCORE_KEYS]


def _fmt_metrics(md):
    """Render only the metrics actually present, in METRIC_KEYS order."""
    return '  '.join(f"{mk}: {md[mk]:{f}}" for mk, _, _, f in METRIC_KEYS if mk in md)


def _append_aligned(history, key, value, ref_key):
    """Append `value` to history[key], padding with None first so the series
    stays index-aligned with history[ref_key] (already appended this epoch).
    A metric added mid-run -- the F-scores, on a run resumed from a checkpoint
    that predates them -- would otherwise line its first value up with epoch 1."""
    s = history.setdefault(key, [])
    s.extend([None] * (len(history.get(ref_key, [])) - 1 - len(s)))
    s.append(value)


@torch.no_grad()
def pc_scores(pred, target, chunk=512):
    """Chamfer plus precision / recall / F1 at PC_THRESHOLDS, batch means.

    pred (B, N, 3), target (B, M, 3). Chunked over pred so it never builds the
    dense (B, N, M, 3) tensor embodied_mae.chamfer_distance does: 12 GiB at
    B=16, N=M=8196, which ran sm_scene_off_s1 out of memory mid-validation on
    an A100-40GB. `pc_chamfer` is chamfer_distance's formula exactly (squared
    NN distances, mean both ways). Precision is the share of predicted points
    within t of the target, recall the share of target points within t of the
    prediction; F1 is their harmonic mean, per sample, then averaged.
    """
    d_pp = []                                                     # pred -> target
    d_pt = torch.full(target.shape[:2], float('inf'), device=pred.device, dtype=pred.dtype)
    for s in range(0, pred.shape[1], chunk):
        d = ((pred[:, s:s + chunk, None] - target[:, None]) ** 2).sum(-1)   # (B, c, M)
        d_pp.append(d.min(dim=2).values)
        d_pt = torch.minimum(d_pt, d.min(dim=1).values)
        del d
    d_pp = torch.cat(d_pp, dim=1)
    out = {'pc_chamfer': d_pp.mean() + d_pt.mean()}
    for t in PC_THRESHOLDS:
        p = (d_pp < t * t).float().mean(1)
        r = (d_pt < t * t).float().mean(1)
        out[f'precision@{t}'] = p.mean()
        out[f'recall@{t}'] = r.mean()
        out[f'f1@{t}'] = (2 * p * r / (p + r).clamp(min=1e-9)).mean()
    return {k: v.item() for k, v in out.items()}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_modalities(v):
    """None | 'rgb,pc' | ['rgb','pc'] -> tuple | None. None means all four."""
    if v is None:
        return None
    if isinstance(v, str):
        v = [t for t in (x.strip() for x in v.split(',')) if t]
    mods = tuple(str(m).strip().lower() for m in v)
    if not mods or set(mods) == set(MODALITY_ORDER):
        return None
    return mods


def merge_config_with_args(config, args):
    mapping = {
        'data': ['data_root', 'img_size', 'num_points',
                 'view_sampling', 'view_seed', 'max_plants',
                 'plant_subset_seed'],
        'model': ['model_size', 'mask_ratio', 'pc_loss_weight',
                  'depth_norm_type', 'spline_loss_weight', 'max_leaves',
                  'loss_name', 'qal_threshold', 'qal_alpha', 'qal_use_squared',
                  'active_modalities', 'text_mask_ratio'],
        'training': ['batch_size', 'epochs', 'lr', 'weight_decay',
                     'warmup_epochs', 'val_freq', 'test_freq', 'seed'],
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
    ns.view_sampling      = bool(config['data'].get('view_sampling', False))
    ns.view_seed          = int(config['data'].get('view_seed', 0))
    # E3 data scaling: cap the TRAIN split at this many plants (nested subsets,
    # see SorghumDataset4M). None -> every plant. val/test are never capped.
    _mp                   = config['data'].get('max_plants', None)
    ns.max_plants         = None if _mp in (None, 0, 'null') else int(_mp)
    ns.plant_subset_seed  = int(config['data'].get('plant_subset_seed', 42))
    # Where the *_spline.yml params are read from. None lets SorghumDataset4M
    # decide: the view folders, unless they are the 2026-09-30 rewrite (no
    # waviness keys), in which case the SorghumData originals.
    _sr                   = config['data'].get('spline_root', None)
    ns.spline_root        = None if _sr in (None, 'null') else str(_sr)
    ns.model_size         = config['model'].get('model_size', 'base')
    ns.mask_ratio         = config['model'].get('mask_ratio', 0.15)
    ns.pc_loss_weight     = config['model'].get('pc_loss_weight', 10.0)
    ns.depth_norm_type    = config['model'].get('depth_norm_type', 'minmax')
    ns.spline_loss_weight = config['model'].get('spline_loss_weight', 1.0)
    ns.max_leaves         = config['model'].get('max_leaves', 24)
    # Point-cloud objective: 'chamfer' (previous behaviour) or 'qal_loss', the
    # sigmoid-weighted two-sided Chamfer ported from the yongyun branch.
    ns.pc_loss_name       = config['model'].get('loss_name', 'chamfer')
    ns.qal_threshold      = config['model'].get('qal_threshold', 0.01)
    ns.qal_alpha          = config['model'].get('qal_alpha', 100.0)
    ns.qal_use_squared    = config['model'].get('qal_use_squared', False)
    ns.pc_sinkhorn_weight = config['model'].get('pc_sinkhorn_weight', 0.0)
    ns.pc_sinkhorn_points = config['model'].get('pc_sinkhorn_points', 2048)
    ns.pc_sinkhorn_blur   = config['model'].get('pc_sinkhorn_blur', 0.01)
    # E2 modality value-add. None -> all four streams (the headline model).
    # A subset names which token streams exist AT ALL: an inactive modality has
    # no embedder, no decoder head and no loss term, so it is absent from the
    # model rather than merely masked. 'pc' is mandatory (it anchors the target).
    # Accepts a YAML list or a comma-separated string.
    ns.active_modalities  = _parse_modalities(config['model'].get('active_modalities'))
    # Mask text at this rate OUTSIDE the Dirichlet budget (the e2_pcrgbdt_tg
    # control). None -> text shares one length-agnostic budget with pc/rgb/depth,
    # which at mask_ratio 0.80 leaves text ~59% visible and cuts vision from 20%
    # to ~18% -- so e2_pcrgbdt pretrained on fewer vision tokens than e2_pcrgbd,
    # a handicap rather than a param-stream effect. None is every run before the
    # control, bit-identical. Stored as the EFFECTIVE value because config.json
    # dumps this namespace: an arm without text records null, not a ratio its
    # model never applied.
    _tmr                  = config['model'].get('text_mask_ratio', None)
    ns.text_mask_ratio    = None if _tmr in (None, 'null') else float(_tmr)
    if ns.text_mask_ratio is not None:
        # `not 0 <= x <= 1` also catches NaN/inf. The model floors to >= 1 visible
        # token, so a percent typo (80) would train silently at 1 of 25 -- refuse it.
        if not 0.0 <= ns.text_mask_ratio <= 1.0:
            raise ValueError(f"text_mask_ratio must be in [0, 1], got {ns.text_mask_ratio}")
        if ns.active_modalities is not None and 'text' not in ns.active_modalities:
            print(f"⚠️  text_mask_ratio={ns.text_mask_ratio} ignored: text is not an "
                  f"active modality ({','.join(ns.active_modalities)})")
            ns.text_mask_ratio = None
    # Neighbour-occlusion masking (structured_mask.py), YAML only. Stored as
    # the validated, fully-defaulted dict -- or None when off -- so config.json
    # records every knob the run used, not just the ones the YAML spelled out.
    _sm                   = NeighbourMaskConfig.from_dict(
        config['model'].get('structured_mask'))
    ns.structured_mask    = None if _sm is None else _sm.to_dict()
    # Occluded-scene training (occlusion_scene.py), YAML only, stored the same
    # way. prob 0 trains clean but still adds the occluded validation pass.
    _sc                   = SceneConfig.from_dict(config['model'].get('occlusion_scene'))
    ns.occlusion_scene    = None if _sc is None else _sc.to_dict()
    if _sc is not None and _sc.prob > 0:
        if _sm is not None:
            raise ValueError("structured_mask and occlusion_scene (prob > 0) both change "
                             "the training inputs' masking; use one")
    # Leaf-weighted masking (leaf_mask.py), YAML only: the target's leaves are
    # masked more often than its stem. Training only; validation stays uniform.
    _lm                   = LeafMaskConfig.from_dict(config['model'].get('leaf_mask'))
    ns.leaf_mask          = None if _lm is None else _lm.to_dict()
    if _lm is not None:
        if _sm is not None:
            raise ValueError("leaf_mask and structured_mask both set the masking; use one")
        if _sc is not None and _sc.prob > 0 and _sc.mask_policy != 'uniform':
            raise ValueError("leaf_mask needs occlusion_scene.mask_policy: uniform -- "
                             "both would set the masking")
        if 'text' in (ns.active_modalities or MODALITY_ORDER) and ns.text_mask_ratio != 0.0:
            print(f"⚠️  occlusion_scene trains with text_mask_ratio={ns.text_mask_ratio}: the "
                  f"target's params are not full conditioning (the meeting's setup is 0.0)")
    ns.batch_size         = config['training'].get('batch_size', 16)
    ns.epochs             = config['training'].get('epochs', 2400)
    ns.lr                 = config['training'].get('lr', 1.5e-4)
    ns.weight_decay       = config['training'].get('weight_decay', 0.05)
    ns.warmup_epochs      = config['training'].get('warmup_epochs', 10)
    ns.val_freq           = config['training'].get('val_freq', 20)
    ns.test_freq          = config['training'].get('test_freq', 50)
    # Seeds model init, masks and loader draws. None = unseeded, as every run
    # before 2026-10-01 was -- two such runs differ by more than their config.
    _seed                 = config['training'].get('seed', None)
    ns.seed               = None if _seed in (None, 'null') else int(_seed)
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

def _unnorm_pix(model, pred_patches, gt_img):
    """Invert norm_pix_loss so a prediction can be rendered.

    With norm_pix_loss=True (the model default) the RGB head predicts PER-PATCH
    standardised values, so the raw prediction is not in image space at all.
    De-normalising it straight with ImageNet statistics — which this code used to
    do — mismatches the spaces and is what turned reconstructed backgrounds blue.
    The per-patch mean/var cannot be recovered from the prediction, so use the
    ground-truth patch statistics, the same convention the original MAE
    visualisations use. Affects the figure only; the loss was always correct.
    """
    if not getattr(model, 'norm_pix_loss', False):
        return pred_patches
    tgt = model.patchify(gt_img, model.patch_size, gt_img.shape[1])
    mean = tgt.mean(dim=-1, keepdim=True)
    var = tgt.var(dim=-1, keepdim=True)
    return pred_patches * (var + 1.e-6) ** .5 + mean



def visualize_reconstruction_4m(model, dataloader, device, epoch, save_dir,
                                  num_samples=4, mask_ratio=0.75):
    """5-row × 3-col grid per sample.
    Row 5 shows target vs predicted text for masked tokens.

    The grid is hard-wired to all four streams. An E2 ablation arm has no RGB,
    depth or param head at all (their predictions come back None), so rather
    than render a grid of blanks the figure is skipped for reduced arms. The
    numbers -- which are what E2 actually compares -- are unaffected.
    """
    model.eval()
    saved_paths = []

    m0 = model.module if hasattr(model, 'module') else model
    if len(getattr(m0, 'active_modalities', MODALITY_ORDER)) < 4:
        print(f"  [viz] skipped: arm is {'+'.join(m0.active_modalities)}, "
              f"the 4-row grid needs all four streams")
        return saved_paths

    batch = next(iter(dataloader))
    # [:6]: a loader built for occluded scenes also yields pc_norm and cam2world.
    rgb_b, depth_b, pc_b, param_floats_b, text_valid_b, names = batch[:6]
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
                rgb_b, depth_b, pc_b, param_floats_b, text_valid_b, mask_ratio=mask_ratio
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

        pred_rgb_vis   = _unnorm_pix(model, pred_rgb, rgb_b)
        pred_rgb_img   = unpatchify(pred_rgb_vis, model.patch_size, 3, model.img_size)
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
            fontsize=11, fontweight='bold', y=0.997
        )
        # reserve headroom for the suptitle - plain tight_layout() ignores it and
        # the first row's axis titles collide with the header
        plt.tight_layout(rect=[0, 0, 1, 0.978])

        path = save_dir / f'epoch_{epoch:03d}_sample_{idx+1}_{names[idx]}.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        saved_paths.append(str(path))
        print(f"  Saved: {path.name}")

    model.train()
    return saved_paths


def visualize_scene_4m(model, dataloader, device, epoch, save_dir, num_samples=4,
                       mask_ratio=0.75, scene=None, seed=0, tag='clean'):
    """Target | input | visible input | reconstruction per image modality, and
    input cloud | predicted cloud | target cloud | scores, for any arm.

    The 5-row grid above needs all four streams and only ever draws a clean
    plant. This one draws what an occluded-scene arm is scored on: with `scene`
    (a SceneConfig) the first val batch becomes occluded scenes exactly as in
    evaluate() (for_validation(): every sample occluded, uniform masking), the
    encoder reads the scene, and every reconstruction is set against the CLEAN
    target. Without `scene` the input is the clean plant, so a 3-modality arm
    (no text head, hence no 5-row grid) still gets a figure.

    The scene draw (neighbours, yaw, spacing), FPS start and token masks are
    seeded by `seed`, so every epoch shows the same plants in the same scenes
    under the same masks. The dataset's own cloud subsample is not seeded
    (load_pointcloud's np.random.choice), so the points drawn vary slightly
    from one epoch to the next. The reconstruction panel shows the prediction on every
    SCORED patch (masked, or showing a neighbour) and the input elsewhere, the
    MAE convention: unscored patches are never trained and are noise. Depth is
    drawn in the loss's own normalised space, the input with the target's
    statistics, and no ground-truth silhouette is applied to the prediction.
    """
    model.eval()
    m = model.module if hasattr(model, 'module') else model
    act = m.active_modalities
    device = torch.device(device)
    batch = next(iter(dataloader))
    rgb, depth, pc, params, valid = (t.to(device) for t in batch[:5])
    names = list(batch[5])
    B = rgb.shape[0]

    x_rgb, x_depth, x_pc, kw, sc = rgb, depth, pc, {}, None
    with torch.no_grad(), torch.random.fork_rng(
            devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        if scene is not None:
            # A 9th item is the per-sample depth near/far (maize, whose renderer
            # sets them per plant); sorghum's are fixed and it has none.
            sc = compose_scene(rgb, depth, pc, batch[6].to(device), batch[7].to(device),
                               scene.for_validation(), patch_size=m.patch_size,
                               generator=torch.Generator().manual_seed(seed),
                               near_far=batch[8].to(device) if len(batch) > 8 else None)
            if sc is None:
                print(f"  [viz] no scene for a batch of {B}; skipped")
                return []
            x_rgb, x_depth, x_pc = sc['rgb'], sc['depth'], sc['pc']
            kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': pc},
                  'loss_tokens': sc['loss_tokens']}
        _, _, (p_rgb, p_depth, p_pc, _), (m_rgb, m_depth, _, _) = model(
            x_rgb, x_depth, x_pc, params, valid, mask_ratio=mask_ratio, **kw)

        n = min(num_samples, B)
        scores = [pc_scores(p_pc[i:i + 1], pc[i:i + 1]) for i in range(n)]
        g = m.img_size // m.patch_size
        shown_tok = (sc['loss_tokens']['rgb'].float() if sc is not None
                     else torch.zeros(B, g * g, device=device))
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img = lambda t: (t * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        up = lambda tok: tok.view(B, 1, g, g).repeat_interleave(
            m.patch_size, 2).repeat_interleave(m.patch_size, 3)

        rows = {}
        if 'rgb' in act:
            scored = up(torch.maximum(m_rgb.float(), shown_tok)) > 0
            rec = unpatchify(_unnorm_pix(m, p_rgb, rgb), m.patch_size, 3, m.img_size)
            rows['RGB'] = dict(
                target=img(rgb), input=img(x_rgb),
                visible=img(x_rgb * (1 - up(m_rgb.float())) - 2 * up(m_rgb.float())),
                recon=img(torch.where(scored, rec, x_rgb)),
                masked=m_rgb.float().mean(1).cpu().numpy(), cmap=None)
        if 'depth' in act:
            # The loss's normalisation, statistics from the clean TARGET image.
            tf = depth.flatten(1)
            if m.depth_norm_type == 'minmax':
                lo = tf.min(1).values.view(B, 1, 1, 1)
                sc_ = (tf.max(1).values.view(B, 1, 1, 1) - lo).clamp(min=1e-6)
            else:
                lo = tf.mean(1).view(B, 1, 1, 1)
                sc_ = tf.std(1).view(B, 1, 1, 1).clamp(min=1e-6)
            nd = lambda d: ((d - lo) / sc_)
            scored = up(torch.maximum(m_depth.float(), shown_tok)) > 0
            rec = unpatchify(p_depth, m.patch_size, 1, m.img_size)
            vis = nd(x_depth).masked_fill(up(m_depth.float()) > 0, float('nan'))
            rows['Depth'] = dict(
                target=nd(depth)[:, 0].cpu().numpy(), input=nd(x_depth)[:, 0].cpu().numpy(),
                visible=vis[:, 0].cpu().numpy(),
                recon=torch.where(scored, rec, nd(x_depth))[:, 0].cpu().numpy(),
                masked=m_depth.float().mean(1).cpu().numpy(), cmap='viridis')
        in_pc, pred_pc, tgt_pc = (t[:n].cpu().numpy() for t in (x_pc, p_pc, pc))
        nb_pt = (sc['nb_point'][:n].cpu().numpy() if sc is not None
                 else np.zeros(in_pc.shape[:2], bool))
        if sc is not None:
            fg = foreground(depth)
            hid = ((fg & sc['shown']).flatten(1).sum(1)
                   / fg.flatten(1).sum(1).clamp(min=1)).cpu().numpy()

    rng = np.random.default_rng(seed)
    sub = lambda a, k=3000: a[rng.choice(len(a), min(k, len(a)), replace=False)]
    what = 'occluded scene' if sc is not None else 'clean input'
    saved = []
    for i in range(n):
        nr = len(rows) + 1
        fig = plt.figure(figsize=(16, 4.2 * nr))
        for r, (lab, d) in enumerate(rows.items()):
            kwi = {'cmap': d['cmap'], 'vmin': 0, 'vmax': 1} if d['cmap'] else {}
            panels = [(d['target'][i], f'{lab} target (clean plant)'),
                      (d['input'][i], f'{lab} input ({what})'),
                      (d['visible'][i], f'{lab} visible tokens ({d["masked"][i]:.0%} masked)'),
                      (d['recon'][i], f'{lab} reconstruction\n(scored patches predicted)')]
            for c, (im, title) in enumerate(panels):
                ax = fig.add_subplot(nr, 4, r * 4 + c + 1)
                ax.imshow(im, **kwi)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
        base = (nr - 1) * 4
        ax = fig.add_subplot(nr, 4, base + 1, projection='3d')
        tgt_i, nb_i = in_pc[i][~nb_pt[i]], in_pc[i][nb_pt[i]]
        _scatter3d(ax, sub(tgt_i), c='#2e8b57', s=1)
        if len(nb_i):
            _scatter3d(ax, sub(nb_i), c='#d62728', s=1)
        ax.set_title(f'PC input: target green, neighbour red\n'
                     f'({nb_pt[i].mean():.0%} neighbour)', fontsize=9)
        for c, (pts, title) in enumerate([(pred_pc[i], 'PC reconstruction'),
                                          (tgt_pc[i], 'PC target (clean plant)')]):
            ax = fig.add_subplot(nr, 4, base + 2 + c, projection='3d')
            p = sub(pts)
            _scatter3d(ax, p, c=p[:, 1], cmap='viridis', s=1)
            ax.set_title(title, fontsize=9)
        for a in fig.axes[base:base + 3]:
            a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_zlim(-1, 1)
        ax = fig.add_subplot(nr, 4, base + 4)
        ax.axis('off')
        s = scores[i]
        lines = [f'{names[i]}   epoch {epoch}', f'input: {what}', '',
                 f'chamfer   {s["pc_chamfer"]:.6f}', '',
                 f'{"t":>6} {"F1":>7} {"prec":>7} {"recall":>7}']
        lines += [f'{t:>6} {s[f"f1@{t}"]:>7.3f} {s[f"precision@{t}"]:>7.3f} '
                  f'{s[f"recall@{t}"]:>7.3f}' for t in PC_THRESHOLDS]
        if sc is not None:
            lines += ['', f'target pixels hidden {hid[i]:.0%}',
                      f'patches scored while visible {int(shown_tok[i].sum())}']
        ax.text(0.02, 0.98, '\n'.join(lines), va='top', family='monospace', fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        fig.suptitle(f'Epoch {epoch} | {names[i]} | {what} -> clean target', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = Path(save_dir) / f'epoch_{epoch:03d}_{tag}_sample_{i + 1}_{names[i]}.png'
        fig.savefig(path, dpi=100)
        plt.close(fig)
        saved.append(str(path))
    print(f"  [viz] {len(saved)} {tag} figures -> {save_dir}")
    model.train()
    return saved


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, epoch, mask_ratio=0.75,
                    scene=None, scene_stats=None, leaf=None):
    """One epoch. `scene` (a SceneConfig with prob > 0) turns a share of each
    batch into occluded scenes -- the encoder reads the scene, the loss scores
    the clean plant -- and `scene_stats`, a dict, collects their mean stats."""
    model.train()
    tot = tot_rgb = tot_depth = tot_pc = tot_txt = 0.0
    m = model.module if hasattr(model, 'module') else model
    sstats, n_scene = {}, 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        rgb, depth, pc, param_floats, text_valid = (t.to(device) for t in batch[:5])
        # A 7th item is the cloud normalisation, present only when the dataset
        # was built for structured masking (return_pc_norm=True); an 8th is the
        # camera pose, for occluded scenes (return_pose=True).
        pc_norm = batch[6].to(device) if len(batch) > 6 else None

        kw = {}
        if scene is not None:
            sc = compose_scene(rgb, depth, pc, pc_norm, batch[7].to(device), scene,
                               patch_size=m.patch_size)
            if sc is not None:
                kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': pc},
                      'loss_tokens': sc['loss_tokens'],
                      'mask_flags': sc.get('mask_flags')}
                rgb, depth, pc = sc['rgb'], sc['depth'], sc['pc']
                for k, v in sc['stats'].items():
                    sstats[k] = sstats.get(k, 0.0) + v
                n_scene += 1
        if leaf is not None:
            # Weights from the TARGET's geometry: its clean cloud for the image
            # patches, the (possibly scene) input cloud for the PC tokens, with
            # neighbour points left at weight 1.
            clean_pc = kw['targets']['pc'] if 'targets' in kw else pc
            kw['mask_flags'] = leaf_mask_weights(
                pc, clean_pc, pc_norm, batch[7].to(device), leaf,
                rgb.shape[-2], rgb.shape[-1], m.patch_size,
                nb_point=sc['nb_point'] if 'targets' in kw else None)

        loss, (lr, ld, lp, lt), _, _ = model(rgb, depth, pc, param_floats, text_valid,
                                              mask_ratio=mask_ratio, pc_norm=pc_norm, **kw)

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
    if scene_stats is not None and n_scene:
        scene_stats.update({k: v / n_scene for k, v in sstats.items()})
    return tot/n, tot_rgb/n, tot_depth/n, tot_pc/n, tot_txt/n


@torch.no_grad()
def evaluate(model, dataloader, device, compute_emd=False, mask_ratio=0.75,
             distributed=False, scene=None, scene_seed=0, oracle=False):
    """Validation / test pass. With `scene` (a SceneConfig) every batch is first
    turned into occluded scenes -- scene.for_validation(): all samples, uniform
    masking -- from a CPU generator seeded by (scene_seed, batch index), so
    every arm is scored on the same scenes. All metrics stay against the CLEAN
    plant, and two more report how occluded the scenes were (occ_*). The loader
    must be built with return_pose=True."""
    model.eval()
    tot = tot_rgb = tot_depth = tot_pc = tot_txt = 0.0
    tot_rgb_mse = tot_depth_mse = tot_chamfer = tot_emd = 0.0
    tot_param_mse = tot_param_mae = tot_param_mae_masked = tot_param_acc05 = 0.0
    tot_hidden = tot_nbpts = n_scene = 0.0
    n_text_masked = 0.0
    tot_fs = dict.fromkeys(PC_FSCORE_KEYS, 0.0)

    m = model.module if hasattr(model, 'module') else model
    if scene is not None:
        scene = scene.for_validation()
        if oracle:
            # ORACLE (eval/eval_scene_oracle.py only): mask the tokens that show a
            # neighbour first, as neighbour_first training does -- i.e. assume a
            # segmentation of the target is available at test time.
            scene = replace(scene, mask_policy='neighbour_first')

    desc = 'Evaluating' + (' (occluded scenes)' if scene is not None else '')
    for bi, batch in enumerate(tqdm(dataloader, desc=desc)):
        rgb, depth, pc, param_floats, text_valid = (t.to(device) for t in batch[:5])

        x_rgb, x_depth, x_pc, kw = rgb, depth, pc, {}
        if scene is not None:
            sc = compose_scene(
                rgb, depth, pc, batch[6].to(device), batch[7].to(device), scene,
                patch_size=m.patch_size,
                generator=torch.Generator().manual_seed(scene_seed * 1_000_003 + bi))
            if sc is not None:
                x_rgb, x_depth, x_pc = sc['rgb'], sc['depth'], sc['pc']
                kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': pc},
                      'loss_tokens': sc['loss_tokens']}
                if oracle:
                    kw['mask_flags'] = sc['mask_flags']
                tot_hidden += sc['stats']['hidden_px']
                tot_nbpts  += sc['stats']['nb_points']
                n_scene    += 1

        loss, (lr, ld, lp, lt), \
            (pred_rgb_p, pred_depth_p, pred_pc, pred_params), \
            (_, _, _, mask_text) = model(
                x_rgb, x_depth, x_pc, param_floats, text_valid, mask_ratio=mask_ratio,
                **kw
        )
        del x_rgb, x_depth, x_pc, kw

        tot       += loss.item()
        tot_rgb   += lr.item()
        tot_depth += ld.item()
        tot_pc    += lp.item()
        tot_txt   += lt.item()

        # Inactive modalities have no decoder head, so their prediction is None.
        # Their pixel metrics are skipped here and dropped from `metrics` below,
        # rather than reported as 0.0 -- a zero MSE would read as a perfect
        # reconstruction of a modality the arm cannot even see.
        act = m.active_modalities
        B, _, H, W = rgb.shape
        p = m.patch_size; h = w = H // p
        if 'rgb' in act:
            pred_rgb_img = pred_rgb_p.reshape(B,h,w,p,p,3)
            pred_rgb_img = torch.einsum('nhwpqc->nchpwq', pred_rgb_img).reshape(B,3,H,W)
            tot_rgb_mse += torch.mean((pred_rgb_img - rgb) ** 2).item()
            del pred_rgb_img
        if 'depth' in act:
            pred_depth_img = pred_depth_p.reshape(B,h,w,p,p,1)
            pred_depth_img = torch.einsum('nhwpqc->nchpwq', pred_depth_img).reshape(B,1,H,W)
            tot_depth_mse += torch.mean((pred_depth_img - depth) ** 2).item()
            del pred_depth_img

        pcs = pc_scores(pred_pc, pc)
        tot_chamfer += pcs['pc_chamfer']
        for k in PC_FSCORE_KEYS:
            tot_fs[k] += pcs[k]
        if compute_emd:
            tot_emd += earth_movers_distance(pred_pc, pc, num_samples=500).item()

        # Param-modality metrics, all on normalised params in [0, 1].
        # Reduce over the 9 slots per token (matches the Smooth-L1 loss reduction),
        # then average over the requested set of tokens.
        if 'text' in act:
            diff_abs = (pred_params - param_floats).abs().mean(-1)   # (B, L)
            diff_sq  = ((pred_params - param_floats) ** 2).mean(-1)  # (B, L)

            real_n   = text_valid.sum().clamp(min=1)
            n_text_masked += (text_valid * mask_text).sum().item()
            masked_n = (text_valid * mask_text).sum().clamp(min=1)

            tot_param_mse        += ((diff_sq  * text_valid).sum() / real_n).item()
            tot_param_mae        += ((diff_abs * text_valid).sum() / real_n).item()
            tot_param_mae_masked += ((diff_abs * text_valid * mask_text).sum() / masked_n).item()
            tot_param_acc05      += (((diff_abs < 0.05).float() * text_valid * mask_text)
                                     .sum() / masked_n).item()

        del pred_pc, pred_params, mask_text
        del rgb, depth, pc, param_floats, text_valid

    n = len(dataloader)

    # Aggregate per-batch sums across all DDP ranks so the reported metric covers
    # the FULL split, not just this rank's DistributedSampler shard. Each rank holds
    # a sum over its own batches; SUM-reduce the sums and the batch counts, then divide.
    if distributed and dist.is_available() and dist.is_initialized():
        packed = torch.tensor(
            [tot, tot_rgb, tot_depth, tot_pc, tot_txt,
             tot_rgb_mse, tot_depth_mse, tot_chamfer, tot_emd,
             tot_param_mse, tot_param_mae, tot_param_mae_masked, tot_param_acc05,
             tot_hidden, tot_nbpts, n_scene, n_text_masked,
             *(tot_fs[k] for k in PC_FSCORE_KEYS),
             float(n)], dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed = packed.tolist()
        fs_vals = packed[17:-1]
        (tot, tot_rgb, tot_depth, tot_pc, tot_txt,
         tot_rgb_mse, tot_depth_mse, tot_chamfer, tot_emd,
         tot_param_mse, tot_param_mae, tot_param_mae_masked, tot_param_acc05,
         tot_hidden, tot_nbpts, n_scene, n_text_masked) = packed[:17]
        n = packed[-1]
        tot_fs = dict(zip(PC_FSCORE_KEYS, fs_vals))

    act = m.active_modalities

    # `loss` sums a different number of terms per E2 arm and is NOT comparable
    # across arms -- use `pc_chamfer`, which is computed identically in every arm
    # because PC is always active. Keys for inactive modalities are omitted
    # entirely so a 0.0 can never be misread as a perfect reconstruction.
    metrics = {
        'loss':              tot / n,
        'pc_loss':           tot_pc / n,
        'pc_chamfer':        tot_chamfer / n,
        **{k: tot_fs[k] / n for k in PC_FSCORE_KEYS},
        # pc_emd only when it was actually computed -- EMD is expensive so it runs
        # at the final epoch only. Reporting 0.0 otherwise would read as a perfect
        # match, the same trap as reporting 0.0 for an inactive modality.
        **({'pc_emd': tot_emd / n} if compute_emd else {}),
    }
    if 'rgb' in act:
        metrics['rgb_loss'] = tot_rgb / n
        metrics['rgb_mse']  = tot_rgb_mse / n
    if 'depth' in act:
        metrics['depth_loss'] = tot_depth / n
        metrics['depth_mse']  = tot_depth_mse / n
    if 'text' in act:
        metrics['text_loss']        = tot_txt / n
        metrics['param_mse']        = tot_param_mse / n
        metrics['param_mae']        = tot_param_mae / n
        # Absent, not 0.0, when no param token was ever masked (text_mask_ratio
        # 0.0: params as conditioning) -- 0.0 MAE would read as perfect.
        if n_text_masked > 0:
            metrics['param_mae_masked'] = tot_param_mae_masked / n
            metrics['param_acc@0.05']   = tot_param_acc05 / n
    if scene is not None and n_scene:
        metrics['occ_hidden_px'] = tot_hidden / n_scene
        metrics['occ_nb_points'] = tot_nbpts / n_scene
    return (tot/n, tot_rgb/n, tot_depth/n, tot_pc/n, tot_txt/n), metrics


# ── Worker ────────────────────────────────────────────────────────────────────

def build_model_from_args(args):
    """The model train_worker trains, built from the parsed namespace (on CPU).

    Factored out so a smoke test builds through these exact kwargs instead of an
    improvised copy -- a guessed kwarg name fails silently here (CLAUDE.md).
    """
    build_fn = {'small': embodied_mae_4m_small,
                'base':  embodied_mae_4m_base,
                'large': embodied_mae_4m_large}[args.model_size]
    return build_fn(
        active_modalities=args.active_modalities,
        img_size=args.img_size,
        num_pc_tokens=196,
        target_points=args.num_points,
        pc_loss_weight=args.pc_loss_weight,
        max_leaves=args.max_leaves,
        spline_loss_weight=args.spline_loss_weight,
        depth_norm_type=args.depth_norm_type,
        pc_loss_name=args.pc_loss_name,
        qal_threshold=args.qal_threshold,
        qal_alpha=args.qal_alpha,
        qal_use_squared=args.qal_use_squared,
        pc_sinkhorn_weight=getattr(args, 'pc_sinkhorn_weight', 0.0),
        pc_sinkhorn_points=getattr(args, 'pc_sinkhorn_points', 2048),
        pc_sinkhorn_blur=getattr(args, 'pc_sinkhorn_blur', 0.01),
        text_mask_ratio=args.text_mask_ratio,
        structured_mask=args.structured_mask,
    )


def check_resume_text_mask_ratio(args):
    """Refuse a resume whose checkpoint was trained under different text masking.

    Called before train_worker because train_worker rewrites config.json first
    thing: a mismatched resume would otherwise switch the masking mid-run and
    leave config.json describing only the last segment. The case that matters is
    e2_pcrgbdt's ungated epoch-600 checkpoint loading strict into the gated
    control -- it resumes cleanly and is no longer a control. Checkpoints
    predating the key were all trained ungated, so a missing key reads as None.
    """
    if not (args.resume and os.path.exists(args.resume)):
        return
    try:            # mmap: reads the pickle, not 1.3-4 GB of tensors
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    trained = ckpt.get('text_mask_ratio', None)
    if trained != args.text_mask_ratio:
        raise ValueError(
            f"--resume {args.resume} was trained with text_mask_ratio={trained}, this run "
            f"asks for {args.text_mask_ratio}. Resuming would change the masking mid-run; "
            f"start from scratch or use a matching config.")
    # Same guard for structured masking; checkpoints predating the key were
    # all trained with uniform masking, so a missing key reads as None.
    trained = ckpt.get('structured_mask', None)
    if trained != args.structured_mask:
        raise ValueError(
            f"--resume {args.resume} was trained with structured_mask={trained}, this "
            f"run asks for {args.structured_mask}. Resuming would change the masking "
            f"mid-run; start from scratch or use a matching config.")
    trained = ckpt.get('occlusion_scene', None)
    if trained != args.occlusion_scene:
        raise ValueError(
            f"--resume {args.resume} was trained with occlusion_scene={trained}, this "
            f"run asks for {args.occlusion_scene}. Resuming would change the inputs "
            f"mid-run; start from scratch or use a matching config.")


def train_worker(rank, world_size, args):
    if world_size > 1:
        setup_distributed(rank, world_size, args.dist_backend, args.dist_url)

    is_main = (rank == 0)
    device = (torch.device(f'cuda:{rank}') if world_size > 1
              else torch.device(args.device if torch.cuda.is_available() else 'cpu'))

    output_dir     = Path(args.output_dir)
    viz_dir        = output_dir / 'visualizations'
    test_viz_dir   = output_dir / 'test_visualizations'
    checkpoint_dir = output_dir / 'checkpoints'
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(exist_ok=True)
        test_viz_dir.mkdir(exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    if args.seed is not None:
        # Same seed on every rank for the init (DDP broadcasts rank 0's weights
        # anyway); reseeded per rank below so ranks do not draw identical masks.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if is_main: print(f"\nLoading data from: {args.data_root}")

    # view_sampling (plan §6.1): an epoch is one drawn view per PLANT, not all
    # ten renders. Train rotates the view each epoch; val/test pin view 0 so the
    # metric moves only when the model does.
    vs = args.view_sampling
    # Occluded scenes: trained on when prob > 0, validated on whenever the
    # block is present (so a clean control arm is scored on the same scenes).
    scene_cfg   = SceneConfig.from_dict(args.occlusion_scene)
    scene_train = scene_cfg if scene_cfg is not None and scene_cfg.prob > 0 else None
    leaf_cfg    = LeafMaskConfig.from_dict(getattr(args, 'leaf_mask', None))
    train_ds = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                 num_points=args.num_points, split='train',
                                 max_leaves=args.max_leaves,
                                 view_sampling=vs, view_seed=args.view_seed,
                                 max_plants=args.max_plants,
                                 plant_subset_seed=args.plant_subset_seed,
                                 return_pc_norm=args.structured_mask is not None,
                                 return_pose=scene_train is not None or leaf_cfg is not None,
                                 spline_root=args.spline_root)
    val_ds   = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                 num_points=args.num_points, split='val',
                                 max_leaves=args.max_leaves,
                                 view_sampling=vs, deterministic_view=True,
                                 return_pose=scene_cfg is not None,
                                 spline_root=args.spline_root)
    test_ds  = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                 num_points=args.num_points, split='test',
                                 max_leaves=args.max_leaves,
                                 view_sampling=vs, deterministic_view=True,
                                 spline_root=args.spline_root)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)
        test_sampler  = DistributedSampler(test_ds,  world_size, rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle_train = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=shuffle_train, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, sampler=test_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    if is_main: print(f"\nInitializing EmbodiedMAE-4M-{args.model_size.capitalize()}...")

    model = build_model_from_args(args).to(device)
    if args.seed is not None:
        torch.manual_seed(args.seed * 1000 + 1 + rank)
    if is_main and args.structured_mask is not None:
        print(f"Structured masking (neighbour occlusion, train only): {args.structured_mask}")
    if is_main and scene_cfg is not None:
        print(f"Occluded scenes ({'train p=' + str(scene_cfg.prob) + ' + ' if scene_train else ''}"
              f"occluded val): {args.occlusion_scene}")
    if is_main and leaf_cfg is not None:
        print(f"Leaf-weighted masking (train only): {args.leaf_mask}")
    if is_main and 'text' in (args.active_modalities or MODALITY_ORDER):
        print("Text masking: " + (
            "shared Dirichlet budget" if args.text_mask_ratio is None else
            f"independent at {args.text_mask_ratio}, outside the Dirichlet budget"))

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
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            wandb_run_id = ckpt.get('wandb_run_id')
        if wandb_run_id:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       id=wandb_run_id, resume='must')
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_name, config={
                           'model_size': args.model_size, 'num_points': args.num_points,
                           'mask_ratio': args.mask_ratio, 'pc_loss_weight': args.pc_loss_weight,
                           'text_mask_ratio': args.text_mask_ratio,
                           'structured_mask': args.structured_mask,
                           'occlusion_scene': args.occlusion_scene,
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
            'test_epoch':[],
            'test_loss':[],  'test_rgb':[],  'test_depth':[],  'test_pc':[],  'test_text':[],
            'test_rgb_mse':[], 'test_depth_mse':[], 'test_pc_chamfer':[],
            'test_param_mse':[], 'test_param_mae':[], 'test_param_mae_masked':[],
            'test_param_acc05':[],
        }
    history = empty_history()

    if args.resume and os.path.exists(args.resume):
        if is_main: print(f"\n📂 Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
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
        # Advance the per-plant view draw. Workers are respawned each epoch
        # (persistent_workers is off), so they pick this up.
        train_ds.set_epoch(epoch)

        if is_main:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*80}")

        tr_scene = {}
        tr_loss, tr_rgb, tr_depth, tr_pc, tr_txt = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mask_ratio=args.mask_ratio,
            scene=scene_train, scene_stats=tr_scene, leaf=leaf_cfg)
        for k, v in zip(['train_loss','train_rgb','train_depth','train_pc','train_text'],
                        [tr_loss, tr_rgb, tr_depth, tr_pc, tr_txt]):
            history[k].append(v)
        for k, v in tr_scene.items():
            history.setdefault(f'train_occ_{k}', []).append(v)

        if is_main:
            print(f"\nTrain — Loss: {tr_loss:.4f}  RGB: {tr_rgb:.4f}  "
                  f"Depth: {tr_depth:.4f}  PC: {tr_pc:.4f}  Text: {tr_txt:.4f}")
            if tr_scene:
                print("Train scenes — " + "  ".join(f"{k}: {v:.3f}" for k, v in tr_scene.items()))

        do_val = (epoch % args.val_freq == 0 or epoch == args.epochs or epoch == 1)
        occ_log = {}
        if do_val:
            if is_main: print("\n🔍 Running validation…")
            compute_emd = (epoch == args.epochs)
            (vl, vr, vd, vp, vt), vm = evaluate(
                model, val_loader, device, compute_emd=compute_emd,
                mask_ratio=args.mask_ratio, distributed=(world_size > 1))
            for k, v in zip(['val_loss','val_rgb','val_depth','val_pc','val_text'],
                            [vl, vr, vd, vp, vt]):
                history[k].append(v)
            for mk, hk, _, _ in METRIC_KEYS:
                if mk in vm:
                    _append_aligned(history, hk, vm[mk], 'val_loss')
            if is_main:
                print(f"Val   — Loss: {vl:.4f}  RGB: {vr:.4f}  Depth: {vd:.4f}  "
                      f"PC: {vp:.4f}  Text: {vt:.4f}")
                print(f"Metrics — {_fmt_metrics(vm)}")
            # The same val plants as occluded scenes, scored against the clean
            # plant -- the number the occlusion arms are compared on. Seeds
            # depend on (val_seed, rank, batch), so arms run at the same GPU
            # count see identical scenes. best_model.pth stays on clean val.
            if scene_cfg is not None:
                _, om = evaluate(
                    model, val_loader, device, mask_ratio=args.mask_ratio,
                    distributed=(world_size > 1), scene=scene_cfg,
                    scene_seed=scene_cfg.val_seed * 1000 + rank)
                history.setdefault('val_occ_epoch', []).append(epoch)
                for k, v in om.items():
                    _append_aligned(history, f'val_occ_{k}', v, 'val_occ_epoch')
                occ_log = {f'val_occ/{k}': v for k, v in om.items()}
                if is_main:
                    print("Val (occluded) — " + "  ".join(
                        f"{k}: {v:.6f}" for k, v in om.items()))
        else:
            vl = history['val_loss'][-1] if history['val_loss'] else float('inf')
            # Carry forward only metrics that have actually been recorded; an
            # inactive modality stays absent rather than defaulting to 0.0.
            vm = {mk: history[hk][-1] for mk, hk, _, _ in METRIC_KEYS if history.get(hk)}

        if is_main and args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': tr_loss, 'train/rgb': tr_rgb, 'train/depth': tr_depth,
                'train/pc': tr_pc, 'train/text': tr_txt,
                **{f'train_occ/{k}': v for k, v in tr_scene.items()},
                'val/loss': vl,
                'learning_rate': scheduler.get_last_lr()[0],
                **{f'metrics/{mk}': vm[mk] for mk, _, _, _ in METRIC_KEYS if mk in vm},
                **occ_log,
            })

        # viz_freq <= 0 disables visualisation entirely (E2 reduced arms cannot
        # fill the 4-row grid anyway). Guard the modulo: `epoch % 0` raises.
        if is_main and args.viz_freq > 0 and (epoch % args.viz_freq == 0 or epoch == 1):
            print(f"\n📊 Generating visualizations for epoch {epoch}…")
            mv = model.module if world_size > 1 else model
            # The 5-row grid needs all four streams; a reduced arm gets the
            # per-modality figure on clean input instead of nothing.
            if len(mv.active_modalities) == len(MODALITY_ORDER):
                figs = {'visualizations': visualize_reconstruction_4m(
                    mv, val_loader, device, epoch, viz_dir, args.num_viz_samples,
                    mask_ratio=args.mask_ratio)}
            else:
                figs = {'visualizations': visualize_scene_4m(
                    mv, val_loader, device, epoch, viz_dir, args.num_viz_samples,
                    mask_ratio=args.mask_ratio, tag='clean')}
            # Occluded val scenes -> clean target, the same fixed scenes every
            # epoch; drawn for the clean control arm too.
            if scene_cfg is not None:
                figs['scene_visualizations'] = visualize_scene_4m(
                    mv, val_loader, device, epoch, viz_dir, args.num_viz_samples,
                    mask_ratio=args.mask_ratio, scene=scene_cfg,
                    seed=scene_cfg.val_seed, tag='scene')
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({**{k: [wandb.Image(p, caption=Path(p).name) for p in v]
                              for k, v in figs.items() if v}, 'epoch': epoch})

        # ── Test-set evaluation + visualizations every test_freq epochs ──────────
        # test_freq <= 0 disables the test pass. The held-out split is scored
        # once at the end by a separate eval, not every N epochs during ablations.
        do_test = (args.test_freq > 0
                   and (epoch % args.test_freq == 0 or epoch == args.epochs))
        if do_test:
            if is_main: print(f"\n🧪 Running TEST-set evaluation (epoch {epoch})…")
            compute_emd_test = (epoch == args.epochs)
            # evaluate() runs on ALL ranks (each on its DistributedSampler shard) to
            # keep DDP in lockstep; only rank 0 logs the result.
            (tl, tr_, td_, tp_, tt_), tmet = evaluate(
                model, test_loader, device, compute_emd=compute_emd_test,
                mask_ratio=args.mask_ratio, distributed=(world_size > 1))
            if is_main:
                history['test_epoch'].append(epoch)
                for k, v in zip(
                        ['test_loss','test_rgb','test_depth','test_pc','test_text'],
                        [tl, tr_, td_, tp_, tt_]):
                    history[k].append(v)
                for mk, _, hk, _ in METRIC_KEYS:
                    if hk is not None and mk in tmet:
                        _append_aligned(history, hk, tmet[mk], 'test_epoch')
                print(f"Test  — Loss: {tl:.4f}  RGB: {tr_:.4f}  Depth: {td_:.4f}  "
                      f"PC: {tp_:.4f}  Text: {tt_:.4f}")
                print(f"Test Metrics — {_fmt_metrics(tmet)}")
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch,
                        'test/loss': tl, 'test/rgb': tr_, 'test/depth': td_,
                        'test/pc': tp_, 'test/text': tt_,
                        **{f'test_metrics/{mk}': tmet[mk]
                           for mk, _, hk, _ in METRIC_KEYS
                           if hk is not None and mk in tmet},
                    })
                print(f"📊 Generating TEST visualizations (epoch {epoch})…")
                mv = model.module if world_size > 1 else model
                tpaths = visualize_reconstruction_4m(
                    mv, test_loader, device, epoch, test_viz_dir,
                    args.num_viz_samples, mask_ratio=args.mask_ratio)
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({'test_visualizations':
                               [wandb.Image(p, caption=Path(p).name) for p in tpaths],
                               'epoch': epoch})

        scheduler.step()

        if is_main and epoch % args.save_freq == 0:
            ms = (model.module if world_size > 1 else model).state_dict()
            torch.save({
                'epoch': epoch, 'model_state_dict': ms,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss, 'history': history,
                'text_mask_ratio': args.text_mask_ratio,
                'structured_mask': args.structured_mask,
                'occlusion_scene': args.occlusion_scene,
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
                'text_mask_ratio': args.text_mask_ratio,
                'structured_mask': args.structured_mask,
                'occlusion_scene': args.occlusion_scene,
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
    parser.add_argument('--view_sampling',      action='store_true', default=None,
                        help='Plan 6.1: one drawn view per plant per epoch '
                             '(10x cheaper epoch, same view diversity).')
    parser.add_argument('--view_seed',          type=int,   default=None)
    parser.add_argument('--max_plants',         type=int,   default=None,
                        help='E3: cap the TRAIN split at N plants (nested subsets)')
    parser.add_argument('--plant_subset_seed',  type=int,   default=None)
    parser.add_argument('--model_size',         type=str,   default=None, choices=['small','base','large'])
    parser.add_argument('--active_modalities',  type=str,   default=None,
                        help="E2 arm, e.g. 'pc' or 'pc,rgb' or 'pc,rgb,depth'. "
                             "Default (unset) = all four. 'pc' is mandatory.")
    parser.add_argument('--mask_ratio',         type=float, default=None)
    parser.add_argument('--text_mask_ratio',    type=float, default=None,
                        help='Mask text at this rate outside the Dirichlet budget '
                             '(e2_pcrgbdt_tg). Unset = shared budget, as every '
                             'earlier run.')
    parser.add_argument('--pc_loss_weight',     type=float, default=None)
    parser.add_argument('--depth_norm_type',    type=str,   default=None, choices=['minmax','standard'])
    parser.add_argument('--spline_loss_weight', type=float, default=None)
    parser.add_argument('--max_leaves',         type=int,   default=None)
    parser.add_argument('--batch_size',         type=int,   default=None)
    parser.add_argument('--epochs',             type=int,   default=None)
    parser.add_argument('--lr',                 type=float, default=None)
    parser.add_argument('--weight_decay',       type=float, default=None)
    parser.add_argument('--warmup_epochs',      type=int,   default=None)
    parser.add_argument('--seed',               type=int,   default=None,
                        help='seed model init, masks and loader draws (default: unseeded)')
    parser.add_argument('--val_freq',           type=int,   default=None)
    parser.add_argument('--test_freq',          type=int,   default=None)
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
                              'weight_decay': 0.05, 'warmup_epochs': 10, 'val_freq': 20,
                              'test_freq': 50},
            'checkpointing': {'output_dir': './outputs/4m_run', 'save_freq': 100, 'resume': None},
            'visualization': {'viz_freq': 50, 'num_viz_samples': 6},
            'distributed':   {'world_size': 4, 'dist_backend': 'nccl', 'dist_url': 'env://'},
            'system':        {'num_workers': 8, 'device': 'cuda'},
            'wandb':         {'use_wandb': True, 'wandb_project': 'embodied-mae-4m',
                              'wandb_entity': None, 'wandb_name': None},
        }
        cfg = config_to_namespace(merge_config_with_args(default_cfg, args))

    check_resume_text_mask_ratio(cfg)

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
