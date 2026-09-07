"""
Cross-modal DISTILLATION training for EmbodiedMAE-4M.

Goal
----
Train the 4M model so that **any single modality (RGB | Depth | PointCloud |
Spline-params), with the other three fully masked, reconstructs all four**.

How
---
Teacher-student distillation (frozen full-modal teacher → single-modal student):

  • TEACHER  : a frozen copy of a trained 4M checkpoint. Every step it sees ALL
               four modalities fully visible and produces token-aligned decoder
               features + a CLS latent — the privileged "all-modality" target.
  • STUDENT  : the trainable model. On a *cross-modal* step it sees ONE source
               modality (the rest fully masked) and must (a) reconstruct the
               masked modalities against ground truth AND (b) match the teacher's
               decoder features + CLS latent.

The decoder restores every token position for every modality regardless of which
tokens were masked, so teacher/student decoder sequences line up position-for-
position — feature distillation is a plain token-wise MSE.

Batch schedule (deterministic, DDP-synchronised from (epoch, batch_idx)):
  • with prob `crossmodal_prob`  → cross-modal step, source modality cycled
                                    through `sources` so all four learn to
                                    generate the rest.
  • otherwise                    → a normal Dirichlet-masked reconstruction step
                                    (keeps in-distribution reconstruction sharp).

Usage
-----
    # single GPU
    python train_sorghum_4m_distill.py --config config_4m_distill.yaml --world_size 1

    # multi-GPU (torchrun)
    torchrun --standalone --nproc_per_node=4 train_sorghum_4m_distill.py \
        --config config_4m_distill.yaml
"""

import os
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    embodied_mae_4m_small,
    embodied_mae_4m_base,
    N_PARAMS,
)
from embodied_mae import chamfer_distance
from sorghum_dataset_4m import SorghumDataset4M
from train_sorghum_4m import (
    setup_distributed, cleanup_distributed, unpatchify,
)


MODALITIES = ['rgb', 'depth', 'pc', 'text']
ALL_VISIBLE = set(MODALITIES)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_namespace(c):
    g = lambda sec, k, d: c.get(sec, {}).get(k, d)
    ns = argparse.Namespace()
    # data / model / training (mirrors train_sorghum_4m)
    ns.data_root          = c['data']['data_root']
    ns.img_size           = g('data', 'img_size', 224)
    ns.num_points         = g('data', 'num_points', 8196)
    ns.model_size         = g('model', 'model_size', 'base')
    ns.mask_ratio         = g('model', 'mask_ratio', 0.75)
    ns.pc_loss_weight     = g('model', 'pc_loss_weight', 10.0)
    ns.depth_norm_type    = g('model', 'depth_norm_type', 'minmax')
    ns.spline_loss_weight = g('model', 'spline_loss_weight', 5.0)
    ns.max_leaves         = g('model', 'max_leaves', 24)
    # Point-cloud objective. 'qal_loss' is the sigmoid-weighted two-sided Chamfer
    # ported from the yongyun branch; it up-weights nearest-neighbour errors past
    # `qal_threshold`, which targets thin-leaf geometry. 'chamfer' = previous behaviour.
    ns.pc_loss_name       = g('model', 'loss_name', 'chamfer')
    ns.qal_threshold      = g('model', 'qal_threshold', 0.01)
    ns.qal_alpha          = g('model', 'qal_alpha', 100.0)
    ns.qal_use_squared    = g('model', 'qal_use_squared', False)
    ns.batch_size         = g('training', 'batch_size', 8)
    ns.epochs             = g('training', 'epochs', 800)
    ns.lr                 = g('training', 'lr', 1.0e-4)
    ns.weight_decay       = g('training', 'weight_decay', 0.05)
    ns.warmup_epochs      = g('training', 'warmup_epochs', 10)
    ns.val_freq           = g('training', 'val_freq', 20)
    # cap the cross-modal val sweep at N batches per source (0 = full split).
    # The 15k split has 22.5k val folders x len(sources) forward passes, which is
    # both wasteful and long enough to trip the NCCL timeout on non-zero ranks.
    ns.val_max_batches    = g('training', 'val_max_batches', 0)
    ns.max_steps          = g('training', 'max_steps', 0)
    # distillation
    ns.teacher_checkpoint     = g('distill', 'teacher_checkpoint',
                                  './outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth')
    ns.student_init           = g('distill', 'student_init',
                                  './outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth')
    ns.crossmodal_prob        = g('distill', 'crossmodal_prob', 0.7)
    # Mask the SOURCE modality during cross-modal steps. 0.0 (default) hands the
    # source over whole, which is how every run up to 2026-09-03 behaved. Above 0,
    # the student must infer the absent modalities from a partial source -- a
    # strictly harder task, and the one rgb_mask_sweep.py probes at eval time.
    # Note this makes the source's own recon loss non-zero: its masked tokens
    # become real targets, so a logged 'rgb' of 0.0000 stops being expected.
    ns.source_mask_ratio      = g('distill', 'source_mask_ratio', 0.0)
    ns.sources                = g('distill', 'sources', MODALITIES)
    ns.feat_distill_weight    = g('distill', 'feat_distill_weight', 1.0)
    ns.cls_distill_weight     = g('distill', 'cls_distill_weight', 0.5)
    ns.output_distill_weight  = g('distill', 'output_distill_weight', 0.0)
    ns.distill_masked_only    = g('distill', 'distill_masked_only', True)
    ns.viz_source             = g('distill', 'viz_source', 'depth')
    # optional single-modality *target*: when set (e.g. 'pc'), only that modality's
    # reconstruction loss is trained, feature distillation is focused on its tokens,
    # and the best-model metric is its own generation quality (chamfer for pc).
    ns.target                 = g('distill', 'target', None)
    # `target` may be a single modality or a list; `targets` is the normalised
    # list form used everywhere below. None/[] means "all four" (no restriction).
    _t = ns.target
    ns.targets = [] if _t is None else ([_t] if isinstance(_t, str) else list(_t))
    # checkpointing / viz / dist / system / wandb
    ns.output_dir         = g('checkpointing', 'output_dir', './outputs/4m_distill')
    ns.save_freq          = g('checkpointing', 'save_freq', 50)
    # Retention: how many periodic checkpoints to keep in checkpoints/.
    # best_model.pth lives at output_dir level and is NEVER pruned, so
    # keep_last=1 means 'last + best', which is all that is needed:
    # best_model.pth carries model+optimizer+scheduler+history and is
    # fully resumable on its own. 0 or negative disables pruning.
    ns.keep_last          = g('checkpointing', 'keep_last', 1)
    ns.resume             = g('checkpointing', 'resume', None)
    ns.viz_freq           = g('visualization', 'viz_freq', 25)
    ns.num_viz_samples    = g('visualization', 'num_viz_samples', 6)
    ns.world_size         = g('distributed', 'world_size', 1)
    ns.dist_backend       = g('distributed', 'dist_backend', 'nccl')
    ns.dist_url           = g('distributed', 'dist_url', 'env://')
    ns.num_workers        = g('system', 'num_workers', 8)
    ns.device             = g('system', 'device', 'cuda')
    ns.use_wandb          = g('wandb', 'use_wandb', True)
    ns.wandb_project      = g('wandb', 'wandb_project', 'embodied-mae-4m-distill')
    ns.wandb_entity       = g('wandb', 'wandb_entity', None)
    ns.wandb_name         = g('wandb', 'wandb_name', None)
    # keep validated source list
    ns.sources = [s for s in ns.sources if s in MODALITIES]
    assert ns.sources, "distill.sources must contain at least one of rgb/depth/pc/text"
    for _m in ns.targets:
        assert _m in MODALITIES, \
            f"distill.target entries must be in {MODALITIES}, got {_m}"
    assert len(set(ns.targets)) == len(ns.targets), \
        f"distill.target has duplicates: {ns.targets}"
    return ns


# ── Model build / checkpoint loading ──────────────────────────────────────────

def build_model(args, device):
    build_fn = embodied_mae_4m_small if args.model_size == 'small' else embodied_mae_4m_base
    return build_fn(
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
    ).to(device)


def load_weights_into(model, ckpt_path, device, tag):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']
    if list(sd)[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ep = ckpt.get('epoch')
    vl = ckpt.get('val_loss', ckpt.get('best_val_loss'))
    print(f"  [{tag}] loaded {ckpt_path} (epoch {ep}"
          + (f", val {vl:.4f}" if isinstance(vl, (int, float)) else "") + ")")
    if missing:    print(f"      missing keys   : {len(missing)}")
    if unexpected: print(f"      unexpected keys: {len(unexpected)}")
    return model


# ── Distillation token weighting ──────────────────────────────────────────────

def token_block_sizes(model):
    """Lengths of the four decoder blocks in concat order."""
    return [model.num_patches, model.num_patches,
            model.num_pc_tokens, model.n_text_tokens]


def masked_token_weight(model, source, device):
    """(L,) weight: 1.0 on tokens of GENERATED (masked) modalities, 0.0 on the
    visible source modality's tokens. Used to focus feature distillation on the
    positions the student must actually generate."""
    sizes = token_block_sizes(model)
    w = []
    for name, n in zip(MODALITIES, sizes):
        w.append(torch.zeros(n) if name == source else torch.ones(n))
    return torch.cat(w).to(device)


def target_token_weight(model, targets, device):
    """(L,) weight: 1.0 on tokens of the TARGET modalities, 0.0 elsewhere.
    Focuses feature distillation on the modalities being specialised for.
    `targets` is a list; a bare string is accepted for older configs."""
    if isinstance(targets, str):
        targets = [targets]
    tset = set(targets)
    sizes = token_block_sizes(model)
    w = []
    for name, n in zip(MODALITIES, sizes):
        w.append(torch.ones(n) if name in tset else torch.zeros(n))
    return torch.cat(w).to(device)


def feature_distill_loss(feats_s, feats_t, weight=None):
    """Token-wise MSE between student and (frozen) teacher decoder features.
    weight : optional (L,) per-token weight; None → uniform over all tokens."""
    diff = (feats_s - feats_t.detach()) ** 2          # (B, L, D)
    per_tok = diff.mean(-1)                            # (B, L)
    if weight is None:
        return per_tok.mean()
    denom = weight.sum().clamp(min=1) * per_tok.shape[0]
    return (per_tok * weight.unsqueeze(0)).sum() / denom


# ── Single distillation step (cross-modal) ────────────────────────────────────

def crossmodal_distill_step(student, teacher, batch_t, source, args, device):
    rgb, depth, pc, params, tv = batch_t
    visible = {source}

    total_recon, (lr, ld, lp, lt), preds_s, _, (feats_s, cls_s) = student(
        rgb, depth, pc, params, tv, visible=visible, return_features=True,
        source_mask_ratio=getattr(args, 'source_mask_ratio', 0.0))

    with torch.no_grad():
        _, _, preds_t, _, (feats_t, cls_t) = teacher(
            rgb, depth, pc, params, tv, visible=ALL_VISIBLE, return_features=True)

    m = student.module if hasattr(student, 'module') else student
    targets = getattr(args, 'targets', None) or []
    if targets:
        w = target_token_weight(m, targets, device)
    elif args.distill_masked_only:
        w = masked_token_weight(m, source, device)
    else:
        w = None
    feat_loss = feature_distill_loss(feats_s, feats_t, w)
    cls_loss  = F.mse_loss(cls_s, cls_t.detach())

    out_loss = torch.zeros((), device=device)
    if args.output_distill_weight > 0:
        pr_s, pd_s, pp_s, pa_s = preds_s
        pr_t, pd_t, pp_t, pa_t = preds_t
        out_loss = (F.mse_loss(pr_s, pr_t.detach())
                    + F.mse_loss(pd_s, pd_t.detach())
                    + F.mse_loss(pa_s, pa_t.detach())
                    + chamfer_distance(pp_s, pp_t.detach()) * m.pc_loss_weight)

    # Restrict reconstruction to the target modalities when requested. Summing
    # the components is the right combination: each is already scaled by its own
    # loss weight (pc_loss_weight, spline_loss_weight) inside forward_loss, so
    # they arrive on a comparable scale.
    _parts = {'rgb': lr, 'depth': ld, 'pc': lp, 'text': lt}
    recon_term = (sum(_parts[t] for t in targets) if targets else total_recon)

    loss = (recon_term
            + args.feat_distill_weight * feat_loss
            + args.cls_distill_weight  * cls_loss
            + args.output_distill_weight * out_loss)

    stats = {
        'loss': loss.item(), 'recon': recon_term.item(),
        'rgb': lr.item(), 'depth': ld.item(), 'pc': lp.item(), 'text': lt.item(),
        'feat': feat_loss.item(), 'cls': cls_loss.item(),
        'out': float(out_loss.item()),
    }
    return loss, stats


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(student, teacher, loader, optimizer, device, epoch, args,
                    n_batches):
    student.train()
    agg = {k: 0.0 for k in
           ['loss', 'recon', 'rgb', 'depth', 'pc', 'text', 'feat', 'cls', 'out']}
    n_xm = 0
    src_count = {s: 0 for s in args.sources}

    n_done = 0
    _warm, _t0 = 10, None      # steady-state timing ignores the first _warm steps
    if args.max_steps:
        torch.cuda.reset_peak_memory_stats(device)
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for bi, (rgb, depth, pc, params, tv, _) in enumerate(pbar):
        rgb = rgb.to(device); depth = depth.to(device); pc = pc.to(device)
        params = params.to(device); tv = tv.to(device)
        batch_t = (rgb, depth, pc, params, tv)

        # Deterministic, rank-identical schedule so DDP stays in lock-step.
        step_id = (epoch - 1) * n_batches + bi
        is_xm   = random.Random(step_id).random() < args.crossmodal_prob

        if is_xm:
            source = args.sources[step_id % len(args.sources)]
            loss, stats = crossmodal_distill_step(
                student, teacher, batch_t, source, args, device)
            n_xm += 1
            src_count[source] += 1
        else:
            # config mask_ratio was parsed but never passed here; the model default
            # (0.75) was silently used on every full-recon step.
            loss, (lr, ld, lp, lt), _, _ = student(rgb, depth, pc, params, tv,
                                                   mask_ratio=args.mask_ratio)
            stats = {'loss': loss.item(), 'recon': loss.item(),
                     'rgb': lr.item(), 'depth': ld.item(),
                     'pc': lp.item(), 'text': lt.item(),
                     'feat': 0.0, 'cls': 0.0, 'out': 0.0}

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        for k in agg:
            agg[k] += stats[k]
        pbar.set_postfix({
            'loss': f"{stats['loss']:.3f}",
            'mode': ('XM:' + source) if is_xm else 'recon',
            'feat': f"{stats['feat']:.3f}",
        })

        n_done = bi + 1
        if args.max_steps and n_done == _warm:
            torch.cuda.synchronize(device); _t0 = time.time()
        # Debug/smoke only: cut the epoch short so the val/viz/checkpoint path is
        # reachable without walking all 105k folders. All ranks break at the same
        # step, so DDP stays in sync.
        if args.max_steps and n_done >= args.max_steps:
            break

    n = max(n_done, 1)
    if args.max_steps:
        torch.cuda.synchronize(device)
        _rate = ((n_done - _warm) / (time.time() - _t0)) if _t0 and n_done > _warm else 0.0
        print(f"PEAKMEM alloc={torch.cuda.max_memory_allocated(device)/2**20:.0f}MiB "
              f"reserved={torch.cuda.max_memory_reserved(device)/2**20:.0f}MiB "
              f"steps={n} steady_it_s={_rate:.3f}", flush=True)
    return {k: v / n for k, v in agg.items()}, n_xm, src_count


# ── Cross-modal validation (per source) ───────────────────────────────────────

@torch.no_grad()
def evaluate_crossmodal(student, loader, device, sources, target=None, max_batches=0):
    """For each source modality, run single-modal → all and measure how well the
    *generated* (masked) modalities match ground truth. Returns per-source dict
    and a scalar `mean_gen` used for best-model. When `target` is set, `mean_gen`
    tracks that modality's own generation quality (chamfer for pc, param MAE for
    text, per-modality recon otherwise); else it is the mean recon-total."""
    student.eval()
    m = student.module if hasattr(student, 'module') else student
    per_source = {}

    for src in sources:
        acc = {'total': 0.0, 'rgb': 0.0, 'depth': 0.0,
               'pc_chamfer': 0.0, 'param_mae_masked': 0.0,
               'pc_loss': 0.0, 'text_loss': 0.0}
        nb = 0
        for rgb, depth, pc, params, tv, _ in loader:
            rgb = rgb.to(device); depth = depth.to(device); pc = pc.to(device)
            params = params.to(device); tv = tv.to(device)
            total, (lr, ld, lp, lt), (pr, pd, ppc, pparam), \
                (mr, md, mpc, mt) = m(
                    rgb, depth, pc, params, tv, visible={src})

            acc['total'] += total.item()
            acc['rgb']   += lr.item()
            acc['depth'] += ld.item()
            acc['pc_loss'] += lp.item()
            acc['text_loss'] += lt.item()
            acc['pc_chamfer'] += chamfer_distance(ppc, pc).item()
            diff_abs = (pparam - params).abs().mean(-1)         # (B, L)
            denom = (tv * mt).sum().clamp(min=1)
            acc['param_mae_masked'] += ((diff_abs * tv * mt).sum() / denom).item()
            nb += 1
            if max_batches and nb >= max_batches:
                break

        per_source[src] = {k: v / max(nb, 1) for k, v in acc.items()}

    # One target keeps its natural, directly-interpretable metric (Chamfer, param
    # MAE). Several targets cannot share one of those - Chamfer ~1e-3 against
    # param MAE ~3e-2 would let the larger scale decide the best model on its own -
    # so fall back to the summed weighted loss components, which is exactly the
    # quantity being optimised.
    tl = [target] if isinstance(target, str) else list(target or [])
    single = {'pc': 'pc_chamfer', 'text': 'param_mae_masked',
              'rgb': 'rgb', 'depth': 'depth'}
    if len(tl) == 1:
        mean_gen = float(np.mean([per_source[s][single[tl[0]]] for s in sources]))
    elif tl:
        comp = {'rgb': 'rgb', 'depth': 'depth', 'pc': 'pc_loss', 'text': 'text_loss'}
        mean_gen = float(np.mean([sum(per_source[s][comp[t]] for t in tl)
                                  for s in sources]))
    else:
        mean_gen = float(np.mean([per_source[s]['total'] for s in sources]))
    student.train()
    return per_source, mean_gen


# ── Visualisation (fixed source → all modalities) ─────────────────────────────

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



def _sub(pc_np, n=1500):
    if pc_np.shape[0] > n:
        return pc_np[np.random.choice(pc_np.shape[0], n, replace=False)]
    return pc_np


@torch.no_grad()
def visualize_crossmodal(model, loader, device, epoch, save_dir, source,
                         num_samples=6):
    model_m = model.module if hasattr(model, 'module') else model
    model_m.eval()
    rgb, depth, pc, params, tv, names = next(iter(loader))
    rgb = rgb[:num_samples].to(device); depth = depth[:num_samples].to(device)
    pc = pc[:num_samples].to(device); params = params[:num_samples].to(device)
    tv = tv[:num_samples].to(device); names = list(names[:num_samples])

    _, _, (pr, pd, ppc, pparam), _ = model_m(
        rgb, depth, pc, params, tv, visible={source})
    pr = _unnorm_pix(model_m, pr, rgb)
    pr_img = unpatchify(pr, model_m.patch_size, 3, model_m.img_size)
    pd_img = unpatchify(pd, model_m.patch_size, 1, model_m.img_size)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rgb_dn  = (rgb * std + mean).clamp(0, 1).cpu().numpy()
    pr_dn   = (pr_img * std + mean).clamp(0, 1).cpu().numpy()
    depth_np = depth.cpu().numpy(); pd_np = pd_img.cpu().numpy()
    pc_np = pc.cpu().numpy(); ppc_np = ppc.cpu().numpy()
    gt_txt  = model_m.decode_params_to_text(params)
    gen_txt = model_m.decode_params_to_text(pparam)

    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(rgb.shape[0]):
        fig = plt.figure(figsize=(16, 7))
        cd = chamfer_distance(ppc[i:i+1], pc[i:i+1]).item()

        def panel(pos, img, title, cmap=None):
            ax = plt.subplot(2, 4, pos)
            ax.imshow(img if cmap is None else img, cmap=cmap); ax.axis('off')
            ax.set_title(title, fontsize=8, fontweight='bold')

        is_in = lambda mod: '  ← INPUT' if mod == source else '  (generated)'
        panel(1, rgb_dn[i].transpose(1, 2, 0).clip(0, 1), f'GT RGB')
        panel(2, pr_dn[i].transpose(1, 2, 0).clip(0, 1), f'RGB{is_in("rgb")}')
        panel(3, depth_np[i, 0], 'GT Depth', cmap='viridis')
        panel(4, pd_np[i, 0], f'Depth{is_in("depth")}', cmap='viridis')

        gt = _sub(pc_np[i]); gen = _sub(ppc_np[i])
        ax = plt.subplot(2, 4, 5, projection='3d')
        ax.scatter(gt[:, 0], gt[:, 2], gt[:, 1], c=gt[:, 1], cmap='viridis', s=2)
        ax.set_title('GT PointCloud', fontsize=8, fontweight='bold'); ax.view_init(20, 45)
        ax = plt.subplot(2, 4, 6, projection='3d')
        ax.scatter(gen[:, 0], gen[:, 2], gen[:, 1], c=gen[:, 1], cmap='plasma', s=2)
        ax.set_title(f'PC{is_in("pc")}\nChamfer {cd:.5f}', fontsize=8); ax.view_init(20, 45)

        # Own panel in the two empty bottom-right cells (cols 2-3 of row 1) so the
        # spline-param text never overlaps the Depth(generated) panel above it.
        ax = plt.subplot2grid((2, 4), (1, 2), colspan=2); ax.axis('off')
        n_real = int(tv[i].sum().item())
        lines = [f"SPLINE PARAMS  (params {is_in('text').strip()})", "=" * 30, "",
                 "PLANT:", f" GT : {gt_txt[i][0]}", f" GEN: {gen_txt[i][0]}", ""]
        for ti in range(1, min(n_real, 4)):
            lines += [f"leaf{ti} GT : {gt_txt[i][ti]}",
                      f"leaf{ti} GEN: {gen_txt[i][ti]}", ""]
        ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left',
                family='monospace', fontsize=7, transform=ax.transAxes)

        plt.suptitle(f"Epoch {epoch} | {names[i]} | source={source.upper()} → all",
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        p = save_dir / f'epoch_{epoch:03d}_src-{source}_sample_{i+1}_{names[i]}.png'
        plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
        saved.append(str(p))

    model_m.train()
    return saved


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
        viz_dir.mkdir(exist_ok=True); checkpoint_dir.mkdir(exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    # Data
    train_ds = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                num_points=args.num_points, split='train',
                                max_leaves=args.max_leaves)
    val_ds   = SorghumDataset4M(args.data_root, img_size=args.img_size,
                                num_points=args.num_points, split='val',
                                max_leaves=args.max_leaves)
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        shuffle_train = False
    else:
        train_sampler = None; shuffle_train = True
    # persistent_workers: without it the 8 ranks tear down and respawn
    # num_workers processes EVERY epoch. On a cold page cache that respawn plus
    # re-reading the 105k-folder index dominated the epoch (measured 2026-08-22:
    # 5 min/epoch warm -> 63 min/epoch cold, node load 354/360). The val loader
    # stays non-persistent and thin: it runs every val_freq epochs and is
    # iterated once per source, so keeping 8x workers resident all run would
    # just steal CPU from training.
    val_workers = min(args.num_workers, 8)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=shuffle_train, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=val_workers, pin_memory=True)
    n_batches = len(train_loader)

    # ── Teacher (frozen, full-modal) ──────────────────────────────────────────
    if is_main: print("\n🧊 Building frozen TEACHER…")
    teacher = build_model(args, device)
    load_weights_into(teacher, args.teacher_checkpoint, device, 'teacher')
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Student (warm-started, trainable) ─────────────────────────────────────
    if is_main: print("\n🎓 Building STUDENT…")
    student = build_model(args, device)
    if args.student_init and os.path.exists(args.student_init):
        load_weights_into(student, args.student_init, device, 'student-init')
    elif is_main:
        print("  [student-init] none found — training from scratch")

    if world_size > 1:
        # cross-modal steps leave the masked modalities' encoder embeds unused,
        # and the unused set changes step-to-step → must allow it.
        student = DDP(student, device_ids=[rank], output_device=rank,
                      find_unused_parameters=True)

    total_params = sum(p.numel() for p in student.parameters())
    if is_main: print(f"\nStudent parameters: {total_params:,}")

    # wandb
    if is_main and args.use_wandb and WANDB_AVAILABLE:
        if args.wandb_name is None:
            args.wandb_name = f"4m_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run_id = None
        if args.resume and os.path.exists(args.resume):
            wandb_run_id = torch.load(args.resume, map_location='cpu', weights_only=False).get('wandb_run_id')
        if wandb_run_id:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       id=wandb_run_id, resume='must')
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_name, config=vars(args))
        print(f"✅ W&B: {wandb.run.url}")

    optimizer = optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (ep - args.warmup_epochs)
                                 / max(1, args.epochs - args.warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch   = 1
    best_val      = float('inf')
    history       = {'train': [], 'val': []}

    if args.resume and os.path.exists(args.resume):
        if is_main: print(f"\n📂 Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt['model_state_dict']
        if world_size > 1 and not list(sd)[0].startswith('module.'):
            sd = {'module.' + k: v for k, v in sd.items()}
        elif world_size == 1 and list(sd)[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        student.load_state_dict(sd)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val_loss', float('inf'))
        history     = ckpt.get('history', history)

    if is_main:
        print(f"\n🚀 Distillation training for {args.epochs} epochs")
        print(f"   crossmodal_prob={args.crossmodal_prob}  sources={args.sources}"
              f"  target={'+'.join(args.targets) if args.targets else 'all'}")
        print(f"   feat_w={args.feat_distill_weight} cls_w={args.cls_distill_weight} "
              f"out_w={args.output_distill_weight} masked_only={args.distill_masked_only}")
        print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n{'='*80}\nEpoch {epoch}/{args.epochs}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}\n{'='*80}")

        tr, n_xm, src_count = train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch, args, n_batches)
        history['train'].append({'epoch': epoch, **tr})

        if is_main:
            print(f"\nTrain — loss {tr['loss']:.4f}  recon {tr['recon']:.4f}  "
                  f"feat {tr['feat']:.4f}  cls {tr['cls']:.4f}  "
                  f"| {n_xm}/{n_batches} crossmodal  sources={src_count}")

        do_val = (epoch % args.val_freq == 0 or epoch == args.epochs or epoch == 1)
        mean_gen = None
        if do_val and is_main:
            print("\n🔍 Cross-modal validation (per source)…")
            per_source, mean_gen = evaluate_crossmodal(
                student, val_loader, device, args.sources, args.targets,
                max_batches=args.val_max_batches)
            history['val'].append({'epoch': epoch, 'mean_gen': mean_gen,
                                   'per_source': per_source})
            for s in args.sources:
                ps = per_source[s]
                print(f"  src={s:<5} → total {ps['total']:.4f}  "
                      f"rgb {ps['rgb']:.4f}  depth {ps['depth']:.4f}  "
                      f"pc_chamfer {ps['pc_chamfer']:.5f}  "
                      f"param_mae(masked) {ps['param_mae_masked']:.4f}")
            _mlbl = ('+'.join(args.targets) + '_loss' if len(args.targets) > 1
                     else {'pc': 'pc_chamfer', 'text': 'param_mae'}.get(
                         args.targets[0], 'recon') if args.targets else 'total')
            print(f"  MEAN generation metric over sources "
                  f"({'+'.join(args.targets) if args.targets else 'all'}→{_mlbl}): {mean_gen:.5f}")

        if is_main and args.use_wandb and WANDB_AVAILABLE:
            logd = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in tr.items()},
                    'train/crossmodal_frac': n_xm / max(n_batches, 1)}
            if mean_gen is not None:
                logd['val/mean_gen'] = mean_gen
                for s in args.sources:
                    for k, v in per_source[s].items():
                        logd[f'val/{s}/{k}'] = v
            wandb.log(logd)

        if is_main and (epoch % args.viz_freq == 0 or epoch == 1):
            print(f"\n📊 Visualising source={args.viz_source} → all…")
            paths = visualize_crossmodal(student, val_loader, device, epoch,
                                         viz_dir, args.viz_source, args.num_viz_samples)
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({'visualizations': [wandb.Image(p, caption=Path(p).name)
                                              for p in paths], 'epoch': epoch})

        scheduler.step()

        # checkpoints
        def state_dict():
            return (student.module if world_size > 1 else student).state_dict()

        if is_main and epoch % args.save_freq == 0:
            # Write to .tmp then os.replace: an atomic rename on POSIX, so a
            # preemption mid-write cannot leave a truncated checkpoint. This
            # matters now that we prune -- without it, keep_last=1 could delete
            # the only good checkpoint in favour of a torn one.
            _final = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            _tmp   = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth.tmp'
            torch.save({'epoch': epoch, 'model_state_dict': state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val, 'history': history,
                        'wandb_run_id': (wandb.run.id if args.use_wandb
                                         and WANDB_AVAILABLE and wandb.run else None)},
                       _tmp)
            os.replace(_tmp, _final)
            print(f"💾 checkpoint_epoch_{epoch}.pth")

            # Prune older periodic checkpoints. best_model.pth is a separate
            # file at output_dir level and is never touched here.
            if args.keep_last and args.keep_last > 0:
                cks = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'),
                             key=lambda p: int(p.stem.rsplit('_', 1)[1]))
                for old_ck in cks[:-args.keep_last]:
                    try:
                        old_ck.unlink()
                        print(f"🗑  pruned {old_ck.name}")
                    except OSError as exc:
                        print(f"⚠️  could not prune {old_ck.name}: {exc}")

        if is_main and mean_gen is not None and mean_gen < best_val:
            best_val = mean_gen
            torch.save({'epoch': epoch, 'model_state_dict': state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': mean_gen, 'best_val_loss': best_val,
                        'history': history,
                        'wandb_run_id': (wandb.run.id if args.use_wandb
                                         and WANDB_AVAILABLE and wandb.run else None)},
                       output_dir / 'best_model.pth')
            print(f"⭐ New best cross-modal model! mean_gen {mean_gen:.4f}")

        if is_main:
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

    if world_size > 1:
        cleanup_distributed()
    if is_main and args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    if is_main:
        print(f"\n{'='*80}\nDistillation complete! best mean_gen {best_val:.4f}\n"
              f"Outputs: {output_dir}\n{'='*80}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Cross-modal distillation for EmbodiedMAE-4M')
    ap.add_argument('--config', type=str, default='config_4m_distill.yaml')
    ap.add_argument('--world_size', type=int, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--teacher_checkpoint', type=str, default=None)
    ap.add_argument('--student_init', type=str, default=None)
    ap.add_argument('--crossmodal_prob', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--val_max_batches', type=int, default=None)
    ap.add_argument('--max_steps', type=int, default=None,
                    help='debug/smoke: stop each epoch after N steps (0 = full epoch)')
    ap.add_argument('--source_mask_ratio', type=float, default=None,
                    help='mask this fraction of the SOURCE modality tokens during '
                         'cross-modal steps (0 = hand the source over whole)')
    ap.add_argument('--no_wandb', action='store_true')
    args_cli = ap.parse_args()

    if not os.path.exists(args_cli.config):
        raise FileNotFoundError(f"config not found: {args_cli.config}")
    print(f"📋 Loading config: {args_cli.config}")
    cfg = config_to_namespace(load_config(args_cli.config))

    # CLI overrides
    for k in ['world_size', 'output_dir', 'resume', 'teacher_checkpoint',
              'student_init', 'crossmodal_prob', 'batch_size', 'epochs',
              'num_workers', 'val_max_batches', 'max_steps', 'source_mask_ratio']:
        v = getattr(args_cli, k)
        if v is not None:
            setattr(cfg, k, v)
    if args_cli.no_wandb:
        cfg.use_wandb = False

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        rank = int(os.environ['RANK']); world_size = int(os.environ['WORLD_SIZE'])
        cfg.world_size = world_size
        if rank == 0:
            print(f"\n🚀 Multi-GPU (torchrun) GPUs={world_size} batch/GPU={cfg.batch_size}")
        train_worker(rank, world_size, cfg)
    elif cfg.world_size > 1:
        print(f"\n🚀 Multi-GPU (mp.spawn) GPUs={cfg.world_size}")
        mp.spawn(train_worker, args=(cfg.world_size, cfg),
                 nprocs=cfg.world_size, join=True)
    else:
        train_worker(0, 1, cfg)


if __name__ == '__main__':
    main()
