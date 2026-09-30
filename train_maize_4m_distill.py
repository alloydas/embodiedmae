"""
Cross-modal DISTILLATION training for EmbodiedMAE-4M — MAIZE.

The maize twin of train_sorghum_4m_distill.py. Deliberately a separate entry
point, not a species flag on the sorghum trainer: maize and sorghum are kept as
independent pipelines (CLAUDE.md, "Second dataset: Maize"), so a change to one
can never silently change the other.

Goal
----
Train the 4M model so that **any single modality (RGB | Depth | PointCloud |
Spline-params), with the other three fully masked, reconstructs all four**.

How (as sorghum, with one declared departure in what the teacher sees)
---
Teacher-student distillation (frozen full-modal teacher -> single-modal student):

  * TEACHER  : a frozen copy of a trained maize 4M checkpoint. Every step it sees
               ALL four modalities and produces token-aligned decoder features +
               a CLS latent -- the privileged target. Sorghum hands the teacher
               every token of every modality; the maize config hands it 50 % of
               each modality's tokens (distill.teacher_source_mask_ratio 0.5),
               because the maize checkpoint collapses its point cloud when all
               196 PC tokens are visible. See config_to_namespace.
  * STUDENT  : the trainable model, warm-started from the same checkpoint. On a
               *cross-modal* step it sees ONE source modality (the rest fully
               masked) and must (a) reconstruct the masked modalities against
               ground truth AND (b) match the teacher's decoder features + CLS.

Batch schedule (deterministic, DDP-synchronised from (epoch, optimiser step)):
  * with prob `crossmodal_prob` -> cross-modal step, source cycled through
                                   `sources`.
  * otherwise                   -> a normal Dirichlet-masked reconstruction step.

What differs from the sorghum trainer, and why
----------------------------------------------
Everything that is maize-specific comes from the maize modules; the distillation
logic (losses, token weighting, schedule, validation metric, visualisation) is a
verbatim copy of the sorghum trainer's.

  * Model     : EmbodiedMAE4MMaize -- N_PARAMS = 14, MAX_LEAVES = 28 (29 text
                tokens), built with EXACTLY the kwargs train_maize_4m.py:711-724
                uses (`target_points`, `pc_loss_name`, `active_modalities` parsed
                to a tuple by _parse_modalities -- never the raw string).
  * Data      : MaizeDataset4M, num_points 8192 (a pointcloud_cam.ply holds
                exactly 8192; more pads with duplicates). ALL TEN VIEWS per
                epoch, no view_sampling -- the sorghum distill run read all
                105,000 train folders per epoch too, so an epoch is the same
                105,000 samples in both species.
  * Config    : NO built-in default. A missing --config is refused, exactly as
                train_maize_4m.py refuses it: a silent fallback would build a
                sorghum-width model.
  * Accum     : the reference run (outputs/4m_distill_15k_all) trained at
                16/GPU x 8 GPUs = 128 with accum 1, i.e. 821 optimiser steps per
                epoch. Here 16/GPU x 2 GPUs x accum 4 = 128. What that buys is
                equivalence to the reference's 8-rank DDP arrangement (a mean of
                eight per-16 means, BatchNorm statistics per 16), NOT to a single
                batch of 128 -- the reference was not one either. 105,000 / 2 ranks /
                16 = 3,282 micro-batches per rank = 820 full groups of 4 + a
                2-micro-batch tail. The sorghum accum path floors
                (n_batches // accum) and drops that tail -- its gradients are
                computed and then zeroed at the next epoch's first zero_grad. Here
                the tail STEPS, scaled by 1/len(tail), so maize takes exactly the
                reference run's 821 optimiser steps per epoch over the same
                105,000 samples, and the (epoch, step) -> (mode, source) schedule
                is the reference run's schedule step for step.
  * LR        : unchanged -- LambdaLR over EPOCHS, stepped once per epoch, so
                accumulation changes neither the number of scheduler steps nor
                the LR any epoch sees.
  * Warm start: `student_init` that does not exist is FATAL (sorghum silently
                trains from scratch), and `expect_init_epoch` pins the teacher
                and the student to the epoch the config names (600 = the
                E2-matched 197,400-step maize_4m point, not best_model.pth).
  * Epoch 0   : the warm-started student is scored with the exact per-source
                metric BEFORE any distillation step, recorded as history['val']
                epoch 0 and in warmstart_eval.json. Sorghum's 0.3439 came from a
                separate script (eval/eval_warmstart.py); doing it in-run makes
                the before->after pair one code path, one val subset.
  * Eval RNG  : validation is run under a fixed seed (FPS picks its first
                centroid with torch.randint). Same metric; before and after are
                then paired draws rather than two independent ones.
  * Writes    : checkpoints, best_model.pth, history and config.json are all
                written temp-file + os.replace, so a preemption cannot leave a
                torn file for the launcher's auto-resume to trip over. Epoch 1 is
                checkpointed as well as every save_freq, and the W&B run id is
                kept in wandb_run_id.txt, so a requeue before the first periodic
                checkpoint rejoins the same W&B run. config.json is rewritten at
                every launch but carries student_init/student_init_epoch forward
                from the checkpoint's run_meta, so it still names the warm start.
  * Teacher   : `distill.teacher_visible` / `teacher_source_mask_ratio`. The
                code default (all four, 0.0) is sorghum's ALL_VISIBLE, and it is
                REFUSED unless `distill.allow_all_visible_teacher: true`: the
                maize epoch-600 checkpoint collapses its decoded point cloud
                whenever all 196 PC tokens are visible (config_to_namespace has
                the numbers). configs/config_maize_distill_all.yaml runs all four
                at teacher_source_mask_ratio 0.5 -- the one departure from the
                sorghum recipe, declared and recorded in config.json/run_meta.
  * Resume    : refused if any experiment-defining value changed (epochs, lr,
                warmup, weight decay, schedule, distill weights, sources, val
                subset/seed, teacher regime, batch layout). `lr_lambda` is rebuilt
                from args.epochs, so a resume with a different --epochs would
                silently reshape the cosine (the maize_4m_1000ep trap).

Usage
-----
    # 2 GPUs, global batch 16 x 2 x 4 = 128 (what slurm/distill_maize.sbatch runs)
    torchrun --standalone --nproc_per_node=2 train_maize_4m_distill.py \
        --config configs/config_maize_distill_all.yaml

    # the 'before' number alone (no training; one process, GPU or CPU)
    python train_maize_4m_distill.py --config configs/config_maize_distill_all.yaml \
        --eval_only --world_size 1
"""

import os
import sys
import json
import math
import time
import random
import argparse
import contextlib
from datetime import datetime, timedelta
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

from embodied_mae_4m_maize import (
    embodied_mae_4m_maize_small,
    embodied_mae_4m_maize_base,
    embodied_mae_4m_maize_large,
    MAX_LEAVES,
    N_PARAMS,
)
from embodied_mae import chamfer_distance
from maize_dataset_4m import MaizeDataset4M
from train_maize_4m import (
    cleanup_distributed, unpatchify, _unnorm_pix, _parse_modalities,
)


MODALITIES = ['rgb', 'depth', 'pc', 'text']
ALL_VISIBLE = set(MODALITIES)
SPECIES = 'maize'


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_namespace(c):
    g = lambda sec, k, d: (c.get(sec) or {}).get(k, d)
    ns = argparse.Namespace()
    # data / model / training (mirrors train_maize_4m for the maize-width keys)
    ns.data_root          = c['data']['data_root']
    ns.img_size           = g('data', 'img_size', 224)
    ns.num_points         = g('data', 'num_points', 8192)
    # Smoke/debug only: cap the TRAIN split at N plants (all ten views each).
    # null = every plant, which is what the reference run used.
    _mp                   = g('data', 'max_plants', None)
    ns.max_plants         = None if _mp in (None, 0, 'null') else int(_mp)
    ns.plant_subset_seed  = int(g('data', 'plant_subset_seed', 42))
    ns.model_size         = g('model', 'model_size', 'base')
    ns.active_modalities  = _parse_modalities(g('model', 'active_modalities', None))
    ns.mask_ratio         = g('model', 'mask_ratio', 0.80)
    ns.pc_loss_weight     = g('model', 'pc_loss_weight', 1.0)
    ns.depth_norm_type    = g('model', 'depth_norm_type', 'minmax')
    ns.spline_loss_weight = g('model', 'spline_loss_weight', 5.0)
    ns.max_leaves         = g('model', 'max_leaves', MAX_LEAVES)
    # The trainer reads YAML `loss_name` into the constructor's `pc_loss_name`
    # (CLAUDE.md: the two names differ and guessing silently does the wrong thing).
    ns.pc_loss_name       = g('model', 'loss_name', 'chamfer')
    ns.qal_threshold      = g('model', 'qal_threshold', 0.01)
    ns.qal_alpha          = g('model', 'qal_alpha', 100.0)
    ns.qal_use_squared    = g('model', 'qal_use_squared', False)
    ns.batch_size         = g('training', 'batch_size', 16)
    ns.epochs             = g('training', 'epochs', 100)
    ns.lr                 = g('training', 'lr', 1.0e-4)
    ns.weight_decay       = g('training', 'weight_decay', 0.05)
    ns.warmup_epochs      = g('training', 'warmup_epochs', 5)
    ns.val_freq           = g('training', 'val_freq', 5)
    # cap the cross-modal val sweep at N batches per source (0 = full split).
    ns.val_max_batches    = g('training', 'val_max_batches', 0)
    ns.max_steps          = g('training', 'max_steps', 0)
    # Gradient accumulation. Global batch = batch_size x world_size x accum_steps.
    ns.accum_steps        = g('training', 'accum_steps', 1)
    # Optional guard: refuse to start unless batch_size x world_size x accum_steps
    # equals this. Catches a 4-GPU launch that keeps accum 4 (= 256) or a 1-GPU
    # debug launch that forgets to raise it. null disables the check.
    ns.global_batch_expected = g('training', 'global_batch', None)
    # Score the warm start (epoch 0) before the first distillation step.
    ns.val_at_start       = bool(g('training', 'val_at_start', True))
    # Seed for validation / epoch-0 eval (null = unseeded, as sorghum ran).
    ns.eval_seed          = g('training', 'eval_seed', 0)
    # distillation
    ns.teacher_checkpoint     = g('distill', 'teacher_checkpoint', None)
    ns.student_init           = g('distill', 'student_init', None)
    ns.expect_init_epoch      = g('distill', 'expect_init_epoch', None)
    ns.crossmodal_prob        = g('distill', 'crossmodal_prob', 0.7)
    ns.source_mask_ratio      = g('distill', 'source_mask_ratio', 0.0)
    ns.sources                = g('distill', 'sources', MODALITIES)
    ns.feat_distill_weight    = g('distill', 'feat_distill_weight', 1.0)
    ns.cls_distill_weight     = g('distill', 'cls_distill_weight', 0.5)
    ns.output_distill_weight  = g('distill', 'output_distill_weight', 0.0)
    ns.distill_masked_only    = g('distill', 'distill_masked_only', True)
    ns.viz_source             = g('distill', 'viz_source', 'pc')
    # What the frozen teacher sees. The code default (all four, 0.0) is exactly
    # the sorghum trainer's ALL_VISIBLE -- and train_worker REFUSES it unless
    # allow_all_visible_teacher is set, because the maize epoch-600 checkpoint
    # COLLAPSES its decoded point cloud when every PC token is visible. Mean PC
    # chamfer of the teacher on val plants (CPU, 4 independent probes, 19 plants):
    #                               maize e600          sorghum teacher_final
    #   all four, every token       0.053-0.089         0.00048 (0.00068)
    #   all four, PC 90 % visible   0.0157              0.00045
    #   all four, PC 75 % visible   0.0022              0.00039
    #   all four, 50 % of each      0.0009-0.0014       0.00035
    #   [rgb, depth, text]          0.0012-0.0021       0.00062
    #   own Dirichlet 0.80 masking  0.0008-0.0013       0.00101
    # Maize degrades monotonically once PC visibility passes the 75 % its
    # pretraining ever showed it (min_mask_ratio 0.25); sorghum never does. With
    # distill_masked_only, an all-visible maize teacher would hand the student
    # collapsed PC-token targets on every rgb/depth/text step while qal pulls the
    # same outputs toward ground truth. 50 % of each is the regime where BOTH
    # teachers are at their best, and unlike [rgb, depth, text] it keeps PC among
    # the teacher's privileged inputs -- so it preserves the teacher's role.
    _tvis = g('distill', 'teacher_visible', MODALITIES)
    _tvis = [_tvis] if isinstance(_tvis, str) else list(_tvis)
    assert _tvis and set(_tvis) <= set(MODALITIES), \
        f"distill.teacher_visible must be a non-empty subset of {MODALITIES}, got {_tvis}"
    ns.teacher_visible        = [m for m in MODALITIES if m in set(_tvis)]
    ns.teacher_source_mask_ratio = float(g('distill', 'teacher_source_mask_ratio', 0.0))
    assert 0.0 <= ns.teacher_source_mask_ratio < 1.0, \
        f"distill.teacher_source_mask_ratio must be in [0, 1), got {ns.teacher_source_mask_ratio}"
    ns.allow_all_visible_teacher = bool(g('distill', 'allow_all_visible_teacher', False))
    ns.target                 = g('distill', 'target', None)
    _t = ns.target
    ns.targets = [] if _t is None else ([_t] if isinstance(_t, str) else list(_t))
    # checkpointing / viz / dist / system / wandb
    ns.output_dir         = g('checkpointing', 'output_dir', './outputs/maize_distill')
    ns.save_freq          = g('checkpointing', 'save_freq', 2)
    # Periodic checkpoints kept in checkpoints/ (best_model.pth is never pruned).
    # 2 rather than sorghum's 1 so the launcher's walk-down always has a fallback.
    ns.keep_last          = g('checkpointing', 'keep_last', 2)
    ns.resume             = g('checkpointing', 'resume', None)
    ns.viz_freq           = g('visualization', 'viz_freq', 10)
    ns.num_viz_samples    = g('visualization', 'num_viz_samples', 6)
    ns.world_size         = g('distributed', 'world_size', 1)
    ns.dist_backend       = g('distributed', 'dist_backend', 'nccl')
    ns.dist_url           = g('distributed', 'dist_url', 'env://')
    # Rank 0 runs validation / visualisation alone while the other ranks block in
    # their next collective. The default 10-minute watchdog is too tight for a
    # cold-cache val pass, so give the process group an hour.
    ns.dist_timeout_min   = g('distributed', 'timeout_min', 60)
    ns.num_workers        = g('system', 'num_workers', 8)
    ns.device             = g('system', 'device', 'cuda')
    ns.use_wandb          = g('wandb', 'use_wandb', True)
    ns.wandb_project      = g('wandb', 'wandb_project', 'embodied-mae-maize')
    ns.wandb_entity       = g('wandb', 'wandb_entity', None)
    ns.wandb_name         = g('wandb', 'wandb_name', None)

    # ── validation ────────────────────────────────────────────────────────────
    ns.sources = [s for s in ns.sources if s in MODALITIES]
    assert ns.sources, "distill.sources must contain at least one of rgb/depth/pc/text"
    for _m in ns.targets:
        assert _m in MODALITIES, \
            f"distill.target entries must be in {MODALITIES}, got {_m}"
    assert len(set(ns.targets)) == len(ns.targets), \
        f"distill.target has duplicates: {ns.targets}"
    assert ns.accum_steps >= 1, f'training.accum_steps must be >= 1, got {ns.accum_steps}'
    if ns.active_modalities is not None:
        # Feature distillation lines teacher and student decoder sequences up
        # token-for-token over the rgb|depth|pc|text layout, and every source
        # must be a modality the model has. A reduced arm has neither.
        raise SystemExit(
            f"model.active_modalities={ns.active_modalities}: cross-modal "
            "distillation needs all four streams (pc,rgb,depth,text).")
    if not ns.teacher_checkpoint:
        raise SystemExit("distill.teacher_checkpoint is required")
    return ns


# ── Model build / checkpoint loading ──────────────────────────────────────────

def build_model(args, device):
    """Exactly the constructor call train_maize_4m.py:708-724 makes."""
    build_fn = {'small': embodied_mae_4m_maize_small,
                'base':  embodied_mae_4m_maize_base,
                'large': embodied_mae_4m_maize_large}[args.model_size]
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
    ).to(device)


def load_weights_into(model, ckpt_path, device, tag, verbose=True,
                      expect_epoch=None):
    """Strict load. Returns the checkpoint's recorded epoch.

    Sorghum loads with strict=False because its older checkpoints predate some
    buffers. Every maize checkpoint comes from this codebase, so a missing or
    unexpected key here means the wrong file (or the wrong width), not history.

    mmap: the file is 1.4 GB, two thirds of it optimiser state this never uses.
    Memory-mapping reads only the pages of the tensors actually copied (~0.46 GB),
    and lets `expect_epoch` refuse a wrong file (e.g. best_model.pth = epoch 300)
    before any weights are read at all. Measured on Nova's Lustre, a cold full
    read of one of these files took over ten minutes.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    ep = ckpt.get('epoch')
    if expect_epoch is not None and ep != int(expect_epoch):
        raise SystemExit(
            f"[{tag}] {ckpt_path} is epoch {ep}, config expects {expect_epoch} "
            "(distill.expect_init_epoch). maize_4m's best_model.pth is epoch 300; "
            "the E2-matched point is checkpoints/checkpoint_epoch_600.pth.")
    sd = ckpt['model_state_dict']
    if list(sd)[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    vl = ckpt.get('val_loss', ckpt.get('best_val_loss'))
    if verbose:
        print(f"  [{tag}] loaded {ckpt_path} (epoch {ep}"
              + (f", val {vl:.4f}" if isinstance(vl, (int, float)) else "") + ")")
    del ckpt, sd
    return ep


# Keys of the source run's config.json that must agree with this config: the
# first five shape the network (a mismatch fails at load anyway, but with a far
# worse message), the rest shape the reconstruction loss the student keeps
# optimising, which a load would NOT catch.
_TEACHER_MATCH_KEYS = ('model_size', 'num_points', 'max_leaves', 'img_size',
                       'active_modalities', 'pc_loss_name', 'qal_threshold',
                       'qal_alpha', 'qal_use_squared', 'pc_loss_weight',
                       'spline_loss_weight', 'depth_norm_type', 'mask_ratio')


def check_against_source_run(args, ckpt_path):
    """Compare this config with the teacher run's config.json, if it has one.

    outputs/<run>/checkpoints/checkpoint_epoch_N.pth -> outputs/<run>/config.json.
    Returns the list of mismatches (empty = consistent or no config.json found).
    """
    p = Path(ckpt_path).resolve()
    cands = [p.parent / 'config.json', p.parent.parent / 'config.json']
    src = next((c for c in cands if c.exists()), None)
    if src is None:
        return None, []
    theirs = json.loads(src.read_text())
    mine = vars(args)
    bad = []
    for k in _TEACHER_MATCH_KEYS:
        if k not in theirs:
            continue
        a, b = mine.get(k), theirs[k]
        if isinstance(a, tuple):
            a = list(a)
        if a != b:
            bad.append(f"{k}: this config {a!r} vs teacher run {b!r}")
    return src, bad


# ── Distillation token weighting (verbatim from the sorghum trainer) ──────────

def token_block_sizes(model):
    """Lengths of the four decoder blocks in concat order."""
    return [model.num_patches, model.num_patches,
            model.num_pc_tokens, model.n_text_tokens]


def masked_token_weight(model, source, device):
    """(L,) weight: 1.0 on tokens of GENERATED (masked) modalities, 0.0 on the
    visible source modality's tokens."""
    sizes = token_block_sizes(model)
    w = []
    for name, n in zip(MODALITIES, sizes):
        w.append(torch.zeros(n) if name == source else torch.ones(n))
    return torch.cat(w).to(device)


def target_token_weight(model, targets, device):
    """(L,) weight: 1.0 on tokens of the TARGET modalities, 0.0 elsewhere."""
    if isinstance(targets, str):
        targets = [targets]
    tset = set(targets)
    sizes = token_block_sizes(model)
    w = []
    for name, n in zip(MODALITIES, sizes):
        w.append(torch.ones(n) if name in tset else torch.zeros(n))
    return torch.cat(w).to(device)


def feature_distill_loss(feats_s, feats_t, weight=None):
    """Token-wise MSE between student and (frozen) teacher decoder features."""
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
        # All four, unmasked, is the sorghum trainer's ALL_VISIBLE; the maize
        # config masks 50 % of each modality instead (see config_to_namespace).
        _, _, preds_t, _, (feats_t, cls_t) = teacher(
            rgb, depth, pc, params, tv,
            visible=set(getattr(args, 'teacher_visible', ALL_VISIBLE)),
            return_features=True,
            source_mask_ratio=getattr(args, 'teacher_source_mask_ratio', 0.0))

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


# ── Accumulation layout ───────────────────────────────────────────────────────

def opt_steps_per_epoch(n_batches, accum):
    """Optimiser steps in an epoch of `n_batches` micro-batches.

    ceil, not floor: a short tail group still steps (see the module docstring).
    At accum=1 this is n_batches, i.e. the sorghum reference run's count.
    """
    return max(1, math.ceil(n_batches / max(1, accum)))


def micro_position(bi, n_batches, accum):
    """(opt_idx, group_size, is_first_micro, is_last_micro) for micro-batch `bi`.

    Groups are [0..accum-1], [accum..2accum-1], ...; only the final group of the
    epoch can be short, and every rank computes the same layout because
    DistributedSampler hands each rank the same number of samples.
    """
    opt_idx = bi // accum
    g0 = opt_idx * accum
    group = min(accum, n_batches - g0)
    return opt_idx, group, bi == g0, (bi - g0) == group - 1


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(student, teacher, loader, optimizer, device, epoch, args,
                    n_batches, is_main=True):
    student.train()
    agg = {k: 0.0 for k in
           ['loss', 'recon', 'rgb', 'depth', 'pc', 'text', 'feat', 'cls', 'out']}
    n_xm = 0
    src_count = {s: 0 for s in args.sources}
    on_cuda = (device.type == 'cuda')

    n_done = 0
    n_opt_done = 0
    # Gradient accumulation: `accum` micro-batches make one optimiser step, so the
    # effective global batch is batch_size x world_size x accum. n_opt keys the
    # cross-modal schedule below; with 2 GPUs x accum 4 it equals the 8-GPU
    # reference run's 821, so (epoch, step) -> (mode, source) is the same schedule.
    accum = max(1, getattr(args, 'accum_steps', 1))
    n_opt = opt_steps_per_epoch(n_batches, accum)
    _warm, _t0 = 10, None      # steady-state timing ignores the first _warm steps
    if args.max_steps and on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    pbar = tqdm(loader, desc=f'Epoch {epoch}', disable=not is_main,
                mininterval=(0.1 if sys.stderr.isatty() else 30.0))
    for bi, (rgb, depth, pc, params, tv, _) in enumerate(pbar):
        rgb = rgb.to(device); depth = depth.to(device); pc = pc.to(device)
        params = params.to(device); tv = tv.to(device)
        batch_t = (rgb, depth, pc, params, tv)

        # Deterministic, rank-identical schedule keyed on the OPTIMISER step, so
        # every micro-batch of a group shares one mode and one source and the
        # effective global batch stays homogeneous (what the 8-GPU run did).
        opt_idx, group, is_first_micro, is_last_micro = micro_position(
            bi, n_batches, accum)
        step_id = (epoch - 1) * n_opt + opt_idx
        is_xm   = random.Random(step_id).random() < args.crossmodal_prob

        if is_first_micro:
            optimizer.zero_grad()

        # DDP all-reduces only on the last micro-batch of the group. The forward
        # sits inside no_sync() too: DDP arms its reducer during forward, so
        # syncing the forward but not the backward silently drops gradients.
        sync_ctx = (student.no_sync()
                    if (not is_last_micro and hasattr(student, 'no_sync'))
                    else contextlib.nullcontext())
        with sync_ctx:
            if is_xm:
                source = args.sources[step_id % len(args.sources)]
                loss, stats = crossmodal_distill_step(
                    student, teacher, batch_t, source, args, device)
                n_xm += 1
                src_count[source] += 1
            else:
                loss, (lr, ld, lp, lt), _, _ = student(rgb, depth, pc, params, tv,
                                                       mask_ratio=args.mask_ratio)
                stats = {'loss': loss.item(), 'recon': loss.item(),
                         'rgb': lr.item(), 'depth': ld.item(),
                         'pc': lp.item(), 'text': lt.item(),
                         'feat': 0.0, 'cls': 0.0, 'out': 0.0}

            # Scale so the accumulated gradient is the MEAN of the group's
            # micro-batch losses -- DDP then averages over ranks, matching the
            # reference run's mean over 8 ranks at accum 1. The tail group
            # divides by its own length, not by accum, so its step is not
            # silently shrunk. `stats` keeps the unscaled loss for logging.
            (loss / group).backward()

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            n_opt_done += 1

        for k in agg:
            agg[k] += stats[k]
        pbar.set_postfix({
            'loss': f"{stats['loss']:.3f}",
            'mode': ('XM:' + source) if is_xm else 'recon',
            'feat': f"{stats['feat']:.3f}",
        })

        n_done = bi + 1
        if args.max_steps and n_done == _warm and on_cuda:
            torch.cuda.synchronize(device); _t0 = time.time()
        # Debug/smoke only: cut the epoch short. All ranks break at the same
        # micro-batch, and only on a group boundary, so DDP stays in sync.
        if args.max_steps and n_done >= args.max_steps and is_last_micro:
            break

    n = max(n_done, 1)
    if args.max_steps and on_cuda:
        torch.cuda.synchronize(device)
        _rate = ((n_done - _warm) / (time.time() - _t0)) if _t0 and n_done > _warm else 0.0
        print(f"PEAKMEM alloc={torch.cuda.max_memory_allocated(device)/2**20:.0f}MiB "
              f"reserved={torch.cuda.max_memory_reserved(device)/2**20:.0f}MiB "
              f"steps={n} steady_it_s={_rate:.3f}", flush=True)
    out = {k: v / n for k, v in agg.items()}
    out['opt_steps'] = n_opt_done
    out['micro_batches'] = n_done
    return out, n_xm, src_count


# ── Cross-modal validation (per source) ───────────────────────────────────────

@contextlib.contextmanager
def fixed_rng(seed, device):
    """Run a block under a fixed seed without perturbing the training RNG stream.

    Validation draws randomness in two places even under eval()/no_grad: FPS
    picks its first centroid with torch.randint, and each DataLoader worker's
    numpy RNG (the loader's point permutation) is seeded from the main process's
    torch generator when the iterator is created. Seeding the main generator
    fixes both. seed=None runs unseeded, as the sorghum trainer does.
    """
    if seed is None:
        yield
        return
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = [device] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed)); random.seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(np_state); random.setstate(py_state)


@torch.no_grad()
def evaluate_crossmodal(student, loader, device, sources, target=None, max_batches=0):
    """Verbatim from the sorghum trainer. For each source modality, run
    single-modal -> all and measure how well the *generated* (masked) modalities
    match ground truth. Returns per-source dict and the scalar `mean_gen` used
    for best-model selection (mean over sources of the recon total when no
    target is set -- the number sorghum reports as 0.3439 -> 0.2781)."""
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
        per_source[src]['n_batches'] = nb

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


def run_val(student, loader, device, args):
    with fixed_rng(args.eval_seed, device):
        return evaluate_crossmodal(student, loader, device, args.sources,
                                   args.targets, max_batches=args.val_max_batches)


def print_val(per_source, mean_gen, args, header=None):
    if header:
        print(header)
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


# ── Visualisation (fixed source → all modalities; verbatim from sorghum) ──────

def _sub(pc_np, n=1500):
    if pc_np.shape[0] > n:
        return pc_np[np.random.choice(pc_np.shape[0], n, replace=False)]
    return pc_np


@torch.no_grad()
def visualize_crossmodal(model, loader, device, epoch, save_dir, source,
                         num_samples=6):
    """Unlike train_maize_4m.visualize_reconstruction_4m, the depth panels here
    render the prediction raw -- no ground-truth silhouette is borrowed."""
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
    # The WHOLE batch, (B, n_text_tokens, N_PARAMS): the maize override raises on
    # anything else (the epoch-1 viz crash of job 16447748).
    gt_txt  = model_m.decode_params_to_text(params)
    gen_txt = model_m.decode_params_to_text(pparam)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(rgb.shape[0]):
        fig = plt.figure(figsize=(16, 7))
        cd = chamfer_distance(ppc[i:i+1], pc[i:i+1]).item()

        def panel(pos, img, title, cmap=None):
            ax = plt.subplot(2, 4, pos)
            ax.imshow(img, cmap=cmap); ax.axis('off')
            ax.set_title(title, fontsize=8, fontweight='bold')

        is_in = lambda mod: '  ← INPUT' if mod == source else '  (generated)'
        panel(1, rgb_dn[i].transpose(1, 2, 0).clip(0, 1), 'GT RGB')
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

        ax = plt.subplot2grid((2, 4), (1, 2), colspan=2); ax.axis('off')
        n_real = int(tv[i].sum().item())
        lines = [f"SPLINE PARAMS  (params {is_in('text').strip()})", "=" * 30, "",
                 "PLANT:", f" GT : {gt_txt[i][0]}", f" GEN: {gen_txt[i][0]}", ""]
        for ti in range(1, min(n_real, 4)):
            lines += [f"leaf{ti} GT : {gt_txt[i][ti]}",
                      f"leaf{ti} GEN: {gen_txt[i][ti]}", ""]
        ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left',
                family='monospace', fontsize=6, transform=ax.transAxes)

        plt.suptitle(f"Epoch {epoch} | {names[i]} | source={source.upper()} → all",
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        p = save_dir / f'epoch_{epoch:03d}_src-{source}_sample_{i+1}_{names[i]}.png'
        plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig)
        saved.append(str(p))

    model_m.train()
    return saved


# ── Small I/O helpers ─────────────────────────────────────────────────────────

def _atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _setup_distributed(rank, local_rank, world_size, args, device_type):
    """train_maize_4m.setup_distributed plus a longer timeout, and gloo on CPU
    (used only by the CPU smoke test; a real run is always nccl)."""
    backend = args.dist_backend if device_type == 'cuda' else 'gloo'
    kw = dict(backend=backend, world_size=world_size, rank=rank,
              timeout=timedelta(minutes=float(args.dist_timeout_min)))
    if int(os.environ.get('LOCAL_RANK', -1)) >= 0:
        dist.init_process_group(init_method='env://', **kw)
    else:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12357')
        dist.init_process_group(
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            **kw)
    if device_type == 'cuda':
        torch.cuda.set_device(local_rank)


def _pick_device(args, local_rank):
    if args.device == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(f'cuda:{local_rank}')


def _make_val_loader(args):
    val_ds = MaizeDataset4M(args.data_root, img_size=args.img_size,
                            num_points=args.num_points, split='val',
                            max_leaves=args.max_leaves)
    val_workers = min(args.num_workers, 8)
    return val_ds, DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=val_workers, pin_memory=True)


# ── Worker ────────────────────────────────────────────────────────────────────

# Values that define the experiment. A resume may not change any of them:
# lr_lambda is rebuilt from `epochs` / `warmup_epochs` (only last_epoch is
# restored), the (epoch, step) -> (mode, source) schedule from the batch layout,
# and the before -> after pairing from the val subset and seed. A key missing from
# an older checkpoint's run_meta is not checked.
_RESUME_PINNED = (
    'species', 'global_batch', 'batch_size', 'opt_steps_per_epoch',
    'num_points', 'max_leaves',
    'epochs', 'lr', 'weight_decay', 'warmup_epochs', 'mask_ratio',
    'crossmodal_prob', 'sources', 'targets', 'source_mask_ratio',
    'feat_distill_weight', 'cls_distill_weight', 'output_distill_weight',
    'distill_masked_only',
    'teacher_visible', 'teacher_source_mask_ratio', 'teacher_epoch',
    'val_max_batches', 'eval_seed',
)


def check_teacher_regime(args):
    """Refuse the sorghum all-visible teacher unless it was asked for by name."""
    if (set(args.teacher_visible) == ALL_VISIBLE
            and args.teacher_source_mask_ratio == 0.0
            and not args.allow_all_visible_teacher):
        raise SystemExit(
            "distill.teacher_visible is all four with teacher_source_mask_ratio 0.0 "
            "-- sorghum's ALL_VISIBLE teacher. The maize epoch-600 checkpoint "
            "collapses its point cloud in that regime (PC chamfer ~0.07 against "
            "~0.001 under its own masking), so every PC-token feature target would "
            "come from a degenerate teacher. configs/config_maize_distill_all.yaml "
            "uses teacher_source_mask_ratio 0.5. To run the literal sorghum recipe "
            "anyway, set distill.allow_all_visible_teacher: true.")


def train_worker(rank, world_size, args, local_rank=None):
    local_rank = rank if local_rank is None else local_rank
    check_teacher_regime(args)
    device = _pick_device(args, local_rank)
    if world_size > 1:
        _setup_distributed(rank, local_rank, world_size, args, device.type)
    is_main = (rank == 0)

    output_dir     = Path(args.output_dir)
    viz_dir        = output_dir / 'visualizations'
    checkpoint_dir = output_dir / 'checkpoints'
    # Resume sanity, before anything expensive. Sorghum (and train_maize_4m)
    # quietly start from scratch on a missing --resume; here that would write a
    # fresh epoch-1 best_model.pth over the previous run's (best_val starts at inf).
    resuming = bool(args.resume)
    if resuming and not os.path.exists(args.resume):
        raise SystemExit(f"--resume {args.resume} does not exist; refusing to "
                         "start from scratch in its place.")
    if not resuming and any(checkpoint_dir.glob('checkpoint_epoch_*.pth')):
        raise SystemExit(
            f"{checkpoint_dir} already holds checkpoints but no --resume was given. "
            "Pass --resume <ckpt> (slurm/distill_maize.sbatch does this itself) or "
            "use a new output_dir; a fresh start here would overwrite best_model.pth.")
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(exist_ok=True); checkpoint_dir.mkdir(exist_ok=True)

    accum = max(1, args.accum_steps)
    global_batch = args.batch_size * world_size * accum
    if args.global_batch_expected and global_batch != int(args.global_batch_expected):
        raise SystemExit(
            f"global batch {args.batch_size} x {world_size} GPUs x accum {accum} = "
            f"{global_batch}, but training.global_batch is "
            f"{args.global_batch_expected}. Keep batch_size at 16 (PointCloudEmbed's "
            "BatchNorm is per rank) and set accum_steps = global_batch / (16 x GPUs).")

    # Data: all ten views per epoch, as the sorghum distill run.
    train_ds = MaizeDataset4M(args.data_root, img_size=args.img_size,
                              num_points=args.num_points, split='train',
                              max_leaves=args.max_leaves,
                              max_plants=args.max_plants,
                              plant_subset_seed=args.plant_subset_seed)
    val_ds, val_loader = _make_val_loader(args)
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        shuffle_train = False
    else:
        train_sampler = None; shuffle_train = True
    # persistent_workers is safe HERE and only here: CLAUDE.md forbids it because a
    # persistent worker never sees set_epoch(), which freezes view_sampling. This
    # trainer does not view-sample (every view, every epoch), so there is nothing
    # for set_epoch to reach, and respawning the workers every epoch is pure cost
    # (the sorghum distill trainer measured 5 -> 63 min/epoch cold without it).
    assert not train_ds.view_sampling
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=shuffle_train, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    n_batches = len(train_loader)
    n_opt = opt_steps_per_epoch(n_batches, accum)

    # ── Teacher (frozen, full-modal) ──────────────────────────────────────────
    if is_main: print("\n🧊 Building frozen TEACHER…")
    teacher = build_model(args, device)
    teacher_epoch = load_weights_into(teacher, args.teacher_checkpoint, device,
                                      'teacher', verbose=is_main,
                                      expect_epoch=args.expect_init_epoch)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    src_cfg, mism = check_against_source_run(args, args.teacher_checkpoint)
    if mism:
        raise SystemExit(f"config disagrees with the teacher run ({src_cfg}):\n  "
                         + "\n  ".join(mism))

    # ── Student (warm-started, trainable) ─────────────────────────────────────
    if is_main: print("\n🎓 Building STUDENT…")
    student = build_model(args, device)
    student_epoch = None
    if resuming:
        if is_main: print("  [student-init] skipped: resuming overwrites it")
    elif args.student_init:
        if not os.path.exists(args.student_init):
            # Sorghum prints "training from scratch" here and carries on. A
            # from-scratch student is a different experiment; refuse instead.
            raise SystemExit(f"distill.student_init not found: {args.student_init}")
        student_epoch = load_weights_into(student, args.student_init, device,
                                          'student-init', verbose=is_main,
                                          expect_epoch=args.expect_init_epoch)
    else:
        raise SystemExit("distill.student_init is not set; the distillation "
                         "student is always warm-started.")

    if world_size > 1:
        # cross-modal steps change which inputs feed the loss step-to-step.
        ddp_kw = dict(device_ids=[local_rank], output_device=local_rank) \
            if device.type == 'cuda' else {}
        student = DDP(student, find_unused_parameters=True, **ddp_kw)

    total_params = sum(p.numel() for p in student.parameters())
    if is_main: print(f"\nStudent parameters: {total_params:,}")

    run_meta = {
        'species': SPECIES, 'n_params': N_PARAMS, 'max_leaves': args.max_leaves,
        'n_text_tokens': 1 + args.max_leaves, 'num_points': args.num_points,
        'batch_size': args.batch_size, 'world_size': world_size,
        'accum_steps': accum, 'global_batch': global_batch,
        'micro_batches_per_epoch': n_batches, 'opt_steps_per_epoch': n_opt,
        'teacher_visible': list(args.teacher_visible),
        'teacher_source_mask_ratio': args.teacher_source_mask_ratio,
        'teacher_epoch': teacher_epoch,
        'epochs': args.epochs, 'lr': args.lr, 'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs, 'mask_ratio': args.mask_ratio,
        'crossmodal_prob': args.crossmodal_prob, 'sources': list(args.sources),
        'targets': list(args.targets), 'source_mask_ratio': args.source_mask_ratio,
        'feat_distill_weight': args.feat_distill_weight,
        'cls_distill_weight': args.cls_distill_weight,
        'output_distill_weight': args.output_distill_weight,
        'distill_masked_only': args.distill_masked_only,
        'val_max_batches': args.val_max_batches, 'eval_seed': args.eval_seed,
        # Provenance: set on the fresh start, carried forward from the checkpoint
        # on every resume (a resume skips the student init, so it cannot re-derive
        # them, and config.json is rewritten at every launch).
        'student_init': args.student_init, 'student_init_epoch': student_epoch,
    }

    # Resume compatibility, before W&B or anything else is touched. mmap: only
    # the small pickled objects are read here; the weights come later.
    prev_wandb_id = None
    if resuming:
        _head = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        prev = _head.get('run_meta') or {}
        prev_wandb_id = _head.get('wandb_run_id')
        del _head
        for k in _RESUME_PINNED:
            if k in prev and prev[k] != run_meta[k]:
                raise SystemExit(
                    f"refusing to resume {args.resume}: it was trained with "
                    f"{k}={prev[k]!r}, this launch has {k}={run_meta[k]!r}. A resume "
                    "must continue the same experiment; start a new output_dir to "
                    "change it.")
        for k in ('student_init', 'student_init_epoch'):
            if k in prev:
                run_meta[k] = prev[k]

    # wandb. The run id is also kept in <output_dir>/wandb_run_id.txt, so a job
    # preempted before its first checkpoint rejoins the same W&B run on requeue
    # instead of opening a second one under the same name.
    wb_id_file = output_dir / 'wandb_run_id.txt'
    if is_main and args.use_wandb and WANDB_AVAILABLE:
        if args.wandb_name is None:
            args.wandb_name = f"maize_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run_id = prev_wandb_id
        if not wandb_run_id and wb_id_file.exists():
            wandb_run_id = wb_id_file.read_text().strip() or None
            if wandb_run_id:
                print(f"  W&B: rejoining run {wandb_run_id} from {wb_id_file.name} "
                      "(restart before the first checkpoint)")
        if wandb_run_id:
            # 'allow', not 'must': a logging hiccup must never kill a requeued run.
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       id=wandb_run_id, resume='allow', name=args.wandb_name)
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_name, config={**vars(args), **run_meta})
        _tmp = wb_id_file.with_name(wb_id_file.name + f'.tmp{os.getpid()}')
        _tmp.write_text(str(wandb.run.id) + '\n')
        os.replace(_tmp, wb_id_file)
        print(f"✅ W&B: {wandb.run.url}")

    def _wb_id():
        return (wandb.run.id if args.use_wandb and WANDB_AVAILABLE and wandb.run
                else None)

    optimizer = optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Per-EPOCH schedule, exactly the sorghum trainer's: LambdaLR stepped once
    # per epoch, so accumulation leaves the LR every epoch sees unchanged.
    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (ep - args.warmup_epochs)
                                 / max(1, args.epochs - args.warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch   = 1
    best_val      = float('inf')
    history       = {'train': [], 'val': []}

    if resuming:
        if is_main: print(f"\n📂 Resuming from {args.resume}")
        # (compatibility with this launch was checked above, before W&B)
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
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
        del ckpt, sd
        if is_main:
            print(f"✅ Resumed at epoch {start_epoch - 1} | best mean_gen {best_val:.4f}")

    if is_main:
        # run_meta last: on a resume its student_init / student_init_epoch are
        # the ones carried forward from the checkpoint, not this launch's None.
        cfg_out = {**vars(args), **run_meta,
                   'total_opt_steps': n_opt * args.epochs,
                   'train_samples': len(train_ds), 'val_samples': len(val_ds),
                   'view_sampling': False, 'text_mask_ratio': None,
                   'resumed_from': args.resume if resuming else None}
        _atomic_json(output_dir / 'config.json', cfg_out)
        print(f"\n🚀 Maize distillation for {args.epochs} epochs")
        print(f"   crossmodal_prob={args.crossmodal_prob}  sources={args.sources}"
              f"  target={'+'.join(args.targets) if args.targets else 'all'}")
        print(f"   feat_w={args.feat_distill_weight} cls_w={args.cls_distill_weight} "
              f"out_w={args.output_distill_weight} masked_only={args.distill_masked_only}")
        print(f"   teacher sees {'+'.join(args.teacher_visible)}"
              f" (source_mask_ratio {args.teacher_source_mask_ratio})")
        print(f"   batch/GPU={args.batch_size} x world_size={world_size} x accum={accum}"
              f"  -> GLOBAL BATCH {global_batch}"
              f"  ({n_batches} micro-batches/rank -> {n_opt} optimiser steps/epoch,"
              f" {n_opt * args.epochs:,} total)")
        print("=" * 80)

    # ── Epoch 0: the warm start, before any distillation step ─────────────────
    # Same function, same val loader (first val_max_batches x batch_size samples
    # in index order), same seed as every later val point, so before -> after is
    # one code path. Recorded, never used for best-model selection.
    have_e0 = any(v.get('epoch') == 0 for v in history['val'])
    if is_main and args.val_at_start and start_epoch == 1 and not have_e0:
        per0, mg0 = run_val(student, val_loader, device, args)
        print_val(per0, mg0, args,
                  header="\n🔍 EPOCH 0 — warm start, no distillation (per source)…")
        history['val'].append({'epoch': 0, 'mean_gen': mg0, 'per_source': per0,
                               'tag': 'warm_start'})
        _atomic_json(output_dir / 'warmstart_eval.json', {
            'epoch': 0, 'checkpoint': args.student_init,
            'checkpoint_epoch': student_epoch, 'mean_gen': mg0,
            'per_source': per0, 'sources': args.sources, 'targets': args.targets,
            'val_max_batches': args.val_max_batches, 'batch_size': args.batch_size,
            'eval_seed': args.eval_seed, 'device': str(device), 'source': 'in-run'})
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({'epoch': 0, 'val/mean_gen': mg0,
                       **{f'val/{s}/{k}': v for s in args.sources
                          for k, v in per0[s].items()}})

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n{'='*80}\nEpoch {epoch}/{args.epochs}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}\n{'='*80}")

        tr, n_xm, src_count = train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch, args,
            n_batches, is_main=is_main)
        history['train'].append({'epoch': epoch, **tr})

        if is_main:
            print(f"\nTrain — loss {tr['loss']:.4f}  recon {tr['recon']:.4f}  "
                  f"feat {tr['feat']:.4f}  cls {tr['cls']:.4f}  "
                  f"| {n_xm}/{tr['micro_batches']} micro-batches crossmodal  "
                  f"sources={src_count}  opt_steps={tr['opt_steps']}")

        do_val = (epoch % args.val_freq == 0 or epoch == args.epochs or epoch == 1)
        mean_gen = None
        if do_val and is_main:
            per_source, mean_gen = run_val(student, val_loader, device, args)
            print_val(per_source, mean_gen, args,
                      header="\n🔍 Cross-modal validation (per source)…")
            history['val'].append({'epoch': epoch, 'mean_gen': mean_gen,
                                   'per_source': per_source})

        if is_main and args.use_wandb and WANDB_AVAILABLE:
            logd = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in tr.items()},
                    'train/crossmodal_frac': n_xm / max(tr['micro_batches'], 1)}
            if mean_gen is not None:
                logd['val/mean_gen'] = mean_gen
                for s in args.sources:
                    for k, v in per_source[s].items():
                        logd[f'val/{s}/{k}'] = v
            wandb.log(logd)

        if is_main and args.viz_freq > 0 and (epoch % args.viz_freq == 0 or epoch == 1):
            print(f"\n📊 Visualising source={args.viz_source} → all…")
            with fixed_rng(args.eval_seed, device):
                paths = visualize_crossmodal(student, val_loader, device, epoch,
                                             viz_dir, args.viz_source,
                                             args.num_viz_samples)
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({'visualizations': [wandb.Image(p, caption=Path(p).name)
                                              for p in paths], 'epoch': epoch})

        scheduler.step()

        def ckpt_obj(**extra):
            return {'epoch': epoch,
                    'model_state_dict': (student.module if world_size > 1
                                         else student).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val, 'history': history,
                    'run_meta': run_meta, 'wandb_run_id': _wb_id(), **extra}

        # Best BEFORE periodic (sorghum does the reverse): a periodic checkpoint
        # written first carries the previous best_val, and a resume from it can
        # then overwrite best_model.pth with a worse model.
        if is_main and mean_gen is not None and mean_gen < best_val:
            best_val = mean_gen
            _atomic_save(ckpt_obj(val_loss=mean_gen), output_dir / 'best_model.pth')
            print(f"⭐ New best cross-modal model! mean_gen {mean_gen:.4f}")

        # Epoch 1 as well: on preemptible scavenger, the epoch-0 eval plus the
        # first two epochs are otherwise ~1.5 h with no resume point.
        if is_main and (epoch % args.save_freq == 0 or epoch == 1):
            _atomic_save(ckpt_obj(), checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"💾 checkpoint_epoch_{epoch}.pth")
            if args.keep_last and args.keep_last > 0:
                cks = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'),
                             key=lambda p: int(p.stem.rsplit('_', 1)[1]))
                for old_ck in cks[:-args.keep_last]:
                    try:
                        old_ck.unlink()
                        print(f"🗑  pruned {old_ck.name}")
                    except OSError as exc:
                        print(f"⚠️  could not prune {old_ck.name}: {exc}")

        if is_main:
            _atomic_json(output_dir / 'training_history.json', history)

    if world_size > 1:
        cleanup_distributed()
    if is_main and args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    if is_main:
        e0 = next((v for v in history['val'] if v.get('epoch') == 0), None)
        last = history['val'][-1] if history['val'] else None
        print(f"\n{'='*80}\nDistillation complete! best mean_gen {best_val:.4f}")
        if e0 and last and last is not e0:
            print(f"mean_gen epoch 0 -> {last['epoch']}: {e0['mean_gen']:.4f} -> "
                  f"{last['mean_gen']:.4f} "
                  f"({100 * (last['mean_gen'] / e0['mean_gen'] - 1):+.1f}%)")
        print(f"Outputs: {output_dir}\n{'='*80}")


# ── Eval-only: the 'before' number without a training run ─────────────────────

def eval_only(args, ckpt_path, out_path, expect_epoch=None):
    """Score one checkpoint with the trainer's own validation and nothing else.

    Default checkpoint is student_init, i.e. the warm start -- the maize
    equivalent of eval/eval_warmstart.py's epoch-0 number, but through this
    file's loader, subset and seed so it is the same number the training run
    records as epoch 0. Needs no teacher and no second GPU. `expect_epoch` is
    the training path's expect_init_epoch pin; main() passes it whenever the
    result is the warm-start number, so this path cannot score (say) maize_4m's
    epoch-300 best_model.pth into warmstart_eval.json either.
    """
    device = _pick_device(args, 0)
    model = build_model(args, device)
    # Weights first: a wrong-epoch file is refused before the val index is built.
    ep = load_weights_into(model, ckpt_path, device, 'eval', expect_epoch=expect_epoch)
    _, val_loader = _make_val_loader(args)
    per, mg = run_val(model, val_loader, device, args)
    print_val(per, mg, args, header=f"\n=== {ckpt_path} (epoch {ep}) — per source ===")
    out = {'checkpoint': str(ckpt_path), 'checkpoint_epoch': ep, 'mean_gen': mg,
           'per_source': per, 'sources': args.sources, 'targets': args.targets,
           'val_max_batches': args.val_max_batches, 'batch_size': args.batch_size,
           'eval_seed': args.eval_seed, 'device': str(device),
           'expect_epoch': expect_epoch, 'source': 'eval_only'}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_path, out)
    print(f"wrote {out_path}\nEVAL_DONE")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Cross-modal distillation for EmbodiedMAE-4M (maize)')
    # No default on purpose (see train_maize_4m.py): a typo'd path must not
    # quietly build some other model.
    ap.add_argument('--config', type=str, required=True)
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
                    help='debug/smoke: stop each epoch after N micro-batches (0 = full epoch)')
    ap.add_argument('--source_mask_ratio', type=float, default=None)
    ap.add_argument('--accum_steps', type=int, default=None,
                    help='micro-batches per optimiser step. Global batch = '
                         'batch_size x world_size x accum_steps.')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--no_wandb', action='store_true')
    ap.add_argument('--eval_only', action='store_true',
                    help="score --eval_ckpt (default: student_init, the warm start) "
                         "with the trainer's validation and exit")
    ap.add_argument('--eval_ckpt', type=str, default=None)
    ap.add_argument('--eval_out', type=str, default=None,
                    help='JSON path (default <output_dir>/warmstart_eval.json for '
                         'the warm start, else <output_dir>/eval_<ckpt-stem>.json)')
    args_cli = ap.parse_args()

    if not os.path.exists(args_cli.config):
        raise SystemExit(
            f"Config not found: {args_cli.config}\n"
            "train_maize_4m_distill.py has no default config on purpose -- maize "
            "width (14 params / 28 leaves / 8192 points) and the teacher live in "
            "the YAML. Pass --config configs/config_maize_distill_all.yaml.")
    print(f"📋 Loading config: {args_cli.config}")
    cfg = config_to_namespace(load_config(args_cli.config))

    for k in ['world_size', 'output_dir', 'resume', 'teacher_checkpoint',
              'student_init', 'crossmodal_prob', 'batch_size', 'epochs',
              'num_workers', 'val_max_batches', 'max_steps', 'source_mask_ratio',
              'accum_steps', 'device']:
        v = getattr(args_cli, k)
        if v is not None:
            setattr(cfg, k, v)
    if args_cli.no_wandb:
        cfg.use_wandb = False
    assert cfg.accum_steps >= 1, f'accum_steps must be >= 1, got {cfg.accum_steps}'

    if args_cli.eval_only:
        if int(os.environ.get('RANK', 0)) != 0:
            return      # launched under torchrun: one process scores, the rest exit
        ck = args_cli.eval_ckpt or cfg.student_init
        if not ck or not os.path.exists(ck):
            raise SystemExit(f"--eval_only: checkpoint not found: {ck}")
        # The warm start = student_init (by file identity, not spelling). Its
        # score is the 'before' number, so it gets the same epoch pin as training.
        is_warm = bool(cfg.student_init) and os.path.exists(cfg.student_init) \
            and os.path.samefile(ck, cfg.student_init)
        out = args_cli.eval_out or (
            Path(cfg.output_dir) / ('warmstart_eval.json' if is_warm
                                    else f'eval_{Path(ck).stem}.json'))
        eval_only(cfg, ck, out,
                  expect_epoch=cfg.expect_init_epoch if is_warm else None)
        return

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        rank = int(os.environ['RANK']); world_size = int(os.environ['WORLD_SIZE'])
        cfg.world_size = world_size
        if rank == 0:
            print(f"\n🚀 Multi-GPU (torchrun) ranks={world_size} batch/GPU={cfg.batch_size}"
                  f" accum={cfg.accum_steps}")
        train_worker(rank, world_size, cfg, local_rank=local_rank)
    elif cfg.world_size > 1:
        print(f"\n🚀 Multi-GPU (mp.spawn) GPUs={cfg.world_size}")
        mp.spawn(train_worker, args=(cfg.world_size, cfg),
                 nprocs=cfg.world_size, join=True)
    else:
        train_worker(0, 1, cfg)


if __name__ == '__main__':
    main()
