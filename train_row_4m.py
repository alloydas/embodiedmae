#!/usr/bin/env python3
"""Train / fine-tune the 4M model on ROW data (Regime III: neighbour occlusion), either species.

The encoder sees the rover's row render (RGB, depth and the scene cloud with both
neighbours in it, masked at `mask_ratio` as in pretraining, the recipe in the shared
budget as the maize runs did); the loss is taken against the TARGET PLANT ALONE
(gt_rgba / gt_depth / gt_pointcloud of row_dataset.py) and its recipe. `data.species`
(maize | sorghum) picks the recipe parser and the model family. So the model
is trained to reconstruct the plant behind its neighbours, not to copy the scene.
`data.mode: isolated` instead feeds the gt_* streams: the occlusion-free control with the
same camera, the same plants and the same targets.

Warm start: `init.checkpoint` (a maize 4M checkpoint, e.g. outputs/maize_4m/checkpoints/
checkpoint_epoch_600.pth on Nova or maize_4m_d2's on Delta) loads strictly; `init.schema`
records which parameter schema it was trained on. The row XML is schema v2 (the trimmed
layout), so a v1-schema warm start keeps its weights but relearns the parameter mapping;
pass a v2 checkpoint when one is on this machine. `init.checkpoint: null` trains from
scratch.

    torchrun --standalone --nproc_per_node=2 train_row_4m.py --config configs/config_maize_row_ft.yaml
    torchrun --standalone --nproc_per_node=2 train_row_4m.py --config configs/config_sorghum_row_ft.yaml
    python train_row_4m.py --config configs/config_maize_row_ft.yaml --smoke 8 --device cpu

Outputs (config.json, checkpoints/, best_model.pth, training_history.json) follow the
other trainers so eval/eval_row.py and the probes can rebuild the model from config.json.
config.json records `trainer: row`, `species`, `row_mode`, `init_from`, and for sorghum the
`rgb_brighten_gain` the images are served at (init.rgb_gain, default 1.0 = the renders).
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from tqdm import tqdm

from embodied_mae import chamfer_distance
from row_dataset import RowDataset, collate, MODES, species_info
from train_maize_4m import (_atomic_torch_save, _atomic_json, _parse_modalities,
                            prune_checkpoints)


def model_factories(species):
    if species == 'maize':
        from embodied_mae_4m_maize import (embodied_mae_4m_maize_small, embodied_mae_4m_maize_base,
                                           embodied_mae_4m_maize_large)
        return {'small': embodied_mae_4m_maize_small, 'base': embodied_mae_4m_maize_base,
                'large': embodied_mae_4m_maize_large}
    from embodied_mae_4m import embodied_mae_4m_small, embodied_mae_4m_base, embodied_mae_4m_large
    return {'small': embodied_mae_4m_small, 'base': embodied_mae_4m_base, 'large': embodied_mae_4m_large}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for sec in ('data', 'model', 'training', 'checkpointing', 'init', 'distributed', 'system', 'wandb'):
        cfg.setdefault(sec, {})
    return cfg


def config_to_namespace(cfg):
    ns = argparse.Namespace()
    d, m, t, c, i, s = cfg['data'], cfg['model'], cfg['training'], cfg['checkpointing'], cfg['init'], cfg['system']
    ns.species = d.get('species', 'maize')
    sp = species_info(ns.species)
    ns.row_root = d['row_root']
    ns.row_mode = d.get('mode', 'row')
    if ns.row_mode not in MODES:
        raise ValueError(f'data.mode must be one of {MODES}')
    ns.train_split = d.get('train_split', 'train')
    ns.val_split = d.get('val_split', 'val')
    # When the row data has no val split (maize: train + test only), validate on a fixed
    # hold-out of `val_holdout` train plants (seeded), never on test.
    ns.val_holdout = int(d.get('val_holdout', 0) or 0)
    ns.holdout_seed = int(d.get('holdout_seed', 0))
    ns.thetas = d.get('thetas', None)
    ns.img_size = int(d.get('img_size', 224))
    ns.num_points = int(d.get('num_points', sp['num_points']))
    ns.view_seed = int(d.get('view_seed', 42))
    ns.sample_per_plant = bool(d.get('sample_per_plant', True))
    ns.param_schema = sp['param_schema']         # maize: the row XML is v2; sorghum: n/a
    ns.rgb_file = 'rgba.png'
    # sorghum: the brighten_gain the images are served at (the renders carry no tag, 1.0);
    # recorded as rgb_brighten_gain so the probes serve this run the same brightness.
    ns.rgb_gain = float(i.get('rgb_gain', 1.0)) if ns.species == 'sorghum' else None
    ns.rgb_brighten_gain = ns.rgb_gain if ns.rgb_gain is not None else 1.0
    if ns.species == 'sorghum':
        from embodied_mae_4m import LEAF_FIELDS
        ns.leaf_fields = list(LEAF_FIELDS)
    ns.model_size = m.get('model_size', 'base')
    ns.mask_ratio = float(m.get('mask_ratio', 0.8))
    ns.active_modalities = _parse_modalities(m.get('active_modalities'))
    _tmr = m.get('text_mask_ratio', None)
    ns.text_mask_ratio = None if _tmr in (None, 'null') else float(_tmr)
    ns.pc_loss_name = m.get('loss_name', 'qal_loss')
    ns.qal_threshold = float(m.get('qal_threshold', 0.01))
    ns.qal_alpha = float(m.get('qal_alpha', 100.0))
    ns.qal_use_squared = bool(m.get('qal_use_squared', False))
    ns.pc_loss_weight = float(m.get('pc_loss_weight', 1.0))
    ns.pc_sinkhorn_weight = float(m.get('pc_sinkhorn_weight', 0.0) or 0.0)
    ns.pc_sinkhorn_points = int(m.get('pc_sinkhorn_points', 1024))
    ns.pc_sinkhorn_blur = float(m.get('pc_sinkhorn_blur', 0.02))
    ns.depth_norm_type = m.get('depth_norm_type', 'minmax')
    ns.spline_loss_weight = float(m.get('spline_loss_weight', 5.0))
    ns.max_leaves = int(m.get('max_leaves', sp['max_leaves']))
    ns.batch_size = int(t.get('batch_size', 16))
    ns.accum_steps = int(t.get('accum_steps', 1) or 1)
    ns.epochs = int(t.get('epochs', 100))
    ns.lr = float(t.get('lr', 1e-4))
    ns.weight_decay = float(t.get('weight_decay', 0.05))
    ns.warmup_epochs = int(t.get('warmup_epochs', 5))
    ns.val_freq = int(t.get('val_freq', 5))
    ns.output_dir = c.get('output_dir', './outputs/maize_row_ft')
    ns.save_freq = int(c.get('save_freq', 5))
    ns.keep_last = c.get('keep_last', None)
    ns.keep_every = c.get('keep_every', None)
    ns.resume = c.get('resume', None)
    ns.init_from = i.get('checkpoint', None)
    ns.init_schema = i.get('schema', None)
    ns.num_workers = int(s.get('num_workers', 8))
    ns.device = s.get('device', 'cuda')
    w = cfg['wandb']
    ns.use_wandb = bool(w.get('use_wandb', False))
    ns.wandb_project = w.get('wandb_project', 'embodied-mae-maize')
    ns.wandb_entity = w.get('wandb_entity', None)
    ns.wandb_name = w.get('wandb_name', None)
    ns.trainer = 'row'
    return ns


def build_model(args, device):
    kw = {'param_schema': args.param_schema} if args.species == 'maize' else {}
    model = model_factories(args.species)[args.model_size](
        active_modalities=args.active_modalities, img_size=args.img_size, num_pc_tokens=196,
        target_points=args.num_points, pc_loss_weight=args.pc_loss_weight, max_leaves=args.max_leaves,
        spline_loss_weight=args.spline_loss_weight, **kw,
        depth_norm_type=args.depth_norm_type, pc_loss_name=args.pc_loss_name,
        qal_threshold=args.qal_threshold, qal_alpha=args.qal_alpha, qal_use_squared=args.qal_use_squared,
        pc_sinkhorn_weight=args.pc_sinkhorn_weight, pc_sinkhorn_points=args.pc_sinkhorn_points,
        pc_sinkhorn_blur=args.pc_sinkhorn_blur, text_mask_ratio=args.text_mask_ratio)
    return model.to(device)


def strip_module(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def warm_start(model, path, is_main):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = strip_module(ckpt['model_state_dict'])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or any('param' not in k and 'text' not in k for k in missing):
        raise RuntimeError(f'warm start {path}: unexpected {unexpected[:5]} missing {missing[:5]}')
    if is_main:
        print(f'🔥 warm start from {path} (epoch {ckpt.get("epoch")}; missing {len(missing)}, unexpected {len(unexpected)})')
    return ckpt.get('epoch')


def to_dev(batch, device):
    rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = batch
    f = lambda x: x.to(device, non_blocking=True)
    return f(rgb), f(depth), f(pc), f(params), f(valid), f(g_rgb), f(g_depth), f(g_pc), info


def train_one_epoch(model, loader, optimizer, device, epoch, mask_ratio, accum_steps):
    model.train()
    sums = np.zeros(5)
    n = len(loader)
    pbar = tqdm(loader, desc=f'Epoch {epoch}', dynamic_ncols=True)
    for bi, batch in enumerate(pbar):
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, _ = to_dev(batch, device)
        g0 = (bi // accum_steps) * accum_steps
        g_size = min(accum_steps, n - g0)
        is_last = (bi + 1 == g0 + g_size)
        if bi == g0:
            optimizer.zero_grad(set_to_none=True)
        sync = model.no_sync() if (not is_last and hasattr(model, 'no_sync')) else contextlib.nullcontext()
        with sync:
            loss, (lr_, ld, lp, lt), _, _ = model(rgb, depth, pc, params, valid, mask_ratio=mask_ratio,
                                                   targets=(g_rgb, g_depth, g_pc))
            (loss / g_size).backward()
        if is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        vals = np.array([loss.item(), lr_.item(), ld.item(), lp.item(), lt.item()])
        sums += vals
        pbar.set_postfix(loss=f'{vals[0]:.4f}', rgb=f'{vals[1]:.4f}', depth=f'{vals[2]:.4f}',
                         pc=f'{vals[3]:.4f}', text=f'{vals[4]:.4f}')
    return sums / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, mask_ratio, distributed, sp):
    """Masked-regime validation against the target plant: loss terms, Chamfer of the
    predicted cloud against the TARGET cloud and against the INPUT cloud (an echo
    detector in row mode), and the recipe error on hidden real tokens, v2 columns only."""
    model.eval()
    m = model.module if hasattr(model, 'module') else model
    acc = torch.zeros(9, device=device)
    cnt = torch.zeros((), device=device)
    for batch in tqdm(loader, desc='Validating', dynamic_ncols=True):
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, _ = to_dev(batch, device)
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        loss, (lr_, ld, lp, lt), (_, _, pred_pc, pred_par), (_, _, _, mt) = model(
            rgb, depth, pc, params, valid, mask_ratio=mask_ratio, targets=(g_rgb, g_depth, g_pc))
        ch_gt = chamfer_distance(pred_pc, g_pc)
        ch_in = chamfer_distance(pred_pc, pc)
        if 'text' in m.active_modalities:
            err = (pred_par - params).abs()
            cols = torch.zeros(sp['n_params'], device=device)
            cols[sp['plant_cols']] = 1
            lcols = torch.zeros(sp['n_params'], device=device)
            lcols[sp['leaf_cols']] = 1
            plant_mae = (err[:, 0] * cols).sum(-1) / cols.sum()                       # (B,)
            leaf_w = (mt[:, 1:] * valid[:, 1:])                                      # hidden real leaves
            leaf_mae = ((err[:, 1:] * lcols).sum(-1) / lcols.sum() * leaf_w).sum() / leaf_w.sum().clamp(min=1)
            pm = plant_mae.mean()
        else:
            pm = leaf_mae = torch.zeros((), device=device)
        acc += torch.stack([loss, lr_, ld, lp, lt, ch_gt, ch_in, pm, leaf_mae]).detach()
        cnt += 1
    if distributed:
        dist.all_reduce(acc); dist.all_reduce(cnt)
    out = (acc / cnt.clamp(min=1)).tolist()
    keys = ['val_loss', 'val_rgb', 'val_depth', 'val_pc', 'val_text', 'val_pc_chamfer',
            'val_pc_chamfer_vs_input', 'val_plant_mae', 'val_leaf_mae_masked']
    return dict(zip(keys, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', type=int, default=0, help='train/val on this many plants, 1 epoch')
    ap.add_argument('--device', default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--no_wandb', action='store_true')
    cli = ap.parse_args()
    cfg = load_config(cli.config)
    args = config_to_namespace(cfg)
    for k in ('device', 'num_workers', 'batch_size', 'resume'):
        if getattr(cli, k) is not None:
            setattr(args, k, getattr(cli, k))
    if cli.no_wandb:
        args.use_wandb = False
    if cli.smoke:
        args.epochs, args.val_freq, args.save_freq = 1, 1, 1
        args.output_dir = str(Path(args.output_dir).with_name(Path(args.output_dir).name + '_smoke'))
        args.use_wandb = False

    # torchrun sets these; otherwise one process
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    distributed = world > 1
    if distributed:
        dist.init_process_group('nccl', init_method='env://')
        torch.cuda.set_device(local)
        device = torch.device(f'cuda:{local}')
    else:
        device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    is_main = rank == 0
    args.world_size = world
    sp = species_info(args.species)

    out_dir = Path(args.output_dir); ckpt_dir = out_dir / 'checkpoints'
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    plants = None
    train_ds = RowDataset(args.row_root, args.train_split, args.species, args.row_mode, args.img_size,
                          args.num_points, args.max_leaves, thetas=args.thetas,
                          sample_per_plant=args.sample_per_plant, seed=args.view_seed,
                          occlusion_stats=False, rgb_gain=args.rgb_gain)
    if args.val_holdout:
        # hold out whole plants from the train split; the two datasets share nothing
        import random as _random
        ids = list(train_ds.plant_ids)
        _random.Random(args.holdout_seed).shuffle(ids)
        held = sorted(ids[:args.val_holdout])
        train_ds.plant_ids = sorted(ids[args.val_holdout:])
        val_ds = RowDataset(args.row_root, args.train_split, args.species, args.row_mode, args.img_size,
                            args.num_points, args.max_leaves, thetas=args.thetas, sample_per_plant=True,
                            deterministic=True, occlusion_stats=False, rgb_gain=args.rgb_gain, plants=held)
        args.val_plants = held
        if is_main:
            print(f'validation = {len(held)} plants held out of {args.train_split} (seed {args.holdout_seed})')
    else:
        val_ds = RowDataset(args.row_root, args.val_split, args.species, args.row_mode, args.img_size,
                            args.num_points, args.max_leaves, thetas=args.thetas, sample_per_plant=True,
                            deterministic=True, occlusion_stats=False, rgb_gain=args.rgb_gain)
    if cli.smoke:
        train_ds.plant_ids = train_ds.plant_ids[:cli.smoke]
        val_ds.plant_ids = val_ds.plant_ids[:cli.smoke]
    if distributed:
        tr_s = DistributedSampler(train_ds, world, rank, shuffle=True)
        va_s = DistributedSampler(val_ds, world, rank, shuffle=False)
    else:
        tr_s = va_s = None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(tr_s is None), sampler=tr_s,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
                              persistent_workers=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=va_s,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
                            persistent_workers=False)

    if is_main:
        _atomic_json(vars(args), out_dir / 'config.json')     # after the hold-out is known
    model = build_model(args, device)
    if args.init_from and not args.resume:
        warm_start(model, args.init_from, is_main)
    if distributed:
        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=False)
    if is_main:
        print(f'{args.species}: params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, mode {args.row_mode}, '
              f'train {len(train_ds)} items/epoch over {len(train_ds.plant_ids)} plants, val {len(val_ds)}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {k: [] for k in ('epoch', 'lr', 'train_loss', 'train_rgb', 'train_depth', 'train_pc', 'train_text',
                               'val_epoch', 'val_loss', 'val_rgb', 'val_depth', 'val_pc', 'val_text',
                               'val_pc_chamfer', 'val_pc_chamfer_vs_input', 'val_plant_mae', 'val_leaf_mae_masked')}
    start, best = 1, float('inf')
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ck['model_state_dict']
        sd = ({'module.' + k: v for k, v in strip_module(sd).items()} if distributed else strip_module(sd))
        model.load_state_dict(sd)
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start, best = ck['epoch'] + 1, ck.get('best_val_loss', float('inf'))
        history = ck.get('history', history)
        if is_main: print(f'📂 resumed from {args.resume} at epoch {ck["epoch"]}')

    wandb = None
    if is_main and args.use_wandb:
        try:
            import wandb as _w
            wandb = _w
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
        except Exception as e:
            print(f'wandb disabled: {e}'); wandb = None

    for epoch in range(start, args.epochs + 1):
        if distributed:
            tr_s.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, epoch, args.mask_ratio, args.accum_steps)
        for k, v in zip(('train_loss', 'train_rgb', 'train_depth', 'train_pc', 'train_text'), tr):
            history[k].append(float(v))
        history['epoch'].append(epoch); history['lr'].append(optimizer.param_groups[0]['lr'])
        if is_main:
            print(f'Epoch {epoch}/{args.epochs}  loss {tr[0]:.4f} rgb {tr[1]:.4f} depth {tr[2]:.4f} '
                  f'pc {tr[3]:.4f} text {tr[4]:.4f}  ({time.time() - t0:.0f}s)')
        do_val = epoch % args.val_freq == 0 or epoch == args.epochs or epoch == 1
        vm = None
        if do_val:
            vm = evaluate(model, val_loader, device, args.mask_ratio, distributed, sp)
            history['val_epoch'].append(epoch)
            for k, v in vm.items():
                history[k].append(float(v))
            if is_main:
                print('  val ' + '  '.join(f'{k[4:]} {v:.4f}' for k, v in vm.items()))
        scheduler.step()
        if is_main:
            if wandb:
                wandb.log({'epoch': epoch, **{k: history[k][-1] for k in ('train_loss', 'train_pc', 'train_text')},
                           **(vm or {})})
            _atomic_json(history, out_dir / 'training_history.json')
            state = lambda: {'epoch': epoch, 'model_state_dict': strip_module(model.state_dict()),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best,
                             'history': history, 'config': vars(args)}
            if epoch % args.save_freq == 0 or epoch == args.epochs:
                _atomic_torch_save(state(), ckpt_dir / f'checkpoint_epoch_{epoch}.pth')
                if args.keep_last:
                    prune_checkpoints(ckpt_dir, int(args.keep_last), args.keep_every)
            if vm is not None and vm['val_loss'] < best:
                best = vm['val_loss']
                _atomic_torch_save(state(), out_dir / 'best_model.pth')
    if distributed:
        dist.barrier(); dist.destroy_process_group()
    if is_main:
        print('done')


if __name__ == '__main__':
    main()
