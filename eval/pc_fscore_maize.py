#!/usr/bin/env python3
"""Point-cloud precision / recall / F1 for the MAIZE 4M runs.

For every plant of a split (val or test, one deterministic view each, view 00)
the model reconstructs the full cloud in ITS OWN TRAINING REGIME -- the trainer's
forward pass, Dirichlet token masking at the run's mask_ratio, the same call
train_maize_4m.evaluate makes. That matters: with every token visible the
decoder is out of distribution and its cloud collapses (chamfer ~15x; see
eval/pc_view_eval.py and the maize distillation teacher table). It also means
the chamfer this script reports should reproduce the run's own
training_history.json val_pc_chamfer, which is the anchor that the harness is
wired correctly. Compare them before trusting the F-scores.

Both clouds live in the loader's frame: centred at the centroid and scaled so
the farthest point sits on the unit sphere. For a point threshold tau:

    precision(tau) = fraction of PREDICTED points within tau of some GT point
    recall(tau)    = fraction of GT points within tau of some predicted point
    F1(tau)        = 2 P R / (P + R)            (0 when both are 0)

computed PER PLANT and then averaged over plants (Knapitsch et al. 2017 /
Tatarchenko et al. 2019). Precision is what chamfer's pred->GT term measures as a
distance; recall is the GT->pred term. A model that drops a leaf keeps high
precision and loses recall; one that smears points keeps recall and loses
precision. Several tau are reported because there is no canonical one: the GT
clouds' own median nearest-neighbour spacing is printed as the natural floor.

Stochastic in three places, all seeded: the Dirichlet masking draw per batch,
the FPS first centroid, and the PLY subsample (the loader pads/subsamples to
num_points with an unseeded np.random.choice; a maize PLY holds exactly 8192
points, so with num_points 8192 it is a no-op, but the worker RNG is seeded
anyway). Text params are the real ones, as in the trainer's val pass.

Per-plant results are cached under outputs/_pc_fscore_maize/, written
atomically; the CSV and the report are rebuilt from the caches, so re-running
after another run finishes costs only that run's forward passes.

    python eval/pc_fscore_maize.py --runs maize_e2_pc_d2 maize_4m_d2 --splits val test
    python eval/pc_fscore_maize.py --runs maize_e3_1k_d2:checkpoints/checkpoint_epoch_6168.pth
"""

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import eval.linear_probe_maize as LP
from embodied_mae_4m_maize import MAX_LEAVES
from maize_dataset_4m import MaizeDataset4M

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TAUS = (0.005, 0.01, 0.02, 0.05)
DEFAULT_CACHE = REPO / 'outputs' / '_pc_fscore_maize'


def newest_checkpoint(run_dir):
    """Highest-numbered checkpoint_epoch_N.pth, as a path relative to run_dir."""
    cks = list((run_dir / 'checkpoints').glob('checkpoint_epoch_*.pth'))
    if not cks:
        raise FileNotFoundError(f'{run_dir}: no checkpoints')
    best = max(cks, key=lambda p: int(re.findall(r'(\d+)\.pth$', p.name)[0]))
    return f'checkpoints/{best.name}'


def parse_run(spec):
    run, _, ckpt = spec.partition(':')
    return run, (ckpt or newest_checkpoint(REPO / 'outputs' / run))


def build_model(run_dir, ckpt, device):
    """LP.build_model plus param_schema, which the probe does not need and this
    script does not either (it never decodes text) -- passed so the model
    object describes the run faithfully."""
    cfg = json.loads((run_dir / 'config.json').read_text())
    model = LP._BUILD[cfg['model_size']](
        active_modalities=cfg.get('active_modalities'),
        img_size=cfg.get('img_size', 224),
        num_pc_tokens=196,
        target_points=cfg.get('num_points', 8192),
        pc_loss_weight=cfg.get('pc_loss_weight', 1.0),
        max_leaves=cfg.get('max_leaves', MAX_LEAVES),
        spline_loss_weight=cfg.get('spline_loss_weight', 5.0),
        depth_norm_type=cfg.get('depth_norm_type', 'minmax'),
        pc_loss_name=cfg.get('pc_loss_name', 'qal_loss'),
        qal_threshold=cfg.get('qal_threshold', 0.01),
        qal_alpha=cfg.get('qal_alpha', 100.0),
        qal_use_squared=cfg.get('qal_use_squared', False),
        param_schema=cfg.get('param_schema', 'v1'),
    )
    state = torch.load(run_dir / ckpt, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in state['model_state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, cfg, int(state.get('epoch', -1))


def cache_path(cache_dir, run, ckpt, split, seed, limit):
    tag = f'{run}__{Path(ckpt).stem}__{split}__seed{seed}'
    if limit:
        tag += f'__limit{limit}'
    return Path(cache_dir) / f'{tag}.npz'


@torch.no_grad()
def nn_dists(pred, gt, chunk=2):
    """(B,N,3),(B,M,3) -> per-point NN distance pred->gt (B,N) and gt->pred (B,M)."""
    fw, bw = [], []
    for i in range(0, pred.shape[0], chunk):
        d = torch.cdist(pred[i:i + chunk], gt[i:i + chunk])          # (b, N, M)
        fw.append(d.min(dim=2).values)
        bw.append(d.min(dim=1).values)
        del d
    return torch.cat(fw), torch.cat(bw)


@torch.no_grad()
def score_run(run, ckpt, split, args):
    out = cache_path(args.cache_dir, run, ckpt, split, args.seed, args.limit)
    if out.exists() and not args.refresh:
        print(f'  cache hit {out.name}')
        return out
    run_dir = REPO / 'outputs' / run
    model, cfg, epoch = build_model(run_dir, ckpt, args.device)
    mask_ratio = float(cfg.get('mask_ratio', 0.8))
    if 'pc' not in model.active_modalities:
        raise SystemExit(f'{run}: pc is not an active modality, nothing to score')
    print(f'  {run} @ {ckpt} (epoch {epoch}, active {list(model.active_modalities)}, '
          f'mask_ratio {mask_ratio}, rgb {cfg.get("rgb_file", "rgb.png")}, '
          f'schema {cfg.get("param_schema", "v1")})')

    ds = MaizeDataset4M(args.data_root or cfg['data_root'],
                        img_size=cfg.get('img_size', 224),
                        num_points=cfg.get('num_points', 8192), split=split,
                        max_leaves=cfg.get('max_leaves', MAX_LEAVES),
                        view_sampling=True, deterministic_view=True,
                        rgb_file=cfg.get('rgb_file', 'rgb.png'),
                        param_schema=cfg.get('param_schema', 'v1'))
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    torch.manual_seed(args.seed)
    loader = DataLoader(LP._RetryTransientIO(ds), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.device != 'cpu',
                        persistent_workers=False, worker_init_fn=LP._worker_init)

    taus = torch.tensor(args.taus, device=args.device)
    names, prec, rec, cham_fw, cham_bw, gt_spacing = [], [], [], [], [], []
    t0 = time.time()
    for bi, (rgb, depth, pc, params, text_valid, name) in enumerate(loader):
        rgb, depth, pc = (t.to(args.device, non_blocking=True) for t in (rgb, depth, pc))
        params, text_valid = params.to(args.device), text_valid.to(args.device)
        # The trainer's own regime (train_maize_4m.evaluate): Dirichlet masking at
        # the run's mask_ratio. Seeded per batch so a rerun reproduces it.
        torch.manual_seed(args.seed + 1000003 * bi + 1)
        _, _, (_, _, pred_pc, _), _ = model(rgb, depth, pc, params, text_valid,
                                            mask_ratio=mask_ratio)
        pred_pc = pred_pc.float()
        fw, bw = nn_dists(pred_pc, pc)                                # (B,N), (B,M)
        prec.append((fw.unsqueeze(-1) < taus).float().mean(dim=1).cpu().numpy())   # (B,T)
        rec.append((bw.unsqueeze(-1) < taus).float().mean(dim=1).cpu().numpy())
        cham_fw.append((fw ** 2).mean(dim=1).cpu().numpy())           # trainer's chamfer terms
        cham_bw.append((bw ** 2).mean(dim=1).cpu().numpy())
        # GT self-spacing: median NN distance of the GT cloud to itself (excluding self).
        g = torch.cdist(pc, pc)
        g.diagonal(dim1=1, dim2=2).fill_(float('inf'))
        gt_spacing.append(g.min(dim=2).values.median(dim=1).values.cpu().numpy())
        del g
        names.extend(name)
        if bi % 20 == 0:
            print(f'    {split}: {min((bi + 1) * args.batch_size, len(ds))}/{len(ds)}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)

    P, R = np.concatenate(prec), np.concatenate(rec)
    F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    arrays = dict(names=np.asarray(names), taus=np.asarray(args.taus, np.float64),
                  precision=P, recall=R, f1=F,
                  chamfer_fw=np.concatenate(cham_fw), chamfer_bw=np.concatenate(cham_bw),
                  gt_spacing=np.concatenate(gt_spacing),
                  epoch=np.int64(epoch), mask_ratio=np.float64(mask_ratio),
                  active=np.asarray(list(model.active_modalities)),
                  rgb_file=str(cfg.get('rgb_file', 'rgb.png')),
                  param_schema=str(cfg.get('param_schema', 'v1')))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f'.tmp{os.getpid()}.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    print(f'  wrote {out.name}  ({time.time() - t0:.0f}s)')
    del model
    if args.device != 'cpu':
        torch.cuda.empty_cache()
    return out


def published_val_chamfer(run):
    """The run's own last val_pc_chamfer and its epoch, for the anchor column."""
    try:
        h = json.loads((REPO / 'outputs' / run / 'training_history.json').read_text())
        c = h.get('val_pc_chamfer') or []
        return (float(c[-1]) if c else float('nan')), len(c)
    except Exception:
        return float('nan'), 0


def summarise(cache_files):
    """One row per (run, split, tau): means and medians over plants."""
    rows = []
    for run, split, f in cache_files:
        z = np.load(f, allow_pickle=False)
        ch = z['chamfer_fw'] + z['chamfer_bw']
        pub, _ = published_val_chamfer(run)
        for ti, tau in enumerate(z['taus']):
            rows.append({
                'run': run, 'split': split, 'epoch': int(z['epoch']),
                'active': '+'.join(z['active'].tolist()),
                'n_plants': len(z['names']), 'tau': float(tau),
                'precision': float(z['precision'][:, ti].mean()),
                'recall': float(z['recall'][:, ti].mean()),
                'f1': float(z['f1'][:, ti].mean()),
                'f1_median': float(np.median(z['f1'][:, ti])),
                'f1_p10': float(np.percentile(z['f1'][:, ti], 10)),
                'chamfer': float(ch.mean()),
                'chamfer_pred_to_gt': float(z['chamfer_fw'].mean()),
                'chamfer_gt_to_pred': float(z['chamfer_bw'].mean()),
                'published_val_chamfer': pub,
                'gt_nn_spacing_median': float(np.median(z['gt_spacing'])),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--runs', nargs='+', required=True, help='run or run:ckpt-relpath')
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    ap.add_argument('--data-root', default=None, help='default: each run\'s config.json data_root')
    ap.add_argument('--taus', nargs='+', type=float, default=list(DEFAULT_TAUS))
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='first N plants only (smoke)')
    ap.add_argument('--cache-dir', default=str(DEFAULT_CACHE))
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--out', default=str(REPO / 'reports' / 'pc_fscore_maize.csv'),
                    help='CSV; rows for the scored (run, split) pairs are replaced, others kept')
    args = ap.parse_args()

    runs = [parse_run(s) for s in args.runs]
    done = []
    for run, ckpt in runs:
        for split in args.splits:
            print(f'\n== {run} / {split}')
            done.append((run, split, score_run(run, ckpt, split, args)))

    df = summarise(done)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.limit:
        old = pd.read_csv(out)
        keep = ~old.set_index(['run', 'split']).index.isin(df.set_index(['run', 'split']).index)
        df = pd.concat([old[keep], df], ignore_index=True)
    if args.limit:
        out = out.with_name(out.stem + f'_limit{args.limit}.csv')
    df = df.sort_values(['split', 'run', 'tau']).reset_index(drop=True)
    tmp = out.with_name(out.name + f'.tmp{os.getpid()}')
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)

    pd.set_option('display.width', 200)
    show = df[df.tau == 0.01][['run', 'split', 'epoch', 'active', 'n_plants', 'precision',
                               'recall', 'f1', 'chamfer', 'published_val_chamfer']]
    print('\n@ tau = 0.01 (unit-sphere frame):')
    print(show.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nGT median NN spacing: {df.gt_nn_spacing_median.iloc[0]:.4f}')
    print(f'📄 {out}')


if __name__ == '__main__':
    main()
