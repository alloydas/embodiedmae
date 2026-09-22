#!/usr/bin/env python3
"""Downstream linear probe on frozen MAIZE encoder features — decision 6.4.

The maize twin of `eval/linear_probe.py`. Separate file on purpose: the two
species are independent pipelines, and the target tables, the plant-id parse and
the model width all differ. Nothing here imports the sorghum probe.

WHY THIS MATTERS MORE THAN THE SORGHUM PROBE
Decision 6.4 names four targets — height, leaf angle, leaf count, biomass. In
Sorghum_15K three of those are the same measurement (the generator sets
stem_length = 0.05 x n_leaves, r = 0.994; the biomass proxy is r = 0.996 with
leaf count) and neither leaf-angle column is learnable — `roll_mean` is the mean
of n near-uniform circular draws and `branch_mean` spans 0.27 degrees. The
sorghum probe therefore reports a rank-2 answer to a rank-4 question.

Maize does not have that problem. Measured on all 15,000 plants:
    height (stem_internodeSum) vs leaf_count   r = 0.199   (sorghum 0.994)
    biomass (leaf_areaProxy)   vs leaf_count   r = 0.763   (sorghum 0.996)
    leaf_angleMean             vs everything   |r| < 0.034 (sorghum: unlearnable)
Condition number of the candidate set 199 against sorghum's 1207; effective rank
7.7 of 12. **Maize is where decision 6.4's metric can actually be reported as
four independent phenotypes.**

THE SAME FIVE RULES AS THE SORGHUM PROBE, for the same reasons:
  1. Text is never visible AND the param tensor is zeroed. The maize plant token
     carries stemRadius, stemShrink, leafCount and stemInternodeSum — leaf count
     and height, verbatim — so an arm that sees the spline stream is handed two
     of the four targets. Zeroing makes the leak structurally impossible.
  2. Features come from `forward_encoder_select(..., source_mask_ratio=0.0)`,
     never `forward_encoder`, which masks.
  3. One deterministic view per plant: `view_sampling=True` AND
     `deterministic_view=True`. All ten views share one label.
  4. Standardiser fit on TRAIN ONLY. Maize val carries 1.5-2.6x the target
     variance of train (Mahalanobis outlier enrichment, seed 0), so fitting on
     val launders the shift into the score.
  5. Extraction is stochastic: FPS picks its first centroid randomly and the PLY
     subsample is unseeded. Seeded here; without it R^2 does not reproduce.

Usage
-----
    python eval/linear_probe_maize.py --runs maize_4m
    python eval/linear_probe_maize.py --runs maize_4m --split-set train,val,test \\
                                      --out reports/probe_maize.csv
"""

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from embodied_mae_4m_maize import (
    embodied_mae_4m_maize_small,
    embodied_mae_4m_maize_base,
    embodied_mae_4m_maize_large,
    MAX_LEAVES,
)
from maize_dataset_4m import MaizeDataset4M

DATA_ROOT = '/work/mech-ai-scratch/alloy/Maize'
REPO = Path(__file__).resolve().parent.parent

_BUILD = {
    'small': embodied_mae_4m_maize_small,
    'base': embodied_mae_4m_maize_base,
    'large': embodied_mae_4m_maize_large,
}

# name, plant_scores.csv column, provenance, unit.
#   '6.4' = named by the locked decision — and in maize all four are real
#   'ind' = extra independent axes maize offers that sorghum does not
TARGETS = [
    ('height',      'stem_internodeSum',          '6.4', 'gen. units'),
    ('leaf_angle',  'leaf_angleMean',             '6.4', 'deg'),
    ('leaf_count',  'leaf_count',                 '6.4', 'leaves'),
    ('biomass',     'leaf_areaProxy',             '6.4', 'len^2'),
    ('leaf_droop',  'leaf_droopMean',             'ind', 'deg'),
    ('leaf_twist',  'leaf_twistAbsMean',          'ind', 'deg'),
    ('leaf_curl',   'leaf_curlMean',              'ind', '-'),
    ('leaf_length', 'leaf_lengthMean',            'ind', 'len'),
    ('leaf_width',  'leaf_widthMean',             'ind', 'len'),
    ('stem_radius', 'stem_radius',                'ind', 'len'),
    ('tassel_droop', 'tassel_matureDroopStrength', 'ind', '-'),
]


# ─────────────────────────────── targets ────────────────────────────────────

def load_targets(data_root):
    """Per-plant targets from plant_scores.csv.

    Unlike sorghum there is no separate assignment.csv — `split` is a column
    here. Index is the STRING key `plant_0000`, which is what a folder name
    `plant_0000_07` reduces to; do not coerce it to int, the zero-padding is
    only 4 wide while ids run to 14999.
    """
    df = pd.read_csv(Path(data_root) / 'plant_scores.csv')
    missing = [c for _, c, _, _ in TARGETS if c not in df.columns]
    if missing:
        raise KeyError(f'plant_scores.csv is missing {missing}')
    return df.set_index('plant')


def redundancy_report(tgt):
    """Show how independent the targets actually are, rather than asserting it."""
    cols = [c for _, c, _, _ in TARGETS]
    corr = tgt[cols].astype(float).corr()
    lines = ['', 'Target redundancy (pearson r with leaf_count):']
    for name, col, prov, _u in TARGETS:
        r = corr.loc[col, 'leaf_count']
        flag = ('  <-- same measurement' if abs(r) > 0.95 and col != 'leaf_count'
                else '  (strong)' if abs(r) > 0.8 and col != 'leaf_count' else '')
        lines.append(f'    {name:13s} ({prov})  r = {r:+.4f}{flag}')
    z = tgt[cols].astype(float)
    ev = np.linalg.eigvalsh(np.corrcoef(((z - z.mean()) / z.std()).to_numpy().T))[::-1]
    lines.append(f'    condition number {ev[0]/max(ev[-1],1e-12):.1f} · '
                 f'effective rank {ev.sum()**2/(ev**2).sum():.2f} of {len(cols)}')
    return '\n'.join(lines)


# ──────────────────────────────── model ─────────────────────────────────────

def build_model(run_dir, ckpt_name, device):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    model = _BUILD[cfg['model_size']](
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
    )
    ckpt = torch.load(run_dir / ckpt_name, map_location='cpu', weights_only=False)
    state = {k[7:] if k.startswith('module.') else k: v
             for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)          # BatchNorm1d in PointCloudEmbed — mandatory
    assert model.text_mask_ratio is None, model.text_mask_ratio
    return model, cfg, int(ckpt.get('epoch', -1))


def _worker_init(worker_id):
    """The PLY subsample uses an unseeded np.random.choice; mirror torch's seed."""
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


@torch.no_grad()
def extract_split(model, cfg, data_root, split, feature, batch_size,
                  num_workers, device, seed, repeats):
    ds = MaizeDataset4M(
        data_root,
        img_size=cfg.get('img_size', 224),
        num_points=cfg.get('num_points', 8192),
        split=split,
        max_leaves=cfg.get('max_leaves', MAX_LEAVES),
        view_sampling=True,
        deterministic_view=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=False, worker_init_fn=_worker_init)

    visible = tuple(m for m in model.active_modalities if m != 'text')
    assert visible, 'no non-text modality active'

    feats, plants = [], []
    torch.manual_seed(seed)
    t0 = time.time()
    for bi, (rgb, depth, pc, params, _valid, names) in enumerate(loader):
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        pc = pc.to(device, non_blocking=True)
        # zeroed: the maize plant token carries leafCount and stemInternodeSum,
        # i.e. two of the four 6.4 targets, verbatim
        params = torch.zeros_like(params).to(device, non_blocking=True)

        acc = None
        for r in range(repeats):
            torch.manual_seed(seed + 1000003 * bi + r)
            latent = model.forward_encoder_select(
                rgb, depth, pc, params, visible=visible, source_mask_ratio=0.0)[0]
            cls = latent[:, 0]
            f = (cls if feature == 'cls'
                 else latent[:, 1:].mean(1) if feature == 'mean'
                 else torch.cat([cls, latent[:, 1:].mean(1)], dim=1))
            acc = f if acc is None else acc + f
        feats.append((acc / repeats).float().cpu().numpy())
        # 'plant_0004_07' -> 'plant_0004', the plant_scores.csv key
        plants.extend(MaizeDataset4M.plant_of(n) for n in names)

        if bi % 50 == 0:
            print(f'    {split}: {min((bi+1)*batch_size, len(ds))}/{len(ds)}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)

    return np.asarray(plants, dtype=object), np.concatenate(feats, axis=0)


def cached_features(model, cfg, args, run, split):
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (f'maize_{run}__{Path(args.ckpt).stem}__{split}__{args.feature}'
                    f'__seed{args.seed}__rep{args.repeats}.npz')
    if path.exists() and not args.refresh:
        z = np.load(path, allow_pickle=True)
        print(f'  ⚡ cache hit {path.name}  ({z["feats"].shape})')
        return z['plants'], z['feats']
    plants, feats = extract_split(model, cfg, args.data_root, split, args.feature,
                                  args.batch_size, args.num_workers, args.device,
                                  args.seed, args.repeats)
    tmp = path.parent / f'{path.stem}.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, plants=plants, feats=feats)
    os.replace(tmp, path)            # atomic: racing jobs cannot corrupt it
    print(f'  💾 cached {path.name}  ({feats.shape})')
    return plants, feats


# ──────────────────────────────── probe ─────────────────────────────────────

def fit_probe(Xtr, ytr, evals, alphas):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    xs = StandardScaler().fit(Xtr)
    ym, ysd = float(ytr.mean()), float(ytr.std()) or 1.0
    gs = GridSearchCV(Ridge(), {'alpha': alphas},
                      cv=KFold(5, shuffle=True, random_state=0),
                      scoring='r2', n_jobs=-1)
    gs.fit(xs.transform(Xtr), (ytr - ym) / ysd)
    est = gs.best_estimator_

    out = {'alpha': gs.best_params_['alpha'], 'cv_r2': gs.best_score_}
    for name, (X, y) in evals.items():
        pred = est.predict(xs.transform(X)) * ysd + ym
        out[f'{name}_r2'] = r2_score(y, pred)
        out[f'{name}_rmse'] = float(np.sqrt(np.mean((y - pred) ** 2)))
        out[f'{name}_mae'] = mean_absolute_error(y, pred)
        out[f'{name}_base_mae'] = mean_absolute_error(y, np.full_like(y, ym))
    return out


def run_one(run, args, tgt):
    run_dir = REPO / 'outputs' / run if not Path(run).exists() else Path(run)
    slug = Path(run_dir).name
    print(f'\n=== {slug} ===')
    model, cfg, epoch = build_model(run_dir, args.ckpt, args.device)
    print(f'  {cfg["model_size"]} · active={",".join(model.active_modalities)} '
          f'· {args.ckpt} @ epoch {epoch}')

    data = {}
    for split in args.split_set:
        plants, feats = cached_features(model, cfg, args, slug, split)
        y = tgt.reindex(plants)
        assert not y.index.isna().any(), 'a plant is missing from plant_scores.csv'
        data[split] = (plants, feats, y)
    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    fit_on = args.split_set[0]
    Xtr = data[fit_on][1]
    rows = []
    for name, col, prov, unit in TARGETS:
        ytr = data[fit_on][2][col].to_numpy(dtype=np.float64)
        evals = {s: (data[s][1], data[s][2][col].to_numpy(dtype=np.float64))
                 for s in args.split_set}
        res = fit_probe(Xtr, ytr, evals, args.alphas)
        rows.append({'run': slug, 'model_size': cfg['model_size'], 'epoch': epoch,
                     'feature': args.feature, 'dim': Xtr.shape[1], 'target': name,
                     'source_col': col, 'provenance': prov, 'unit': unit, **res})
        ev = 'val' if 'val' in args.split_set else fit_on
        print(f'  {name:13s} ({prov})  {ev} R2 {res[f"{ev}_r2"]:+.4f}   '
              f'{ev} MAE {res[f"{ev}_mae"]:.4g} vs {res[f"{ev}_base_mae"]:.4g} '
              f'base [{unit}]  alpha {res["alpha"]:g}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--ckpt', default='best_model.pth',
                    help='best_model.pth is selected on TOTAL val loss and can sit '
                         'anywhere in the schedule; pass an explicit '
                         'checkpoints/checkpoint_epoch_N.pth to compare runs at a '
                         'matched cut')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split-set', default='train,val,test',
                    help='comma-separated; the FIRST is what the probe fits on')
    ap.add_argument('--feature', default='cls', choices=('cls', 'mean', 'cls+mean'))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--alphas', type=float, nargs='+',
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--cache-dir', default=str(REPO / 'outputs' / '_probe_cache_maize'))
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.split_set = [s.strip() for s in args.split_set.split(',') if s.strip()]

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('⚠️  no CUDA, falling back to CPU (slow)')
        args.device = 'cpu'

    tgt = load_targets(args.data_root)
    print(redundancy_report(tgt))
    print("""
    Unlike sorghum, all four of decision 6.4's targets are real and largely
    independent here — height vs leaf count r = 0.20 (sorghum 0.99) and
    leaf angle is uncorrelated with everything (sorghum's was unlearnable).
    Maize val/test are outlier-enriched (1.5-2.6x train variance), so R2 is
    comparable BETWEEN runs on this split but is not a portable absolute
    number; MAE in native units is printed for that.""")

    rows = []
    for run in args.runs:
        rows.extend(run_one(run, args, tgt))
    df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'\n📄 wrote {args.out}  ({len(df)} rows)')

    ev = 'val' if 'val' in args.split_set else args.split_set[0]
    print('\n' + '=' * 78)
    print(f'{ev} R2 by run (rows = target)')
    print('=' * 78)
    piv = df.pivot_table(index='target', columns='run', values=f'{ev}_r2')
    print(piv.reindex([n for n, _, _, _ in TARGETS]).to_string(
        float_format=lambda v: f'{v:+.4f}'))


if __name__ == '__main__':
    main()
