#!/usr/bin/env python3
"""Downstream linear probe on frozen encoder features -- locked decision 6.4.

6.4 makes the modality value-add metric a linear probe on plant phenotypes,
explicitly *not* reconstruction loss. `eval/analyze_e2.py` compares arms on
val_pc_chamfer, which is the right arm-invariant monitoring signal and is not
this. This script is that.

Five rules this script exists to enforce -- each one silently produces a
plausible-looking wrong number if you drop it:

1. THE TEXT STREAM IS NEVER VISIBLE, AND THE PARAMS ARE ZEROED. `param_floats`
   row 0 is the plant token and its first element is `stem_length` -- the height
   target, verbatim (embodied_mae_4m.py `_plant_to_params`). An arm allowed to
   see the spline stream at probe time is handed the answer, and the
   four-modality arm would "win" E2 on a leak. We pass `visible = active - text`
   AND feed zeros in the param slot, so the leak is structurally impossible
   rather than merely avoided.

2. FEATURES COME FROM `forward_encoder_select`, NOT `forward_encoder`. The
   latter applies Dirichlet masking (default 0.75) and would drop three
   quarters of the tokens. Note `forward_encoder(mask_ratio=0.0)` does *not*
   raise and does keep every token, but it still permutes them per sample --
   harmless for CLS, wrong for anything per-token. `forward_encoder_select`
   with `source_mask_ratio=0.0` returns natural order and consumes no RNG.

3. ONE DETERMINISTIC VIEW PER PLANT. All ten views of a plant share one label,
   so feeding view rows inflates n tenfold and narrows every error bar by ~3.2x
   with no extra information. `view_sampling=True, deterministic_view=True`
   gives view 00 for every plant. Both flags are required -- deterministic_view
   alone is a silent no-op.

4. THE SCALER IS FIT ON TRAIN ONLY. The split is extreme-enriched: val and test
   carry ~1.83x the target variance of train. Fitting the standardiser on them
   launders that shift into the score.

5. R^2 IS COMPARABLE BETWEEN ARMS, NOT AGAINST LITERATURE. Same variance ratio:
   identical absolute error reads as R^2 0.50 on train and 0.73 on val. Every
   arm is scored on the same val plants so the ranking is sound, but the
   absolute number is not portable. RMSE and MAE in native units are printed
   alongside for that reason.

Usage
-----
    # one arm, validate the probe end to end
    python eval/linear_probe.py --runs e2_pcrgbdt

    # the E2 value-add comparison decision 6.4 actually asks for
    python eval/linear_probe.py --runs e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt \
                                --out reports/probe_e2.csv

    # data and model scaling
    python eval/linear_probe.py --runs e3_1k e3_3k e3_10k e2_pcrgbdt
    python eval/linear_probe.py --runs e4_small e2_pcrgbdt e4_large

Features are cached per (run, checkpoint, split, feature, seed) under
`--cache-dir`, so re-fitting with different targets or alphas costs nothing.
"""

# Repo root on sys.path: this script lives in eval/ but imports the top-level
# modules (embodied_mae_4m, sorghum_dataset_4m).
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

from embodied_mae_4m import (
    embodied_mae_4m_small,
    embodied_mae_4m_base,
    embodied_mae_4m_large,
)
from sorghum_dataset_4m import SorghumDataset4M

DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
REPO = Path(__file__).resolve().parent.parent

_BUILD = {
    'small': embodied_mae_4m_small,
    'base': embodied_mae_4m_base,
    'large': embodied_mae_4m_large,
}

# name, source column, provenance, unit.
#   '6.4' = named by the locked decision
#   'alt' = the other reading of an ambiguous 6.4 target
#   'ind' = not in 6.4, carried because it is a genuinely independent axis
TARGETS = [
    ('height',          'stem_length',    '6.4', 'gen. units'),
    ('leaf_count',      'n_leaves',       '6.4', 'leaves'),
    ('leaf_angle',      'branch_mean',    '6.4', 'deg'),
    ('biomass',         'biomass',        '6.4', 'leaves x len'),
    ('leaf_angle_roll', 'roll_mean',      'alt', 'deg'),
    ('leaf_len_mean',   'leaf_len_mean',  'ind', 'len'),
    ('leaf_len_max',    'leaf_len_max',   'ind', 'len'),
    ('waviness',        'wav_mean',       'ind', '-'),
]


# ─────────────────────────────── targets ────────────────────────────────────

def load_targets(data_root):
    """Per-plant target table joined to the split assignment.

    `biomass` is derived: it is in neither features.csv nor the spline YAMLs,
    and n_leaves x leaf_len_mean is the cheapest defensible proxy. It is also
    r = 0.996 with n_leaves -- see the redundancy report this script prints.
    """
    root = Path(data_root)
    feats = pd.read_csv(root / 'features.csv')
    # `extremeness` comes along because it is the criterion the split was built
    # on, so it is the one label that should NOT be recoverable from shape --
    # eval/latent_analysis.py colours by it for exactly that reason. It is not a
    # probe target: TARGETS is an explicit list, so extra columns are inert here.
    assign = pd.read_csv(root / 'assignment.csv')[['plant', 'split', 'extremeness']]
    df = feats.merge(assign, on='plant', validate='1:1')
    df['biomass'] = df['n_leaves'] * df['leaf_len_mean']
    return df.set_index('plant')


def redundancy_report(tgt):
    """How much of 6.4's four-target set is actually one measurement.

    Printed rather than silently corrected: the probe still reports every
    target 6.4 names, but a reader has to be able to see that three of them
    move together.
    """
    cols = [src for _, src, _, _ in TARGETS]
    sub = tgt[cols].astype(float)
    corr = sub.corr()
    lines = ['', 'Target redundancy (pearson r with leaf_count = n_leaves):']
    for name, src, prov, unit in TARGETS:
        r = corr.loc[src, 'n_leaves']
        flag = '  <-- same measurement' if abs(r) > 0.95 and src != 'n_leaves' else ''
        lines.append(f'    {name:16s} ({prov})  r = {r:+.4f}{flag}')
    return '\n'.join(lines), corr


# ──────────────────────────────── model ─────────────────────────────────────

def build_model(run_dir, ckpt_name, device):
    """Rebuild the exact model a run used and load its weights strictly.

    Every architectural choice comes from the run's own config.json -- the E2
    arms differ in `active_modalities` (fewer embedders, structurally different
    state_dict) and the E4 arms in `model_size`.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())

    model = _BUILD[cfg['model_size']](
        active_modalities=cfg.get('active_modalities'),   # null -> all four
        img_size=cfg.get('img_size', 224),
        num_pc_tokens=196,                # hardcoded in train_sorghum_4m.py
        target_points=cfg.get('num_points', 8196),        # not the 10000 default
        pc_loss_weight=cfg.get('pc_loss_weight', 1.0),
        max_leaves=cfg.get('max_leaves', 24),
        spline_loss_weight=cfg.get('spline_loss_weight', 5.0),
        depth_norm_type=cfg.get('depth_norm_type', 'minmax'),
        pc_loss_name=cfg.get('pc_loss_name', 'qal_loss'),
        qal_threshold=cfg.get('qal_threshold', 0.01),
        qal_alpha=cfg.get('qal_alpha', 100.0),
        qal_use_squared=cfg.get('qal_use_squared', False),
    )

    ckpt = torch.load(run_dir / ckpt_name, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    # PointCloudEmbed holds BatchNorm1d: in train mode the features would depend
    # on batch composition and the running stats would be mutated in place.
    model.eval().to(device)

    # forward_encoder_select ignores text_mask_ratio, but a model built with it
    # set is not the model the checkpoint was trained as -- assert, don't assume.
    assert model.text_mask_ratio is None, model.text_mask_ratio
    return model, cfg, int(ckpt.get('epoch', -1))


def _worker_init(worker_id):
    """Make load_pointcloud's unseeded np.random.choice reproducible.

    sorghum_dataset.load_pointcloud subsamples ~104k raw points down to 8196
    with np.random.choice and no seed, so two reads of the same sample give
    different clouds. Torch seeds each worker deterministically from the base
    seed; we mirror that into numpy.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


@torch.no_grad()
def extract_split(model, cfg, data_root, split, feature, batch_size,
                  num_workers, device, seed, repeats):
    """CLS (and/or mean-pooled) features for one deterministic view per plant."""
    ds = SorghumDataset4M(
        data_root,
        img_size=cfg.get('img_size', 224),
        num_points=cfg.get('num_points', 8196),
        split=split,
        max_leaves=cfg.get('max_leaves', 24),
        view_sampling=True,        # both flags required: deterministic_view
        deterministic_view=True,   # alone is a silent no-op
        # max_plants is deliberately NOT passed -- a capped probe set would
        # score arms on different yardsticks.
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
        persistent_workers=False,   # same rule as the training loaders
        worker_init_fn=_worker_init,
    )

    # Text never visible. The param slot is zeroed too, so even a future change
    # to `visible` cannot leak stem_length into the features.
    visible = tuple(m for m in model.active_modalities if m != 'text')
    assert visible, 'no non-text modality active'

    feats, plants = [], []
    torch.manual_seed(seed)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        rgb, depth, pc, params, _text_valid, names = batch
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        pc = pc.to(device, non_blocking=True)
        params = torch.zeros_like(params).to(device, non_blocking=True)

        # FPS picks its first centroid at random (embodied_mae.py), so the token
        # centres jitter between identical calls. repeats>1 averages it out.
        acc = None
        for r in range(repeats):
            torch.manual_seed(seed + 1000003 * bi + r)
            latent = model.forward_encoder_select(
                rgb, depth, pc, params, visible=visible, source_mask_ratio=0.0)[0]
            cls = latent[:, 0]
            if feature == 'cls':
                f = cls
            elif feature == 'mean':
                f = latent[:, 1:].mean(dim=1)
            elif feature == 'cls+mean':
                f = torch.cat([cls, latent[:, 1:].mean(dim=1)], dim=1)
            else:
                raise ValueError(f'unknown feature {feature!r}')
            acc = f if acc is None else acc + f
        feats.append((acc / repeats).float().cpu().numpy())

        # Folders are Sorghum_<plant>_<view>; ds.plant_ids is sorted as STRINGS
        # so row i is not plant i -- carry the parsed id with every row.
        plants.extend(int(n.rsplit('_', 1)[0].split('_')[-1]) for n in names)

        if bi % 50 == 0:
            done = (bi + 1) * batch_size
            print(f'    {split}: {min(done, len(ds))}/{len(ds)}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)

    return np.asarray(plants, dtype=np.int64), np.concatenate(feats, axis=0)


def cached_features(model, cfg, args, run, split):
    """Extract once, reuse for every target and every alpha sweep."""
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    stem = (f'{run}__{Path(args.ckpt).stem}__{split}__{args.feature}'
            f'__seed{args.seed}__rep{args.repeats}.npz')
    path = cache / stem
    if path.exists() and not args.refresh:
        z = np.load(path)
        print(f'  ⚡ cache hit {path.name}  ({z["feats"].shape})')
        return z['plants'], z['feats']
    plants, feats = extract_split(
        model, cfg, args.data_root, split, args.feature, args.batch_size,
        args.num_workers, args.device, args.seed, args.repeats)
    # Write-then-rename, so the cache is safe when two copies of this job race
    # (e.g. one queued on nova and one on scavenger, whichever gets a GPU
    # first). os.replace is atomic on POSIX: a reader either sees the previous
    # file or the complete new one, never a half-written array. Worst case the
    # two jobs duplicate the extraction and one overwrites the other with
    # byte-identical content, which costs time but cannot corrupt.
    tmp = path.parent / f'{path.stem}.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, plants=plants, feats=feats)
    os.replace(tmp, path)
    print(f'  💾 cached {path.name}  ({feats.shape})')
    return plants, feats


# ──────────────────────────────── probe ─────────────────────────────────────

def fit_probe(Xtr, ytr, evals, alphas):
    """Ridge with train-only standardisation; alpha by 5-fold CV on train.

    One row per plant, so plain KFold is already plant-disjoint.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    xs = StandardScaler().fit(Xtr)
    Xtr_s = xs.transform(Xtr)

    ym, ysd = float(ytr.mean()), float(ytr.std())
    ysd = ysd if ysd > 0 else 1.0

    gs = GridSearchCV(
        Ridge(), {'alpha': alphas}, cv=KFold(5, shuffle=True, random_state=0),
        scoring='r2', n_jobs=-1)
    gs.fit(Xtr_s, (ytr - ym) / ysd)
    model = gs.best_estimator_

    out = {'alpha': gs.best_params_['alpha'], 'cv_r2': gs.best_score_}
    for name, (X, y) in evals.items():
        pred = model.predict(xs.transform(X)) * ysd + ym
        out[f'{name}_r2'] = r2_score(y, pred)
        out[f'{name}_rmse'] = float(np.sqrt(np.mean((y - pred) ** 2)))
        out[f'{name}_mae'] = mean_absolute_error(y, pred)
        # the number a probe has to beat: predict the train mean every time
        out[f'{name}_base_mae'] = mean_absolute_error(y, np.full_like(y, ym))
    return out


def run_one(run, args, tgt):
    run_dir = REPO / 'outputs' / run if not Path(run).exists() else Path(run)
    slug = Path(run_dir).name
    print(f'\n=== {slug} ===')
    model, cfg, epoch = build_model(run_dir, args.ckpt, args.device)
    act = ','.join(model.active_modalities)
    print(f'  {cfg["model_size"]} · active={act} · {args.ckpt} @ epoch {epoch}')

    data = {}
    for split in ('train', 'val', 'test'):
        plants, feats = cached_features(model, cfg, args, slug, split)
        # align targets to the feature rows
        y = tgt.reindex(plants)
        assert not y.index.isna().any(), 'plant missing from features.csv'
        data[split] = (plants, feats, y)

    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    Xtr = data['train'][1]
    rows = []
    for name, src, prov, unit in TARGETS:
        ytr = data['train'][2][src].to_numpy(dtype=np.float64)
        evals = {s: (data[s][1], data[s][2][src].to_numpy(dtype=np.float64))
                 for s in ('train', 'val', 'test')}
        res = fit_probe(Xtr, ytr, evals, args.alphas)
        rows.append({
            'run': slug, 'model_size': cfg['model_size'], 'active': act,
            'epoch': epoch, 'feature': args.feature, 'dim': Xtr.shape[1],
            'target': name, 'source_col': src, 'provenance': prov, 'unit': unit,
            **res,
        })
        print(f'  {name:16s} ({prov})  val R2 {res["val_r2"]:+.4f}   '
              f'test R2 {res["test_r2"]:+.4f}   '
              f'val MAE {res["val_mae"]:.4g} vs {res["val_base_mae"]:.4g} base '
              f'[{unit}]  alpha {res["alpha"]:g}')
    return rows


# ───────────────────────────────── main ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True,
                    help='run slugs under outputs/, or explicit paths')
    ap.add_argument('--ckpt', default='best_model.pth',
                    help="checkpoint file inside the run dir. NOTE best_model.pth "
                         "is selected on total val loss and sits at 42-100%% of "
                         "schedule depending on arm; pass an explicit "
                         "checkpoints/checkpoint_epoch_N.pth for a matched cut.")
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--feature', default='cls', choices=('cls', 'mean', 'cls+mean'),
                    help='CLS is the only feature whose dimension is arm-invariant')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=1,
                    help='forward passes per batch, averaged, to damp FPS jitter')
    ap.add_argument('--alphas', type=float, nargs='+',
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--cache-dir', default=str(REPO / 'outputs' / '_probe_cache'))
    ap.add_argument('--refresh', action='store_true', help='ignore cached features')
    ap.add_argument('--out', default=None, help='write the full table to this CSV')
    args = ap.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('⚠️  no CUDA, falling back to CPU (slow)')
        args.device = 'cpu'

    tgt = load_targets(args.data_root)
    report, _corr = redundancy_report(tgt)
    print(report)
    print("""
    Three of decision 6.4's four targets are one measurement: the generator
    sets stem_length = 0.05 x n_leaves, and the biomass proxy is n_leaves
    rescaled by a near-constant leaf length. They are all reported below, but
    read them as one axis. `leaf_angle` (branch_mean) varies over 0.27 deg and
    is ~80%% leaf count; `leaf_angle_roll` is the mean of n near-uniform
    circular draws and has an R2 ceiling of ~0 -- a score near zero there is
    the target, not the encoder. The 'ind' rows are the independent axes.""")

    rows = []
    for run in args.runs:
        rows.extend(run_one(run, args, tgt))

    df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'\n📄 wrote {args.out}  ({len(df)} rows)')

    print('\n' + '=' * 78)
    print('val R2 by arm (rows = target, cols = run)')
    print('=' * 78)
    piv = df.pivot_table(index='target', columns='run', values='val_r2')
    order = [n for n, _, _, _ in TARGETS]
    print(piv.reindex(order).to_string(float_format=lambda v: f'{v:+.4f}'))
    print('\nReminder: val is extreme-enriched (~1.83x train variance), so these '
          'R2 are\ncomparable BETWEEN arms on this split but are not portable '
          'absolute numbers.')


if __name__ == '__main__':
    main()
