#!/usr/bin/env python3
"""E8 baselines through the decision-6.4 linear probe, using the arms' own protocol.

The E8 table is only readable if a baseline row and an E2/E3/E4 arm row were
produced the same way. This script therefore imports the probe's own pieces
from eval/linear_probe.py (sorghum) and eval/linear_probe_maize.py (maize):
the target tables, the TARGETS lists, `fit_probe` (train-only standardiser,
ridge alphas, 5-fold CV), `_worker_init` and the dataset class. Only feature
extraction is new. So baseline rows concatenate directly under the arm rows:
same plants, same single deterministic view, same loader seeding, same targets,
same scaler, same alphas, same CSV columns.

The adapters live in eval/baselines/. The contract is in
eval/baselines/__init__.py.

WHAT IS COPIED, NOT IMPORTED, AND WHY
`extract_split`, `cached_features` and `run_one`'s row assembly in the probe
build the model themselves (from outputs/<run>/config.json), so they cannot
take an adapter. The copies below differ from the probe only in these places:
  * adapter.features(model, batch) replaces forward_encoder_select;
  * modality slots not in the adapter's INPUTS are passed as None, and so is
    text_valid, whose sum is the leaf count (the probe never passes it on);
  * several --feature modes share one data pass, each call after the probe's
    own reseed, so each mode's rows equal a single-mode extraction;
  * img_size / num_points / max_leaves are the arms' values as constants,
    because a baseline has no config.json (--check-parity asserts that a real
    arm's config agrees);
  * numpy is seeded in-process when --num-workers 0 (the probe's _worker_init
    never runs then, and the PLY subsample would be unseeded);
  * --max-plants-per-split wraps the dataset in Subset(range(N));
  * maize plant keys are cached as fixed-width unicode, not dtype=object.
`--check-parity` proves the copy has not drifted: it runs the probe's OWN
extract_split and this file's on the same plants and requires bitwise-equal
features, then equal fits. Rerun it after any edit to either file.

Verified 2026-09-24 on CPU, 48 plants per split:
  * --check-parity on e2_pcrgbdt and on maize_4m (checkpoint_epoch_600):
    features bitwise-equal on train/val/test, and every fit_probe output
    identical.
  * A full-row comparison: probe.run_one against run_baseline, with the
    random_init weights saved as a run dir. Every CSV column except `run` is
    identical and in the same order, for both species.
  * The production cache e2_pcrgbdt__best_model__val (A100) is reproduced at
    batch 32 to a median per-row max|diff| of 4.9e-5 (device numerics). At
    seed 1 that figure is 0.087, and at batch 16 it is 0.074, so the rows
    really are the same point subsets.

THE PROBE'S FIVE RULES STILL HOLD, and three are enforced here structurally:
text is never an input (INPUTS may not name it, the params are zeroed, and
text_valid is blanked),
there is one deterministic view per plant (the probe's own dataset flags), and
the scaler is fit on train only (the probe's own fit_probe).

THE PC SUBSAMPLE DEPENDS ON THE LOADER SHAPE. load_pointcloud draws from each
worker's numpy RNG, and batch b goes to worker b % num_workers. So the 8196
points a plant gets are a function of (seed, batch_size, num_workers). The arm
caches were written by slurm/linear_probe.sbatch at --batch-size 32 and
--num-workers 32 ($SLURM_CPUS_PER_TASK), so those are the defaults here. At
other settings a baseline sees DIFFERENT point subsets of the same plants.
This matters little for the R2 values, but it breaks the "same inputs" claim.
Every cache records the settings it was made with: the weights (SOURCE), a
hash of the adapter's code (its input conversion), and for a PC adapter the
loader shape. A mismatch is FATAL on read (--refresh re-extracts;
--allow-stale-cache overrides), because a CPU debug run at 4 workers pointed at
the default cache dir would otherwise hand the real job other point subsets.
On Delta, slurm/delta/probe.sbatch runs the arms at CPUS-1 workers, so arm
caches written there are NOT this shape: pass the same --num-workers the arms
you compare against were extracted with (the Nova arm caches: 32).

--max-plants-per-split IS FOR TESTING ONLY. It keeps the first N plants of each
split, in the dataset's (string-sorted) order. Batching is sequential and
workers are assigned round-robin, so the capped rows are an exact prefix of a
full run. It also tags the CSV `run` column (`<name>__TESTONLY_capN`) and the
cache name (`__capN`), so a capped result can never be mistaken for, or served
in place of, a full one.

CACHE NAMESPACE. Files are named
  base_<name>__<species>__<split>__<feature>__seed<n>__rep<n>[__capN].npz
in the probe's own cache dirs. Probe caches are `<run>__<ckpt-stem>__...`
(sorghum) and `maize_<run>__...` (maize). A collision would therefore need a
sorghum run directory named `base_<name>` AND a checkpoint named
`sorghum.pth` or `maize.pth`. The first half is refused at startup. Every
write is atomic (tmp + os.replace) and only ever targets a `base_` name.

DELIBERATELY NOT BUILT
  * 4M (Apple ml-4m) takes pre-tokenised inputs from its own discrete
    tokenizer stack (one VQ tokenizer per modality) and has no point-cloud
    tokenizer. It could enter only as an RGB(+D) row fed through tokenizers
    trained on other data. MultiMAE already covers "multi-modal MAE over
    RGB+D" with continuous patches like ours.
  * PointGPT is the same family as Point-MAE: FPS+kNN groups, mini-PointNet
    tokens, a transformer over masked groups. A second row from it would test
    the pretext objective, not a different kind of representation.
  * "EmbodiedMAE's recipe retrained on sorghum" is not a separate run:
    e2_pcrgbd is RGB+D+PC under Dirichlet token allocation, trained on sorghum,
    and is labelled that way in the table. What it actually shares with
    EmbodiedMAE is the modality set and the Dirichlet allocation. The PC
    tokenizer (K=32, random-start FPS vs DP3 K=64 + centre MLP), the
    CLS/modality embeddings, LayerScale, the decoders, per-batch vs per-sample
    Dirichlet, and scratch init vs DINOv2 init + distillation all differ.
    Caption it as "EmbodiedMAE-style masking in our architecture", not as
    "EmbodiedMAE retrained".
  * The supervised baseline trains, so it is not an adapter row by default.
    train_supervised_4m.py writes an ordinary run dir, and eval/linear_probe.py
    scores its frozen CLS unchanged.

POOLING IS PART OF THE BASELINE, AND OF THE CSV `feature` COLUMN.
EmbodiedMAE has no CLS token (FEATURES=('mean',)); MultiMAE's 'cls' is its
global token; Point-MAE has none, so its 'cls' is a [max || mean] token pool
and its rows say feature='max+mean' (FEATURE_LABELS). By default every
baseline runs every mode it has, in one data pass, so a single command
yields all rows. Compare a row only with arms probed at the SAME `feature`:
the arm caches in outputs/_probe_cache* are 'cls' only, so the mean-pool
comparison needs the arms re-probed with linear_probe.py --feature mean.

THE SUPERVISED BASELINE is not an adapter: train_supervised_4m.py trains it
into outputs/e8_supervised, and eval/linear_probe.py scores that run dir.

Usage
-----
    # login node, with internet: fetch and verify every checkpoint, then exit
    python eval/baseline_probe.py --prefetch

    # the E8 frozen baselines (1 GPU, 32 CPUs; dataloader-bound). Default
    # --feature is 'cls mean'; each baseline skips a mode it lacks.
    python eval/baseline_probe.py --species sorghum \\
        --baselines random_init embodiedmae multimae dinov2_vitb14 pointmae pointmae_upright \\
        --batch-size 32 --num-workers 32 --out reports/probe_e8_sorghum.csv
    python eval/baseline_probe.py --species maize --baselines ... \\
        --out reports/probe_e8_maize.csv

    # smoke test on CPU (tagged TESTONLY everywhere)
    python eval/baseline_probe.py --species sorghum --baselines random_init \\
        --max-plants-per-split 48 --device cpu --num-workers 4 --cache-dir /tmp/x

    # prove this file's extraction equals the probe's own, on a real arm
    python eval/baseline_probe.py --species sorghum --check-parity e2_pcrgbdt \\
        --ckpt checkpoints/checkpoint_epoch_600.pth --max-plants-per-split 48
"""

# Repo root and eval/ on sys.path: this script imports the top-level modules
# (datasets, models) and its siblings linear_probe*/baselines.
import sys as _sys
import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))   # repo root
_sys.path.insert(0, str(_HERE.parent))          # eval/, for linear_probe and baselines

import argparse
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import baselines as B

REPO = _HERE.parent.parent
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]     # linear_probe's default grid


# ─────────────────────────────── species ────────────────────────────────────

@dataclasses.dataclass
class Species:
    name: str
    probe: object          # the linear_probe(_maize) module
    dataset: type          # the exact class the probe builds, captured at import
    ds_attr: str           # its attribute name in `probe` (for --check-parity)
    num_points: int
    max_leaves: int
    plant_of: object       # folder name -> target-table key
    plant_dtype: object
    data_root: str
    cache_dir: Path
    active_col: bool       # sorghum rows carry `active`; maize rows do not


def get_species(name):
    """Import only the requested species' probe: maize's pulls in open3d + XML."""
    if name == 'sorghum':
        import linear_probe as P
        return Species(
            name='sorghum', probe=P, dataset=P.SorghumDataset4M,
            ds_attr='SorghumDataset4M',
            # the E2/E3/E4 arms' config.json values (e2_pcrgbd, e2_pcrgbdt)
            num_points=8196, max_leaves=24,
            # the probe's own parse (linear_probe.extract_split)
            plant_of=lambda n: int(n.rsplit('_', 1)[0].split('_')[-1]),
            plant_dtype=np.int64,
            data_root=P.DATA_ROOT, cache_dir=REPO / 'outputs' / '_probe_cache',
            active_col=True)
    if name == 'maize':
        import linear_probe_maize as P
        return Species(
            name='maize', probe=P, dataset=P.MaizeDataset4M,
            ds_attr='MaizeDataset4M',
            num_points=8192,             # a pointcloud_cam.ply holds exactly 8192
            max_leaves=P.MAX_LEAVES,
            plant_of=P.MaizeDataset4M.plant_of,      # 'plant_0004_07' -> 'plant_0004'
            # fixed-width unicode rather than the probe's dtype=object, so the
            # cache loads without allow_pickle
            plant_dtype=str,
            data_root=P.DATA_ROOT, cache_dir=REPO / 'outputs' / '_probe_cache_maize',
            active_col=False)
    raise ValueError(f'unknown species {name!r}')


# ─────────────────────────────── extraction ─────────────────────────────────

def build_frozen(mod, device, hub):
    """adapter.build, plus the checks the contract promises the table."""
    t0 = time.time()
    model = mod.build(device, hub)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f'{mod.NAME}.build returned {type(model).__name__}, not nn.Module')
    # BatchNorm (PointCloudEmbed, Point-MAE's mini-PointNet) in train mode makes
    # features depend on batch composition and mutates running stats: fatal.
    training = [n or '<root>' for n, m in model.named_modules() if m.training]
    if training:
        raise RuntimeError(f'{mod.NAME}.build returned modules in train mode: {training[:5]}')
    model.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f'  built {mod.NAME}: {n:,} params, eval mode ({time.time() - t0:.0f}s)')
    return model


@torch.no_grad()
def extract_split(mod, model, sp, data_root, split, features, batch_size,
                  num_workers, device, seed, repeats, max_plants=None):
    """linear_probe.extract_split with adapter.features in place of the encoder call.

    Returns (plants, {feature: X}). Several feature modes share ONE pass over
    the data: loading is the bottleneck, and each mode's call is made after the
    same reseed the probe does, so each X equals a single-mode run's
    (--check-parity checks exactly that against the probe).

    Keep every line that touches the dataset, loader or RNG identical to the
    probe's. --check-parity fails if they drift.
    """
    features = list(features)
    ds = sp.dataset(
        data_root,
        img_size=224,
        num_points=sp.num_points,
        split=split,
        max_leaves=sp.max_leaves,
        view_sampling=True,        # both flags required: deterministic_view
        deterministic_view=True,   # alone is a silent no-op
        # max_plants is deliberately NOT passed: it draws a random TRAIN
        # subset for E3; the cap below is a prefix, for testing only.
    )
    if max_plants is not None and max_plants < len(ds):
        ds = Subset(ds, range(max_plants))
    loader = DataLoader(
        sp.probe._RetryTransientIO(ds), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=False,   # same rule as the training loaders
        worker_init_fn=sp.probe._worker_init,
    )

    keep = set(mod.INPUTS)
    ctx = B.ProbeContext(species=sp.name, split=split, data_root=Path(data_root),
                         feature=features[0], seed=seed,
                         uses_camera_pose=bool(getattr(mod, 'USES_CAMERA_POSE', False)))
    ctxs = {f: dataclasses.replace(ctx, feature=f) for f in features}

    feats, plants, dim = {f: [] for f in features}, [], {}
    torch.manual_seed(seed)
    if num_workers == 0:
        # In-process loading never calls worker_init_fn, so the probe's seeding
        # of load_pointcloud's np.random.choice would not happen. The features
        # are then reproducible but NOT the subsets a num_workers>0 run draws.
        np.random.seed(seed)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        rgb, depth, pc, params, _text_valid, names = batch

        def _dev(x, m):
            return x.to(device, non_blocking=True) if m in keep else None
        # The spline stream is never an input: params zeroed exactly as the
        # probe does it (an adapter wrapping an arm model needs a tensor
        # there), and text_valid blanked, because its sum is the leaf count.
        b = (_dev(rgb, 'rgb'), _dev(depth, 'depth'), _dev(pc, 'pc'),
             torch.zeros_like(params).to(device, non_blocking=True), None, names)

        for feat in features:
            model.probe_ctx = ctxs[feat]
            acc = None
            for r in range(repeats):
                torch.manual_seed(seed + 1000003 * bi + r)
                f = mod.features(model, b)
                if not (torch.is_tensor(f) and f.ndim == 2 and f.shape[0] == len(names)
                        and f.is_floating_point()):
                    raise TypeError(f'{mod.NAME}.features must return a float (B, D) tensor, '
                                    f'got {type(f).__name__} '
                                    f'{tuple(f.shape) if torch.is_tensor(f) else ""}')
                if dim.setdefault(feat, f.shape[1]) != f.shape[1]:
                    raise RuntimeError(f'{mod.NAME}: {feat} dim changed {dim[feat]} -> {f.shape[1]}')
                acc = f if acc is None else acc + f
            feats[feat].append((acc / repeats).float().cpu().numpy())
        # Folder order is string-sorted, so row i is not plant i: carry the id.
        plants.extend(sp.plant_of(n) for n in names)

        if bi % 50 == 0:
            done = (bi + 1) * batch_size
            print(f'    {split}: {min(done, len(ds))}/{len(ds)}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)

    out = {}
    for feat in features:
        X = np.concatenate(feats[feat], axis=0)
        if not np.isfinite(X).all():
            raise RuntimeError(f'{mod.NAME}: non-finite {feat} features in {split}')
        # A model that ignores its input (wrong key names loaded "fine", an input
        # slot never read) gives one row repeated: the ridge then scores the train
        # mean, which is a plausible-looking R2 of ~0 rather than an error.
        if len(X) > 1:
            sd = X.std(axis=0)
            if not (sd > 0).any():
                raise RuntimeError(f'{mod.NAME}: every {split} {feat} row is identical: '
                                   f'the features do not depend on the input')
            print(f'    {split} {feat}: {X.shape}  per-dim std across plants: median '
                  f'{np.median(sd):.4g} (|feature| median {np.median(np.abs(X)):.4g})')
        out[feat] = X
    print(f'    {split}: {len(plants)} plants in {time.time() - t0:.0f}s', flush=True)
    return np.asarray(plants, dtype=sp.plant_dtype), out


def cache_path(args, name, split, feature):
    stem = (f'base_{name}__{args.species}__{split}__{feature}'
            f'__seed{args.seed}__rep{args.repeats}')
    if args.max_plants_per_split is not None:
        stem += f'__cap{args.max_plants_per_split}'
    path = Path(args.cache_dir) / f'{stem}.npz'
    assert path.name.startswith('base_'), path      # never a run cache's name
    return path


def cache_settings(mod, args):
    """What a cached feature file must have been made with to be served.

    The name carries species/split/feature/seed/repeats/cap. These are the rest:
    the weights (SOURCE), the input conversion (the adapter's code), and, for a
    PC adapter, the loader shape, which picks each plant's point subset.
    """
    s = {'source': mod.SOURCE, 'code': B.code_fingerprint(mod)}
    if 'pc' in mod.INPUTS:
        s.update(batch_size=args.batch_size, num_workers=args.num_workers)
    return s


def cached_features(mod, get_model, sp, args, split, features):
    """Extract once per (split, feature), reuse for every target and alpha sweep.

    Returns (plants, {feature: X}). Cached modes are read; missing ones are
    extracted together in one data pass. The model is built only on a miss.
    """
    settings = cache_settings(mod, args)
    out, plants, todo = {}, None, []
    for feat in features:
        path = cache_path(args, mod.NAME, split, feat)
        if not path.exists() or args.refresh:
            todo.append(feat)
            continue
        z = np.load(path)                      # plants are int64 / unicode: no pickle
        meta = json.loads(str(z['meta'])) if 'meta' in z else {}
        stale = {k: (meta.get(k), v) for k, v in settings.items() if meta.get(k) != v}
        if stale:
            # A warning here was not enough: a CPU debug run at 4 workers
            # writing the default cache dir would hand the real job different
            # point subsets from the arms', and an edited conversion would be
            # served its old features, both silently.
            msg = (f'{path.name} was made with different settings {stale} (cached, now): '
                   f'point subsets, weights or input conversion differ')
            if not args.allow_stale_cache:
                raise RuntimeError(msg + '. --refresh to re-extract, or --allow-stale-cache')
            print(f'  ⚠️  {msg}; used anyway (--allow-stale-cache)')
        p = z['plants']
        if plants is not None and not np.array_equal(p, plants):
            raise RuntimeError(f'{path.name}: plant order differs from the other feature '
                               f'caches of this split: --refresh')
        plants, out[feat] = p, z['feats']
        print(f'  ⚡ cache hit {path.name}  ({z["feats"].shape})')

    if todo:
        p, X = extract_split(
            mod, get_model(), sp, args.data_root, split, todo,
            args.batch_size, args.num_workers, args.device, args.seed, args.repeats,
            max_plants=args.max_plants_per_split)
        if plants is not None and not np.array_equal(p, plants):
            raise RuntimeError(f'{split}: fresh plant order differs from the cached '
                               f'features of another mode: --refresh')
        plants = p
        for feat in todo:
            path = cache_path(args, mod.NAME, split, feat)
            meta = {'baseline': mod.NAME, 'species': sp.name, 'split': split,
                    'feature': feat, 'seed': args.seed, 'repeats': args.repeats,
                    'max_plants_per_split': args.max_plants_per_split,
                    'inputs': list(B.inputs_canonical(mod)), 'device': str(args.device),
                    'torch': torch.__version__,
                    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                    # recorded even for a non-PC adapter, where it is not checked
                    'loader': {'batch_size': args.batch_size, 'num_workers': args.num_workers},
                    **settings}
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, as the probe does: a reader sees the old file or
            # the complete new one, never a half-written array, so racing jobs are safe.
            tmp = path.parent / f'{path.stem}.tmp{os.getpid()}.npz'
            np.savez_compressed(tmp, plants=p, feats=X[feat], meta=np.array(json.dumps(meta)))
            os.replace(tmp, path)
            out[feat] = X[feat]
            print(f'  💾 cached {path.name}  ({X[feat].shape})')
    return plants, out


def check_rows(plants, tgt, split, cap):
    """The feature rows must be exactly this split's plants, once each.

    Stronger than the probe's own assertion (reindex never yields a NaN
    index), and it catches the two ways a cache goes stale: a capped test
    cache read by a full run, and a truncated split.
    """
    n_split = int((tgt['split'] == split).sum())
    want = n_split if cap is None else min(cap, n_split)
    if len(plants) != want:
        raise RuntimeError(f'{split}: {len(plants)} feature rows, expected {want} '
                           f'(of {n_split} {split} plants): stale cache? --refresh')
    if len(set(plants.tolist())) != len(plants):
        raise RuntimeError(f'{split}: duplicate plants in the feature rows')
    got = tgt.reindex(plants)['split']
    if got.isna().any() or (got != split).any():
        bad = [p for p, s in zip(plants.tolist(), got.tolist()) if s != split][:5]
        raise RuntimeError(f'{split}: plants not assigned to {split} in the target '
                           f'table, e.g. {bad}')


# ──────────────────────────────── probe ─────────────────────────────────────

def run_baseline(mod, args, sp, tgt, hub):
    name = mod.NAME
    label = name if args.max_plants_per_split is None else \
        f'{name}__TESTONLY_cap{args.max_plants_per_split}'
    feats_ok = tuple(getattr(mod, 'FEATURES', B.VALID_FEATURES))
    features = [f for f in args.feature if f in feats_ok]
    skipped = [f for f in args.feature if f not in feats_ok]
    print(f'\n=== {label} ===')
    print(f'  inputs={",".join(B.inputs_canonical(mod))} · features='
          + ', '.join(f'{f} (CSV "{B.feature_label(mod, f)}")' for f in features))
    print(f'  source: {mod.SOURCE}')
    if skipped:
        # Skipped, never substituted: a stand-in pooling under the requested
        # label would sit in the wrong column of the table.
        print(f'  · skipping {skipped}: {name} has no such feature (supports {feats_ok})')
    if not features:
        raise ValueError(f'{name} supports none of --feature {args.feature} (it has {feats_ok})')
    if getattr(mod, 'USES_CAMERA_POSE', False):
        print(f'  ⚠️  {name} READS PER-VIEW CAMERA POSE, which no arm gets: a sensitivity '
              f'row, to be captioned as using camera pose, never a headline row')
    if sp.name == 'sorghum' and (REPO / 'outputs' / f'base_{name}').exists():
        raise RuntimeError(f'outputs/base_{name} exists: a run with that slug could '
                           f'share a cache name with this baseline; rename one')

    box = {}

    def get_model():
        if 'm' not in box:
            box['m'] = build_frozen(mod, args.device, hub)
        return box['m']

    data = {}
    for split in args.split_set:
        plants, feats = cached_features(mod, get_model, sp, args, split, features)
        check_rows(plants, tgt, split, args.max_plants_per_split)
        data[split] = (plants, feats, tgt.reindex(plants))
    box.clear()
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()

    fit_on = args.split_set[0]
    ev = 'val' if 'val' in args.split_set else fit_on
    rows = []
    for feat in features:
        flabel = B.feature_label(mod, feat)
        Xtr = data[fit_on][1][feat]
        print(f'  -- feature {flabel} ({Xtr.shape[1]}-D)')
        for tname, src, prov, unit in sp.probe.TARGETS:
            ytr = data[fit_on][2][src].to_numpy(dtype=np.float64)
            evals = {s: (data[s][1][feat], data[s][2][src].to_numpy(dtype=np.float64))
                     for s in args.split_set}
            res = sp.probe.fit_probe(Xtr, ytr, evals, args.alphas)
            # Column order is the probe's (linear_probe.run_one / _maize.run_one).
            row = {'run': label, 'model_size': getattr(mod, 'MODEL_SIZE', '-')}
            if sp.active_col:
                row['active'] = ','.join(B.inputs_canonical(mod))
            row.update({'epoch': getattr(mod, 'EPOCH', -1), 'feature': flabel,
                        'dim': Xtr.shape[1], 'target': tname, 'source_col': src,
                        'provenance': prov, 'unit': unit, **res})
            rows.append(row)
            print(f'  {tname:16s} ({prov})  {ev} R2 {res[f"{ev}_r2"]:+.4f}   '
                  f'{ev} MAE {res[f"{ev}_mae"]:.4g} vs {res[f"{ev}_base_mae"]:.4g} base '
                  f'[{unit}]  alpha {res["alpha"]:g}')
    return rows


# ─────────────────────────────── parity check ───────────────────────────────

def check_parity(args, sp, hub):
    """Bitwise check that extract_split here == the probe's own, on the same plants.

    Uses random_init.features, which is the probe's feature step for ANY
    EmbodiedMAE4M, on either the untrained model or a real arm. The probe's
    dataset class is capped by a temporary subclass; this file's path is capped
    by Subset. So the check also proves the two capping routes agree.
    """
    P = sp.probe
    if args.num_workers == 0:
        raise SystemExit('--check-parity needs --num-workers > 0: with 0 the probe '
                         'leaves the PLY subsample unseeded, so nothing reproduces')
    N = args.max_plants_per_split or 48
    ours = {'img_size': 224, 'num_points': sp.num_points, 'max_leaves': sp.max_leaves}
    adapter = B.load('random_init')
    if args.check_parity == 'random_init':
        model, cfg = build_frozen(adapter, args.device, hub), dict(ours)
    else:
        run = args.check_parity
        run_dir = Path(run) if Path(run).exists() else REPO / 'outputs' / run
        model, cfg, epoch = P.build_model(run_dir, args.ckpt, args.device)
        print(f'  {run_dir.name} · {args.ckpt} @ epoch {epoch} · '
              f'active={",".join(model.active_modalities)}')
        diff = {k: (cfg.get(k, v), v) for k, v in ours.items() if cfg.get(k, v) != v}
        if diff:
            raise SystemExit(f'{run_dir.name} was probed at {diff} (run, baselines): '
                             f'its rows and the baseline rows are not the same protocol')

    orig = getattr(P, sp.ds_attr)

    class Capped(orig):
        def __len__(self):
            return min(N, super().__len__())

    # The probe extracts ONE feature per pass; the framework extracts every
    # requested feature in one pass. Each must still equal the probe's own.
    got = {}
    ok = True
    for split in args.split_set:
        p_new, f_new = extract_split(adapter, model, sp, args.data_root, split,
                                     args.feature, args.batch_size, args.num_workers,
                                     args.device, args.seed, args.repeats, max_plants=N)
        for feat in args.feature:
            setattr(P, sp.ds_attr, Capped)
            try:
                p_ref, f_ref = P.extract_split(model, cfg, args.data_root, split,
                                               feat, args.batch_size,
                                               args.num_workers, args.device,
                                               args.seed, args.repeats)
            finally:
                setattr(P, sp.ds_attr, orig)
            fn = f_new[feat]
            same_p = [str(p) for p in p_ref] == [str(p) for p in p_new]
            bit = f_ref.shape == fn.shape and np.array_equal(f_ref, fn)
            md = float(np.abs(f_ref - fn).max()) if f_ref.shape == fn.shape else float('nan')
            ok &= same_p and bit
            print(f'  PARITY {split} {feat}: plants identical={same_p}  features {f_ref.shape} '
                  f'bitwise-equal={bit}  max|diff|={md:.3g}')
            got[(split, feat)] = (p_ref, f_ref, fn)

    splits = [s for s in args.split_set]
    if 'train' in splits and ok:
        tgt = P.load_targets(args.data_root)
        for feat in args.feature:
            g = {s: got[(s, feat)] for s in splits}
            for tname, src, _prov, _u in P.TARGETS:
                y = {s: tgt.reindex(g[s][0])[src].to_numpy(dtype=np.float64) for s in g}
                r_ref = P.fit_probe(g['train'][1], y['train'],
                                    {s: (g[s][1], y[s]) for s in g}, args.alphas)
                r_new = P.fit_probe(g['train'][2], y['train'],
                                    {s: (g[s][2], y[s]) for s in g}, args.alphas)
                same = r_ref == r_new
                ok &= same
                ev = 'val' if 'val' in g else 'train'
                print(f'  PARITY fit {feat:4s} {tname:16s} {ev} R2 probe {r_ref[f"{ev}_r2"]:+.6f} '
                      f'framework {r_new[f"{ev}_r2"]:+.6f}  identical={same}')
    print(f'\nPARITY {"PASS" if ok else "FAIL"}: {sp.name}, {args.check_parity}, '
          f'features {args.feature}, {N} plants/split, batch {args.batch_size}, '
          f'workers {args.num_workers}')
    return 0 if ok else 1


# ─────────────────────────────── prefetch ───────────────────────────────────

def prefetch(names, hub):
    """Fetch + verify weights (and prove the strict load) on a node with internet."""
    failed = []
    for name in names:
        print(f'\n=== prefetch {name} ===')
        try:
            mod = B.load(name)
            print(f'  source: {mod.SOURCE}')
            if hasattr(mod, 'prefetch'):
                mod.prefetch(hub)
            else:
                # build() downloads on a miss, verifies and strict-loads, which
                # proves an offline compute node will get the same network.
                model = mod.build('cpu', hub)
                del model
            print(f'  ✓ {name} ready')
        except Exception:
            traceback.print_exc()
            failed.append(name)
    print(f'\nprefetch: {len(names) - len(failed)}/{len(names)} ready'
          + (f'; FAILED {failed}' if failed else ''))
    print(f'Compute jobs must see the same TORCH_HOME={os.environ["TORCH_HOME"]} and '
          f'HF_HOME={os.environ["HF_HOME"]} (or the same --weights-dir), and can set '
          f'HF_HUB_OFFLINE=1.')
    return 1 if failed else 0


# ───────────────────────────────── main ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species', choices=('sorghum', 'maize'))
    ap.add_argument('--baselines', nargs='+', metavar='NAME',
                    help=f'registered: {", ".join(B.NAMES)}')
    ap.add_argument('--data-root', default=None, help='default: the probe\'s DATA_ROOT')
    ap.add_argument('--split-set', default='train,val,test',
                    help='comma-separated; the FIRST is what the probe fits on. '
                         'Sorghum must be train,val,test (linear_probe.py hardcodes it)')
    ap.add_argument('--feature', nargs='+', default=list(B.VALID_FEATURES),
                    choices=B.VALID_FEATURES,
                    help='pooling modes to probe (default: both). Each baseline runs the '
                         'ones it has and skips the rest; all share one data pass')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='32 = slurm/linear_probe.sbatch; changes the PC subsample')
    ap.add_argument('--num-workers', type=int, default=32,
                    help='32 = the arm caches ($SLURM_CPUS_PER_TASK); changes the PC subsample')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=1,
                    help='forward passes per batch, averaged (only matters for random FPS)')
    ap.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    ap.add_argument('--cache-dir', default=None,
                    help='feature cache; default the probe\'s own (outputs/_probe_cache[_maize])')
    ap.add_argument('--weights-dir', default=None,
                    help='root for torch/ and huggingface/ weight caches; default '
                         '$TORCH_HOME/$HF_HOME, else ~/.cache (never XDG_CACHE_HOME)')
    ap.add_argument('--refresh', action='store_true', help='ignore cached features')
    ap.add_argument('--allow-stale-cache', action='store_true',
                    help='serve a cache whose weights, adapter code or (PC) loader shape '
                         'differ from this run (default: refuse)')
    ap.add_argument('--out', default=None, help='write the full table to this CSV')
    ap.add_argument('--max-plants-per-split', type=int, default=None, metavar='N',
                    help='TESTING ONLY: first N plants per split; tags run and cache')
    ap.add_argument('--prefetch', action='store_true',
                    help='download + verify weights for --baselines (default: all), then exit')
    ap.add_argument('--check-parity', default=None, metavar='RUN|random_init',
                    help='assert this file extracts bitwise what the probe does, then exit')
    ap.add_argument('--ckpt', default='best_model.pth',
                    help='checkpoint inside RUN, for --check-parity only')
    args = ap.parse_args()
    args.feature = list(dict.fromkeys(args.feature))      # dedupe, keep order

    # Before any adapter (or huggingface_hub) is imported.
    hub = B.configure_weight_cache(args.weights_dir)
    print(f'weights: TORCH_HOME={os.environ["TORCH_HOME"]}  HF_HOME={os.environ["HF_HOME"]}')

    if args.prefetch:
        names = args.baselines or list(B.NAMES)
        unknown = [n for n in names if n not in B.REGISTRY]
        if unknown:
            ap.error(f'unknown baselines {unknown}; registered: {list(B.NAMES)}')
        raise SystemExit(prefetch(names, hub))

    if args.species is None:
        ap.error('--species is required (except with --prefetch)')
    sp = get_species(args.species)
    args.data_root = args.data_root or sp.data_root
    args.cache_dir = args.cache_dir or str(sp.cache_dir)
    args.split_set = [s.strip() for s in args.split_set.split(',') if s.strip()]
    if sp.name == 'sorghum' and args.split_set != ['train', 'val', 'test']:
        ap.error('sorghum --split-set must be train,val,test: linear_probe.py fits on '
                 'train and reports all three, and the CSV columns must match its rows')
    if args.max_plants_per_split is not None and args.max_plants_per_split < 1:
        ap.error('--max-plants-per-split must be >= 1')

    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        print('⚠️  no CUDA, falling back to CPU (slow)')
        args.device = 'cpu'

    capped = args.max_plants_per_split is not None
    banner = ('!' * 78 + f'\n!!  --max-plants-per-split {args.max_plants_per_split}: '
              f'TESTING ONLY. First {args.max_plants_per_split} plants per split.\n'
              f'!!  Rows are tagged "__TESTONLY_cap{args.max_plants_per_split}" and are '
              f'NOT comparable to any arm row.\n' + '!' * 78)
    if capped and not args.check_parity:
        print(banner)

    if args.check_parity:
        raise SystemExit(check_parity(args, sp, hub))

    if not args.baselines:
        ap.error('--baselines is required')
    unknown = [n for n in args.baselines if n not in B.REGISTRY]
    if unknown:
        ap.error(f'unknown baselines {unknown}; registered: {list(B.NAMES)}')

    tgt = sp.probe.load_targets(args.data_root)
    rep = sp.probe.redundancy_report(tgt)
    print(rep[0] if isinstance(rep, tuple) else rep)    # sorghum returns (str, corr)

    rows, failed = [], []
    for name in args.baselines:
        # One adapter's failure (an unfetched checkpoint, a strict-load error)
        # must not cost the other baselines their GPU slot. It is still fatal
        # for that baseline, with no rows, and the exit code says so.
        try:
            rows.extend(run_baseline(B.load(name), args, sp, tgt, hub))
        except Exception:
            traceback.print_exc()
            failed.append(name)

    df = pd.DataFrame(rows)
    if args.out and len(df):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'\n📄 wrote {args.out}  ({len(df)} rows)')

    if len(df):
        ev = 'val' if 'val' in args.split_set else args.split_set[0]
        print('\n' + '=' * 78)
        print(f'{ev} R2 by baseline (rows = target, cols = baseline / feature)')
        print('=' * 78)
        # Feature is part of the column: a CLS row and a mean-pool row are
        # different measurements and are compared only with arms probed the
        # same way (linear_probe.py --feature).
        piv = df.pivot_table(index='target', columns=['run', 'feature'], values=f'{ev}_r2')
        print(piv.reindex([t[0] for t in sp.probe.TARGETS]).to_string(
            float_format=lambda v: f'{v:+.4f}'))
        print(f'\nReminder: {ev} is extreme-enriched, so R2 is comparable BETWEEN rows '
              f'on this split, not as an absolute number.')
    if capped:
        print(banner)
    if failed:
        print(f'\n❌ FAILED baselines (no rows written for them): {failed}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
