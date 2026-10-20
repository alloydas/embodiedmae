#!/usr/bin/env python3
"""Regime III evaluation: a 4M checkpoint (either species) on the ROW data (row_dataset.py),
the same plants under the occluded row render and, paired, under the isolated render
from the same rover camera.

For every item (plant, theta, shot), every mode in --modes and every input set in
--sources, the model generates the target plant's cloud and recipe with the recipe
NEVER an input (the param tensor is zeroed and the text stream contributes no encoder
token, as in every probe here):

  all   : rgb + depth + pc visible, token-masked at --mask-ratio (the regime the model
          trained in; with every token visible the decoder is out of distribution and the
          cloud collapses, see CLAUDE.md "Reconstruction must run in the masked regime")
  rgb / depth / pc : that stream alone, every token visible, the others absent

Scores per item:
  chamfer_gt : squared Chamfer (mean nearest-neighbour distance both ways, per item) of the predicted cloud against
               the TARGET plant's cloud, in the input cloud's normalised frame
  chamfer_in : the same against the INPUT cloud; in row mode a model that merely echoes
               the scene scores well here and badly on chamfer_gt
  miss_0p01  : fraction of target points with no predicted point within 0.01 (normalised units)
  *_m2 / miss_5mm : the same in metres (x scale^2 / 5 mm), comparable between row and isolated
               inputs, whose unit-sphere frames differ in scale (scene vs target alone)
  plant_mae / leaf_mae : recipe error on the real columns of the species' current
               layout (maize: schema v2, plant leafCount / stemInternodeSum / height and the
               7 LEAF_FIELDS_V2; sorghum: the 9 plant fields and the 5 width-layout leaf
               fields), normalised units; leaf tokens weighted by validity. Only for
               checkpoints trained on that layout (maize config.json param_schema v2;
               sorghum config.json leaf_fields containing 'width'); an older-layout model
               predicts different columns and is scored on geometry only.
  occ_frac / nbr_frac / theta : the occlusion measures from the dataset, for the bins.

Outputs <out>.csv (one row per item x mode x source) and <out>.json: per mode x source
mean / median of each score, the same per theta and per occlusion quartile, and the
PAIRED row - isolated delta per (plant, theta, source) with a bootstrap CI over plants.
Deterministic: numpy and torch are seeded per item, so a subset run reproduces rows.

    python eval/eval_row.py --species maize --run maize_4m --ckpt checkpoints/checkpoint_epoch_600.pth \\
        --split val --n-plants 64 --out reports/row_eval_maize4m
    python eval/eval_row.py --species sorghum --root /work/mech-ai/alloydas/sorghum_row \\
        --run e2_pcrgbdt_d2 --ckpt checkpoints/checkpoint_epoch_600.pth --out reports/row_eval_sorghum_pcrgbdt
The sorghum images are served at the run's recorded rgb_brighten_gain (1.0 when unrecorded).
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from row_dataset import RowDataset, collate, MODES, species_info, SPECIES
from train_row_4m import model_factories

REPO = Path(__file__).resolve().parent.parent
MODALITIES = ('rgb', 'depth', 'pc', 'text')
SOURCES = ('all', 'rgb', 'depth', 'pc')
DEFAULT_ROOT = {'maize': '/work/mech-ai/alloydas/maize_row', 'sorghum': '/work/mech-ai/alloydas/sorghum_row'}


def build_model(run_dir, ckpt_name, device, species):
    """The run's own config.json decides every constructor kwarg (as the probes do)."""
    cfg = json.loads((run_dir / 'config.json').read_text())
    sp = species_info(species)
    kw = {'param_schema': cfg.get('param_schema') or 'v1'} if species == 'maize' else {}
    model = model_factories(species)[cfg.get('model_size', 'base')](
        active_modalities=cfg.get('active_modalities'), img_size=cfg.get('img_size', 224),
        num_pc_tokens=196, target_points=cfg.get('gt_points', cfg.get('num_points', sp['num_points'])),
        pc_loss_weight=cfg.get('pc_loss_weight', 1.0), max_leaves=cfg.get('max_leaves', sp['max_leaves']),
        spline_loss_weight=cfg.get('spline_loss_weight', 5.0), **kw,
        depth_norm_type=cfg.get('depth_norm_type', 'minmax'), pc_loss_name=cfg.get('pc_loss_name', 'qal_loss'),
        qal_threshold=cfg.get('qal_threshold', 0.01), qal_alpha=cfg.get('qal_alpha', 100.0),
        qal_use_squared=cfg.get('qal_use_squared', False),
        pc_sinkhorn_weight=0.0, text_mask_ratio=cfg.get('text_mask_ratio'))
    ck = torch.load(run_dir / ckpt_name, map_location='cpu', weights_only=False)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg, ck.get('epoch')


def _unpack(out):
    latent = out[0]
    masks = dict(zip(MODALITIES, out[1:5]))
    rests = out[5:9]
    nvis = dict(zip(MODALITIES, out[9:13]))
    return latent, masks, rests, nvis


@torch.no_grad()
def predict(model, src, rgb, depth, pc, params, mask_ratio, seed):
    torch.manual_seed(seed)
    act = set(model.active_modalities)
    zeros = torch.zeros_like(params)
    if src == 'all':
        vis = {'rgb', 'depth', 'pc'} & act
        out = model.forward_encoder_select(rgb, depth, pc, zeros, visible=vis, source_mask_ratio=mask_ratio)
    else:
        if src not in act:
            return None, None
        out = model.forward_encoder_select(rgb, depth, pc, zeros, visible={src}, source_mask_ratio=0.0)
    latent, masks, rests, nvis = _unpack(out)
    if 'text' in act:
        assert nvis['text'] == 0, 'the recipe reached the encoder'
    _, _, pred_pc, pred_par = model.forward_decoder(latent, *rests, *nvis.values())
    return pred_pc, pred_par


def score_cloud(pred, gt, scale):
    """Per-item squared Chamfer and miss rates, one item at a time: a batched (B, Ngt, Npred)
    distance matrix against the 24,576-point scene cloud is ~13 GB at batch 16, and the repo's
    chamfer_distance expands (B, N, M, 3) on top of it (36 GB; OOM on a 40 GB A100, job 17016779).

    Clouds sit in the INPUT cloud's unit-sphere frame, whose scale differs between row (scene,
    ~0.52-0.61 m) and isolated (target alone, ~0.50-0.54 m) inputs, so a normalised Chamfer is
    not comparable across modes; `scale` (B,) metres per unit converts to metric units:
      per       squared Chamfer in the normalised frame   per_m2    the same in m^2
      miss      target points > 0.01 (normalised) from any prediction   miss_5mm  > 5 mm
    """
    per, per_m2, miss, miss_5mm = [], [], [], []
    for b in range(pred.shape[0]):
        d = torch.cdist(gt[b], pred[b])                        # (Ngt, Npred)
        dg = d.min(dim=1).values                               # target -> nearest prediction
        dp = d.min(dim=0).values                               # prediction -> nearest target
        cd = (dg ** 2).mean() + (dp ** 2).mean()
        per.append(cd); per_m2.append(cd * scale[b] ** 2)
        miss.append((dg > 0.01).float().mean()); miss_5mm.append((dg * scale[b] > 0.005).float().mean())
        del d
    return torch.stack(per), torch.stack(per_m2), torch.stack(miss), torch.stack(miss_5mm)


def recipe_scored(cfg, species, model):
    """Does this checkpoint predict the species' CURRENT recipe layout?"""
    if 'text' not in model.active_modalities:
        return False
    if species == 'maize':
        return (cfg.get('param_schema') or 'v1') == 'v2'
    return 'width' in (cfg.get('leaf_fields') or [])


def recipe_error(pred_par, params, valid, sp):
    pc_, lc = sp['plant_cols'], sp['leaf_cols']
    err = (pred_par - params).abs()
    plant = err[:, 0, pc_].mean(-1)
    w = valid[:, 1:]
    leaf = ((err[:, 1:][..., lc].mean(-1) * w).sum(1) / w.sum(1).clamp(min=1))
    return plant, leaf


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--species', default='maize', choices=sorted(SPECIES))
    ap.add_argument('--run', required=True, help='outputs/<run> (its config.json builds the model)')
    ap.add_argument('--ckpt', default='best_model.pth')
    ap.add_argument('--root', default=None, help=f'default per species: {DEFAULT_ROOT}')
    ap.add_argument('--split', default='val')
    ap.add_argument('--modes', default='row,isolated')
    ap.add_argument('--sources', default='all,rgb,depth,pc')
    ap.add_argument('--thetas', default=None, help='comma list, default all')
    ap.add_argument('--n-plants', type=int, default=None)
    ap.add_argument('--num-points', type=int, default=None,
                    help='points sampled from the INPUT cloud (default: the run\'s; a row run records 24576, '
                         'an isolated run 8192, which thins the target to a third of its density)')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--mask-ratio', type=float, default=None, help="'all' source masking; default the run's")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True, help='prefix: writes <out>.csv and <out>.json')
    a = ap.parse_args()

    device = torch.device(a.device)
    a.root = a.root or DEFAULT_ROOT[a.species]
    sp = species_info(a.species)
    run_dir = REPO / 'outputs' / a.run if not Path(a.run).exists() else Path(a.run)
    model, cfg, epoch = build_model(run_dir, a.ckpt, device, a.species)
    schema = (cfg.get('param_schema') or 'v1') if a.species == 'maize' else \
        ('width' if 'width' in (cfg.get('leaf_fields') or []) else 'waviness')
    recipe = recipe_scored(cfg, a.species, model)
    rgb_gain = float(cfg.get('rgb_brighten_gain', 1.0)) if a.species == 'sorghum' else None
    mask_ratio = a.mask_ratio if a.mask_ratio is not None else float(cfg.get('mask_ratio', 0.8))
    modes = [m for m in a.modes.split(',') if m]
    sources = [s for s in a.sources.split(',') if s]
    thetas = None if a.thetas is None else [int(t) for t in a.thetas.split(',')]
    print(f'{a.species} run {run_dir.name} @ {a.ckpt} (epoch {epoch}), recipe layout {schema}, recipe scored: '
          f'{recipe}, modes {modes}, sources {sources}, mask_ratio {mask_ratio}'
          + (f', rgb gain {rgb_gain:g}' if rgb_gain is not None else ''))

    rows = []
    t0 = time.time()
    for mode in modes:
        ds = RowDataset(a.root, a.split, a.species, mode, cfg.get('img_size', 224),
                        a.num_points or cfg.get('num_points', sp['num_points']), cfg.get('max_leaves', sp['max_leaves']),
                        thetas=thetas, rgb_gain=rgb_gain,
                        gt_points=cfg.get('gt_points', cfg.get('num_points', sp['num_points'])))
        if a.n_plants:
            keep = set(ds.plant_ids[:a.n_plants])
            ds.items = [it for it in ds.items if it[0] in keep]
            ds.plant_ids = sorted(keep)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                            collate_fn=collate, persistent_workers=False,
                            worker_init_fn=lambda w: np.random.seed(a.seed + 7919 * w))
        for bi, batch in enumerate(loader):
            rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = [x.to(device) if torch.is_tensor(x) else x
                                                                       for x in batch]
            for src in sources:
                pred_pc, pred_par = predict(model, src, rgb, depth, pc, params, mask_ratio, a.seed + 1000003 * bi)
                if pred_pc is None:
                    continue
                scale = torch.tensor([float(inf['scale']) for inf in info], device=pred_pc.device)
                cg, cg_m2, miss, miss5 = score_cloud(pred_pc, g_pc, scale)
                ci, ci_m2, _, _ = score_cloud(pred_pc, pc, scale)
                pm, lm = recipe_error(pred_par, params, valid, sp) if recipe else (None, None)
                for k, inf in enumerate(info):
                    rows.append({'species': a.species, 'run': run_dir.name, 'epoch': epoch, 'mode': mode, 'source': src,
                                 'plant': inf['plant'], 'theta': inf['theta'], 'shot': inf['shot'],
                                 'occ_frac': inf['occ_frac'], 'nbr_frac': inf['nbr_frac'], 'scale_m': float(inf['scale']),
                                 'chamfer_gt': float(cg[k]), 'chamfer_in': float(ci[k]), 'miss_0p01': float(miss[k]),
                                 'chamfer_gt_m2': float(cg_m2[k]), 'chamfer_in_m2': float(ci_m2[k]), 'miss_5mm': float(miss5[k]),
                                 'plant_mae': float(pm[k]) if recipe else float('nan'),
                                 'leaf_mae': float(lm[k]) if recipe else float('nan')})
            if bi % 10 == 0:
                print(f'  {mode}: {min((bi + 1) * a.batch_size, len(ds))}/{len(ds)}  ({time.time() - t0:.0f}s)', flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ── summary ──────────────────────────────────────────────────────────────
    metrics = ['chamfer_gt', 'chamfer_in', 'miss_0p01', 'chamfer_gt_m2', 'chamfer_in_m2', 'miss_5mm'] + (['plant_mae', 'leaf_mae'] if recipe else [])
    def agg(sel):
        return {m: {'mean': float(np.mean([r[m] for r in sel])), 'median': float(np.median([r[m] for r in sel])),
                    'n': len(sel)} for m in metrics} if sel else {}
    summary = {'species': a.species, 'run': run_dir.name, 'ckpt': a.ckpt, 'epoch': epoch, 'recipe_layout': schema,
               'recipe_scored': recipe, 'rgb_gain': rgb_gain, 'split': a.split,
               'mask_ratio': mask_ratio, 'n_rows': len(rows), 'by_mode_source': {}, 'by_theta': {},
               'by_occlusion_quartile': {}, 'paired_row_minus_isolated': {}}
    occ = np.array([r['occ_frac'] for r in rows if r['mode'] == 'row'])
    qs = np.quantile(occ, [0.25, 0.5, 0.75]) if len(occ) else np.array([0, 0, 0])
    for mode in modes:
        for src in sources:
            sel = [r for r in rows if r['mode'] == mode and r['source'] == src]
            if not sel:
                continue
            key = f'{mode}/{src}'
            summary['by_mode_source'][key] = agg(sel)
            summary['by_theta'][key] = {str(t): agg([r for r in sel if r['theta'] == t])
                                        for t in sorted({r['theta'] for r in sel})}
            if mode == 'row':
                bins = np.digitize([r['occ_frac'] for r in sel], qs)
                summary['by_occlusion_quartile'][key] = {
                    f'q{b + 1}': {**agg([r for r, bb in zip(sel, bins) if bb == b]),
                                  'occ_range': [float(min([r['occ_frac'] for r, bb in zip(sel, bins) if bb == b], default=np.nan)),
                                                float(max([r['occ_frac'] for r, bb in zip(sel, bins) if bb == b], default=np.nan))]}
                    for b in range(4)}
    if 'row' in modes and 'isolated' in modes:
        for src in sources:
            iso = {(r['plant'], r['theta'], r['shot']): r for r in rows if r['mode'] == 'isolated' and r['source'] == src}
            per_plant = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if r['mode'] != 'row' or r['source'] != src:
                    continue
                j = iso.get((r['plant'], r['theta'], r['shot']))
                if j is None:
                    continue
                for m in metrics:
                    per_plant[m][r['plant']].append(r[m] - j[m])
            summary['paired_row_minus_isolated'][src] = {
                m: dict(zip(('mean', 'ci_lo', 'ci_hi'), bootstrap_ci([np.mean(v) for v in d.values()], seed=a.seed)))
                   | {'n_plants': len(d)}
                for m, d in per_plant.items()}
    summary['seconds'] = time.time() - t0
    with open(out.with_suffix('.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ('by_mode_source', 'paired_row_minus_isolated')}, indent=1))
    print(f'wrote {out.with_suffix(".csv")} and {out.with_suffix(".json")} in {summary["seconds"]:.0f}s')


if __name__ == '__main__':
    main()
