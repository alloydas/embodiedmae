#!/usr/bin/env python3
"""Cross-modal POINT-CLOUD generation from ONE source modality, over a whole split.

For every plant of the split (one deterministic view per plant: view 00, via
`view_sampling=True` AND `deterministic_view=True`, exactly as the probes and
the trainers' val loaders do) and for each requested source in {rgb, depth}:

  * the source modality is FULLY visible (every token, source_mask_ratio=0.0);
  * every other modality -- including text -- contributes ZERO encoder tokens
    and is reconstructed entirely by the decoder;
  * the param tensor fed in is all ZEROS. The plant token carries targets
    verbatim (maize: leafCount, stemInternodeSum; sorghum: stem_length), so
    zeroing makes a param leak structurally impossible.

This is the same masking path as export/export_maize_views.py
(`forward_encoder_select(visible={src}, source_mask_ratio=0.0)` then
`forward_decoder`), and the same visibility regime eval/eval_views_one_plant.py
and eval/eval_rgb2pc_calib.py use through `model(..., visible={src})`. The
per-modality token counts and masks it returns are asserted on every plant.

The decoded cloud is scored against the loader's GT cloud with the repo's own
`embodied_mae.chamfer_distance` (squared NN distances, mean both ways -- the
function behind val_pc_chamfer), plus `miss_0p01`: the fraction of GT points
whose nearest predicted point is farther than 0.01 (Euclidean, loader-normalised
units: the cloud is centred and scaled onto the unit sphere).

REFERENCE ("ref" rows). The same plants are also run under the model's normal
training-time masking: all active modalities, `forward_encoder` with the run's
`mask_ratio` (0.80) and one Dirichlet draw per plant, with the REAL params --
i.e. what `evaluate()` in the trainers does for val_pc_chamfer (there the draw
is per batch of 16, here per plant). Its mean should land near the run's
recorded val_pc_chamfer at the same checkpoint, which checks this script's data
and model path end to end.

DETERMINISM. Each plant gets seed = --seed + plant id. numpy is seeded with it
immediately before the dataset read (load_pointcloud subsamples/permutes with an
unseeded np.random.choice) and torch is re-seeded with it before EVERY forward
(FPS draws its first centroid with torch.randint; the reference also draws its
Dirichlet split and token shuffles from torch). Rows therefore do not depend on
batch composition, worker count or plant subset: an --n-plants 8 run reproduces
the first 8 rows of the full run.

LEAK CHECK. On the first --leak-plants plants every source pass is re-run with
the REAL params and with every non-source image/cloud replaced by noise; the
predicted cloud and params must match the zero-param pass bit for bit
(max|diff| = 0.0), or the script aborts.

RESUMABLE. Rows are flushed atomically (temp file + os.replace) to
<out>.partial.csv every --flush plants; a requeued job with the same --out
reloads them and continues. Deterministic seeding makes the resumed rows
identical to an uninterrupted run.

Outputs: <out>.csv (one row per plant x mode) and <out>.json (summary: mean,
median, p10, p90 of chamfer and mean miss fraction per mode, the leak check,
the run's recorded val_pc_chamfer at the checkpoint epoch, runtime).

    python eval/eval_single_source_pc.py --species maize --n-plants 8 \\
        --out reports/single_source_pc_maize_test
    python eval/eval_single_source_pc.py --species sorghum --run e2_pcrgbdt \\
        --ckpt checkpoints/checkpoint_epoch_600.pth --sources rgb,depth \\
        --split val --out reports/single_source_pc_sorghum_$SLURM_JOB_ID
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, *_dataset_4m, eval/*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from embodied_mae import chamfer_distance

REPO = Path(__file__).resolve().parent.parent

SPECIES = {
    'maize': {'run': 'maize_4m',
              'data_root': '/work/mech-ai-scratch/alloy/Maize',
              'num_points': 8192, 'max_leaves': 28},
    'sorghum': {'run': 'e2_pcrgbdt',
                'data_root': '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K',
                'num_points': 8196, 'max_leaves': 24},
}
MODALITIES = ('rgb', 'depth', 'pc', 'text')      # the 13-tuple's order
# F-score thresholds, Euclidean on the unit-sphere-normalised cloud -- the
# same three train_sorghum_4m.pc_scores and eval/eval_occlusion.py report.
THRESHOLDS = (0.01, 0.02, 0.03)
FSCORE = [f'{m}_{t:.2f}'.replace('0.', '0p') for t in THRESHOLDS
          for m in ('f1', 'precision', 'recall')]
FIELDS = ['plant', 'plant_id', 'view', 'seed', 'mode', 'chamfer',
          'acc_pred_to_gt', 'comp_gt_to_pred', 'miss_0p01', 'miss_0p02',
          'nn_gt_median', *FSCORE]


def load_species(species, run_dir, ckpt, device):
    """(model, cfg, epoch, dataset class) from the species' own code path."""
    if species == 'maize':
        from eval.linear_probe_maize import build_model
        from maize_dataset_4m import MaizeDataset4M as DS
    else:
        from eval.linear_probe import build_model
        from sorghum_dataset_4m import SorghumDataset4M as DS
    model, cfg, epoch = build_model(run_dir, ckpt, device)
    return model, cfg, epoch, DS


class SeededPlants(Dataset):
    """Wrap a view-sampled split so every read is seeded by the plant id."""

    def __init__(self, ds, seed, n):
        self.ds, self.seed = ds, seed
        self.keys = list(ds.plant_ids)[:n]

    def __len__(self):
        return len(self.keys)

    def pid(self, i):
        return int(self.keys[i].rsplit('_', 1)[1])

    def __getitem__(self, i):
        s = self.seed + self.pid(i)
        np.random.seed(s)                        # load_pointcloud's np.random.choice
        rgb, depth, pc, params, valid, name = self.ds[i]
        return rgb, depth, pc, params, valid, name, self.keys[i], self.pid(i), s


def _unpack(out):
    latent = out[0]
    masks = dict(zip(MODALITIES, out[1:5]))
    rests = out[5:9]
    nvis = dict(zip(MODALITIES, out[9:13]))
    return latent, masks, rests, nvis


@torch.no_grad()
def predict_single_source(model, src, rgb, depth, pc, params, seed):
    """`src` fully visible, every other active modality 0 encoder tokens."""
    torch.manual_seed(seed)
    out = model.forward_encoder_select(rgb, depth, pc, params, visible={src},
                                       source_mask_ratio=0.0)
    latent, masks, rests, nvis = _unpack(out)
    for m in model.active_modalities:
        if m == src:
            assert nvis[m] == model._token_len[m], (m, nvis[m])
            assert bool((masks[m] == 0).all()), f'source {m} is not fully visible'
        else:
            assert nvis[m] == 0, f'{m} contributed {nvis[m]} encoder tokens'
            assert bool((masks[m] == 1).all()), f'{m} is not fully masked'
    assert latent.shape[1] == 1 + model._token_len[src], latent.shape
    _, _, pred_pc, pred_par = model.forward_decoder(latent, *rests, *nvis.values())
    return pred_pc, pred_par


@torch.no_grad()
def predict_training_masking(model, rgb, depth, pc, params, mask_ratio, seed):
    """The trainer's val path: all active modalities, Dirichlet mask at mask_ratio."""
    torch.manual_seed(seed)
    out = model.forward_encoder(rgb, depth, pc, params, mask_ratio)
    latent, masks, rests, nvis = _unpack(out)
    _, _, pred_pc, _ = model.forward_decoder(latent, *rests, *nvis.values())
    n_vis = {m: int(nvis[m]) for m in model.active_modalities}
    return pred_pc, n_vis


def score(pred, gt):
    """Repo chamfer + per-GT-point Euclidean NN stats. pred, gt: (1, N, 3)."""
    cd = float(chamfer_distance(pred, gt))
    d = torch.cdist(gt[0], pred[0], compute_mode='donot_use_mm_for_euclid_dist')
    nn_gt = d.min(dim=1).values                   # per GT point -> nearest pred
    nn_pr = d.min(dim=0).values                   # per pred point -> nearest GT
    acc, comp = float((nn_pr ** 2).mean()), float((nn_gt ** 2).mean())
    assert abs((acc + comp) - cd) <= 1e-7 + 1e-4 * cd, (cd, acc + comp)
    out = {'chamfer': cd, 'acc_pred_to_gt': acc, 'comp_gt_to_pred': comp,
           'miss_0p01': float((nn_gt > 0.01).float().mean()),
           'miss_0p02': float((nn_gt > 0.02).float().mean()),
           'nn_gt_median': float(nn_gt.median())}
    for t in THRESHOLDS:
        k = f'{t:.2f}'.replace('0.', '0p')
        prec = float((nn_pr < t).float().mean())   # predicted points near the GT
        rec = float((nn_gt < t).float().mean())    # GT points the prediction covers
        out[f'precision_{k}'], out[f'recall_{k}'] = prec, rec
        out[f'f1_{k}'] = 2 * prec * rec / max(prec + rec, 1e-9)
    return out


def noise_like(x, g, kind):
    if kind == 'pc':
        n = torch.randn(x.shape, generator=g)
        return n / n.norm(dim=-1).max()
    return torch.rand(x.shape, generator=g) if kind == 'depth' else torch.randn(x.shape, generator=g)


def stats(rows, mode):
    r = [x for x in rows if x['mode'] == mode]
    c = np.array([x['chamfer'] for x in r], np.float64)
    ms = np.array([x['miss_0p01'] for x in r], np.float64)
    m2 = np.array([x['miss_0p02'] for x in r], np.float64)
    return {'n': int(len(c)),
            'chamfer_mean': float(c.mean()), 'chamfer_median': float(np.median(c)),
            'chamfer_p10': float(np.percentile(c, 10)),
            'chamfer_p90': float(np.percentile(c, 90)),
            'chamfer_std': float(c.std(ddof=1)) if len(c) > 1 else None,
            'chamfer_sem': float(c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else None,
            'miss_0p01_mean': float(ms.mean()), 'miss_0p01_median': float(np.median(ms)),
            'miss_0p02_mean': float(m2.mean()),
            'acc_pred_to_gt_mean': float(np.mean([x['acc_pred_to_gt'] for x in r])),
            'comp_gt_to_pred_mean': float(np.mean([x['comp_gt_to_pred'] for x in r])),
            **{f'{k}_mean': float(np.mean([x[k] for x in r])) for k in FSCORE}}


def write_rows(path, rows):
    tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_rows(path):
    ints, strs = {'plant_id', 'seed'}, {'plant', 'view', 'mode'}
    with open(path) as f:
        return [{k: (v if k in strs else int(v) if k in ints else float(v))
                 for k, v in r.items()} for r in csv.DictReader(f)]


def recorded_val_chamfer(run_dir, cfg, epoch):
    """val_pc_chamfer the trainer logged at `epoch`, or None.

    training_history.json keeps no val-epoch column; the trainers validate at
    epoch 1, every val_freq, and the last epoch, so the epochs are rebuilt from
    config.json and used only if their count matches the logged series.
    """
    try:
        h = json.loads((run_dir / 'training_history.json').read_text())
    except Exception:
        return None
    ch = h.get('val_pc_chamfer') or []
    vf, E = cfg.get('val_freq'), cfg.get('epochs')
    if not (ch and vf and E):
        return None
    eps = sorted({1, E, *range(vf, E + 1, vf)})
    if len(eps) != len(ch) or epoch not in eps:
        return {'note': f'{len(ch)} logged vs {len(eps)} expected val epochs',
                'last_logged': ch[-1]}
    return {'epoch': epoch, 'val_pc_chamfer': ch[eps.index(epoch)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species', required=True, choices=sorted(SPECIES))
    ap.add_argument('--run', default=None, help='outputs/<run> (default per species)')
    ap.add_argument('--outputs-root', default=None,
                    help='directory holding <run>/ (default: this repo\'s outputs/)')
    ap.add_argument('--ckpt', default='checkpoints/checkpoint_epoch_600.pth')
    ap.add_argument('--sources', default='rgb,depth')
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True, help='path prefix; writes <out>.csv/.json')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-plants', type=int, default=None, help='first N plants only')
    ap.add_argument('--leak-plants', type=int, default=4)
    ap.add_argument('--no-ref', action='store_true', help='skip the training-masking reference')
    ap.add_argument('--flush', type=int, default=50)
    args = ap.parse_args()

    if 'best_model' in args.ckpt:
        raise SystemExit('best_model.pth is selected on total val loss at a '
                         'run-dependent epoch; pass checkpoints/checkpoint_epoch_N.pth')
    torch.set_num_threads(args.threads)
    sp = SPECIES[args.species]
    run = args.run or sp['run']
    run_dir = Path(args.outputs_root or REPO / 'outputs') / run
    data_root = args.data_root or sp['data_root']
    sources = [s.strip() for s in args.sources.split(',') if s.strip()]
    device = torch.device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model, cfg, epoch, DS = load_species(args.species, run_dir, args.ckpt, device)
    for s in sources:
        assert s in model.active_modalities and s in ('rgb', 'depth', 'pc'), s
    assert 'pc' in model.active_modalities
    mask_ratio = float(cfg['mask_ratio'])
    print(f'[{time.time()-t0:.0f}s] {args.species} {run} {args.ckpt} (epoch {epoch}), '
          f'{cfg["model_size"]}, active={model.active_modalities}, '
          f'target_points={model.target_points}, mask_ratio={mask_ratio}, '
          f'sources={sources}, ref={not args.no_ref}', flush=True)

    ds = DS(data_root, img_size=cfg.get('img_size', 224),
            num_points=cfg.get('num_points', sp['num_points']), split=args.split,
            max_leaves=cfg.get('max_leaves', sp['max_leaves']),
            view_sampling=True, deterministic_view=True)
    n = len(ds) if args.n_plants is None else min(args.n_plants, len(ds))
    wrap = SeededPlants(ds, args.seed, n)
    print(f'[{time.time()-t0:.0f}s] {args.split}: {len(ds)} plants, scoring {n}', flush=True)

    partial = out.with_name(out.name + '.partial.csv')
    rows = read_rows(partial) if partial.exists() else []
    modes = sources + ([] if args.no_ref else ['ref'])
    # a plant counts as done only if every mode is present
    per = {}
    for r in rows:
        per.setdefault(r['plant'], set()).add(r['mode'])
    done = {p for p, ms in per.items() if set(modes) <= ms}
    rows = [r for r in rows if r['plant'] in done]
    if done:
        print(f'resuming: {len(done)} plants already scored in {partial.name}', flush=True)
    todo = [i for i in range(n) if wrap.keys[i] not in done]
    # leak-check plants are always re-run (cheap) so the check survives a requeue
    leak_idx = set(range(min(args.leak_plants, n)))
    todo = sorted(set(todo) | leak_idx)

    class _Sub(Dataset):
        def __len__(self): return len(todo)
        def __getitem__(self, j): return wrap[todo[j]]

    dl = DataLoader(_Sub(), batch_size=1, shuffle=False, num_workers=args.workers,
                    persistent_workers=False, collate_fn=lambda b: b[0])

    leak = {s: {'pred_pc': 0.0, 'pred_params': 0.0, 'plants': 0} for s in sources}
    ref_nvis = []
    t_loop = time.time()
    new = 0
    for j, (rgb, depth, pc, params, valid, name, key, pid, seed) in enumerate(dl):
        i = todo[j]
        assert name.rsplit('_', 1)[1] == '00', f'{key}: deterministic view is {name}, not 00'
        b = lambda t: t.unsqueeze(0).to(device)
        rgb, depth, pc, params = b(rgb), b(depth), b(pc), b(params)
        zeros = torch.zeros_like(params)
        assert float(zeros.abs().max()) == 0.0
        res = {}
        for s in sources:
            pred, ppar = predict_single_source(model, s, rgb, depth, pc, zeros, seed)
            res[s] = score(pred, pc)
            if i in leak_idx:
                g = torch.Generator().manual_seed(seed + 7)
                ins = {'rgb': rgb, 'depth': depth, 'pc': pc}
                for m in ins:
                    if m != s:
                        ins[m] = noise_like(ins[m].cpu(), g, m).to(device)
                pred2, ppar2 = predict_single_source(model, s, ins['rgb'], ins['depth'],
                                                     ins['pc'], params, seed)
                leak[s]['pred_pc'] = max(leak[s]['pred_pc'],
                                         float((pred2 - pred).abs().max()))
                if ppar is not None:
                    leak[s]['pred_params'] = max(leak[s]['pred_params'],
                                                 float((ppar2 - ppar).abs().max()))
                leak[s]['plants'] += 1
        if not args.no_ref:
            pred, nv = predict_training_masking(model, rgb, depth, pc, params,
                                                mask_ratio, seed)
            res['ref'] = score(pred, pc)
            ref_nvis.append(nv)
        if key not in done:
            for mode, sc in res.items():
                rows.append({'plant': key, 'plant_id': pid, 'view': name[-2:],
                             'seed': seed, 'mode': mode, **sc})
            done.add(key)
            new += 1
        if (j + 1) % 10 == 0 or j == 0:
            el = time.time() - t_loop
            print(f'[{time.time()-t0:.0f}s] {j+1}/{len(todo)} {key}  '
                  + '  '.join(f'{m} {res[m]["chamfer"]:.6f}/{res[m]["miss_0p01"]:.3f}'
                              for m in res)
                  + f'   {el/(j+1):.2f} s/plant', flush=True)
        if new and new % args.flush == 0:
            write_rows(partial, rows)

    for s in sources:
        assert leak[s]['pred_pc'] == 0.0 and leak[s]['pred_params'] == 0.0, \
            f'LEAK: a masked input reached the {s}-only output: {leak[s]}'

    order = {k: i for i, k in enumerate(wrap.keys)}
    rows.sort(key=lambda r: (order[r['plant']], modes.index(r['mode'])))
    assert len({r['plant'] for r in rows}) == n, (len({r['plant'] for r in rows}), n)
    csv_path = out.with_name(out.name + '.csv')
    write_rows(csv_path, rows)
    if partial.exists():
        partial.unlink()

    summary = {
        'species': args.species, 'run': run, 'ckpt': args.ckpt, 'ckpt_epoch': epoch,
        'split': args.split, 'n_plants': n, 'view': '00 (view_sampling + deterministic_view)',
        'sources': sources, 'model_size': cfg['model_size'],
        'active_modalities': list(model.active_modalities),
        'target_points': model.target_points, 'mask_ratio_ref': mask_ratio,
        'data_root': data_root,
        'regime': {s: f'{s} fully visible (source_mask_ratio 0.0); every other '
                      f'modality incl. text 0 encoder tokens; params zeroed'
                   for s in sources},
        'ref_regime': ('forward_encoder(mask_ratio) over all active modalities, one '
                       'Dirichlet draw per plant, REAL params (the trainer val path)'),
        'metric': ('chamfer = embodied_mae.chamfer_distance (squared NN, mean both ways); '
                   'miss_0p01 = frac of GT points whose nearest pred point is > 0.01 '
                   '(Euclidean, unit-sphere-normalised cloud); precision_t = frac of '
                   'pred points within t of the GT, recall_t = frac of GT points within '
                   't of the pred (= 1 - miss_t), f1_t their harmonic mean, per plant'),
        'seed_scheme': f'{args.seed} + plant id; numpy before the read, torch before each forward',
        'stats': {m: stats(rows, m) for m in modes},
        'recorded_val_pc_chamfer': recorded_val_chamfer(run_dir, cfg, epoch),
        'leak_check_max_abs_diff': leak,
        'ref_mean_visible_tokens': ({m: float(np.mean([v[m] for v in ref_nvis]))
                                     for m in model.active_modalities} if ref_nvis else None),
        'runtime_s': time.time() - t0,
        'loop_s_per_plant': (time.time() - t_loop) / max(1, len(todo)),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
        'host': socket.gethostname(), 'threads': args.threads, 'workers': args.workers,
        'torch': torch.__version__, 'python': platform.python_version(),
    }
    json_path = out.with_name(out.name + '.json')
    tmp = json_path.with_name(json_path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, json_path)

    print(f'\n{args.species} {run} @ {args.ckpt}: {n} {args.split} plants, view 00')
    print(f"{'mode':>6}{'n':>6}{'mean':>11}{'median':>11}{'p10':>11}{'p90':>11}"
          f"{'miss>0.01':>11}{'miss>0.02':>11}")
    for m in modes:
        st = summary['stats'][m]
        print(f"{m:>6}{st['n']:>6}{st['chamfer_mean']:>11.6f}{st['chamfer_median']:>11.6f}"
              f"{st['chamfer_p10']:>11.6f}{st['chamfer_p90']:>11.6f}"
              f"{st['miss_0p01_mean']:>11.4f}{st['miss_0p02_mean']:>11.4f}")
    print(f"\n{'mode':>6}" + ''.join(f"{'F1/P/R@' + str(t):>24}" for t in THRESHOLDS))
    for m in modes:
        st = summary['stats'][m]
        print(f"{m:>6}" + ''.join(
            f"{st[f'f1_{k}_mean']:>10.4f}/{st[f'precision_{k}_mean']:.4f}/{st[f'recall_{k}_mean']:.4f}"
            .rjust(24) for k in (f'{t:.2f}'.replace('0.', '0p') for t in THRESHOLDS)))
    print(f'recorded val_pc_chamfer: {summary["recorded_val_pc_chamfer"]}')
    print(f'leak check: {leak}')
    print(f'wrote {csv_path} and {json_path}  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
