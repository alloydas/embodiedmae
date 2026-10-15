#!/usr/bin/env python3
"""1 camera vs 3 vs 10: fused point clouds, MAIZE, evaluation only.

The maize twin of eval/pc_view_eval.py, with the premise turned around. A sorghum
`_nc_cam.ply` is the COMPLETE plant in a camera's frame, so that study built
visibility masks to take points AWAY. A maize `pointcloud_cam.ply` is what ITS OWN
camera sees (tools/check_maize_cloud_views.py, 20 val plants: 2 % of a cloud's
points are hidden from the camera that made it, 25 % from another camera, and the
depth image back-projects to within 1 mm of the cloud), so every maize model in
this repo trained on single-camera clouds and no complete cloud is on disk. The
question is therefore what a model GAINS when the rig has more cameras:

  1view   view 00's cloud as on disk: the regime the model trained on
  3view   the clouds of views 00, 01 and 02 moved into camera 00's frame and fused
  full    all ten views' clouds fused, the most complete cloud available; also the
          ground truth the reconstructions are scored against

The maize views are random cameras (no elevation ladder), so views 01 and 02 are
"two more cameras", not a fixed rig. RGB and depth stay view 00's, as in the probe,
so only the cloud changes. Each condition's cloud goes through the loader's own
steps: subsample to num_points (8,192), centre on its own centroid, scale so the
farthest point lands on the unit sphere.

Per run and condition:
  probe   decision 6.4's ridge probe on the frozen CLS token, eval/linear_probe_maize.py's
          own code (text never visible, params zeroed, train-only standardiser, alpha
          by 5-fold CV). Two fits per condition:
            matched   fitted on train features of the same condition
            transfer  the probe fitted on the TRAINING regime (1view) applied to this
                      condition's features: a readout fitted on one camera, run on a
                      multi-camera rig
  recon   (val) chamfer between the model's reconstructed cloud and the ten-camera
          cloud in the input's normalised frame, both directions, next to the same
          distances for the input itself (the "echo the input" baseline). From one
          camera this asks whether the decoder completes what that camera did not see.
          It runs in the model's own masked regime (every visible modality token-masked
          at the run's mask_ratio); with nothing masked the decoder collapses.
  coverage  the fraction of the ten-camera cloud within 5 mm of the condition's cloud.

    python eval/pc_view_eval_maize.py --runs maize_e2_pc_d2 maize_e2_pcrgb_d2 \\
        maize_e2_pcrgbd_d2 maize_4m_d2 --out-prefix reports/pcview_maize_d2

A run given without ':<ckpt>' uses checkpoints/checkpoint_epoch_600.pth. Features
(and val reconstruction distances) are cached per (run, ckpt, split) under
outputs/_probe_cache_pcview_maize/, written atomically. Point subsets are seeded per
plant, so a rerun reproduces them and every run sees the same clouds.
"""
# Repo root on sys.path: this script lives in eval/ but imports top-level modules.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from embodied_mae_4m_maize import N_PARAMS, MAX_LEAVES
from maize_dataset_4m import MaizeDataset4M, PC_FILENAME

REPO = Path(__file__).resolve().parent.parent


def _import_file(modname, path):
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


LP = _import_file('linear_probe_maize', REPO / 'eval' / 'linear_probe_maize.py')

# condition -> the views whose clouds are fused (in camera 00's frame)
CONDITIONS = {'1view': (0,), '3view': (0, 1, 2), 'full': tuple(range(10))}
ANCHOR = '1view'              # the training regime: the transfer probe is fitted here
GT_COND = 'full'              # the reconstruction target
DEFAULT_CKPT = 'checkpoints/checkpoint_epoch_600.pth'
CACHE_DIR = REPO / 'outputs' / '_probe_cache_pcview_maize'
DATA_ROOT = LP.DATA_ROOT
COVER_TAU = 0.005             # 5 mm in the renderer's metres (clouds sit ~1-4 mm apart)


def _pose(folder):
    m = json.loads((folder / 'camera_pose.json').read_text())
    return (np.asarray(m['cameraToWorld'], dtype=np.float64).reshape(4, 4),
            np.asarray(m['worldToCamera'], dtype=np.float64).reshape(4, 4))


def _apply(M, P):
    return (M[:3, :3] @ P.T).T + M[:3, 3]


class PCViewSetMaize(Dataset):
    """View 00 of every plant, with its cloud under each CONDITION."""

    def __init__(self, data_root, split, num_points, img_size, max_leaves, seed,
                 limit=None, rgb_file='rgb.png', param_schema='v1'):
        self.base = MaizeDataset4M(data_root, img_size=img_size, num_points=num_points,
                                   split=split, max_leaves=max_leaves,
                                   view_sampling=True, deterministic_view=True,
                                   rgb_file=rgb_file, param_schema=param_schema)
        self.folders = []
        for views in self.base.plant_views:
            folder = self.base.samples[views[0]]
            if not folder.name.endswith('_00'):
                raise RuntimeError(f'{folder.name}: expected view 00 first')
            if len(views) != 10:
                raise RuntimeError(f'{folder.name}: {len(views)} views, expected 10')
            self.folders.append(folder)
        if limit:
            self.folders = self.folders[:limit]
        self.num_points, self.seed = num_points, seed

    def __len__(self):
        return len(self.folders)

    def _take(self, rng, n):
        """MaizeDataset4M.load_pointcloud's subsample-or-pad, with a given RNG."""
        N = self.num_points
        if n >= N:
            return rng.choice(n, N, replace=False)
        return np.concatenate([np.arange(n), rng.choice(n, N - n, replace=True)])

    def __getitem__(self, i):
        import open3d as o3d
        from scipy.spatial import cKDTree
        folder = self.folders[i]
        plant = MaizeDataset4M.plant_of(folder.name)
        rgb = self.base.load_rgb(folder)
        depth = self.base.load_depth(folder / 'depth.png')
        # every view's cloud, moved into camera 00's frame
        _, W2C0 = _pose(folder)
        clouds = []
        for v in range(10):
            f = folder.with_name(f'{plant}_{v:02d}')
            P = np.asarray(o3d.io.read_point_cloud(str(f / PC_FILENAME)).points, dtype=np.float64)
            if len(P) == 0:
                raise RuntimeError(f'{f}: empty cloud')
            if v:
                C2W, _ = _pose(f)
                P = _apply(W2C0, _apply(C2W, P))
            clouds.append(P)
        pid = int(plant.split('_')[-1])
        fused = {c: np.concatenate([clouds[v] for v in views]) for c, views in CONDITIONS.items()}
        # one ten-camera subsample per plant: the ground truth for every condition
        gt_raw = fused[GT_COND][self._take(np.random.default_rng([self.seed, pid, 99]), len(fused[GT_COND]))]
        out = {'plant': plant, 'rgb': rgb, 'depth': depth}
        for ci, (cond, views) in enumerate(CONDITIONS.items()):
            sel = fused[cond]
            x = sel[self._take(np.random.default_rng([self.seed, pid, ci]), len(sel))]
            c = x.mean(axis=0)
            x = x - c
            s = float(np.linalg.norm(x, axis=1).max()) or 1.0
            out[f'pc_{cond}'] = torch.from_numpy((x / s).astype(np.float32))
            out[f'gt_{cond}'] = torch.from_numpy(((gt_raw - c) / s).astype(np.float32))
            # coverage: how much of the ten-camera cloud this condition's cloud reaches
            d = cKDTree(sel).query(gt_raw, workers=1)[0]
            out[f'cov_{cond}'] = float((d <= COVER_TAU).mean())
        return out


def chamfer_both(a, b, chunk=4):
    """(B,N,3),(B,M,3) -> per-sample mean squared NN distance a->b and b->a."""
    fw, bw = [], []
    for i in range(0, a.shape[0], chunk):
        d = (torch.cdist(a[i:i + chunk], b[i:i + chunk]) ** 2).clamp_min(0)
        fw.append(d.min(dim=2).values.mean(dim=1))
        bw.append(d.min(dim=1).values.mean(dim=1))
    return torch.cat(fw), torch.cat(bw)


def parse_run(spec):
    run, _, ckpt = spec.partition(':')
    return run, (ckpt or DEFAULT_CKPT)


def run_data(run):
    """(rgb_file, param_schema) a run trained on, from its config.json."""
    cfg = json.loads((REPO / 'outputs' / run / 'config.json').read_text())
    return cfg.get('rgb_file', 'rgb.png'), cfg.get('param_schema', 'v1')


def cache_path(args, run, ckpt, split):
    tag = f'maize_{run}__{Path(ckpt).stem}__{split}__cls__pcview_v{"".join(f"{v:02d}" for v in CONDITIONS["3view"])}__seed{args.seed}'
    rgb_file, _ = run_data(run)
    if rgb_file != 'rgb.png':
        tag += f'__{Path(rgb_file).stem}'      # rgba: data version 2 images
    if args.limit:
        tag += f'__limit{args.limit}'
    return Path(args.cache_dir) / f'{tag}.npz'


@torch.no_grad()
def extract(runs, args, split, do_recon):
    """One pass over a split's data, every missing run and condition at once."""
    todo = [(r, c) for r, c in runs if not cache_path(args, r, c, split).exists() or args.refresh]
    if not todo:
        return
    models = {}
    for run, ckpt in todo:
        model, cfg, epoch = LP.build_model(REPO / 'outputs' / run, ckpt, args.device)
        mr = float(args.recon_mask_ratio if args.recon_mask_ratio is not None
                   else cfg.get('mask_ratio', 0.8))
        models[run] = (model, ckpt, 1 + int(cfg.get('max_leaves', MAX_LEAVES)), epoch, mr)
        print(f'  loaded {run} @ {ckpt} (epoch {epoch}, active {list(model.active_modalities)}, '
              f'recon source mask ratio {mr})')
    # One data pass serves every run, so they must read the same images (rule: a run
    # is fed its own data version).
    versions = {run_data(run) for run, _ in todo}
    if len(versions) != 1:
        raise SystemExit(f'runs in one pass read different data versions {sorted(versions)}: '
                         f'evaluate them separately')
    rgb_file, param_schema = versions.pop()
    ds = PCViewSetMaize(args.data_root, split, args.num_points, 224, MAX_LEAVES, args.seed,
                        args.limit, rgb_file=rgb_file, param_schema=param_schema)
    loader = DataLoader(LP._RetryTransientIO(ds), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=False)
    feats = {(r, c): [] for r in models for c in CONDITIONS}
    rec = {(r, c): [] for r in models for c in CONDITIONS}
    echo = {c: [] for c in CONDITIONS}
    plants, cov = [], {c: [] for c in CONDITIONS}
    dev = args.device
    t0 = time.time()
    for bi, b in enumerate(loader):
        rgb = b['rgb'].to(dev, non_blocking=True)
        depth = b['depth'].to(dev, non_blocking=True)
        B = rgb.shape[0]
        plants.extend(b['plant'])
        pcs = {c: b[f'pc_{c}'].to(dev, non_blocking=True) for c in CONDITIONS}
        gts = {c: b[f'gt_{c}'].to(dev, non_blocking=True) for c in CONDITIONS}
        for c in CONDITIONS:
            cov[c].extend(float(v) for v in b[f'cov_{c}'])
            if do_recon:
                fw, bw = chamfer_both(pcs[c], gts[c])
                echo[c].append(torch.stack([fw, bw], 1).cpu().numpy())
        for run, (model, _ckpt, n_text, _ep, mr) in models.items():
            params = torch.zeros(B, n_text, N_PARAMS, device=dev)
            visible = tuple(m for m in model.active_modalities if m != 'text')
            for c in CONDITIONS:
                # Same FPS start for every run and condition of a batch, so the
                # conditions differ only in the input cloud.
                torch.manual_seed(args.seed + 1000003 * bi)
                enc = model.forward_encoder_select(rgb, depth, pcs[c], params,
                                                   visible=visible, source_mask_ratio=0.0)
                feats[(run, c)].append(enc[0][:, 0].float().cpu().numpy())
                if do_recon and 'pc' in model.active_modalities:
                    torch.manual_seed(args.seed + 1000003 * bi + 1)
                    enc = model.forward_encoder_select(rgb, depth, pcs[c], params,
                                                       visible=visible, source_mask_ratio=mr)
                    pred_pc = model.forward_decoder(enc[0], *enc[5:9], *enc[9:13])[2]
                    fw, bw = chamfer_both(pred_pc.float(), gts[c])
                    rec[(run, c)].append(torch.stack([fw, bw], 1).cpu().numpy())
        if bi % 25 == 0:
            print(f'    {split}: {min((bi + 1) * args.batch_size, len(ds))}/{len(ds)}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)
    plants = np.asarray(plants, dtype=str)      # fixed-width unicode: loads without pickle
    for run, (model, ckpt, _n, epoch, _mr) in models.items():
        arrays = {'plants': plants, 'epoch': np.int64(epoch)}
        for c in CONDITIONS:
            arrays[f'feats_{c}'] = np.concatenate(feats[(run, c)])
            arrays[f'cov_{c}'] = np.asarray(cov[c])
            if do_recon and rec[(run, c)]:
                arrays[f'rec_{c}'] = np.concatenate(rec[(run, c)])
            if do_recon:
                arrays[f'echo_{c}'] = np.concatenate(echo[c])
        out = cache_path(args, run, ckpt, split)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + f'.tmp{os.getpid()}.npz')
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
        print(f'  💾 {out.name}')
    del models
    if dev.startswith('cuda'):
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help="run slug, or slug:ckpt")
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--num-points', type=int, default=8192)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--cache-dir', default=str(CACHE_DIR))
    ap.add_argument('--recon-mask-ratio', type=float, default=None,
                    help="token mask ratio for reconstruction (default: each run's own mask_ratio)")
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='first N plants per split (smoke)')
    ap.add_argument('--out-prefix', default=None, help='write <prefix>_{probe,recon,coverage}.csv')
    args = ap.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('⚠️  no CUDA, falling back to CPU (slow)')
        args.device = 'cpu'

    runs = [parse_run(s) for s in args.runs]
    for split in ('val', 'train', 'test'):
        print(f'\n== {split}')
        extract(runs, args, split, do_recon=(split == 'val'))

    tgt = LP.load_targets(args.data_root)
    data = {}
    for run, ckpt in runs:
        for split in ('train', 'val', 'test'):
            data[(run, split)] = np.load(cache_path(args, run, ckpt, split))

    cov_rows = []
    for split in ('train', 'val', 'test'):
        zz = data[(runs[0][0], split)]
        for c in CONDITIONS:
            v = zz[f'cov_{c}']
            cov_rows.append({'split': split, 'condition': c, 'n_plants': len(v), 'mean': v.mean(),
                             'std': v.std(), 'min': v.min(), 'p05': np.percentile(v, 5),
                             'median': np.median(v), 'max': v.max()})
    cov_df = pd.DataFrame(cov_rows)
    print(f'\nFraction of the ten-camera cloud within {COVER_TAU * 1000:.0f} mm of each condition\'s cloud:')
    print(cov_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    rec_rows = []
    for run, ckpt in runs:
        z = data[(run, 'val')]
        for c in CONDITIONS:
            e = z[f'echo_{c}']
            row = {'run': run, 'ckpt': ckpt, 'epoch': int(z['epoch']), 'condition': c,
                   'n': len(e), 'echo_chamfer': (e[:, 0] + e[:, 1]).mean(),
                   'echo_pred_to_gt': e[:, 0].mean(), 'echo_gt_to_pred': e[:, 1].mean()}
            if f'rec_{c}' in z:
                r = z[f'rec_{c}']
                row.update({'chamfer': (r[:, 0] + r[:, 1]).mean(),
                            'chamfer_median': np.median(r[:, 0] + r[:, 1]),
                            'pred_to_gt': r[:, 0].mean(), 'gt_to_pred': r[:, 1].mean()})
            rec_rows.append(row)
    rec_df = pd.DataFrame(rec_rows)
    print('\nReconstruction vs the TEN-CAMERA cloud, val (squared distances, unit-sphere units):')
    print(rec_df.drop(columns=['ckpt']).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

    probe_rows = []
    for run, ckpt in runs:
        tr, va, te = (data[(run, s)] for s in ('train', 'val', 'test'))
        y = {s: tgt.reindex(data[(run, s)]['plants']) for s in ('train', 'val', 'test')}
        for s, yy in y.items():
            assert not yy.index.isna().any() and not yy[[c for _, c, _, _ in LP.TARGETS]].isna().any().any(), s
        for name, src, prov, unit in LP.TARGETS:
            ytr = y['train'][src].to_numpy(np.float64)
            yv, yt = y['val'][src].to_numpy(np.float64), y['test'][src].to_numpy(np.float64)
            for c in CONDITIONS:
                res = LP.fit_probe(tr[f'feats_{c}'], ytr,
                                   {'val': (va[f'feats_{c}'], yv), 'test': (te[f'feats_{c}'], yt)},
                                   args.alphas)
                probe_rows.append({'run': run, 'ckpt': ckpt, 'epoch': int(va['epoch']),
                                   'condition': c, 'fit': 'matched', 'target': name,
                                   'source_col': src, 'provenance': prov, 'unit': unit, **res})
            evals = {}
            for c in CONDITIONS:
                evals[f'val_{c}'] = (va[f'feats_{c}'], yv)
                evals[f'test_{c}'] = (te[f'feats_{c}'], yt)
            res = LP.fit_probe(tr[f'feats_{ANCHOR}'], ytr, evals, args.alphas)
            for c in CONDITIONS:
                row = {'run': run, 'ckpt': ckpt, 'epoch': int(va['epoch']), 'condition': c,
                       'fit': 'transfer', 'target': name, 'source_col': src,
                       'provenance': prov, 'unit': unit,
                       'alpha': res['alpha'], 'cv_r2': res['cv_r2']}
                for s in ('val', 'test'):
                    for k in ('r2', 'rmse', 'mae', 'base_mae'):
                        row[f'{s}_{k}'] = res[f'{s}_{c}_{k}']
                probe_rows.append(row)
    probe_df = pd.DataFrame(probe_rows)
    piv = probe_df[probe_df.fit == 'matched'].pivot_table(
        index=['target'], columns=['run', 'condition'], values='val_r2')
    print('\nVal R2, probe fitted per condition (matched):')
    print(piv.to_string(float_format=lambda x: f'{x:.3f}'))
    piv = probe_df[probe_df.fit == 'transfer'].pivot_table(
        index=['target'], columns=['run', 'condition'], values='val_r2')
    print(f'\nVal R2, the {ANCHOR} probe applied to each condition (transfer):')
    print(piv.to_string(float_format=lambda x: f'{x:.3f}'))

    if args.out_prefix:
        pre = Path(args.out_prefix)
        pre.parent.mkdir(parents=True, exist_ok=True)
        cov_df.to_csv(f'{pre}_coverage.csv', index=False)
        rec_df.to_csv(f'{pre}_recon.csv', index=False)
        probe_df.to_csv(f'{pre}_probe.csv', index=False)
        print(f'\n📄 {pre}_{{coverage,recon,probe}}.csv')


if __name__ == '__main__':
    main()
