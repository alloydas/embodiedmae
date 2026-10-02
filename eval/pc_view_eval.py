#!/usr/bin/env python3
"""1-view vs 3-view point clouds, sorghum, evaluation only.

Every model in this repo trained on the COMPLETE plant cloud. This asks what each
one loses when its point-cloud input is what a camera rig would actually capture:
the points visible from camera 00 alone (1 view), or from cameras 00, 01 and 02
together (3 views). The three cameras sit at azimuths of about 0, +88 and -138 deg,
all above the plant, looking down at 64, 44 and 30 deg. Visibility is exact
occlusion against the plant's source mesh, built by eval/pc_view_masks.py; run that
first. One camera sees ~48 % of a plant's points and three see ~77 %.

RGB and depth are camera 00's, as in the probe, so only the point cloud changes.
A partial cloud goes through the loader's own steps: subsample (or pad) to
num_points, centre on its own centroid, and scale so its farthest point lands on
the unit sphere. That is what a model would be handed in deployment.

Per run and condition (full / 3view / 1view):

  probe  decision 6.4's ridge probe on the frozen CLS token, using
         eval/linear_probe.py's own code: text never visible and params zeroed,
         standardiser on train only, alpha by 5-fold CV on train. Two fits:
           matched   fitted on train features of the same condition
           transfer  the full-cloud probe applied to this condition's features,
                     i.e. a probe trained on complete clouds and then run on scans
  recon  (val) the chamfer distance between the model's reconstructed cloud and the
         complete ground-truth cloud in the
         input's normalised frame, in both directions. Shown next to the same two
         distances for the partial input itself, the "echo the input" baseline.
         Each decoded patch is tied to one input token, so this asks whether
         the decoder adds anything the cameras did not see. The reconstruction
         runs in the model's own masked regime: each visible (non-text) modality
         is token-masked at the run's training mask_ratio (0.8), which reproduces
         the published val chamfer on the full cloud. With nothing masked the
         decoder is out of distribution and its cloud collapses.

Point subsets are seeded per plant, not per worker, so every run sees the same
clouds and a rerun reproduces them exactly. The full condition is re-extracted
here instead of being read from outputs/_probe_cache: those caches used
worker-seeded subsets, and one comparison should not mix the two schemes.

    python eval/pc_view_eval.py --runs e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt \\
        4m_pretrain_15k_v2_depthfix_qal:checkpoints/checkpoint_epoch_1000.pth \\
        --out-prefix reports/pcview

A run given without ':<ckpt>' uses checkpoints/checkpoint_epoch_600.pth, the E2
matched cut. Features (and val reconstruction distances) are cached per
(run, ckpt, split) under outputs/_probe_cache_pcview/, written atomically.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from embodied_mae_4m import N_PARAMS
from sorghum_dataset_4m import SorghumDataset4M

REPO = Path(__file__).resolve().parent.parent


def _import_file(modname, path):
    """Import eval/<file> by path (eval/ is not a package), reusing a loaded copy."""
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


LP = _import_file('linear_probe', REPO / 'eval' / 'linear_probe.py')
PM = _import_file('pc_view_masks', REPO / 'eval' / 'pc_view_masks.py')

# condition -> camera views whose visible points are kept (None = every point)
CONDITIONS = {'full': None, '3view': (0, 1, 2), '1view': (0,)}
MASK_VIEWS = (0, 1, 2)
DEFAULT_CKPT = 'checkpoints/checkpoint_epoch_600.pth'


class PCViewSet(Dataset):
    """View 00 of every plant, with its cloud under each CONDITION."""

    def __init__(self, data_root, split, masks, num_points, img_size, max_leaves, seed,
                 limit=None, rgb_file='rgb.png'):
        self.base = SorghumDataset4M(data_root, img_size=img_size, num_points=num_points,
                                     split=split, max_leaves=max_leaves,
                                     view_sampling=True, deterministic_view=True,
                                     rgb_file=rgb_file)
        self.folders = []
        for views in self.base.plant_views:
            folder = self.base.samples[views[0]]
            if not folder.name.endswith('_00'):
                raise RuntimeError(f'{folder.name}: expected view 00 first')
            self.folders.append(folder)
        has = [f for f in self.folders if int(f.name.split('_')[1]) in masks]
        if limit:
            self.folders = has[:limit]
        elif len(has) != len(self.folders):
            raise RuntimeError(f'{split}: {len(self.folders) - len(has)} of {len(self.folders)} '
                               f'plants have no visibility mask; rerun eval/pc_view_masks.py')
        self.masks, self.num_points, self.seed = masks, num_points, seed

    def __len__(self):
        return len(self.folders)

    def _take(self, rng, n):
        """sorghum_dataset.load_pointcloud's subsample-or-pad, with a given RNG."""
        N = self.num_points
        if n >= N:
            return rng.choice(n, N, replace=False)
        return np.concatenate([np.arange(n), rng.choice(n, N - n, replace=True)])

    def __getitem__(self, i):
        import open3d as o3d
        folder = self.folders[i]
        plant = int(folder.name.split('_')[1])
        rgb = self.base.load_rgb(folder)
        depth = self.base.load_depth(folder / 'depth.png')
        pts = np.asarray(o3d.io.read_point_cloud(
            str(self.base.find_pointcloud_file(folder))).points, dtype=np.float64)
        n = len(pts)
        if n != self.masks.n_points(plant):
            raise RuntimeError(f'{folder.name}: cloud has {n} points, mask {self.masks.n_points(plant)}')
        # one complete-cloud subsample per plant, the ground truth for every condition
        gt = pts[self._take(np.random.default_rng([self.seed, plant, 99]), n)]
        out = {'plant': plant, 'rgb': rgb, 'depth': depth}
        for ci, (cond, views) in enumerate(CONDITIONS.items()):
            sel = pts if views is None else pts[self.masks.get(plant, views)]
            if len(sel) == 0:
                raise RuntimeError(f'{folder.name}: no point visible under {cond}')
            x = sel[self._take(np.random.default_rng([self.seed, plant, ci]), len(sel))]
            c = x.mean(axis=0)
            x = x - c
            s = float(np.linalg.norm(x, axis=1).max()) or 1.0
            out[f'pc_{cond}'] = torch.from_numpy((x / s).astype(np.float32))
            out[f'gt_{cond}'] = torch.from_numpy(((gt - c) / s).astype(np.float32))
            out[f'cov_{cond}'] = len(sel) / n
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


def cache_path(args, run, ckpt, split):
    tag = f'{run}__{Path(ckpt).stem}__{split}__cls__pcview_v{"".join(f"{v:02d}" for v in MASK_VIEWS)}__seed{args.seed}'
    if args.limit:
        tag += f'__limit{args.limit}'
    return Path(args.cache_dir) / f'{tag}.npz'


@torch.no_grad()
def extract(runs, args, split, do_recon):
    """One pass over a split's data, every missing run and condition at once."""
    todo = [(r, c) for r, c in runs if not cache_path(args, r, c, split).exists() or args.refresh]
    if not todo:
        return
    masks = PM.Masks(split, MASK_VIEWS, args.mask_dir)
    models = {}
    for run, ckpt in todo:
        model, cfg, epoch = LP.build_model(REPO / 'outputs' / run, ckpt, args.device)
        mr = float(args.recon_mask_ratio if args.recon_mask_ratio is not None
                   else cfg.get('mask_ratio', 0.8))
        models[run] = (model, ckpt, 1 + int(cfg.get('max_leaves', 24)), epoch, mr)
        print(f'  loaded {run} @ {ckpt} (epoch {epoch}, active {list(model.active_modalities)}, '
              f'recon source mask ratio {mr})')
    # One data pass serves every run, so they must have trained on the same images
    # (runs before 2026-10-01 have no rgb_file in config.json: rgb.png).
    rgb_files = {json.loads((REPO / 'outputs' / run / 'config.json').read_text()).get('rgb_file', 'rgb.png')
                 for run, _ in todo}
    if len(rgb_files) != 1:
        raise SystemExit(f'runs in one pass read different RGB files {sorted(rgb_files)}: '
                         f'evaluate them separately')
    ds = PCViewSet(args.data_root, split, masks, args.num_points, 224, 24, args.seed, args.limit,
                   rgb_file=rgb_files.pop())
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
        plants.extend(int(p) for p in b['plant'])
        pcs = {c: b[f'pc_{c}'].to(dev, non_blocking=True) for c in CONDITIONS}
        gts = {c: b[f'gt_{c}'].to(dev, non_blocking=True) for c in CONDITIONS}
        for c in CONDITIONS:
            cov[c].extend(float(v) for v in b[f'cov_{c}'])
            if do_recon:
                fw, bw = chamfer_both(pcs[c], gts[c])
                echo[c].append(torch.stack([fw, bw], 1).cpu().numpy())
        for run, (model, _ckpt, n_text, _ep, _mr) in models.items():
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
                    # Reconstruction in the model's own regime: every visible
                    # modality token-masked at the run's training mask ratio. With
                    # nothing masked the decoder is far outside what it trained on
                    # and its cloud collapses (chamfer ~15x the published val
                    # number, checked on e2_pc / e2_pcrgbdt). Same seed per batch,
                    # so every condition masks the same token positions.
                    torch.manual_seed(args.seed + 1000003 * bi + 1)
                    enc = model.forward_encoder_select(rgb, depth, pcs[c], params,
                                                       visible=visible,
                                                       source_mask_ratio=models[run][4])
                    pred_pc = model.forward_decoder(enc[0], *enc[5:9], *enc[9:13])[2]
                    fw, bw = chamfer_both(pred_pc.float(), gts[c])
                    rec[(run, c)].append(torch.stack([fw, bw], 1).cpu().numpy())
        if bi % 25 == 0:
            print(f'    {split}: {min((bi + 1) * args.batch_size, len(ds))}/{len(ds)}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)
    plants = np.asarray(plants, dtype=np.int64)
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
    ap.add_argument('--data-root', default=PM.DATA_ROOT)
    ap.add_argument('--num-points', type=int, default=8196)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--cache-dir', default=str(REPO / 'outputs' / '_probe_cache_pcview'))
    ap.add_argument('--mask-dir', default=str(PM.OUT_DIR))
    ap.add_argument('--recon-mask-ratio', type=float, default=None,
                    help='token mask ratio for reconstruction (default: each run\'s own mask_ratio)')
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

    # coverage (identical for every run: it is a property of the data)
    cov_rows = []
    z = data[(runs[0][0], 'val')]
    for split in ('train', 'val', 'test'):
        zz = data[(runs[0][0], split)]
        for c in CONDITIONS:
            v = zz[f'cov_{c}']
            cov_rows.append({'split': split, 'condition': c, 'n_plants': len(v), 'mean': v.mean(),
                             'std': v.std(), 'min': v.min(), 'p05': np.percentile(v, 5),
                             'median': np.median(v), 'max': v.max()})
    cov_df = pd.DataFrame(cov_rows)
    print('\nFraction of each plant cloud visible:')
    print(cov_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # reconstruction on val
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
    print('\nReconstruction vs the COMPLETE cloud, val (squared distances, unit-sphere units):')
    print(rec_df.drop(columns=['ckpt']).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

    # the probe
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
            res = LP.fit_probe(tr['feats_full'], ytr, evals, args.alphas)
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
    print('\nVal R2, full-cloud probe applied to each condition (transfer):')
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
