"""Point-cloud precision / recall / F1 of the masked reconstruction, sorghum.

For every plant of a split (view 00, one seeded subsample of its complete cloud as the input,
another as the ground truth, both in the input's centred unit-sphere frame) the model
reconstructs the cloud in its own regime (every visible modality token-masked at the run's
training mask ratio, the parameter stream zeroed and hidden: exactly eval/pc_view_eval.py's
'full' condition, which reproduces the published val Chamfer). Then, per plant and threshold tau:

  precision(tau) = fraction of PREDICTED points within tau of some ground-truth point
  recall(tau)    = fraction of GROUND-TRUTH points within tau of some predicted point
  F1(tau)        = 2 P R / (P + R)

tau is in the normalised frame; `scale_m` (metres per unit, the cloud's max radius) is written
per plant so tau can be read in metres (tau 0.01 is ~5-7 mm on a typical plant). Chamfer (both
directions, squared) is written alongside as the anchor to the published numbers.

  python eval/pc_fscore.py --runs e2_pc_d2 e2_pcrgbd_d2 4m_pretrain_15k_d2_sink:checkpoints/checkpoint_epoch_1000.pth \
      --splits val test --out-prefix reports/pcfscore_<job>

Writes <prefix>_items.csv (one row per run x split x plant) and <prefix>_summary.csv (means and
medians per run x split, plus the plant-level bootstrap CI of the F1 means).
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sorghum_dataset_4m import SorghumDataset4M            # noqa: E402
from embodied_mae_4m import N_PARAMS                       # noqa: E402


def _import_file(modname, path):
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


LP = _import_file('linear_probe', REPO / 'eval' / 'linear_probe.py')
DEFAULT_CKPT = 'checkpoints/checkpoint_epoch_600.pth'
TAUS = (0.005, 0.01, 0.02)


class View0Set(Dataset):
    """View 00 of every plant: RGB, depth, a seeded input subsample and a seeded GT subsample."""

    def __init__(self, data_root, split, num_points, seed, limit=None, rgb_file='rgb.png', rgb_gain=None):
        self.base = SorghumDataset4M(data_root, img_size=224, num_points=num_points, split=split,
                                     max_leaves=24, view_sampling=True, deterministic_view=True,
                                     rgb_file=rgb_file, rgb_gain=rgb_gain)
        self.folders = []
        for views in self.base.plant_views:
            folder = self.base.samples[views[0]]
            if not folder.name.endswith('_00'):
                raise RuntimeError(f'{folder.name}: expected view 00 first')
            self.folders.append(folder)
        if limit:
            self.folders = self.folders[:limit]
        self.num_points, self.seed = num_points, seed

    def __len__(self):
        return len(self.folders)

    def _take(self, rng, n):
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
        pts = np.asarray(o3d.io.read_point_cloud(str(self.base.find_pointcloud_file(folder))).points,
                         dtype=np.float64)
        n = len(pts)
        x = pts[self._take(np.random.default_rng([self.seed, plant, 0]), n)]
        gt = pts[self._take(np.random.default_rng([self.seed, plant, 99]), n)]
        c = x.mean(axis=0)
        x = x - c
        s = float(np.linalg.norm(x, axis=1).max()) or 1.0
        return {'plant': plant, 'rgb': rgb, 'depth': depth, 'scale_m': np.float32(s),
                'pc': torch.from_numpy((x / s).astype(np.float32)),
                'gt': torch.from_numpy(((gt - c) / s).astype(np.float32))}


@torch.no_grad()
def prf(pred, gt, taus):
    """(B,N,3),(B,M,3) -> dict of per-sample arrays: chamfer fw/bw and P/R/F1 per tau."""
    out = {f'{k}_{t:g}': [] for t in taus for k in ('precision', 'recall', 'f1')}
    out['chamfer_pred_to_gt'], out['chamfer_gt_to_pred'] = [], []
    for b in range(pred.shape[0]):
        d = torch.cdist(pred[b], gt[b])                       # (N, M)
        dp = d.min(dim=1).values                               # prediction -> nearest GT
        dg = d.min(dim=0).values                               # GT -> nearest prediction
        out['chamfer_pred_to_gt'].append(float((dp ** 2).mean()))
        out['chamfer_gt_to_pred'].append(float((dg ** 2).mean()))
        for t in taus:
            p = float((dp <= t).float().mean()); r = float((dg <= t).float().mean())
            out[f'precision_{t:g}'].append(p); out[f'recall_{t:g}'].append(r)
            out[f'f1_{t:g}'].append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        del d
    return {k: np.asarray(v) for k, v in out.items()}


def parse_run(spec):
    run, _, ckpt = spec.partition(':')
    return run, (ckpt or DEFAULT_CKPT)


def run_rgb(run):
    cfg = json.loads((REPO / 'outputs' / run / 'config.json').read_text())
    return cfg.get('rgb_file', 'rgb.png'), LP.run_rgb_gain(cfg)


@torch.no_grad()
def evaluate(runs, args, split):
    rows = []
    groups = {}
    for run, ckpt in runs:
        groups.setdefault(run_rgb(run), []).append((run, ckpt))
    for (rgb_file, gain), members in groups.items():
        models = {}
        for run, ckpt in members:
            model, cfg, epoch = LP.build_model(REPO / 'outputs' / run, ckpt, args.device)
            mr = float(args.mask_ratio if args.mask_ratio is not None else cfg.get('mask_ratio', 0.8))
            models[run] = (model, ckpt, 1 + int(cfg.get('max_leaves', 24)), epoch, mr)
            print(f'  loaded {run} @ {ckpt} (epoch {epoch}, active {list(model.active_modalities)}, '
                  f'mask ratio {mr}, rgb {rgb_file} gain {gain:g})', flush=True)
        ds = View0Set(args.data_root, split, args.num_points, args.seed, args.limit, rgb_file, gain)
        loader = DataLoader(LP._RetryTransientIO(ds), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.device != 'cpu')
        dev = args.device; t0 = time.time()
        for bi, b in enumerate(loader):
            rgb = b['rgb'].to(dev); depth = b['depth'].to(dev); pc = b['pc'].to(dev); gt = b['gt'].to(dev)
            B = rgb.shape[0]
            for run, (model, ckpt, n_text, epoch, mr) in models.items():
                if 'pc' not in model.active_modalities:
                    continue
                params = torch.zeros(B, n_text, N_PARAMS, device=dev)
                visible = tuple(m for m in model.active_modalities if m != 'text')
                torch.manual_seed(args.seed + 1000003 * bi + 1)      # as pc_view_eval: same token mask per batch
                enc = model.forward_encoder_select(rgb, depth, pc, params, visible=visible, source_mask_ratio=mr)
                pred = model.forward_decoder(enc[0], *enc[5:9], *enc[9:13])[2].float()
                m = prf(pred, gt, TAUS)
                for k in range(B):
                    rows.append({'run': run, 'ckpt': ckpt, 'epoch': epoch, 'split': split,
                                 'plant': int(b['plant'][k]), 'scale_m': float(b['scale_m'][k]),
                                 **{key: float(v[k]) for key, v in m.items()}})
            if bi % 20 == 0:
                print(f'    {split} [{rgb_file} g{gain:g}]: {min((bi + 1) * args.batch_size, len(ds))}/{len(ds)}'
                      f'  ({time.time() - t0:.0f}s)', flush=True)
    return rows


def summarise(rows, seed=0, n_boot=2000):
    import pandas as pd
    df = pd.DataFrame(rows)
    df['chamfer'] = df.chamfer_pred_to_gt + df.chamfer_gt_to_pred
    metrics = ['chamfer'] + [f'{k}_{t:g}' for t in TAUS for k in ('precision', 'recall', 'f1')]
    rng = np.random.default_rng(seed)
    out = []
    for (run, split), g in df.groupby(['run', 'split']):
        rec = {'run': run, 'epoch': int(g.epoch.iloc[0]), 'split': split, 'n_plants': len(g),
               'scale_m_median': float(g.scale_m.median())}
        for m in metrics:
            v = g[m].values
            rec[f'{m}_mean'] = float(v.mean()); rec[f'{m}_median'] = float(np.median(v))
        for t in TAUS:
            v = g[f'f1_{t:g}'].values
            boots = np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)])
            rec[f'f1_{t:g}_lo'], rec[f'f1_{t:g}_hi'] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        out.append(rec)
    return pd.DataFrame(out), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='run slug, or slug:ckpt')
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    ap.add_argument('--data-root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--num-points', type=int, default=8196)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mask-ratio', type=float, default=None, help="token mask ratio (default: each run's own)")
    ap.add_argument('--limit', type=int, default=None, help='first N plants per split (smoke)')
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()
    runs = [parse_run(s) for s in args.runs]
    rows = []
    for split in args.splits:
        print(f'== {split} ==', flush=True)
        rows += evaluate(runs, args, split)
    summ, items = summarise(rows, args.seed)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    items.to_csv(f'{args.out_prefix}_items.csv', index=False)
    summ.to_csv(f'{args.out_prefix}_summary.csv', index=False)
    cols = ['run', 'split', 'n_plants', 'chamfer_mean'] + [f'{k}_0.01_mean' for k in ('precision', 'recall', 'f1')] + ['f1_0.02_mean']
    print(summ[cols].round(4).to_string(index=False))
    print(f'wrote {args.out_prefix}_items.csv and _summary.csv')


if __name__ == '__main__':
    main()
