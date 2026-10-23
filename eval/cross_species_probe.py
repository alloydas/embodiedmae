"""Z1 (PLAN_cross_species.md): one species' frozen encoder probed on the OTHER species' plants.

The model never sees the recipe stream (zeros of its own width, text never visible), RGB is
served as the data species ships it, clouds are already unit-sphere, and DEPTH is re-encoded
from the data species' render convention into the model's training convention
(species_conventions.py) -- fed raw it would be ~10x out of range. The ridge fit, splits and
targets are the data species' own probe protocol, so a cross row is directly comparable to
the native probe rows and to the species-agnostic E8 baselines (DINOv2 etc.).

  python eval/cross_species_probe.py --model-run e2_pcrgbd_d2 --data-species maize --out reports/xprobe_<job>.csv
  python eval/cross_species_probe.py --model-run maize_4m:checkpoints/checkpoint_epoch_600.pth \
      --data-species sorghum --out reports/xprobe_<job>.csv

Caveats to carry into any table: a sorghum model on maize also crosses the CLOUD CONTENT gap
(complete plant vs single-camera capture), and on Nova the maize images are data v1 (rgb.png,
background kept) while sorghum models trained background-free. Run with --data-species = the
model's own species to anchor the harness against the native probe.
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import species_conventions as SC                      # noqa: E402
from sorghum_dataset_4m import SorghumDataset4M       # noqa: E402
from maize_dataset_4m import MaizeDataset4M           # noqa: E402


def _import_file(modname, path):
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


LP = _import_file('linear_probe', REPO / 'eval' / 'linear_probe.py')
LPM = _import_file('linear_probe_maize', REPO / 'eval' / 'linear_probe_maize.py')
MOD = {'sorghum': LP, 'maize': LPM}
N_PARAMS = {'sorghum': 9, 'maize': 14}
DEFAULT_ROOT = {'sorghum': LP.DATA_ROOT, 'maize': LPM.DATA_ROOT}
DEFAULT_CKPT = 'checkpoints/checkpoint_epoch_600.pth'


def infer_model_species(run, cfg):
    if 'maize' in Path(run).name or 'param_schema' in cfg:
        return 'maize'
    return 'sorghum'


def make_dataset(species, root, split, rgb_file, rgb_gain):
    if species == 'sorghum':
        return SorghumDataset4M(root, img_size=224, num_points=8196, split=split, max_leaves=24,
                                view_sampling=True, deterministic_view=True,
                                rgb_file=rgb_file or 'rgb_nobg.png', rgb_gain=rgb_gain)
    return MaizeDataset4M(root, img_size=224, num_points=8192, split=split,
                          view_sampling=True, deterministic_view=True,
                          rgb_file=rgb_file or 'rgb.png',
                          param_schema='v2' if (rgb_file or '') == 'rgba.png' else 'v1')


def plant_key(species, name):
    if species == 'sorghum':
        return int(name.rsplit('_', 1)[0].split('_')[-1])
    return MaizeDataset4M.plant_of(name)


@torch.no_grad()
def extract(model, cfg, a, model_species, split):
    tag = (f'{Path(a.run_dir).name}__{Path(a.ckpt).stem}__on_{a.data_species}'
           f'__{split}__{a.feature}__seed{a.seed}__rep{a.repeats}'
           + (f'__limit{a.limit}' if a.limit else '') + (('__drop-' + '-'.join(sorted(a.drop))) if a.drop else ''))
    cache = Path(a.cache_dir) / f'{tag}.npz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not a.refresh:
        z = np.load(cache, allow_pickle=True)
        print(f'  ⚡ cache hit {cache.name} ({z["feats"].shape})')
        return list(z['plants']), z['feats']

    ds = make_dataset(a.data_species, a.data_root, split, a.rgb_file, a.rgb_gain)
    if a.limit:
        # one deterministic view per plant: truncating plant_views truncates plants
        ds.plant_views = ds.plant_views[:a.limit]
    loader = DataLoader(LP._RetryTransientIO(ds), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=a.device.startswith('cuda'),
                        persistent_workers=False, worker_init_fn=LP._worker_init)
    cross = model_species != a.data_species
    enc = SC.depth_encoder(SC.model_depth_convention(model_species, cfg)) if cross else None
    visible = tuple(m for m in model.active_modalities if m != 'text' and m not in a.drop)
    assert visible, 'no non-text modality active'
    n_text = 1 + int(cfg.get('max_leaves', {'sorghum': 24, 'maize': 28}[model_species]))

    feats, plants = [], []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        rgb, depth, pc, _params, _valid, names = batch
        if cross:
            for i, name in enumerate(names):
                if a.data_species == 'sorghum':
                    dec = SC.depth_decoder('sorghum_std')
                else:
                    dec = SC.depth_decoder('maize_std', Path(a.data_root) / split / name)
                depth[i] = SC.reencode_depth(depth[i], dec, enc)
        rgb, depth, pc = (t.to(a.device, non_blocking=True) for t in (rgb, depth, pc))
        params = torch.zeros(rgb.shape[0], n_text, N_PARAMS[model_species], device=a.device)
        acc = None
        for r in range(a.repeats):
            torch.manual_seed(a.seed + 1000003 * bi + r)
            latent = model.forward_encoder_select(rgb, depth, pc, params,
                                                  visible=visible, source_mask_ratio=0.0)[0]
            f = latent[:, 0] if a.feature == 'cls' else latent[:, 1:].mean(1)
            acc = f if acc is None else acc + f
        feats.append((acc / a.repeats).float().cpu().numpy())
        plants.extend(plant_key(a.data_species, n) for n in names)
        if bi % 25 == 0:
            print(f'    {split}: {min((bi + 1) * a.batch_size, len(ds))}/{len(ds)} ({time.time() - t0:.0f}s)', flush=True)
    feats = np.concatenate(feats, 0)
    arr = np.asarray(plants, dtype=np.int64 if a.data_species == 'sorghum' else object)
    tmp = cache.parent / f'{cache.stem}.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, plants=arr, feats=feats)
    os.replace(tmp, cache)
    return plants, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-run', nargs='+', required=True, help='run slug under outputs/, or slug:ckpt')
    ap.add_argument('--model-species', choices=('sorghum', 'maize'), default=None,
                    help='default: inferred from the run name / config')
    ap.add_argument('--data-species', choices=('sorghum', 'maize'), required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--rgb-file', default=None, help='data-side RGB file (default per species)')
    ap.add_argument('--rgb-gain', type=float, default=None, help='sorghum data only; None = as on disk')
    ap.add_argument('--feature', choices=('cls', 'mean'), default='cls')
    ap.add_argument('--drop', nargs='*', default=[], help='modalities to hide at probe time, e.g. depth (leak / ablation checks)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--limit', type=int, default=None, help='first N plants per split (smoke)')
    ap.add_argument('--cache-dir', default=str(REPO / 'outputs' / '_probe_cache_xspecies'))
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.data_root = a.data_root or DEFAULT_ROOT[a.data_species]
    dmod = MOD[a.data_species]
    tgt = dmod.load_targets(a.data_root)
    rows = []
    for spec in a.model_run:
        run, _, ckpt = spec.partition(':')
        a.ckpt = ckpt or DEFAULT_CKPT
        a.run_dir = str(REPO / 'outputs' / run)
        cfg = json.loads((Path(a.run_dir) / 'config.json').read_text())
        ms = a.model_species or infer_model_species(run, cfg)
        model, cfg, epoch = MOD[ms].build_model(a.run_dir, a.ckpt, a.device)
        print(f'=== {run} ({ms} model, epoch {epoch}) on {a.data_species} data '
              f'{"(CROSS: depth re-encoded to " + SC.model_depth_convention(ms, cfg) + ")" if ms != a.data_species else "(anchor)"} ===')
        data = {}
        for split in ('train', 'val', 'test'):
            plants, feats = extract(model, cfg, a, ms, split)
            y = tgt.reindex(plants)
            keep = ~y.index.isna()
            data[split] = (feats[np.asarray(keep)], y[keep])
        del model
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
        for name, src, prov, unit in dmod.TARGETS:
            ytr = data['train'][1][src].to_numpy(dtype=np.float64)
            evals = {s: (data[s][0], data[s][1][src].to_numpy(dtype=np.float64)) for s in data}
            res = LP.fit_probe(data['train'][0], ytr, evals, a.alphas)
            rows.append({'run': run, 'model_species': ms, 'data_species': a.data_species,
                         'epoch': epoch, 'feature': a.feature, 'target': name, 'source_col': src,
                         'provenance': prov, 'unit': unit, **res})
            print(f'  {name:16s} val R2 {res["val_r2"]:+.4f}  test R2 {res["test_r2"]:+.4f}  '
                  f'val MAE {res["val_mae"]:.4g} vs {res["val_base_mae"]:.4g} base [{unit}]')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f'wrote {a.out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
