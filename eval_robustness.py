"""How far does each model hold up as the input corruption gets worse?

The occluded validation pass inside training uses one fixed reference
corruption (1x noise, 9-16 leaves). That answers "is this model robust at the
strength we trained with", not "which model wins when the input gets worse".
This sweeps the strength on a finished checkpoint and prints one table per
metric, so two models' curves can be checked for a crossover.

Every condition uses the same fixed subset and the same eval seed, so the
leaves land in the same places for every model and every strength level.

Report on TEST, not val: best_model.pth was chosen on val loss, so val numbers
carry that selection. `--split test` scores one view per test plant (2 250
plants, a fixed seed picks which view) -- ten times the 225-view val subset,
with no plant counted twice.

    python eval_robustness.py --split test --runs paper_maskoff_s1 paper_maskboth_s1
    python eval_robustness.py --runs paper_maskoff_s1 --levels clean 1x noise2x
"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from plant_occlusion import ProceduralOcclusion, StructuredMaskConfig
from sorghum_dataset_4m import SorghumDataset4M
from train_sorghum_4m import build_model, evaluate, fixed_subset
from view_sampler import plant_groups

# name -> (noise multiplier, leaf-count multiplier); None = clean input
LEVELS = {'clean': None, '1x': (1.0, 1.0), 'noise2x': (2.0, 1.0), 'noise4x': (4.0, 1.0),
          'leaves2x': (1.0, 2.0), 'leaves2x_noise2x': (2.0, 2.0),
          'blobmask': None, 'blobmask_1x': (1.0, 1.0),
          'rgbd2pc': None, 'rgb2pc': None}

# Cross-modal 3D reconstruction: the point cloud is (all but one token) hidden
# and must be rebuilt from the images. Structured masking hides the same region
# in every modality, so it trains exactly this -- filling one modality's hole
# from the others -- while uniform masking almost never leaves a modality empty.
# Params are zeroed for these levels (they carry height and leaf count, i.e.
# the answer), so nothing but RGB (and depth) informs the reconstruction.
FORCE_VISIBLE = {'rgbd2pc': {'pc': 1, 'text': 1, 'rgb': 196, 'depth': 196},
                 'rgb2pc':  {'pc': 1, 'text': 1, 'depth': 1, 'rgb': 196}}


class _ZeroParams:
    def __init__(self, loader): self.loader = loader
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            batch = list(batch); batch[3] = torch.zeros_like(batch[3]); yield batch

# Levels whose TOKEN mask is the structured blob mask instead of the uniform one.
# Every other evaluation masks tokens uniformly -- the regime a maskoff model
# trained on -- so it cannot show what structured masking is for: filling large
# contiguous holes. These levels impose the same blob geometry on every model
# (maskoff included), from the same eval seed, so each sees identical holes.
BLOB_LEVELS = {'blobmask', 'blobmask_1x'}
BLOB_MASK = StructuredMaskConfig.from_dict({
    'enabled': True, 'prob': 1.0, 'center_bias': 1.5, 'length_scale': 0.20,
    'modalities': ['rgb', 'depth', 'pc'], 'shared_field': True, 'apply_in_eval': True})
REPORT = [('pc_chamfer', 'PC Chamfer', '{:.5f}'), ('pc_f1@0.03', 'F1@0.03', '{:.4f}'),
          ('pc_recall@0.03', 'recall@0.03', '{:.4f}'), ('pc_precision@0.03', 'prec@0.03', '{:.4f}'),
          ('pc_f1@0.01', 'F1@0.01', '{:.4f}'), ('rgb_mse', 'RGB MSE', '{:.4f}'),
          ('depth_mse', 'Depth MSE', '{:.4f}')]


def occluder_for(args, level):
    """The run's own occlusion block, pinned to the shared eval reference, then
    scaled. None for the clean pass."""
    if LEVELS[level] is None:
        return None
    noise_mul, leaf_mul = LEVELS[level]
    cfg = dict(args.occlusion or {})
    cfg.update(getattr(args, 'eval_occlusion', None) or {})
    cfg['enabled'] = True
    cfg['prob'] = 1.0
    for key in ('rgb_noise_std', 'depth_noise_std', 'pc_noise_std'):
        if cfg.get(key):
            cfg[key] = float(cfg[key]) * noise_mul
    lo, hi = cfg.get('num_leaves', [9, 16])
    cfg['num_leaves'] = [int(round(lo * leaf_mul)), int(round(hi * leaf_mul))]
    return ProceduralOcclusion.from_config(cfg)


def one_view_per_plant(ds, seed):
    """Subset with exactly one view of every plant; the view is fixed by seed."""
    g = torch.Generator().manual_seed(seed)
    keep = [idx[int(torch.randint(len(idx), (1,), generator=g))]
            for _, idx in plant_groups(ds)]
    return torch.utils.data.Subset(ds, sorted(keep))


def load_run(run, device, which='last'):
    """`which='best'` loads best_model.pth, which was picked on CLEAN val loss --
    a selection that favours clean accuracy and so tilts a robustness comparison
    against the structured arms. `which='last'` (default) loads the newest epoch
    checkpoint: the end of the cosine schedule, the same rule for every arm."""
    out = Path('outputs') / run
    args = Namespace(**json.loads((out / 'config.json').read_text()))
    if which == 'best':
        ckpt_path = out / 'best_model.pth'
    else:
        ckpts = sorted((out / 'checkpoints').glob('checkpoint_epoch_*.pth'),
                       key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict({k.replace('module.', '', 1): v
                           for k, v in ckpt['model_state_dict'].items()})
    model.eval()
    return args, model, ckpt.get('epoch', '?')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='run directory names under outputs/')
    ap.add_argument('--levels', nargs='+', default=list(LEVELS), choices=list(LEVELS))
    # (a structured model's own training mask is train-only, so evaluation must
    #  set the mask explicitly for every level -- uniform unless a blob level)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--num_samples', type=int, default=None,
                    help='val: default = the run\'s num_val_samples (225); '
                         'test: default = one view per plant (all plants)')
    ap.add_argument('--checkpoint', default='last', choices=['last', 'best'],
                    help='last (default): newest epoch checkpoint, one rule for all arms; '
                         'best: best_model.pth, selected on clean val loss')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--output', default='reports/robustness.json')
    cli = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    base_ds = None          # scanned once; every run shares the same val split
    for run in cli.runs:
        args, model, epoch = load_run(run, device, cli.checkpoint)
        if base_ds is None:
            base_ds = SorghumDataset4M(args.data_root, split=cli.split, img_size=args.img_size,
                                       num_points=args.num_points, max_leaves=args.max_leaves)
        # Only __getitem__ depends on these two, so switch them per run.
        base_ds.param_encoding = getattr(args, 'param_encoding', 'v1')
        base_ds.geometry_cond = bool(getattr(args, 'geometry_cond', False))
        if cli.split == 'test' and cli.num_samples is None:
            ds = one_view_per_plant(base_ds, args.val_mask_seed)
        else:
            ds = fixed_subset(base_ds, cli.num_samples or args.num_val_samples,
                              args.val_mask_seed)
        loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False,
                            num_workers=cli.num_workers, pin_memory=True)
        print(f"\n=== {run} ({cli.checkpoint} checkpoint, epoch {epoch}, {len(ds)} {cli.split} samples) ===",
              flush=True)
        results[run] = {}
        own_mask = model.structured_mask
        for level in cli.levels:
            model.structured_mask = BLOB_MASK if level in BLOB_LEVELS else None
            model.force_visible = FORCE_VISIBLE.get(level)
            lvl_loader = _ZeroParams(loader) if level in FORCE_VISIBLE else loader
            _, m = evaluate(model, lvl_loader, device, mask_ratio=args.mask_ratio,
                            val_mask_seed=args.val_mask_seed,
                            pc_metric_thresholds=args.pc_metric_thresholds,
                            pc_metric_chunk_size=args.pc_metric_chunk_size,
                            occlusion=occluder_for(args, level))
            model.structured_mask = own_mask
            model.force_visible = None
            results[run][level] = {k: float(m[k]) for k, _, _ in REPORT if k in m}
            print(f"  {level:18s}" + '  '.join(
                f"{name} {fmt.format(m[key])}" for key, name, fmt in REPORT if key in m), flush=True)

    Path(cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.output).write_text(json.dumps(results, indent=1))
    for key, name, fmt in REPORT:
        print(f"\n### {name} by corruption strength")
        print(f"{'run':24s}" + ''.join(f"{lv:>18s}" for lv in cli.levels))
        for run in cli.runs:
            print(f"{run:24s}" + ''.join(
                fmt.format(results[run][lv][key]).rjust(18) for lv in cli.levels))
    print(f"\nwritten to {cli.output}")


if __name__ == '__main__':
    main()
