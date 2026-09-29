"""Do the spline params actually help reconstruction?  Measure it directly.

Two validation passes over the same checkpoint and the same fixed val set:

  ON   every param token visible, carrying the sample's own params
  OFF  every param token visible, but carrying the val-set MEAN params

Everything else is identical: the eval seed fixes the Dirichlet split and the
RGB / depth / PC token masks, so both passes hide exactly the same pixels and
points.  The only difference is whether the params say anything about this
particular plant.  OFF - ON on the PC / RGB / depth metrics is therefore the
information the params contribute; ~0 means the model ignores them.

    python eval_param_oracle.py --config config_4m_recipe_b_ls020_noise1x.yaml \\
        --checkpoint outputs/recipe_b_ls020_noise1x/best_model.pth --occluded

`--occluded` adds the same comparison under the config's reference occlusion
(occlusion.eval_occlusion), where the image evidence is weaker and the params
should matter more.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from embodied_mae_4m import load_spline_params
from plant_occlusion import ProceduralOcclusion
from sorghum_dataset_4m import SorghumDataset4M
from train_sorghum_4m import (build_model, config_to_namespace, evaluate,
                              fixed_subset, load_config)

METRICS = [('loss', 'total loss', True), ('pc_chamfer', 'PC Chamfer', True),
           ('pc_f1@0.01', 'PC F1@0.01', False), ('pc_f1@0.02', 'PC F1@0.02', False),
           ('rgb_mse', 'RGB MSE', True), ('depth_mse', 'Depth MSE', True)]


class _MeanParams:
    """Dataloader wrapper that swaps every sample's params for the val mean."""

    def __init__(self, loader, mean):
        self.loader, self.mean = loader, mean

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            batch = list(batch)
            batch[3] = self.mean.expand_as(batch[3]).clone()
            yield batch


def dataset_folders(ds):
    if hasattr(ds, 'indices'):
        return [ds.dataset.samples[i] for i in ds.indices]
    return list(ds.samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--occluded', action='store_true',
                    help='also compare under the reference eval occlusion')
    ap.add_argument('--num_val_samples', type=int, default=None,
                    help='override the config (default: same fixed val set)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=8)
    cli = ap.parse_args()

    args = config_to_namespace(load_config(cli.config))
    if cli.num_val_samples:
        args.num_val_samples = cli.num_val_samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_ds = SorghumDataset4M(args.data_root, split='val', img_size=args.img_size,
                              num_points=args.num_points, max_leaves=args.max_leaves,
                              param_encoding=args.param_encoding,
                              geometry_cond=args.geometry_cond)
    val_ds = fixed_subset(val_ds, args.num_val_samples, args.val_mask_seed)
    loader = DataLoader(val_ds, batch_size=cli.batch_size, shuffle=False,
                        num_workers=cli.num_workers, pin_memory=True)

    # Val-set mean params: the "knows the typical plant, not this plant" input.
    stack = torch.stack([
        load_spline_params(next(f.glob('*_spline.yml')), args.max_leaves,
                           encoding=args.param_encoding)[1]
        for f in dataset_folders(val_ds)])
    mean_params = stack.mean(0, keepdim=True)

    model = build_model(args).to(device)
    # Our own training checkpoints: they pickle the metric history (numpy
    # scalars), which the weights-only loader refuses.
    ckpt = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    state = {k.replace('module.', '', 1): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.eval()
    model.force_visible = {'text': model.n_text_tokens}

    passes = [('clean', None)]
    if cli.occluded:
        occ = dict(args.occlusion or {})
        occ.update(args.eval_occlusion or {})
        occ['enabled'] = True
        passes.append(('occluded', ProceduralOcclusion.from_config(occ)))

    kw = dict(mask_ratio=args.mask_ratio, val_mask_seed=args.val_mask_seed,
              pc_metric_thresholds=args.pc_metric_thresholds,
              pc_metric_chunk_size=args.pc_metric_chunk_size)
    print(f"\n{cli.checkpoint}  (epoch {ckpt.get('epoch', '?')})  "
          f"encoding={args.param_encoding}  geometry_cond={args.geometry_cond}  "
          f"val samples={len(val_ds)}")
    for label, occluder in passes:
        (on_loss, *_), on = evaluate(model, loader, device, occlusion=occluder, **kw)
        (off_loss, *_), off = evaluate(model, _MeanParams(loader, mean_params), device,
                                       occlusion=occluder, **kw)
        on['loss'], off['loss'] = on_loss, off_loss
        print(f"\n[{label}] params ON (own) vs OFF (val mean); "
              f"positive 'OFF worse' = params help")
        print(f"{'metric':<12}{'ON':>12}{'OFF':>12}{'OFF worse by':>15}")
        for key, name, lower_better in METRICS:
            if key not in on:
                continue
            a, b = on[key], off[key]
            worse = (b - a) if lower_better else (a - b)
            rel = 100.0 * worse / abs(a) if a else float('nan')
            print(f"{name:<12}{a:12.6f}{b:12.6f}{worse:+12.6f} ({rel:+.1f}%)")


if __name__ == '__main__':
    main()
