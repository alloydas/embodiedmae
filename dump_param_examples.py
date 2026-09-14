"""Dump worked examples for the ->params distillation arms: for each of the
first N validation plants, the parameters the model recovers from ONE modality
against the parameters the plant was actually grown from.

The metric in the run logs is a single normalised MAE. This turns that back into
the quantities it was computed over -- a stem length in the generator's own
units, a branching angle in degrees -- so an error of 0.03 can be read as what
it is for each field.

Runs on CPU by default: the model has no CUDA-only ops, and a handful of
forward passes costs less than waiting out a GPU queue.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

import train_sorghum_4m_distill as DIS
from embodied_mae_4m import _PLANT_SCALE, _PLANT_SHIFT, _LEAF_SCALE, _LEAF_SHIFT
from sorghum_dataset_4m import SorghumDataset4M

# Field names as the spline YAMLs spell them, with the unit the raw value is in.
PLANT_FIELDS = [('stem_length', ''), ('stem_dir_x', ''), ('stem_dir_y', ''),
                ('stem_dir_z', ''), ('panicle_size_x', ''), ('panicle_size_y', ''),
                ('panicle_size_z', ''), ('panicle_seed_amount', ''),
                ('panicle_seed_radius', '')]
LEAF_FIELDS = [('starting_point', ''), ('length', ''), ('roll_angle', '°'),
               ('branching_angle', '°'), ('waviness_frequency', ''),
               ('waviness_period_start_0', '°'), ('waviness_period_start_1', '°')]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def denorm_rgb(t):
    """(3, H, W) ImageNet-normalised tensor -> uint8 HWC."""
    a = t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--source', required=True, choices=['rgb', 'depth', 'pc', 'text'])
    ap.add_argument('--n', type=int, default=6)
    # Samples are <plant>_<view>: there are 10 consecutive views of each plant,
    # so consecutive indices are one plant from 10 angles. Pass explicit indices
    # to get distinct plants, spread across the first 1600 samples -- the subset
    # val_max_batches:100 actually measures (160 plants, not 1600).
    ap.add_argument('--indices', default='',
                    help='comma-separated dataset indices; default = first --n')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    args = DIS.config_to_namespace(cfg)

    ds = SorghumDataset4M(f"{cfg['data']['data_root']}/val",
                          img_size=args.img_size, num_points=args.num_points)
    # shuffle=False and the first N samples: the same plants, in the same order,
    # the run's own validation metric was averaged over.
    if a.indices:
        idx = [int(x) for x in a.indices.split(',')]
        ds = torch.utils.data.Subset(ds, idx)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = DIS.build_model(args, a.device)
    DIS.load_weights_into(model, a.checkpoint, a.device, 'example')
    model.eval()

    records = []
    with torch.no_grad():
        kept = 0
        for i, (rgb, depth, pc, params, tv, name) in enumerate(dl):
            if kept >= a.n:
                break
            kept += 1
            _, _, (_, _, _, pparam), (_, _, _, mt) = model(
                rgb, depth, pc, params, tv, visible={a.source})

            gt, pred = params[0].numpy(), pparam[0].numpy()
            valid, masked = tv[0].numpy().astype(bool), mt[0].numpy().astype(bool)

            plant_gt = gt[0] * _PLANT_SCALE - _PLANT_SHIFT
            plant_pr = np.clip(pred[0], 0, 1) * _PLANT_SCALE - _PLANT_SHIFT
            leaves = []
            for li in range(1, len(gt)):
                if not valid[li]:
                    continue
                g = gt[li][:7] * _LEAF_SCALE[:7] - _LEAF_SHIFT[:7]
                p = np.clip(pred[li][:7], 0, 1) * _LEAF_SCALE[:7] - _LEAF_SHIFT[:7]
                leaves.append({'index': li, 'unseen': bool(masked[li]),
                               'gt': [round(float(x), 4) for x in g],
                               'pred': [round(float(x), 4) for x in p],
                               'norm_mae': round(float(np.abs(
                                   gt[li][:7] - np.clip(pred[li][:7], 0, 1)).mean()), 5),
                               # The 2 padding dims are always 0 for a leaf and are
                               # INSIDE the reported mean over N_PARAMS=9. Kept so the
                               # dilution they cause can be measured, not assumed.
                               'pad_pred': [round(float(x), 5)
                                            for x in np.clip(pred[li][7:9], 0, 1)]})

            nm = name[0]
            Image.fromarray(denorm_rgb(rgb[0])).save(out / f'{nm}_rgb.png')
            records.append({
                'name': nm, 'source': a.source, 'n_leaves': len(leaves),
                'plant': {'gt': [round(float(x), 4) for x in plant_gt],
                          'pred': [round(float(x), 4) for x in plant_pr]},
                'leaves': leaves,
            })
            print(f"[{kept}/{a.n}] {nm}  leaves={len(leaves)}  "
                  f"leaf_mae={np.mean([l['norm_mae'] for l in leaves]):.5f}")

    json.dump({'plant_fields': PLANT_FIELDS, 'leaf_fields': LEAF_FIELDS,
               'checkpoint': a.checkpoint, 'records': records},
              open(out / f'examples_{a.source}.json', 'w'), indent=1)
    print('wrote', out / f'examples_{a.source}.json')


if __name__ == '__main__':
    main()
