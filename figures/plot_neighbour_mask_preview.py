#!/usr/bin/env python3
"""Preview neighbour-occlusion structured masking on real samples.

One row per sample:
  target RGB | with two neighbours composited in (what the TEST input looks
  like) | target plant pixels the neighbours hide (red) | the image tokens the
  structured mask forces (red) vs plant patches it leaves to the uniform fill |
  the cloud seen from the camera, hidden points red.

    python figures/plot_neighbour_mask_preview.py --split train --n 6 \\
        --out reports/neighbour_mask_preview.png
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sorghum_dataset_4m import SorghumDataset4M
from structured_mask import (
    NeighbourMaskConfig, draw_params, hidden_points, image_token_flags,
    occluders, project_points,
)
from eval.eval_occlusion import DATA_ROOT, test_occlusion_config

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def show_rgb(t):
    return (t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split', default='train')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--placement', choices=('train', 'test'), default='train')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ds = SorghumDataset4M(args.data_root, split=args.split, return_pc_norm=True)
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(ds)), 3 * args.n)
    items = [ds[i] for i in picks]
    rgb = torch.stack([x[0] for x in items])
    depth = torch.stack([x[1] for x in items])
    pc = torch.stack([x[2] for x in items])
    pn = torch.stack([x[6] for x in items])
    n = args.n
    tgt = slice(0, n)

    flags_cfg = NeighbourMaskConfig()
    place_cfg = (NeighbourMaskConfig(prob=1.0) if args.placement == 'train'
                 else test_occlusion_config())
    params = draw_params(n, place_cfg, 'cpu',
                         generator=torch.Generator().manual_seed(args.seed))
    occ = occluders(depth[tgt], [depth[n:2 * n], depth[2 * n:]], params,
                    nb_rgbs=[rgb[n:2 * n], rgb[2 * n:]])
    img_flags = image_token_flags(occ, 16, flags_cfg).view(n, 14, 14)
    plant = (F.avg_pool2d(occ['tgt_fg'].float(), 16) >= flags_cfg.min_patch_plant).view(n, 14, 14)
    hid = hidden_points(pc[tgt], pn[tgt], occ['occ_z'])
    composite = torch.where(occ['shown'], occ['rgb'], rgb[tgt])

    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    for i in range(n):
        fg_px = int(occ['tgt_fg'][i].sum())
        frac = int(occ['hidden'][i].sum()) / max(fg_px, 1)
        a = axes[i]
        a[0].imshow(show_rgb(rgb[i])); a[0].set_title(items[i][5], fontsize=8)
        a[1].imshow(show_rgb(composite[i])); a[1].set_title('with neighbours', fontsize=8)
        over = np.zeros((224, 224, 3))
        over[occ['tgt_fg'][i, 0].numpy()] = (0.2, 0.6, 0.2)
        over[occ['hidden'][i, 0].numpy()] = (0.9, 0.1, 0.1)
        a[2].imshow(over); a[2].set_title(f'hidden plant px {frac:.0%}', fontsize=8)
        grid = np.zeros((14, 14, 3)) + 0.15
        grid[plant[i].numpy()] = (0.2, 0.6, 0.2)
        grid[img_flags[i].numpy()] = (0.9, 0.1, 0.1)
        a[3].imshow(grid, interpolation='nearest')
        a[3].set_title(f'forced patches {int(img_flags[i].sum())} / plant {int(plant[i].sum())}',
                       fontsize=8)
        r, c, _, _ = project_points(pc[i:i + 1], pn[i:i + 1], 224, 224)
        h = hid[i].numpy()
        a[4].scatter(c[0][~h], r[0][~h], s=0.2, c='g')
        a[4].scatter(c[0][h], r[0][h], s=0.2, c='r')
        a[4].set_xlim(0, 224); a[4].set_ylim(224, 0); a[4].set_aspect('equal')
        a[4].set_title(f'hidden points {h.mean():.0%}', fontsize=8)
        for ax in a:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Neighbour-occlusion masking ({args.placement} placement, {args.split} split)')
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
