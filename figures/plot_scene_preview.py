#!/usr/bin/env python3
"""Preview occluded-scene inputs (occlusion_scene.py) on real samples.

One row per sample:
  clean target RGB (the RGB target) | scene RGB (the input) | scene depth |
  target plant pixels a neighbour covers (red) with the image patches scored
  while visible outlined | the scene cloud from the camera, target green and
  neighbour red | the same cloud from the side in world coordinates, with the
  crop cylinder.

Prints the per-sample stats compose() reports, plus two geometry checks: the
target's stem should sit at the world origin, and the crop should keep nearly
all of the target.

    python figures/plot_scene_preview.py --split val --n 8 \\
        --out reports/scene_preview_val.png
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import random
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from occlusion_scene import SceneConfig, camera_pixels, compose, to_world
from sorghum_dataset_4m import SorghumDataset4M
from structured_mask import foreground

DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def show_rgb(t):
    return (t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--spacing', type=float, nargs=2, default=None)
    ap.add_argument('--crop-radius', type=float, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds = SorghumDataset4M(args.data_root, split=args.split, return_pose=True)
    picks = random.Random(args.seed).sample(range(len(ds)), args.n)
    items = [ds[i] for i in picks]
    rgb, depth, pc = (torch.stack([x[k] for x in items]) for k in (0, 1, 2))
    pn = torch.stack([x[6] for x in items])
    c2w = torch.stack([x[7] for x in items])
    names = [x[5] for x in items]

    cfg = SceneConfig(prob=1.0)
    if args.spacing:
        cfg = replace(cfg, spacing=tuple(args.spacing))
    if args.crop_radius:
        cfg = replace(cfg, crop_radius=args.crop_radius)
    sc = compose(rgb, depth, pc, pn, c2w, cfg,
                 generator=torch.Generator().manual_seed(args.seed))
    print('config', cfg.to_dict())
    print('batch stats', {k: round(v, 3) for k, v in sc['stats'].items()})

    # Geometry checks on the clean targets.
    _, world = to_world(pc, pn, c2w)
    r = torch.hypot(world[..., 0], world[..., 2])
    low = world[..., 1] < world[..., 1].min(1, keepdim=True).values + 0.02
    base = [(world[b][low[b]][:, [0, 2]].mean(0)).tolist() for b in range(len(items))]
    print('target stem base xz (want ~0):', [[round(v, 3) for v in xz] for xz in base])
    print('target points within crop:',
          [round(float((r[b] <= cfg.crop_radius).float().mean()), 3) for b in range(len(items))])

    nb_pt = sc['nb_point']
    shown = sc['shown']
    fg = foreground(depth)
    loss_tok = sc['loss_tokens']['rgb'].view(-1, 14, 14)

    n = len(items)
    fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        a = axes[i]
        hid = (fg[i, 0] & shown[i, 0]).float().sum() / fg[i, 0].float().sum().clamp(min=1)
        a[0].imshow(show_rgb(rgb[i])); a[0].set_title(f'{names[i]} (target)', fontsize=8)
        a[1].imshow(show_rgb(sc['rgb'][i])); a[1].set_title('scene RGB (input)', fontsize=8)
        dd = sc['depth'][i, 0].numpy().copy(); dd[dd <= 0.02] = np.nan
        a[2].imshow(dd, cmap='viridis'); a[2].set_title('scene depth', fontsize=8)
        over = np.zeros((224, 224, 3)) + 0.1
        over[fg[i, 0].numpy()] = (0.2, 0.6, 0.2)
        over[(fg[i, 0] & shown[i, 0]).numpy()] = (0.9, 0.1, 0.1)
        over[(~fg[i, 0] & shown[i, 0]).numpy()] = (0.5, 0.3, 0.3)
        a[3].imshow(over)
        for (pr, pcl) in loss_tok[i].nonzero().tolist():
            a[3].add_patch(plt.Rectangle((pcl * 16 - .5, pr * 16 - .5), 16, 16,
                                         fill=False, ec='y', lw=0.4))
        a[3].set_title(f'target hidden {hid:.0%}; scored-while-visible '
                       f'{int(loss_tok[i].sum())} patches', fontsize=8)
        cam = sc['pc'][i] * pn[i, 3] + pn[i, :3]
        row, col, _ = camera_pixels(cam[None], 224, 224)
        m = nb_pt[i].numpy()
        a[4].scatter(col[0][~m], row[0][~m], s=0.2, c='g')
        a[4].scatter(col[0][m], row[0][m], s=0.2, c='r')
        a[4].set_xlim(0, 224); a[4].set_ylim(224, 0); a[4].set_aspect('equal')
        a[4].set_title(f'scene cloud: neighbour {m.mean():.0%}', fontsize=8)
        R, t = c2w[i, :3, :3], c2w[i, :3, 3]
        w = cam @ R.T + t
        a[5].scatter(w[~m, 0], w[~m, 1], s=0.2, c='g')
        a[5].scatter(w[m, 0], w[m, 1], s=0.2, c='r')
        for x in (-cfg.crop_radius, cfg.crop_radius):
            a[5].axvline(x, c='k', lw=0.5, ls='--')
        a[5].set_aspect('equal'); a[5].set_title('world x-y (crop dashed)', fontsize=8)
        for ax in a[:5]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Occluded scenes ({args.split}): spacing {cfg.spacing} m, '
                 f'crop {cfg.crop_radius} m, {cfg.neighbours} neighbours')
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
