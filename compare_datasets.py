"""
compare_datasets.py — side-by-side comparison of the render domains the 4M model
sees: the old 10k (`Dataset/new_data`) it was trained on vs Sorghum_15K (the OOD
set). Produces two figures:

  <out>_samples.png — qualitative grid: RGB / depth / point cloud per dataset
  <out>_stats.png   — aggregate distributions that would explain a domain shift
                      (depth encoding, plant framing, raw PC scale, point count)

Usage:
    python compare_datasets.py --n_stats 300 --n_show 4
"""
import argparse
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

PC_COLOR = '#2a9d8f'

GROUPS = [
    ('old10k / train', '/work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/train'),
    ('old10k / val',   '/work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/val'),
    ('15K / train',    '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/train'),
    ('15K / test',     '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/test'),
]


def folders(root, n, seed=0):
    fs = sorted(p for p in Path(root).iterdir() if p.is_dir())
    rng = random.Random(seed)
    return rng.sample(fs, min(n, len(fs)))


def read_sample(folder):
    """Raw, un-normalised load — this is what differs between domains."""
    rgb = np.asarray(Image.open(folder / 'rgb.png').convert('RGB'))
    depth = np.asarray(Image.open(folder / 'depth.png'))
    ply = list(folder.glob('*_nc_cam.ply'))[0]
    pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    return rgb, depth, pts


def stats(rgb, depth, pts):
    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    dmax = 65535.0 if depth.dtype == np.uint16 else 255.0
    d = d / dmax
    fg = d > 0.01                       # non-background pixels
    ext = pts.max(0) - pts.min(0)       # raw bbox extent, before unit-sphere scaling
    return dict(
        depth_fg_mean=float(d[fg].mean()) if fg.any() else 0.0,
        depth_fg_frac=float(fg.mean()),          # how much of the frame the plant fills
        depth_bits=16 if depth.dtype == np.uint16 else 8,
        rgb_mean=float(rgb.mean() / 255.0),
        n_points=len(pts),
        pc_extent=float(np.linalg.norm(ext)),    # raw diagonal — absolute scale
        pc_height=float(ext[1]),
        depth_vals=d[fg] if fg.any() else np.array([0.0]),
    )


def pc_panel(ax, pts, title=None):
    """Front view (x-y) of the point cloud, unit-sphere normalised as the model sees it."""
    p = pts - pts.mean(0)
    m = np.linalg.norm(p, axis=1).max()
    if m > 0:
        p = p / m
    ax.scatter(p[:, 0], p[:, 1], s=0.4, c=PC_COLOR, alpha=0.55, linewidths=0)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def fig_samples(groups, n_show, out):
    ncols = n_show * 3
    fig, axes = plt.subplots(len(groups), ncols,
                             figsize=(2.0 * ncols, 2.25 * len(groups)))
    for r, (name, root) in enumerate(groups):
        for i, folder in enumerate(folders(root, n_show, seed=1)):
            rgb, depth, pts = read_sample(folder)
            d = depth if depth.ndim == 2 else depth[..., 0]

            ax = axes[r, i * 3]
            ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{folder.name}\nRGB", fontsize=6)
            if i == 0:
                ax.set_ylabel(name, fontsize=10, fontweight='bold')

            ax = axes[r, i * 3 + 1]
            im = ax.imshow(d, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"depth ({depth.dtype})", fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

            pc_panel(axes[r, i * 3 + 2], pts, f"PC ({len(pts):,} pts)")
    fig.suptitle('Render-domain comparison — raw RGB / depth / point cloud',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved {out}")


def fig_stats(groups, n_stats, out):
    coll = {}
    for name, root in groups:
        rows = []
        for folder in folders(root, n_stats, seed=2):
            try:
                rows.append(stats(*read_sample(folder)))
            except Exception as e:
                print(f"  skip {folder.name}: {e}")
        coll[name] = rows
        s = rows
        print(f"{name:16s} n={len(s):4d}  "
              f"depth_bits={s[0]['depth_bits']}  "
              f"depth_fg_mean={np.mean([r['depth_fg_mean'] for r in s]):.3f}  "
              f"frame_fill={np.mean([r['depth_fg_frac'] for r in s]):.3f}  "
              f"pc_extent={np.mean([r['pc_extent'] for r in s]):.2f}  "
              f"n_points={np.mean([r['n_points'] for r in s]):.0f}  "
              f"rgb_mean={np.mean([r['rgb_mean'] for r in s]):.3f}")

    panels = [
        ('depth_fg_mean', 'mean depth value (foreground)', 'depth encoding / camera distance'),
        ('depth_fg_frac', 'fraction of frame filled by plant', 'framing / zoom'),
        ('pc_extent',     'raw PC bbox diagonal', 'absolute plant scale (pre-normalisation)'),
        ('n_points',      'points per cloud', 'PC density'),
        ('rgb_mean',      'mean RGB intensity', 'lighting / exposure'),
        ('pc_height',     'raw PC height (y extent)', 'plant height'),
    ]
    colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, label, why) in zip(axes.ravel(), panels):
        data = [[r[key] for r in coll[n]] for n, _ in groups]
        lo = min(min(d) for d in data); hi = max(max(d) for d in data)
        bins = np.linspace(lo, hi, 40) if hi > lo else 20
        for (name, _), d, c in zip(groups, data, colors):
            ax.hist(d, bins=bins, alpha=0.55, label=name, color=c, zorder=3)
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(why, fontsize=9, fontweight='bold')
        ax.grid(alpha=0.25, zorder=0)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Why RGB→PC does not transfer: old 10k vs Sorghum_15K render statistics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_show',  type=int, default=4, help='samples shown per dataset')
    ap.add_argument('--n_stats', type=int, default=300, help='samples per dataset for histograms')
    ap.add_argument('--out',     default='dataset_compare')
    args = ap.parse_args()

    fig_samples(GROUPS, args.n_show, f'{args.out}_samples.png')
    fig_stats(GROUPS, args.n_stats, f'{args.out}_stats.png')
