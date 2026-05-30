"""Build the gallery figure for EmbodiedMAE-4M run v3 — completed 2400 epochs.

Like make_slide_figure_v3.py but reads metrics_summary_2400.json so the
hard-coded "best epoch 1100, val 0.1268" strings are replaced with the real
2400-epoch values. Also drops the early-stop progress bar.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from PIL import Image


REPO = Path('/work/mech-ai-scratch/alloy/embodiedmae')
SUMMARY_PATH = REPO / 'outputs/4m_run_v3/metrics_summary_2400.json'


# ── I/O (cloned from make_slide_figure_v3.py) ────────────────────────────────

def load_sample(sample_dir: Path):
    s = sample_dir
    out = {'name': s.name.replace('sample_', '')}
    out['rgb_in']   = np.array(Image.open(s / 'inputs' / 'rgb.png'))
    out['rgb_pred'] = np.array(Image.open(s / 'outputs' / 'rgb.png'))
    out['depth_in']   = np.load(s / 'inputs' / 'depth.npy')
    out['depth_pred'] = np.load(s / 'outputs' / 'depth.npy')
    out['pc_in']   = np.load(s / 'inputs' / 'pointcloud.npy')
    out['pc_pred'] = np.load(s / 'outputs' / 'pointcloud.npy')
    out['m_rgb']   = np.load(s / 'masks' / 'rgb_mask.npy').astype(bool)
    out['m_depth'] = np.load(s / 'masks' / 'depth_mask.npy').astype(bool)
    out['m_pc']    = np.load(s / 'masks' / 'pc_mask.npy').astype(bool)
    out['m_text']  = np.load(s / 'masks' / 'text_mask.npy').astype(bool)
    with open(s / 'metrics.json') as f:
        out['metrics'] = json.load(f)
    with open(s / 'outputs' / 'params.json') as f:
        out['params'] = json.load(f)
    return out


def apply_patch_mask(img, mask, patch=16, fill=0):
    out = img.copy()
    h, w = out.shape[:2]
    nh, nw = h // patch, w // patch
    mask_2d = mask.reshape(nh, nw)
    for i in range(nh):
        for j in range(nw):
            if mask_2d[i, j]:
                out[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = fill
    return out


def norm_depth(arr):
    arr = arr.squeeze()
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def plot_pc(ax, pts, color='seagreen', s=0.4, alpha=0.7,
            elev=20, azim=-70, lim=None):
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=color, s=s, alpha=alpha,
               depthshade=False, edgecolors='none')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    if lim is None:
        lim = max(np.abs(pts).max(), 1e-3) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    for spine in ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane:
        spine.set_visible(False)
    ax.grid(False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir',   default='slide_dump_v3_2400')
    ap.add_argument('--out',        default='slide_figure_v3_2400.png')
    ap.add_argument('--n_plants',   type=int, default=15)
    ap.add_argument('--dpi',        type=int, default=140)
    args = ap.parse_args()

    summary = json.load(open(SUMMARY_PATH))
    best     = summary['best']
    best_ep  = summary['best_epoch']
    final_ep = summary['final_epoch']
    final    = summary['final']

    dump = Path(args.dump_dir)
    sample_dirs = sorted(dump.glob('sample_*'))[:args.n_plants]
    if not sample_dirs:
        raise SystemExit(f'No samples found in {dump}')
    samples = [load_sample(sd) for sd in sample_dirs]
    n = len(samples)
    print(f'Loaded {n} samples')

    agg_path = dump / 'metrics_aggregate.json'
    agg = json.load(open(agg_path)) if agg_path.exists() else {}

    # Pre-compute
    pc_lim = max((max(np.abs(s['pc_in']).max(), np.abs(s['pc_pred']).max())
                  for s in samples)) * 1.05
    pc_color = 'seagreen'
    for s in samples:
        s['rgb_in_masked']   = apply_patch_mask(s['rgb_in'],   s['m_rgb'],   16, 0)
        s['depth_in_n']      = norm_depth(s['depth_in'])
        s['depth_pred_n']    = norm_depth(s['depth_pred'])
        s['depth_in_masked'] = apply_patch_mask(
            (s['depth_in_n'] * 255).astype(np.uint8), s['m_depth'], 16, 0)

    row_labels = [
        'RGB ground-truth',
        'RGB input (masked)',
        'RGB reconstruction',
        'Depth ground-truth',
        'Depth reconstruction',
        'PointCloud recon',
    ]
    n_rows = len(row_labels)

    col_w = 1.75
    fig_w = max(22, 2.5 + n * col_w)
    fig_h = 16.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        nrows=n_rows + 2, ncols=n + 1,
        height_ratios=[1.8] + [1.0]*n_rows + [1.5],
        width_ratios=[0.48] + [1.0]*n,
        hspace=0.08, wspace=0.05,
        left=0.005, right=0.995, top=0.985, bottom=0.015,
    )

    # ── Title row ────────────────────────────────────────────────────────────
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(
        0.5, 0.88,
        f'EmbodiedMAE-4M  ·  {n}-plant reconstruction gallery  ·  run v3 (completed)',
        ha='center', va='center', fontsize=26, weight='bold',
        transform=title_ax.transAxes,
    )
    sub = (f'best epoch {best_ep} (val loss {best["val_loss"]:.4f}, '
           f'param acc@0.05 = {best["val_param_acc05"]*100:.1f}%)   ·   '
           f'mask_ratio at eval = {agg.get("mask_ratio", 0):.2f}   ·   '
           f'samples = {agg.get("n_samples", n)}   ·   '
           f'split = val   ·   wandb d0lfi27q')
    title_ax.text(0.5, 0.66, sub, ha='center', va='center',
                  fontsize=14, color='dimgray', transform=title_ax.transAxes)

    # Completion banner (replaces the progress bar from the v3 figure)
    bar_left, bar_right = 0.18, 0.82
    bar_y, bar_h = 0.18, 0.22
    title_ax.add_patch(Rectangle(
        (bar_left, bar_y), bar_right - bar_left, bar_h,
        transform=title_ax.transAxes,
        facecolor='#d4edda', edgecolor='#7fbf86', linewidth=1.0))
    title_ax.text(
        0.5, bar_y + bar_h / 2,
        f'training complete: {final_ep} / {final_ep} epochs  '
        f'(final val_loss = {final["val_loss"]:.4f})',
        ha='center', va='center', fontsize=13, weight='bold',
        color='#1d3a14', transform=title_ax.transAxes,
    )

    # ── Row-label cells ──────────────────────────────────────────────────────
    for r, lbl in enumerate(row_labels):
        lax = fig.add_subplot(gs[1 + r, 0])
        lax.axis('off')
        lax.text(0.95, 0.5, lbl, ha='right', va='center',
                 fontsize=14, weight='bold', transform=lax.transAxes)

    # ── Image grid ───────────────────────────────────────────────────────────
    for c, s in enumerate(samples):
        head_ax = fig.add_subplot(gs[1, 1 + c])
        head_ax.set_xticks([]); head_ax.set_yticks([])
        head_ax.imshow(s['rgb_in'])
        head_ax.set_title(s['name'].replace('Sorghum_', 'S'),
                          fontsize=10, pad=2)

        ax = fig.add_subplot(gs[2, 1 + c])
        ax.imshow(s['rgb_in_masked']); ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[3, 1 + c])
        ax.imshow(s['rgb_pred']); ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[4, 1 + c])
        ax.imshow(s['depth_in_n'], cmap='viridis')
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[5, 1 + c])
        ax.imshow(s['depth_pred_n'], cmap='viridis')
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[6, 1 + c], projection='3d')
        plot_pc(ax, s['pc_pred'], color=pc_color, s=0.4, alpha=0.7,
                lim=pc_lim)
        ax.margins(0)

    # ── Bottom metrics strip ─────────────────────────────────────────────────
    msax = fig.add_subplot(gs[-1, :])
    msax.axis('off')

    agg_text = (
        f'Aggregate over {agg.get("n_samples", n)} plants  ·  '
        f'mask_ratio={agg.get("mask_ratio", 0):.2f}        '
        f'RGB MSE = {agg.get("rgb_mse", 0):.4f}     '
        f'Depth MSE = {agg.get("depth_mse", 0):.4f}     '
        f'PC Chamfer = {agg.get("pc_chamfer", 0):.6f}     '
        f'Param MAE (all) = {agg.get("param_mae_all", 0):.4f}     '
        f'Param MAE (masked) = {agg.get("param_mae_masked", 0):.6f}'
    )
    msax.text(0.5, 0.78, agg_text, ha='center', va='center', fontsize=14,
              family='monospace', color='#0b3954',
              transform=msax.transAxes,
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#eaf3f7',
                        edgecolor='#9ec5d4'))

    best_text = (
        f'Best epoch (training val, mask=0.15) ep {best_ep}       '
        f'val_loss = {best["val_loss"]:.4f}     '
        f'RGB MSE = {best["val_rgb_mse"]:.4f}     '
        f'Depth MSE = {best["val_depth_mse"]:.4f}     '
        f'PC Chamfer = {best["val_pc_chamfer"]:.6f}     '
        f'Param MAE (masked) = {best["val_param_mae_masked"]:.6f}     '
        f'Param acc@0.05 = {best["val_param_acc05"]:.4f}'
    )
    msax.text(0.5, 0.32, best_text, ha='center', va='center', fontsize=14,
              family='monospace', color='#1d3a14',
              transform=msax.transAxes,
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f4e3',
                        edgecolor='#9fc792'))

    out = Path(args.out)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
