"""Build a slide-ready composite figure for EmbodiedMAE-4M.

Rows: RGB / Depth / PointCloud / Procedural params
Cols: Incomplete input | Reconstruction | Ground truth

Reads from a validate_4m.py dump (`slide_dump/sample_<name>/`) and writes
`slide_figure.png` at the project root.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from PIL import Image


def load_sample(sample_dir: Path):
    """Return all tensors / masks / params dicts for one sample."""
    s = sample_dir
    rgb_in   = np.array(Image.open(s / 'inputs' / 'rgb.png'))
    rgb_pred = np.array(Image.open(s / 'outputs' / 'rgb.png'))
    depth_in   = np.load(s / 'inputs' / 'depth.npy')
    depth_pred = np.load(s / 'outputs' / 'depth.npy')
    pc_in   = np.load(s / 'inputs' / 'pointcloud.npy')
    pc_pred = np.load(s / 'outputs' / 'pointcloud.npy')
    m_rgb   = np.load(s / 'masks' / 'rgb_mask.npy').astype(bool)   # (196,)
    m_depth = np.load(s / 'masks' / 'depth_mask.npy').astype(bool)
    m_pc    = np.load(s / 'masks' / 'pc_mask.npy').astype(bool)
    m_text  = np.load(s / 'masks' / 'text_mask.npy').astype(bool)
    pc_vis_idx_path = s / 'masks' / 'pc_vis_point_idx.npy'
    pc_vis_idx = np.load(pc_vis_idx_path) if pc_vis_idx_path.exists() else None
    with open(s / 'outputs' / 'params.json') as f:
        params = json.load(f)
    with open(s / 'metrics.json') as f:
        metrics = json.load(f)
    return dict(
        name=s.name.replace('sample_', ''),
        rgb_in=rgb_in, rgb_pred=rgb_pred, depth_in=depth_in, depth_pred=depth_pred,
        pc_in=pc_in, pc_pred=pc_pred, pc_vis_idx=pc_vis_idx,
        m_rgb=m_rgb, m_depth=m_depth, m_pc=m_pc, m_text=m_text,
        params=params, metrics=metrics,
    )


def apply_patch_mask(img: np.ndarray, mask: np.ndarray, patch=16, fill=128):
    """Black out 16x16 patches of `img` where `mask[i]` is True (i.e. masked).
    `mask` is length 196 = 14x14 patches."""
    out = img.copy()
    h, w = out.shape[:2]
    nh = h // patch
    nw = w // patch
    mask_2d = mask.reshape(nh, nw)
    for i in range(nh):
        for j in range(nw):
            if mask_2d[i, j]:
                out[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = fill
    return out


def fps_numpy(xyz: np.ndarray, npoint: int, seed=42):
    """Farthest point sampling on a (N, 3) numpy array → indices (npoint,)."""
    rng = np.random.default_rng(seed)
    N = xyz.shape[0]
    centroids = np.empty(npoint, dtype=np.int64)
    distance = np.full(N, np.inf, dtype=np.float64)
    farthest = int(rng.integers(0, N))
    for i in range(npoint):
        centroids[i] = farthest
        c = xyz[farthest]
        d = np.sum((xyz - c) ** 2, axis=1)
        distance = np.minimum(distance, d)
        farthest = int(np.argmax(distance))
    return centroids


def knn_assign(xyz: np.ndarray, centers_idx: np.ndarray, k=32):
    """For each center, get the indices of its k nearest neighbours.
    Returns (n_centers, k) array of point indices."""
    centers = xyz[centers_idx]                       # (T, 3)
    d = np.sum((xyz[None] - centers[:, None]) ** 2, axis=-1)  # (T, N)
    return np.argpartition(d, k, axis=1)[:, :k]      # (T, k)


def pc_visible_indices(xyz: np.ndarray, m_pc: np.ndarray, num_tokens=196, group_size=32):
    """Reproduce the model's FPS+kNN grouping and return the indices of points
    that belong to *visible* tokens (m_pc[i] == False)."""
    centers_idx = fps_numpy(xyz, num_tokens, seed=0)
    knn_idx = knn_assign(xyz, centers_idx, k=group_size)
    visible_pts = set()
    for ti, masked in enumerate(m_pc):
        if not masked:
            visible_pts.update(knn_idx[ti].tolist())
    return np.array(sorted(visible_pts), dtype=np.int64)


def plot_pc(ax, pts, title, color='C0', s=0.5, alpha=0.6, elev=20, azim=-60,
            lim=None):
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=color, s=s, alpha=alpha,
               depthshade=False, edgecolors='none')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    if lim is None:
        lim = max(np.abs(pts).max(), 1e-3) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)


_PLANT_SHOW = [('sl', 'Plant.sl',     '{:.2f}', ''),     # stem length
               ('ps_x', 'Plant.ps_x', '{:.2f}', ''),     # position x
               ('ps_y', 'Plant.ps_y', '{:.2f}', '')]     # position y
_LEAF_SHOW  = [('sp',  'sp',  '{:.2f}',  ''),            # split position
               ('ln',  'ln',  '{:.2f}',  ''),            # length
               ('ba',  'ba',  '{:.1f}', '°')]            # bend angle


def _row(key_fmt, key, val_dict, fmt, unit):
    return (key_fmt, fmt.format(val_dict.get(key, 0)) + unit)


def format_params_table(params: dict, mode: str, max_leaves: int = 2):
    """mode: 'visible' | 'pred' | 'gt'. Returns list of (label, value_str)."""
    rows = []
    plant = params.get('plant') or {}
    if plant:
        if mode == 'visible' and plant.get('masked'):
            rows.append(('Plant', '[MASKED]'))
        else:
            src = plant['pred_raw'] if mode == 'pred' else plant['target_raw']
            for k, lbl, fmt, unit in _PLANT_SHOW:
                rows.append(_row(lbl, k, src, fmt, unit))

    leaves = params.get('leaves', [])[:max_leaves]
    for lf in leaves:
        idx = lf['leaf_index']
        if mode == 'visible' and lf.get('masked'):
            rows.append((f'Leaf{idx}', '[MASKED]'))
            continue
        src = lf['pred_raw'] if mode == 'pred' else lf['target_raw']
        for k, lbl, fmt, unit in _LEAF_SHOW:
            rows.append(_row(f'L{idx}.{lbl}', k, src, fmt, unit))
    return rows


def draw_param_panel(ax, rows, title, highlight_masked=False):
    ax.axis('off')
    ax.set_title(title, fontsize=11)
    y = 0.95
    line_h = 0.08
    for label, val in rows:
        color = 'crimson' if (val == '[MASKED]' and highlight_masked) else 'black'
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=10,
                family='monospace', color='black')
        ax.text(0.55, y, val, transform=ax.transAxes, fontsize=10,
                family='monospace', color=color, weight='bold' if val == '[MASKED]' else 'normal')
        y -= line_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir', default='slide_dump')
    ap.add_argument('--sample',   default=None,
                    help='sample name (e.g. Sorghum_0_00). If unset, picks first.')
    ap.add_argument('--out',      default='slide_figure.png')
    ap.add_argument('--dpi',      type=int, default=180)
    ap.add_argument('--layout',   choices=['portrait', 'landscape'],
                    default='landscape',
                    help='portrait: modalities are rows; landscape: modalities are columns')
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    if args.sample:
        sd = dump / f'sample_{args.sample}'
    else:
        sd = sorted(dump.glob('sample_*'))[0]
    print(f'Using sample dir: {sd}')

    d = load_sample(sd)
    name = d['name']

    # Aggregate eval metrics for the caption
    agg_path = dump / 'metrics_aggregate.json'
    agg = json.load(open(agg_path)) if agg_path.exists() else None

    # ── Pre-compute everything that goes into each (modality, stage) cell ─
    rgb_in_masked = apply_patch_mask(d['rgb_in'], d['m_rgb'], patch=16, fill=0)

    def norm_depth(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)
    depth_in_n   = norm_depth(d['depth_in'].squeeze())
    depth_pred_n = norm_depth(d['depth_pred'].squeeze())
    depth_in_masked = apply_patch_mask(
        (depth_in_n * 255).astype(np.uint8), d['m_depth'], patch=16, fill=0
    )

    pc_in, pc_pred = d['pc_in'], d['pc_pred']
    if d['pc_vis_idx'] is not None:
        pc_vis = pc_in[d['pc_vis_idx']]
    else:
        vis_idx = pc_visible_indices(pc_in, d['m_pc'], num_tokens=196, group_size=32)
        pc_vis = pc_in[vis_idx]
    pc_elev, pc_azim = 18, -75
    pc_lim = max(np.abs(pc_in).max(), np.abs(pc_pred).max()) * 1.05

    rows_in   = format_params_table(d['params'], 'visible', max_leaves=2)
    rows_pred = format_params_table(d['params'], 'pred',    max_leaves=2)
    rows_gt   = format_params_table(d['params'], 'gt',      max_leaves=2)

    stage_labels = ['Incomplete input  (masked)', 'Reconstruction', 'Ground truth']
    mod_labels   = ['RGB', 'Depth', 'Point Cloud', 'Procedural params']

    # Cells, indexed [modality][stage] → (kind, payload)
    # Single PC color across stages so audience reads them as the same object.
    # Visible cloud is rendered larger to stay legible despite having fewer points.
    pc_color  = 'seagreen'
    pc_colors = [pc_color, pc_color, pc_color]
    pc_sizes  = [1.2, 0.6, 0.6]
    pc_alphas = [0.75, 0.6, 0.6]
    cells = [
        # RGB
        [('img', dict(arr=rgb_in_masked, cmap=None)),
         ('img', dict(arr=d['rgb_pred'], cmap=None)),
         ('img', dict(arr=d['rgb_in'],   cmap=None))],
        # Depth
        [('img', dict(arr=depth_in_masked, cmap='viridis')),
         ('img', dict(arr=depth_pred_n,    cmap='viridis')),
         ('img', dict(arr=depth_in_n,      cmap='viridis'))],
        # Point Cloud
        [('pc',  dict(pts=pc_vis,  color=pc_colors[0], s=pc_sizes[0], alpha=pc_alphas[0])),
         ('pc',  dict(pts=pc_pred, color=pc_colors[1], s=pc_sizes[1], alpha=pc_alphas[1])),
         ('pc',  dict(pts=pc_in,   color=pc_colors[2], s=pc_sizes[2], alpha=pc_alphas[2]))],
        # Params
        [('txt', dict(rows=rows_in,   highlight=True)),
         ('txt', dict(rows=rows_pred, highlight=False)),
         ('txt', dict(rows=rows_gt,   highlight=False))],
    ]

    def render_cell(ax, kind, payload, title=None, ylabel=None):
        if kind == 'img':
            ax.imshow(payload['arr'], cmap=payload['cmap'])
            ax.set_xticks([]); ax.set_yticks([])
        elif kind == 'pc':
            plot_pc(ax, payload['pts'], title='',
                    color=payload['color'], s=payload['s'],
                    alpha=payload['alpha'],
                    elev=pc_elev, azim=pc_azim, lim=pc_lim)
            # tighten 3D axes so the plot fills its allotted rectangle
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax.margins(0)
        elif kind == 'txt':
            draw_param_panel(ax, payload['rows'], title='',
                             highlight_masked=payload.get('highlight', False))
        if title:
            ax.set_title(title, fontsize=13, weight='bold', pad=8)
        if ylabel:
            if kind == 'pc':
                ax.text2D(-0.06, 0.5, ylabel, transform=ax.transAxes,
                          fontsize=13, weight='bold', rotation=90,
                          va='center', ha='center')
            elif kind == 'txt':
                # text-panel ax has axis off; use ax.text rather than ylabel
                ax.text(-0.12, 0.5, ylabel, transform=ax.transAxes,
                        fontsize=13, weight='bold', rotation=90,
                        va='center', ha='center')
            else:
                ax.set_ylabel(ylabel, fontsize=13, weight='bold')

    # --- figure layout ---
    if args.layout == 'landscape':
        # 4 modality cols × 3 stage rows, 16:9-ish
        fig = plt.figure(figsize=(20, 10.6))
        gs = gridspec.GridSpec(
            nrows=4, ncols=4,
            height_ratios=[0.55, 1.0, 1.0, 1.0],
            width_ratios=[1.0, 1.0, 1.0, 1.25],   # params column wider
            hspace=0.10, wspace=0.05,
            left=0.075, right=0.985, top=0.97, bottom=0.04,
        )
    else:
        fig = plt.figure(figsize=(14, 15))
        gs = gridspec.GridSpec(
            nrows=5, ncols=3,
            height_ratios=[0.4, 1.0, 1.0, 1.2, 0.9],
            hspace=0.32, wspace=0.10,
            left=0.04, right=0.98, top=0.96, bottom=0.03,
        )

    # ── Title row ─────────────────────────────────────────────────────────
    # Training-progress numbers (current epoch from history; planned total is
    # fixed by the run config and is not stored in training_history.json).
    try:
        cur_epoch = len(json.load(open('outputs/4m_run_v2_again/training_history.json'))['train_loss'])
    except Exception:
        cur_epoch = 850
    total_epochs = 2400
    frac = min(max(cur_epoch / total_epochs, 0.0), 1.0)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(
        0.5, 0.88,
        'EmbodiedMAE-4M : reconstructions across four modalities',
        ha='center', va='center', fontsize=20, weight='bold',
        transform=title_ax.transAxes,
    )
    sub = f'Sample {name}    ·    checkpoint epoch 760 (best val so far)'
    title_ax.text(0.5, 0.62, sub, ha='center', va='center',
                  fontsize=12, color='dimgray', transform=title_ax.transAxes)

    # Progress bar — visually communicates that training is still under way.
    bar_left, bar_right = 0.22, 0.78
    bar_y, bar_h = 0.10, 0.30
    title_ax.add_patch(Rectangle(
        (bar_left, bar_y), bar_right - bar_left, bar_h,
        transform=title_ax.transAxes,
        facecolor='#e9ecef', edgecolor='#adb5bd', linewidth=1.0))
    title_ax.add_patch(Rectangle(
        (bar_left, bar_y), (bar_right - bar_left) * frac, bar_h,
        transform=title_ax.transAxes,
        facecolor='#2a9d8f', edgecolor='none'))
    title_ax.text(
        0.5, bar_y + bar_h / 2,
        f'Training in progress — epoch {cur_epoch} / {total_epochs}  ({frac*100:.0f}%)',
        ha='center', va='center', fontsize=12, weight='bold',
        color='#0b3954', transform=title_ax.transAxes,
    )
    title_ax.text(
        bar_left - 0.005, bar_y + bar_h / 2, '0',
        ha='right', va='center', fontsize=9, color='gray',
        transform=title_ax.transAxes)
    title_ax.text(
        bar_right + 0.005, bar_y + bar_h / 2, f'{total_epochs}',
        ha='left', va='center', fontsize=9, color='gray',
        transform=title_ax.transAxes)

    if args.layout == 'landscape':
        # rows = stages (Input / Pred / GT), cols = modalities
        for r, stage_label in enumerate(stage_labels):
            for c, mod_label in enumerate(mod_labels):
                kind, payload = cells[c][r]
                if kind == 'pc':
                    ax = fig.add_subplot(gs[1 + r, c], projection='3d')
                else:
                    ax = fig.add_subplot(gs[1 + r, c])
                # column header (modality name) on the top row only
                title = mod_label if r == 0 else None
                # row label (stage) on the leftmost column only
                ylabel = stage_label if c == 0 else None
                render_cell(ax, kind, payload, title=title, ylabel=ylabel)

    else:
        # portrait: rows = modalities, cols = stages
        for r, mod_label in enumerate(mod_labels):
            for c, stage_label in enumerate(stage_labels):
                kind, payload = cells[r][c]
                row_idx = 1 + r
                if kind == 'pc':
                    ax = fig.add_subplot(gs[row_idx, c], projection='3d')
                else:
                    ax = fig.add_subplot(gs[row_idx, c])
                title = stage_label if r == 0 else None
                ylabel = mod_label if c == 0 else None
                render_cell(ax, kind, payload, title=title, ylabel=ylabel)

    out = Path(args.out)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
