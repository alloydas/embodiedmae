"""Render a slide-ready architecture diagram for EmbodiedMAE-4M.

Six columns, all visualised with real data from
slide_dump3/sample_Sorghum_0_06 (the same sample used in slide_figure.png):

    Inputs (clean)  →  Embedders  →  Masked input  →  Encoder + latent  →
    Decoders  →  Reconstructions
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

REPO = Path('/work/mech-ai-scratch/alloy/embodiedmae')
SAMPLE = REPO / 'slide_dump3' / 'sample_Sorghum_0_06'
OUT = REPO / 'arch_diagram.png'


# ── Style ────────────────────────────────────────────────────────────────
MOD_COLORS = {
    'RGB':    '#e76f51',
    'Depth':  '#f4a261',
    'PC':     '#2a9d8f',
    'Params': '#577590',
}
MOD_ORDER = ['RGB', 'Depth', 'PC', 'Params']
DARK = '#264653'
ENC_FILL = '#e0e7ff'
ENC_EDGE = '#3a4cb1'


def box(ax, x, y, w, h, text, fill='white', edge=DARK, lw=1.6, fs=10,
        bold=False, rounding=0.04):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(p)
    if text:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fs,
            weight='bold' if bold else 'normal', color=DARK,
        )


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=14,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
    ))


def data_to_fig(fig, ax, x, y):
    disp = ax.transData.transform((x, y))
    return fig.transFigure.inverted().transform(disp)


def add_inset(fig, ax, x, y, w, h, projection=None):
    fx0, fy0 = data_to_fig(fig, ax, x, y)
    fx1, fy1 = data_to_fig(fig, ax, x + w, y + h)
    return fig.add_axes(
        [fx0, fy0, fx1 - fx0, fy1 - fy0],
        projection=projection,
    )


# ── Data loaders ────────────────────────────────────────────────────────
def load_sample():
    rgb_in    = np.array(Image.open(SAMPLE / 'inputs'  / 'rgb.png'))
    rgb_pred  = np.array(Image.open(SAMPLE / 'outputs' / 'rgb.png'))
    depth_in   = np.load(SAMPLE / 'inputs'  / 'depth.npy').squeeze()
    depth_pred = np.load(SAMPLE / 'outputs' / 'depth.npy').squeeze()
    pc_in   = np.load(SAMPLE / 'inputs'  / 'pointcloud.npy')
    pc_pred = np.load(SAMPLE / 'outputs' / 'pointcloud.npy')
    m_rgb   = np.load(SAMPLE / 'masks' / 'rgb_mask.npy').astype(bool)
    m_depth = np.load(SAMPLE / 'masks' / 'depth_mask.npy').astype(bool)
    m_pc    = np.load(SAMPLE / 'masks' / 'pc_mask.npy').astype(bool)
    m_text  = np.load(SAMPLE / 'masks' / 'text_mask.npy').astype(bool)
    pc_vis_idx = np.load(SAMPLE / 'masks' / 'pc_vis_point_idx.npy')
    with open(SAMPLE / 'inputs'  / 'params.json') as f:
        params_in = json.load(f)
    with open(SAMPLE / 'outputs' / 'params.json') as f:
        params_pred = json.load(f)
    return dict(
        rgb_in=rgb_in, rgb_pred=rgb_pred,
        depth_in=depth_in, depth_pred=depth_pred,
        pc_in=pc_in, pc_pred=pc_pred,
        m_rgb=m_rgb, m_depth=m_depth, m_pc=m_pc, m_text=m_text,
        pc_vis_idx=pc_vis_idx,
        params_in=params_in, params_pred=params_pred,
    )


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


# ── Inset renderers ─────────────────────────────────────────────────────
def render_rgb(ax, rgb):
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])


def render_depth(ax, depth):
    lo, hi = depth.min(), depth.max()
    ax.imshow((depth - lo) / (hi - lo + 1e-8), cmap='viridis')
    ax.set_xticks([]); ax.set_yticks([])


def render_pc(ax, pc, color):
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1],
               c=color, s=0.7, alpha=0.7,
               depthshade=False, edgecolors='none')
    lim = max(np.abs(pc).max(), 1e-3) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=18, azim=-75)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render_params(ax, params, kind='in'):
    """kind: 'in' (target_raw) | 'pred' (pred_raw) | 'masked' (show MASKED)."""
    ax.axis('off')
    plant = params.get('plant', {})
    leaf0 = (params.get('leaves') or [{}])[0]
    if kind == 'masked':
        rows = [('Plant',   '[MASKED]'),
                ('L0.sp',   '[MASKED]'),
                ('L0.ln',   '[MASKED]'),
                ('L0.ba',   '[MASKED]'),
                ('...',     '')]
    else:
        src_p = plant.get('pred_raw' if kind == 'pred' else 'target_raw', {}) or {}
        src_l = leaf0.get('pred_raw' if kind == 'pred' else 'target_raw', {}) or {}
        rows = [
            ('Plant.sl',   f"{src_p.get('sl', 0):.2f}"),
            ('Plant.ps_x', f"{src_p.get('ps_x', 0):.2f}"),
            ('L0.sp',      f"{src_l.get('sp', 0):.2f}"),
            ('L0.ln',      f"{src_l.get('ln', 0):.2f}"),
            ('L0.ba',      f"{src_l.get('ba', 0):.1f}°"),
        ]
    y = 0.92
    for k, v in rows:
        masked = v == '[MASKED]'
        ax.text(0.05, y, k, fontsize=7.5, family='monospace',
                transform=ax.transAxes, va='top', color=DARK)
        ax.text(0.98, y, v, fontsize=7.5, family='monospace',
                transform=ax.transAxes, va='top', ha='right',
                color='crimson' if masked else DARK,
                weight='bold' if masked else 'normal')
        y -= 0.165


def render_latent(ax):
    """A schematic of the latent token sequence after the encoder.

    Shows CLS + visible tokens (mask_ratio = 0.15) from each modality, coloured
    by source modality so the audience can see what the encoder operates on.
    """
    # Counts approximated by mask_ratio = 0.15 on per-modality token totals
    n_visible = {
        'RGB':    30,   # 196 × 0.15
        'Depth':  30,
        'PC':     30,
        'Params':  4,   # 25 × 0.15
    }
    seq = [(DARK, 'CLS')]
    for mod in MOD_ORDER:
        seq.extend([(MOD_COLORS[mod], mod)] * n_visible[mod])

    n_cols = 16
    n_rows = (len(seq) + n_cols - 1) // n_cols
    cell = 0.9
    for i, (c, _) in enumerate(seq):
        r = i // n_cols
        col = i % n_cols
        ax.add_patch(Rectangle((col, n_rows - 1 - r), cell, cell,
                               facecolor=c, edgecolor='white', linewidth=0.4))
    # CLS gets a black border to mark it
    ax.add_patch(Rectangle((0, n_rows - 1), cell, cell,
                           facecolor='none', edgecolor='black', linewidth=1.2))
    ax.set_xlim(-0.5, n_cols + 0.5)
    ax.set_ylim(-0.5, n_rows + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    s = load_sample()

    fig = plt.figure(figsize=(17, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_aspect('auto')
    ax.axis('off')

    # Column anchors
    COL_INPUT  =  2
    COL_EMBED  = 16
    COL_MASK   = 32
    COL_ENC    = 47
    COL_DEC    = 70
    COL_OUTPUT = 87

    ROW_Y = [40, 30, 20, 10]
    BOX_W, BOX_H = 11, 7
    EMB_W = 13
    DEC_W = 13

    embedders = ['PatchEmbed\n→ 196 tok',
                 'PatchEmbed\n→ 196 tok',
                 'PointNet + FPS\n→ 196 tok',
                 'Char-MLP\n→ 25 tok']
    decoders  = ['Linear unpatchify\n+ MSE',
                 'Linear unpatchify\n+ MSE (norm)',
                 'FoldingNet upsamp.\n+ Chamfer',
                 'MLP regressor\n+ Smooth-L1']

    # ── Column headers ────────────────────────────────────────────────────
    headers = [
        (COL_INPUT  + BOX_W / 2, 'Inputs'),
        (COL_EMBED  + EMB_W / 2, 'Embedders'),
        (COL_MASK   + BOX_W / 2, 'Masked input'),
        (COL_ENC    + 18 / 2,    'Encoder  →  latent'),
        (COL_DEC    + DEC_W / 2, 'Decoders'),
        (COL_OUTPUT + BOX_W / 2, 'Reconstructions'),
    ]
    for x, h in headers:
        ax.text(x, 47.0, h, ha='center', va='center',
                fontsize=12, weight='bold', color=DARK)

    # ── Per-modality rows (everything except encoder) ─────────────────────
    for mod, emb_t, dec_t, y in zip(MOD_ORDER, embedders, decoders, ROW_Y):
        color = MOD_COLORS[mod]

        # Input frame
        box(ax, COL_INPUT, y - BOX_H / 2, BOX_W, BOX_H, '',
            fill=color + '22', edge=color)
        ax.text(COL_INPUT + BOX_W / 2, y + BOX_H / 2 - 0.7, mod,
                ha='center', va='top', fontsize=9, weight='bold', color=color, zorder=10)

        # Embedder
        box(ax, COL_EMBED, y - BOX_H / 2, EMB_W, BOX_H, emb_t,
            fill='white', edge=color, fs=9.5)

        # Masked-input frame
        box(ax, COL_MASK, y - BOX_H / 2, BOX_W, BOX_H, '',
            fill=color + '22', edge=color)
        ax.text(COL_MASK + BOX_W / 2, y + BOX_H / 2 - 0.7, mod,
                ha='center', va='top', fontsize=9, weight='bold', color=color, zorder=10)

        # Arrows: input → embedder → mask
        arrow(ax, COL_INPUT + BOX_W, y, COL_EMBED, y, color=color)
        arrow(ax, COL_EMBED + EMB_W, y, COL_MASK, y, color=color)

    # ── Shared encoder (one tall block) ───────────────────────────────────
    enc_x, enc_y, enc_w, enc_h = COL_ENC, 6, 18, 38
    box(ax, enc_x, enc_y, enc_w, enc_h, '',
        fill=ENC_FILL, edge=ENC_EDGE, lw=1.8)
    ax.text(enc_x + enc_w / 2, enc_y + enc_h - 1.8,
            'ViT-base  (depth = 12,  dim = 768)',
            ha='center', va='top', fontsize=10, weight='bold', color=DARK)
    ax.text(enc_x + enc_w / 2, enc_y + enc_h - 4.0,
            'union of visible tokens + CLS',
            ha='center', va='top', fontsize=8.5, style='italic', color=DARK)
    # CLS / latent legend underneath the latent grid
    ax.text(enc_x + enc_w / 2, enc_y + 1.2,
            'each tile = one latent token  (colour = source modality)',
            ha='center', va='center', fontsize=7.5, style='italic', color=DARK)

    # Encoder ← mask arrows: each masked row routes into the encoder
    for y in ROW_Y:
        arrow(ax, COL_MASK + BOX_W, y, enc_x, y, color=ENC_EDGE, lw=1.2)

    # ── Decoder + Reconstruction columns ──────────────────────────────────
    for mod, dec_t, y in zip(MOD_ORDER, decoders, ROW_Y):
        color = MOD_COLORS[mod]
        arrow(ax, enc_x + enc_w, y, COL_DEC, y, color=color)
        box(ax, COL_DEC, y - BOX_H / 2, DEC_W, BOX_H, dec_t,
            fill='white', edge=color, fs=9.5)
        arrow(ax, COL_DEC + DEC_W, y, COL_OUTPUT, y, color=color)
        # Reconstruction frame
        box(ax, COL_OUTPUT, y - BOX_H / 2, BOX_W, BOX_H, '',
            fill=color + '22', edge=color)
        ax.text(COL_OUTPUT + BOX_W / 2, y + BOX_H / 2 - 0.7, mod,
                ha='center', va='top', fontsize=9, weight='bold', color=color, zorder=10)

    # ── Footnote ──────────────────────────────────────────────────────────
    ax.text(
        50, 2.0,
        'per-modality positional + modality embeddings  ·  '
        'CLS token shared across modalities  ·  losses summed with per-modality weights',
        ha='center', va='center', fontsize=9, style='italic', color='#495057',
    )

    # ── Inset axes for the actual previews ────────────────────────────────
    fig.canvas.draw()

    pad = 0.6
    inset_h = BOX_H - 2 * pad - 1.0
    inset_w = BOX_W - 2 * pad

    # Pre-compute masked previews
    rgb_masked = apply_patch_mask(s['rgb_in'], s['m_rgb'])
    lo, hi = s['depth_in'].min(), s['depth_in'].max()
    depth_norm = (s['depth_in'] - lo) / (hi - lo + 1e-8)
    depth_masked = apply_patch_mask(
        (depth_norm * 255).astype(np.uint8), s['m_depth'], fill=0
    )
    pc_visible = s['pc_in'][s['pc_vis_idx']]

    # Per-row insets: (input, masked, output)
    INPUT_INSETS = [
        ('rgb',    lambda a: render_rgb(a, s['rgb_in'])),
        ('depth',  lambda a: render_depth(a, s['depth_in'])),
        ('pc',     lambda a: render_pc(a, s['pc_in'], MOD_COLORS['PC'])),
        ('params', lambda a: render_params(a, s['params_in'], kind='in')),
    ]
    MASKED_INSETS = [
        ('rgb',    lambda a: render_rgb(a, rgb_masked)),
        ('depth',  lambda a: (a.imshow(depth_masked, cmap='viridis'),
                              a.set_xticks([]), a.set_yticks([]))),
        ('pc',     lambda a: render_pc(a, pc_visible, MOD_COLORS['PC'])),
        ('params', lambda a: render_params(a, s['params_in'], kind='masked')),
    ]
    OUTPUT_INSETS = [
        ('rgb',    lambda a: render_rgb(a, s['rgb_pred'])),
        ('depth',  lambda a: render_depth(a, s['depth_pred'])),
        ('pc',     lambda a: render_pc(a, s['pc_pred'], MOD_COLORS['PC'])),
        ('params', lambda a: render_params(a, s['params_pred'], kind='pred')),
    ]

    for triple, y in zip(zip(INPUT_INSETS, MASKED_INSETS, OUTPUT_INSETS), ROW_Y):
        (in_kind, in_fn), (msk_kind, msk_fn), (out_kind, out_fn) = triple
        iy = y - BOX_H / 2 + pad
        in_ax  = add_inset(fig, ax, COL_INPUT  + pad, iy, inset_w, inset_h,
                           projection='3d' if in_kind == 'pc' else None)
        msk_ax = add_inset(fig, ax, COL_MASK   + pad, iy, inset_w, inset_h,
                           projection='3d' if msk_kind == 'pc' else None)
        out_ax = add_inset(fig, ax, COL_OUTPUT + pad, iy, inset_w, inset_h,
                           projection='3d' if out_kind == 'pc' else None)
        in_fn(in_ax)
        msk_fn(msk_ax)
        out_fn(out_ax)

    # Latent-space inset (inside the encoder block, lower half)
    latent_ax = add_inset(
        fig, ax, enc_x + 1.0, enc_y + 3.0, enc_w - 2.0, enc_h - 9.0,
    )
    render_latent(latent_ax)

    fig.savefig(OUT, dpi=200, facecolor='white')
    print(f'Wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
