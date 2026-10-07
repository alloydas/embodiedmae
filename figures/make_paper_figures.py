"""Paper figures: the teaser (Fig. 1) and the architecture diagram (Fig. 2).

Both are drawn from REAL data and REAL model outputs on one held-out val plant,
following the house style (show the modality, not a label; one colour for the
point cloud at every stage).  Writes paper/fig_teaser.{pdf,png} and
paper/fig_architecture.{pdf,png}.

Two species.  Sorghum uses DATA VERSION 2 (rgb_nobg.png over black, the width
leaf layout) and the finished four-stream arm e2_pcrgbdt_d2 (epoch 600); its
RGB-only generation in the teaser is therefore zero-shot (no _d2 distillation
exists yet; pass --sorghum_ckpt when one does).  Maize uses the distilled
generalist (outputs/maize_distill_all, epoch 58, the last checkpoint before the
run was cancelled).  The architecture figure is drawn on the sorghum sample.

    conda activate det
    python figures/make_paper_figures.py                   # CPU, ~2 min
    python figures/make_paper_figures.py --sorghum_view 4 --maize_plant 0 --maize_view 4
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import embodied_mae_4m as sorghum_model
import embodied_mae_4m_maize as maize_model
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M
from train_sorghum_4m import _unnorm_pix, unpatchify, _pc_token_membership

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'paper'

# ── species ─────────────────────────────────────────────────────────────────
# recipe rows: (label, token slot, scale, shift, format) on the un-normalised
# value raw = p * scale - shift.  Only fields that read directly as a trait.
SPECIES = {
    'sorghum': dict(
        title='Sorghum', ds=lambda root: SorghumDataset4M(root, split='val', num_points=8196, rgb_file='rgb_nobg.png'),
        root='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K',
        model=lambda: sorghum_model.embodied_mae_4m_base(target_points=8196),
        ckpt=str(REPO / 'outputs/e2_pcrgbdt_d2/checkpoints/checkpoint_epoch_600.pth'),
        name=lambda plant, view: f'Sorghum_{plant}_{view:02d}', n_leaf_tokens=24,
        plant_rows=[('stem len', 0, 3.0, 0.0, '{:.2f} m')],
        leaf_rows=[('len', 1, 1.25, 0.0, '{:.2f} m'), ('width', 4, 0.2, 0.0, '{:.3f} m'), ('angle', 3, 180.0, 0.0, '{:.0f}°')],
        zero_shot=True),
    'maize': dict(
        title='Maize', ds=lambda root: MaizeDataset4M(root, split='val', num_points=8192),
        root='/work/mech-ai-scratch/alloy/Maize',
        model=lambda: maize_model.embodied_mae_4m_maize_base(target_points=8192),
        ckpt=str(REPO / 'outputs/maize_distill_all/checkpoints/checkpoint_epoch_58.pth'),
        name=lambda plant, view: f'plant_{plant:04d}_{view:02d}', n_leaf_tokens=28,
        plant_rows=[('leaf count', 0, 32.0, 0.0, '{:.0f}'), ('internodes', 4, 2.0, 0.0, '{:.2f} m')],
        leaf_rows=[('len', 1, 0.85, -0.05, '{:.2f} m'), ('width', 2, 0.15, 0.0, '{:.3f} m'), ('angle', 3, 180.0, 90.0, '{:.0f}°')],
        zero_shot=False),
}


# ── style ───────────────────────────────────────────────────────────────────
C = {'rgb': '#d1495b', 'depth': '#edae49', 'pc': '#2a9d8f', 'text': '#30638e'}
NAME = {'rgb': 'RGB', 'depth': 'Depth', 'pc': 'Point cloud', 'text': 'Procedural recipe'}
DARK = '#222831'
GREY = '#6c757d'
ENC_FILL, ENC_EDGE = '#e8ecf7', '#3a4cb1'
DEC_FILL, DEC_EDGE = '#f1f3f5', '#495057'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                     'axes.linewidth': 0.6, 'pdf.fonttype': 42})


# ── drawing helpers (all in figure coordinates, 0..1) ───────────────────────
def overlay(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    return ax


def box(ax, x, y, w, h, text='', fill='white', edge=DARK, lw=0.8, fs=7, bold=False,
        rounding=0.012, z=2, color=None, style='normal', pad=0.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad},rounding_size={rounding}',
                                lw=lw, ec=edge, fc=fill, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
                weight='bold' if bold else 'normal', color=color or DARK, zorder=z + 1,
                style=style, linespacing=1.15)


def arrow(ax, p, q, color=DARK, lw=0.8, style='-|>', z=3, ls='-', shrink=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=lw,
                                 color=color, zorder=z, linestyle=ls, shrinkA=shrink, shrinkB=shrink))


def label(ax, x, y, s, fs=7, bold=False, color=DARK, ha='center', va='center', z=4, style='normal'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, weight='bold' if bold else 'normal',
            color=color, zorder=z, style=style, linespacing=1.15)


def img_axes(fig, x, y, w, h, projection=None):
    ax = fig.add_axes([x, y, w, h], projection=projection)
    if projection is None:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    return ax


def show_rgb(ax, rgb):
    ax.imshow(np.clip(rgb, 0, 1))


def show_depth(ax, d, bg=None):
    d = d.copy().astype(float)
    if bg is not None:
        d[bg] = np.nan
    cm = matplotlib.colormaps['viridis'].copy(); cm.set_bad('white')
    ax.imshow(d, cmap=cm)


def show_pc(ax, pc, color=C['pc'], s=0.35, alpha=0.75, lim=None, elev=12, azim=-62, zoom=1.25):
    # camera frame: x right, y up, z toward the camera -> plot (x, z, y)
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=color, s=s, alpha=alpha,
               depthshade=False, edgecolors='none', rasterized=True)
    lim = lim or 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.patch.set_alpha(0)


def recipe_rows(sp, params, valid, leaves=(1,), with_tokens=False):
    """Human-readable rows for a species from a (1+K, N) [0,1] tensor: the plant
    token's trait fields and the given leaf tokens.  with_tokens also returns
    the token index behind each row (None for the leaf count)."""
    p = params.clamp(0, 1).numpy() if torch.is_tensor(params) else np.clip(params, 0, 1)
    rows, toks = [], []
    for lab, slot, sc, sh, fmt in sp['plant_rows']:
        rows.append((lab, fmt.format(p[0, slot] * sc - sh))); toks.append(0)
    for i, t in enumerate(leaves):
        if valid[t] < 0.5:
            continue
        for lab, slot, sc, sh, fmt in sp['leaf_rows']:
            rows.append((f'leaf{i + 1} {lab}', fmt.format(p[t, slot] * sc - sh))); toks.append(t)
    if not any(lab == 'leaf count' for lab, *_ in sp['plant_rows']):
        rows.append(('leaves', f'{int(valid[1:].sum())}')); toks.append(None)
    return (rows, toks) if with_tokens else rows


def show_recipe(ax, rows, color=C['text'], mask=None, dim=None, fs=5.4, stacked=False):
    """mask: per-row bools -> rendered as hidden.  dim: per-row bools -> value
    greyed (a visible token passed through, not a prediction).  stacked puts
    the value under its key, for narrow boxes."""
    ax.axis('off')
    y, dy = 0.95, 1.0 / (len(rows) + 0.6)
    for i, (k, v) in enumerate(rows):
        hidden = mask is not None and mask[i]
        dimmed = dim is not None and dim[i]
        vc = '#adb5bd' if (hidden or dimmed) else DARK
        vt = '■■■■■■' if hidden else v
        ax.text(0.03, y, k, fontsize=fs, family='monospace', va='top', color=color, transform=ax.transAxes)
        if stacked:
            ax.text(0.97, y - dy * 0.45, vt, fontsize=fs, family='monospace', va='top', ha='right', color=vc, transform=ax.transAxes)
        else:
            ax.text(0.99, y, vt, fontsize=fs, family='monospace', va='top', ha='right', color=vc, transform=ax.transAxes)
        y -= dy


def token_strip(ax, counts, total=None, width=1.0, height=1.0, y0=0.0, cell_gap=0.08, lw=0.25):
    """Row of coloured cells, one per visible token (CLS first)."""
    seq = [('#111111', 'cls')] + [(C[m], m) for m, n in counts for _ in range(n)]
    n = len(seq)
    cw = width / n
    for i, (c, _) in enumerate(seq):
        ax.add_patch(Rectangle((i * cw, y0), cw * (1 - cell_gap), height, fc=c, ec='white', lw=lw))
    ax.set_xlim(0, width); ax.set_ylim(y0 - 0.02, y0 + height + 0.02); ax.axis('off')


# ── data + model ────────────────────────────────────────────────────────────
def load(species, plant, view, ckpt=None):
    sp = SPECIES[species]
    torch.manual_seed(0); np.random.seed(0)
    ds = sp['ds'](sp['root'])
    want = sp['name'](plant, view)
    idx = next(i for i, f in enumerate(ds.samples) if Path(f).name == want)
    rgb, depth, pc, par, tv, name = ds[idx]
    print(species, 'sample', name)

    model = sp['model']().eval()
    ckpt = ckpt or sp['ckpt']
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck.get('model_state_dict', ck).items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (len(miss), len(unexp))
    print(species, 'model', ckpt, 'epoch', ck.get('epoch'))

    b = [t.unsqueeze(0) for t in (rgb, depth, pc, par, tv)]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = {'species': species, 'sp': sp, 'name': name, 'epoch': ck.get('epoch'),
           'rgb': (rgb * std + mean).permute(1, 2, 0).numpy(), 'depth': depth[0].numpy(),
           'pc': pc.numpy(), 'par': par, 'tv': tv.numpy()}
    out['bg'] = out['depth'] < 1e-3

    with torch.no_grad():
        # (a) Dirichlet-masked pretraining step at the training ratio.  Paper
        # design: the parameter stream is a reconstruction target only, so it is
        # taken out of the shared budget and fully hidden (text_mask_ratio 1).
        torch.manual_seed(1)
        model.text_mask_ratio = 1.0
        _, _, (pr, pd, pp, pt), (mr, md, mp, mt) = model(*b, mask_ratio=0.8)
        model.text_mask_ratio = None
        fps = model.pc_embed.fps(b[2], model.num_pc_tokens)
        members = _pc_token_membership(b[2], fps, model.pc_embed.group_size)
        out['masked'] = dict(
            m_rgb=mr[0].numpy(), m_depth=md[0].numpy(), m_pc=mp[0].numpy(), m_text=mt[0].numpy(),
            members=members,
            rgb=(unpatchify(_unnorm_pix(model, pr, b[0]), 16, 3, 224)[0] * std + mean).permute(1, 2, 0).numpy(),
            depth=unpatchify(pd, 16, 1, 224)[0, 0].numpy(), pc=pp[0].numpy(), par=pt[0])
        # (b) single-sensor: RGB alone generates the other three
        torch.manual_seed(2)
        _, _, (pr, pd, pp, pt), _ = model(*b, visible=['rgb'])
        d2 = torch.cdist(pp, b[2]) ** 2
        out['rgb_only'] = dict(
            depth=unpatchify(pd, 16, 1, 224)[0, 0].numpy(), pc=pp[0].numpy(), par=pt[0],
            chamfer=float(d2.min(2).values.mean() + d2.min(1).values.mean()))
    print(species, 'RGB-only chamfer', out['rgb_only']['chamfer'])
    return out


def pc_visible_points(s):
    """Points belonging to the VISIBLE PC tokens of the masked step."""
    m = s['masked']; keep = np.where(m['m_pc'] < 0.5)[0]
    idx = np.unique(np.concatenate([np.asarray(m['members'][t]) for t in keep])) if len(keep) else np.zeros(0, int)
    return s['pc'][idx]


def masked_image(img, mask, fill=1.0):
    g = int(round(mask.shape[0] ** 0.5))
    up = np.kron(mask.reshape(g, g), np.ones((224 // g, 224 // g)))
    out = img.copy()
    if out.ndim == 3:
        out[up > 0.5] = fill
    else:
        out = out.astype(float); out[up > 0.5] = np.nan
    return out


# ── Figure 1: teaser ────────────────────────────────────────────────────────
def teaser(S, path):
    """S: list of two species dicts (sorghum, maize)."""
    fig = plt.figure(figsize=(6.9, 3.45))
    ax = overlay(fig)
    label(ax, 0.228, 0.972, 'In simulation: two crops, each plant with its recipe', fs=7.5, bold=True)
    label(ax, 0.745, 0.972, 'In the field: one camera, everything else inferred', fs=7.5, bold=True)
    ax.plot([0.455, 0.455], [0.05, 0.93], color='#ced4da', lw=0.6, ls=(0, (3, 3)))

    # ── left: two species × four streams → shared encoder ──
    tw, th = 0.088, 0.33
    xs = [0.012 + i * (tw + 0.004) for i in range(4)]
    ys = {0: 0.515, 1: 0.10}
    mods = ['rgb', 'depth', 'pc', 'text']
    for k, s in enumerate(S):
        y = ys[k]
        label(ax, 0.012, y + th + 0.012, s['sp']['title'], fs=6.2, bold=True, color=DARK, ha='left', va='bottom')
        for m, x in zip(mods, xs):
            box(ax, x, y, tw, th, fill='white', edge=C[m], lw=0.8, rounding=0.008)
            if k == 0:
                label(ax, x + tw / 2, y + th + 0.048, NAME[m].replace('Procedural ', ''), fs=5.4, color=C[m], va='bottom')
            iw, ih = tw - 0.01, th - 0.02
            if m == 'rgb':
                show_rgb(img_axes(fig, x + 0.005, y + 0.01, iw, ih), s['rgb'])
            elif m == 'depth':
                show_depth(img_axes(fig, x + 0.005, y + 0.01, iw, ih), s['depth'], s['bg'])
            elif m == 'pc':
                show_pc(img_axes(fig, x + 0.002, y + 0.005, iw + 0.006, ih + 0.01, projection='3d'), s['pc'], s=0.25, zoom=1.35)
            else:
                show_recipe(img_axes(fig, x + 0.003, y + 0.006, iw + 0.004, ih), recipe_rows(s['sp'], s['par'], s['tv'], (1,)), fs=3.9, stacked=True)
    # species name sits above its row; the top row's name shares the line with the stream labels
    ex, ey, ew, eh = 0.388, 0.22, 0.06, 0.56
    box(ax, ex, ey, ew, eh, 'shared\nencoder\n+\ndecoder', fill=ENC_FILL, edge=ENC_EDGE, fs=6.2, bold=True)
    for k in (0, 1):
        yy = ys[k] + th / 2
        arrow(ax, (xs[3] + tw, yy), (ex, yy), color=DARK, lw=0.8, style='<|-|>')
    label(ax, ex + ew / 2, 0.905, '80 % of tokens\nhidden; any\nstream completes\nany other', fs=4.5, color=GREY)
    label(ax, ex + ew / 2, 0.80, 'pretrain', fs=6, color=ENC_EDGE, style='italic')

    # ── right: per species, one RGB photo → student → cloud + recipe ──
    rows_y = {0: 0.515, 1: 0.10}
    rw, rh = 0.105, 0.345
    mx, mw, mh = 0.60, 0.07, 0.22
    ow, oh = 0.135, rh
    for k, s in enumerate(S):
        y = rows_y[k]; r = s['rgb_only']
        rx = 0.472
        box(ax, rx, y, rw, rh, fill='white', edge=C['rgb'], lw=0.8, rounding=0.008)
        label(ax, rx, y + rh + 0.015, f"{s['sp']['title']}: one RGB photo", fs=5.6, bold=True, color=C['rgb'], ha='left', va='bottom')
        show_rgb(img_axes(fig, rx + 0.005, y + 0.01, rw - 0.01, rh - 0.02), s['rgb'])
        my = y + (rh - mh) / 2
        box(ax, mx, my, mw, mh, 'single-\nsensor\nstudent', fill=ENC_FILL, edge=ENC_EDGE, fs=5.6, bold=True)
        arrow(ax, (rx + rw, y + rh / 2), (mx, y + rh / 2), color=C['rgb'])
        # cloud
        ox = 0.69
        box(ax, ox, y, ow, oh, fill='white', edge=C['pc'], lw=0.8, rounding=0.008)
        label(ax, ox + 0.005, y + oh - 0.012, 'point cloud', fs=5.4, bold=True, color=C['pc'], ha='left', va='top')
        show_pc(img_axes(fig, ox + 0.03, y + 0.02, ow - 0.035, oh - 0.07, projection='3d'), r['pc'], lim=1.0, s=0.3, zoom=1.35)
        label(ax, ox + 0.005, y + 0.012, f"Chamfer {r['chamfer']:.4f} vs. true", fs=4.6, color=GREY, ha='left', va='bottom')
        arrow(ax, (mx + mw, y + rh / 2), (ox, y + rh / 2), color=C['pc'], lw=0.7)
        # recipe
        px = ox + ow + 0.012; pw = 0.985 - px
        box(ax, px, y, pw, oh, fill='white', edge=C['text'], lw=0.8, rounding=0.008)
        label(ax, px + 0.005, y + oh - 0.012, 'procedural recipe', fs=5.4, bold=True, color=C['text'], ha='left', va='top')
        show_recipe(img_axes(fig, px + 0.004, y + 0.006, pw - 0.008, oh - 0.05), recipe_rows(s['sp'], r['par'], s['tv'], (1, 2)), fs=4.2)
        arrow(ax, (ox + ow, y + rh / 2), (px, y + rh / 2), color=C['text'], lw=0.7)

    zs = ', '.join(f"{s['sp']['title'].lower()} {'zero-shot from the pretrained arm' if s['sp']['zero_shot'] else 'distilled student'}" for s in S)
    label(ax, 0.5, 0.012, f"held-out plants {S[0]['name']} and {S[1]['name']}; real model outputs on the right ({zs}); no ground truth shown",
          fs=4.8, color=GREY, va='bottom', style='italic')
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    print('wrote', path)


# ── Figure 2: architecture ──────────────────────────────────────────────────
def architecture(s, path):
    fig = plt.figure(figsize=(6.9, 5.3))
    ax = overlay(fig)
    m = s['masked']
    n_vis = {k: int((m[f'm_{k}'] < 0.5).sum()) for k in ('rgb', 'depth', 'pc', 'text')}
    L = {'rgb': 196, 'depth': 196, 'pc': 196, 'text': 25}

    # ---------- (a) pretraining ----------
    TOP, BOT = 0.985, 0.455
    label(ax, 0.012, TOP, '(a)  Four-stream masked pretraining', fs=8, bold=True, ha='left', va='top')
    cols = {'in': 0.012, 'tok': 0.14, 'mask': 0.283, 'enc': 0.41, 'dec': 0.62, 'out': 0.86}
    heads = [('in', 'input'), ('tok', 'tokeniser'), ('mask', 'visible tokens'),
             ('enc', 'encoder'), ('dec', 'decoder'), ('out', 'reconstruction')]
    hy = 0.935
    cw = {'in': 0.11, 'tok': 0.125, 'mask': 0.11, 'enc': 0.125, 'dec': 0.22, 'out': 0.125}
    for k, t in heads:
        label(ax, cols[k] + cw[k] / 2, hy, t, fs=6.5, bold=True, color=GREY)

    rows = ['rgb', 'depth', 'pc', 'text']
    rh = 0.088; gap = 0.01
    ry = {r: hy - 0.03 - (i + 1) * (rh + gap) + gap for i, r in enumerate(rows)}
    tok_txt = {'rgb': '16×16 patches\n→ 196 tokens', 'depth': '16×16 patches\n→ 196 tokens',
               'pc': 'FPS → 196 centres\nkNN 32, mini-PointNet\n→ 196 tokens',
               'text': '1 plant + 24 leaf tokens\n9 floats each (maize 28 / 14)\nLinear–GELU–LN'}
    head_txt = {'rgb': 'linear → 16²·3 per patch\nnorm-pixel MSE on hidden patches',
                'depth': 'linear → 16² per patch\nMSE on per-image min–max target',
                'pc': 'fold a 7×7 grid → 41 pts per token\nQAL Chamfer on the whole cloud',
                'text': 'MLP → 9 floats per token\nSmooth-L1 on hidden, real tokens'}
    vis_pts = pc_visible_points(s)
    hidden_leaf = next((t for t in range(1, 25) if s['tv'][t] > 0.5 and m['m_text'][t] > 0.5), 1)
    n_vis['text'] = int((m['m_text'] < 0.5).sum())
    rows_gt, toks = recipe_rows(s['sp'], s['par'], s['tv'], (hidden_leaf,), with_tokens=True)
    tok_hidden = [bool(m['m_text'][t] > 0.5) if t is not None else False for t in toks]
    par_rec = torch.where(torch.as_tensor(m['m_text'])[:, None] > 0.5, m['par'], s['par'])
    for r in rows:
        y = ry[r]; c = C[r]
        # input
        box(ax, cols['in'], y, cw['in'], rh, fill='white', edge=c, lw=0.8, rounding=0.008)
        label(ax, cols['in'] + 0.004, y + rh - 0.012, NAME[r], fs=5.2, bold=True, color=c, ha='left', va='top')
        iw, ih = cw['in'] - 0.04, rh - 0.03
        ix = cols['in'] + 0.034
        if r == 'rgb':
            show_rgb(img_axes(fig, ix, y + 0.006, iw, ih), s['rgb'])
        elif r == 'depth':
            show_depth(img_axes(fig, ix, y + 0.006, iw, ih), s['depth'], s['bg'])
        elif r == 'pc':
            show_pc(img_axes(fig, ix - 0.004, y + 0.002, iw + 0.008, ih + 0.012, projection='3d'), s['pc'], s=0.25)
        else:
            show_recipe(img_axes(fig, cols['in'] + 0.004, y + 0.004, cw['in'] - 0.008, rh - 0.03), rows_gt, fs=4.2)
        # tokeniser
        box(ax, cols['tok'], y, cw['tok'], rh, tok_txt[r], fill='white', edge=c, fs=5.2, rounding=0.008)
        arrow(ax, (cols['in'] + cw['in'], y + rh / 2), (cols['tok'], y + rh / 2), color=c, lw=0.7)
        # masked
        box(ax, cols['mask'], y, cw['mask'], rh, fill='white', edge=c, lw=0.8, rounding=0.008)
        label(ax, cols['mask'] + 0.004, y + rh - 0.012,
              f'{n_vis[r]} / {L[r]} visible' + (' — target' if r == 'text' else ''), fs=5.0, color=c, ha='left', va='top')
        mx = cols['mask'] + 0.034
        if r == 'rgb':
            show_rgb(img_axes(fig, mx, y + 0.006, iw, ih), masked_image(s['rgb'], m['m_rgb']))
        elif r == 'depth':
            show_depth(img_axes(fig, mx, y + 0.006, iw, ih), masked_image(s['depth'], m['m_depth']), s['bg'])
        elif r == 'pc':
            show_pc(img_axes(fig, mx - 0.004, y + 0.002, iw + 0.008, ih + 0.012, projection='3d'), vis_pts, s=0.25)
        else:
            show_recipe(img_axes(fig, cols['mask'] + 0.004, y + 0.004, cw['mask'] - 0.008, rh - 0.03), rows_gt, mask=tok_hidden, fs=4.2)
        arrow(ax, (cols['tok'] + cw['tok'], y + rh / 2), (cols['mask'], y + rh / 2), color=c, lw=0.7)
        arrow(ax, (cols['mask'] + cw['mask'], y + rh / 2), (cols['enc'], y + rh / 2), color=c, lw=0.7)
        # decoder head + output
        hx = cols['dec']
        box(ax, hx, y, cw['dec'], rh, head_txt[r], fill='white', edge=c, fs=5.0, rounding=0.008)
        arrow(ax, (hx + cw['dec'], y + rh / 2), (cols['out'], y + rh / 2), color=c, lw=0.7)
        box(ax, cols['out'], y, cw['out'], rh, fill='white', edge=c, lw=0.8, rounding=0.008)
        ox = cols['out'] + 0.034
        if r == 'rgb':
            show_rgb(img_axes(fig, ox, y + 0.006, iw, ih), m['rgb'])
        elif r == 'depth':
            show_depth(img_axes(fig, ox, y + 0.006, iw, ih), m['depth'], s['bg'])
        elif r == 'pc':
            show_pc(img_axes(fig, ox - 0.004, y + 0.002, iw + 0.008, ih + 0.012, projection='3d'), m['pc'], s=0.25)
        else:
            show_recipe(img_axes(fig, cols['out'] + 0.004, y + 0.004, cw['out'] - 0.008, rh - 0.03),
                        recipe_rows(s['sp'], par_rec, s['tv'], (hidden_leaf,)),
                        dim=[(t is not None and not h) for t, h in zip(toks, tok_hidden)], fs=4.2)

    # encoder block
    ey0 = ry['text']; ey1 = ry['rgb'] + rh
    box(ax, cols['enc'], ey0, cw['enc'], ey1 - ey0, fill=ENC_FILL, edge=ENC_EDGE, lw=0.9, rounding=0.01)
    label(ax, cols['enc'] + cw['enc'] / 2, ey1 - 0.014, 'ViT-B encoder', fs=6.2, bold=True, color=ENC_EDGE, va='top')
    label(ax, cols['enc'] + cw['enc'] / 2, ey1 - 0.04, '12 blocks, d = 768\nsees CLS + visible\ntokens only', fs=5.0, color=DARK, va='top')
    total_vis = sum(n_vis.values())
    tax = img_axes(fig, cols['enc'] + 0.008, ey0 + 0.1, cw['enc'] - 0.016, 0.02)
    token_strip(tax, [(r, n_vis[r]) for r in rows])
    label(ax, cols['enc'] + cw['enc'] / 2, ey0 + 0.09, f'{total_vis + 1} tokens in', fs=4.8, color=GREY, va='top')
    label(ax, cols['enc'] + cw['enc'] / 2, ey0 + 0.012,
          'Dirichlet(α=1) splits the\nvisible budget over the three\nsensor streams; ≥ 25 % hidden', fs=4.6, color=DARK, va='bottom')
    # decoder trunk (thin, between encoder and heads)
    dx = cols['enc'] + cw['enc'] + 0.012; dw = cols['dec'] - dx - 0.012
    box(ax, dx, ey0, dw, ey1 - ey0, fill=DEC_FILL, edge=DEC_EDGE, lw=0.8, rounding=0.01)
    arrow(ax, (cols['enc'] + cw['enc'], (ey0 + ey1) / 2), (dx, (ey0 + ey1) / 2), color=ENC_EDGE, lw=0.9)
    ax.text(dx + dw / 2, (ey0 + ey1) / 2, 'decoder: 8 blocks, d = 512\nmask tokens restore all 613 positions',
            rotation=90, ha='center', va='center', fontsize=5.0, color=DARK, zorder=5)
    for r in rows:
        arrow(ax, (dx + dw, ry[r] + rh / 2), (cols['dec'], ry[r] + rh / 2), color=C[r], lw=0.7)
    label(ax, 0.5, ry['text'] - 0.018, 'The decoder layout never depends on the mask, so two forward passes with different visible sets '
          'give position-aligned features — the hook for (b).', fs=5.4, color=GREY, style='italic', va='top')

    # ---------- (b) teacher / student ----------
    label(ax, 0.012, BOT - 0.005, '(b)  A vision-complete teacher and single-sensor students', fs=8, bold=True, ha='left', va='top')
    by = 0.05; bh = BOT - 0.07 - by
    # teacher
    tx, tw_ = 0.012, 0.36
    box(ax, tx, by, tw_, bh, fill='#fafafa', edge='#adb5bd', lw=0.7, rounding=0.012)
    label(ax, tx + 0.008, by + bh - 0.012, 'frozen teacher — all four streams, nothing hidden', fs=5.8, bold=True, ha='left', va='top')
    thumbs = [('rgb', lambda a: show_rgb(a, s['rgb'])), ('depth', lambda a: show_depth(a, s['depth'], s['bg'])),
              ('pc', lambda a: show_pc(a, s['pc'], s=0.2, lim=1.0)), ('text', None)]
    tw0 = 0.058; ty = by + 0.065; th0 = bh - 0.115
    for i, (mo, fn) in enumerate(thumbs):
        x = tx + 0.012 + i * (tw0 + 0.008)
        box(ax, x, ty, tw0, th0, fill='white', edge=C[mo], lw=0.7, rounding=0.006)
        if mo == 'pc':
            show_pc(img_axes(fig, x + 0.002, ty + 0.004, tw0 - 0.004, th0 - 0.008, projection='3d'), s['pc'], s=0.2, lim=1.0, zoom=1.3)
        elif mo == 'text':
            show_recipe(img_axes(fig, x + 0.002, ty + 0.004, tw0 - 0.004, th0 - 0.012), recipe_rows(s['sp'], s['par'], s['tv'], (1,)), fs=4.0, stacked=True)
        else:
            fn(img_axes(fig, x + 0.004, ty + 0.004, tw0 - 0.008, th0 - 0.008))
    tex = tx + 0.012 + 4 * (tw0 + 0.008) + 0.004; tew = tx + tw_ - tex - 0.01
    box(ax, tex, ty, tew, th0, 'encoder\n+\ndecoder', fill=ENC_FILL, edge=ENC_EDGE, fs=5.4, bold=True)
    arrow(ax, (tex - 0.006, ty + th0 / 2), (tex, ty + th0 / 2), color=DARK, lw=0.7)
    label(ax, tx + tw_ / 2, by + 0.012, 'sees the recipe (privileged);  emits  c$^T$ (CLS)  and  F$^T$ ∈ ℝ$^{613×512}$', fs=5.2, color=DARK, va='bottom')

    # student
    sx, sw_ = 0.40, 0.36
    box(ax, sx, by, sw_, bh, fill='#fafafa', edge='#adb5bd', lw=0.7, rounding=0.012)
    label(ax, sx + 0.008, by + bh - 0.012, 'student — one source stream, the rest absent', fs=5.8, bold=True, ha='left', va='top')
    x = sx + 0.012
    box(ax, x, ty, tw0, th0, fill='white', edge=C['rgb'], lw=0.7, rounding=0.006)
    show_rgb(img_axes(fig, x + 0.004, ty + 0.004, tw0 - 0.008, th0 - 0.008), s['rgb'])
    for i, mo in enumerate(['depth', 'pc', 'text']):
        xx = x + (i + 1) * (tw0 + 0.008)
        box(ax, xx, ty, tw0, th0, '∅', fill='white', edge=C[mo], lw=0.7, rounding=0.006, fs=9, color='#ced4da')
        label(ax, xx + tw0 / 2, ty + 0.012, NAME[mo].split()[0].lower(), fs=4.4, color=C[mo], va='bottom')
    label(ax, x + tw0 / 2, ty - 0.006, 'optionally 50 % of\nits tokens hidden', fs=4.0, color=GREY, va='top')
    sex = sx + 0.012 + 4 * (tw0 + 0.008) + 0.004; sew = sx + sw_ - sex - 0.01
    box(ax, sex, ty, sew, th0, 'encoder\n+\ndecoder', fill=ENC_FILL, edge=ENC_EDGE, fs=5.4, bold=True)
    arrow(ax, (sex - 0.006, ty + th0 / 2), (sex, ty + th0 / 2), color=C['rgb'], lw=0.7)
    label(ax, sx + sw_ / 2, by + 0.012, 'init = teacher weights;  emits  c$^S$, F$^S$ and all four reconstructions', fs=5.2, color=DARK, va='bottom')

    # losses
    lx = 0.775; lw_ = 0.213
    box(ax, lx, by, lw_, bh, fill='white', edge=DARK, lw=0.8, rounding=0.012)
    label(ax, lx + lw_ / 2, by + bh - 0.012, 'losses, 80 % of steps', fs=5.8, bold=True, va='top')
    lines = [
        ('reconstruct all four', 'ground truth, same ℓ$_m$ as (a)', DARK),
        ('match F$^T$', 'MSE on the 417 generated\npositions only, ×1.0', C['pc']),
        ('match c$^T$', 'MSE on CLS, ×0.5', ENC_EDGE),
        ('other 20 % of steps', 'a plain step from (a) —\nkeeps every head alive', GREY),
    ]
    yy = by + bh - 0.045
    for t, d, c in lines:
        label(ax, lx + 0.01, yy, t, fs=5.6, bold=True, color=c, ha='left', va='top')
        label(ax, lx + 0.01, yy - 0.022, d, fs=4.9, color=DARK, ha='left', va='top')
        yy -= 0.062 if '\n' in d else 0.05
    arrow(ax, (sx + sw_, by + bh * 0.55), (lx, by + bh * 0.55), color=DARK, lw=0.8)
    arrow(ax, (tx + tw_, by + bh * 0.3), (sx, by + bh * 0.3), color='#adb5bd', lw=0.8, ls=(0, (2, 2)))
    label(ax, (tx + tw_ + sx) / 2, by + bh * 0.3 + 0.012, 'targets', fs=4.8, color=GREY)

    label(ax, 0.5, 0.012, f'all panels: held-out sorghum plant {s["name"]} (data v2), real inputs, masks and model outputs; '
          'the maize model is identical apart from its recipe tokens (1 + 28 tokens of 14 floats)', fs=5, color=GREY, va='bottom', style='italic')
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sorghum_ckpt', default=None); ap.add_argument('--maize_ckpt', default=None)
    ap.add_argument('--sorghum_plant', type=int, default=10); ap.add_argument('--sorghum_view', type=int, default=4)
    ap.add_argument('--maize_plant', type=int, default=0); ap.add_argument('--maize_view', type=int, default=4)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    so = load('sorghum', args.sorghum_plant, args.sorghum_view, args.sorghum_ckpt)
    ma = load('maize', args.maize_plant, args.maize_view, args.maize_ckpt)
    teaser([so, ma], out / 'fig_teaser.pdf')
    architecture(so, out / 'fig_architecture.pdf')




if __name__ == '__main__':
    main()
