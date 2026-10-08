"""Paper figures: the teaser (Fig. 1) and the architecture diagram (Fig. 2).

Both are drawn from REAL data and REAL model outputs on one held-out val plant,
following the house style (show the modality, not a label; one colour for the
point cloud at every stage).  Writes paper/fig_teaser.{pdf,png} and
paper/fig_architecture.{pdf,png}.

Two species, read from the raw sample folders committed under paper/data/
(so the figures rebuild off-cluster; only the checkpoints are not in git):
  paper/data/Sorghum_10_04         sorghum DATA VERSION 2 (rgb_nobg.png, width layout)
  paper/data/Maize_1_plant_0004_04 the Maize_1 re-render (rgba.png, simplified XML)
Sorghum runs through the finished four-stream arm e2_pcrgbdt_d2 (epoch 600), so
its RGB-only generation is zero-shot (no _d2 distillation yet).  Maize runs
through the distilled generalist trained on the first maize render
(outputs/maize_distill_all, epoch 58): Maize_1 keeps that render's rgb.png,
depth packing and 8192-point cloud, so the model reads it, but no model has
been trained on Maize_1 itself and its recipe tokens follow the OLD schema;
the ground-truth recipe shown comes from the Maize_1 XML.

    conda activate det
    python figures/make_paper_figures.py                   # CPU, ~2 min
    python figures/make_paper_figures.py --sorghum_dir <folder> --maize_dir <folder>
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

import xml.etree.ElementTree as ET
import open3d as o3d
from PIL import Image

import embodied_mae_4m as sorghum_model
import embodied_mae_4m_maize as maize_model
from train_sorghum_4m import _unnorm_pix, unpatchify, _pc_token_membership

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'paper'

# ── species ─────────────────────────────────────────────────────────────────
# recipe rows: (label, token slot, scale, shift, format) on the un-normalised
# value raw = p * scale - shift.  Only fields that read directly as a trait.
def _maize1_gt(folder):
    """Ground-truth recipe rows from a Maize_1 XML (<plant height> + per-leaf
    attributes).  Returns (rows_by_leaf_index, n_leaves, height)."""
    root = ET.parse(next(Path(folder).glob('*_spline.xml'))).getroot()
    leaves = root.findall('.//leaf')
    rows = {int(l.get('id')): dict(len=float(l.get('leafLength')), width=float(l.get('leafWidth')),
                                   angle=float(l.get('leafAngle'))) for l in leaves}
    return rows, len(leaves), float(root.get('height'))


SPECIES = {
    'sorghum': dict(
        title='Sorghum', folder=str(REPO / 'paper/data/Sorghum_10_04'), num_points=8196,
        rgb_file='rgb_nobg.png', pc_glob='*_nc_cam.ply',
        model=lambda: sorghum_model.embodied_mae_4m_base(target_points=8196),
        ckpt=str(REPO / 'outputs/e2_pcrgbdt_d2/checkpoints/checkpoint_epoch_600.pth'),
        n_leaf_tokens=24,
        params=lambda folder: sorghum_model.load_spline_params(next(Path(folder).glob('*_spline.yml')), 24),
        plant_rows=[('stem len', 0, 3.0, 0.0, '{:.2f} m')],
        leaf_rows=[('len', 1, 1.25, 0.0, '{:.2f} m'), ('width', 4, 0.2, 0.0, '{:.3f} m'), ('angle', 3, 180.0, 0.0, '{:.0f}°')],
        zero_shot=True),
    'maize': dict(
        title='Maize', folder=str(REPO / 'paper/data/Maize_1_plant_0004_04'), num_points=8192,
        rgb_file='rgb.png', pc_glob='pointcloud_cam.ply',
        model=lambda: maize_model.embodied_mae_4m_maize_base(target_points=8192),
        ckpt=str(REPO / 'outputs/maize_distill_all/checkpoints/checkpoint_epoch_58.pth'),
        n_leaf_tokens=28, n_params=14,
        params=None,                      # Maize_1 XML is not the model's schema: GT rows come from _maize1_gt
        plant_rows=[('leaf count', 0, 32.0, 0.0, '{:.0f}')],
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
    """Human-readable rows for a species from a (1+K, N) [0,1] tensor in the
    MODEL's schema: the plant token's trait fields and the given leaf tokens.
    with_tokens also returns the token index behind each row (None for the
    leaf count)."""
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


def gt_rows(s, leaves=(1,)):
    """Ground-truth recipe rows for display.  Sorghum: from the model-schema
    tensor.  Maize_1: straight from its XML (same row labels as the model's)."""
    if s['gt_xml'] is None:
        return recipe_rows(s['sp'], s['par'], s['tv'], leaves)
    rows_by_leaf, n, height = s['gt_xml']
    rows = [('leaf count', f'{n}'), ('height', f'{height:.2f} m')]
    for i, t in enumerate(leaves):
        r = rows_by_leaf.get(t - 1)       # token t holds leaf id t-1
        if r is None:
            continue
        rows += [(f'leaf{i + 1} len', f"{r['len']:.2f} m"), (f'leaf{i + 1} width', f"{r['width']:.3f} m"),
                 (f'leaf{i + 1} angle', f"{r['angle']:.0f}°")]
    return rows


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
def load_folder(sp):
    """Build the model's input tensors from one raw sample folder, the way the
    datasets do: RGB (RGBA composited over black) -> 224, ImageNet-normalised;
    big-endian packed depth -> [0,1] -> 224; cloud sampled to num_points,
    centred, unit sphere; recipe tokens from the spline file when the folder
    carries the model's schema, else zeros with a validity mask."""
    folder = Path(sp['folder'])
    im = Image.open(folder / sp['rgb_file'])
    if im.mode == 'RGBA':
        a = np.asarray(im).astype(np.float32) / 255.0
        rgb8 = (a[..., :3] * a[..., 3:4] * 255).round().astype(np.uint8)
        im = Image.fromarray(rgb8)
    im = im.convert('RGB').resize((224, 224), Image.BILINEAR)
    rgb = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = (rgb - mean) / std

    dp = np.asarray(Image.open(folder / 'depth.png'))
    if dp.ndim == 3 and dp.shape[2] == 4:
        d = dp.astype(np.uint64)
        v = (d[..., 0] * 256 ** 3 + d[..., 1] * 256 ** 2 + d[..., 2] * 256 + d[..., 3]) / float(256 ** 4 - 1)
    else:
        v = dp.astype(np.float64) / np.iinfo(dp.dtype).max
    depth = torch.from_numpy(np.asarray(Image.fromarray(v.astype(np.float32), mode='F')
                                        .resize((224, 224), Image.BILINEAR))).unsqueeze(0)

    pts = np.asarray(o3d.io.read_point_cloud(str(next(folder.glob(sp['pc_glob'])))).points, dtype=np.float32)
    rng = np.random.default_rng(0)
    n = sp['num_points']
    idx = rng.choice(len(pts), n, replace=len(pts) < n)
    pts = pts[idx]; pts -= pts.mean(0); pts /= max(np.linalg.norm(pts, axis=1).max(), 1e-8)
    pc = torch.from_numpy(pts)

    K = sp['n_leaf_tokens']
    if sp['params'] is not None:
        tv, par = sp['params'](folder); gt_xml = None
    else:
        gt_xml = _maize1_gt(folder)
        n_params = sp['n_params']
        par = torch.zeros(1 + K, n_params); tv = torch.zeros(1 + K); tv[:1 + min(K, gt_xml[1])] = 1.0
    return rgb, depth, pc, par, tv, gt_xml


def load(species, ckpt=None, folder=None):
    sp = dict(SPECIES[species]); sp['folder'] = folder or sp['folder']
    torch.manual_seed(0); np.random.seed(0)
    rgb, depth, pc, par, tv, gt_xml = load_folder(sp)
    name = Path(sp['folder']).name
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
    out = {'species': species, 'sp': sp, 'name': name, 'epoch': ck.get('epoch'), 'gt_xml': gt_xml,
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
                show_recipe(img_axes(fig, x + 0.003, y + 0.006, iw + 0.004, ih), gt_rows(s, (1,)), fs=3.9, stacked=True)
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

    zs = ', '.join(f"{s['sp']['title'].lower()} {'zero-shot (pretrained arm)' if s['sp']['zero_shot'] else 'distilled'}" for s in S)
    label(ax, 0.5, 0.012, f"held-out {S[0]['name']} and {S[1]['name']} (raw files in paper/data); right: real outputs, {zs}; no ground truth shown",
          fs=4.8, color=GREY, va='bottom', style='italic')
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    print('wrote', path)


# ── Figure 2 (old column layout, kept for comparison: --style columns) ──────
def architecture_columns(s, path):
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


# ── Figure 2: architecture, EmbodiedMAE-style (arXiv 2505.10105, Fig. 1) ────
# Rows of real images: input with the 14x14 patch grid -> masked input (dark
# hidden patches) -> the visible patches -> tokens -> one tall encoder -> the
# full 613-position sequence with mask tokens -> a Decoder per stream -> the
# reconstruction.  A compact teacher/student panel on the right.
MASK_FILL = (0.33, 0.33, 0.33)
MASK_TOKEN = '#9aa0a6'


def _pill(ax, x, y, w, h, text, color, fs=5.2, rot=90):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.006',
                                lw=0, fc=color, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, rotation=rot, ha='center', va='center',
            fontsize=fs, color='white', weight='bold', zorder=4)


def _grid(ax, color, n=14, px=224, lw=0.25, alpha=0.75):
    for k in range(n + 1):
        v = k * px / n - 0.5
        ax.axvline(v, color=color, lw=lw, alpha=alpha); ax.axhline(v, color=color, lw=lw, alpha=alpha)
    ax.set_xlim(-0.5, px - 0.5); ax.set_ylim(px - 0.5, -0.5)


def _depth_rgb(d, bg):
    """viridis over the per-image min-max normalised depth (what the loss sees), white background."""
    d = np.asarray(d, float); d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    im = matplotlib.colormaps['viridis'](np.clip(d, 0, 1))[..., :3]
    im[bg] = 1.0
    return im


def _mask_fill(img, mask, fill=MASK_FILL):
    g = int(round(mask.shape[0] ** 0.5)); up = np.kron(mask.reshape(g, g), np.ones((224 // g, 224 // g)))
    out = img.copy(); out[up > 0.5] = fill
    return out


def _composite(gt, pred, mask):
    g = int(round(mask.shape[0] ** 0.5)); up = np.kron(mask.reshape(g, g), np.ones((224 // g, 224 // g)))
    out = gt.copy(); out[up > 0.5] = pred[up > 0.5]
    return out


def _patch_stack(img, mask, fg, k=6):
    """k VISIBLE patches of img with the most foreground, stacked vertically (16k x 16)."""
    g = 14; vis = np.where(mask < 0.5)[0]
    score = [fg[(i // g) * 16:(i // g + 1) * 16, (i % g) * 16:(i % g + 1) * 16].mean() for i in vis]
    vis = [v for _, v in sorted(zip(score, vis), key=lambda t: -t[0])[:k]]
    tiles = [img[(i // g) * 16:(i // g + 1) * 16, (i % g) * 16:(i % g + 1) * 16] for i in sorted(vis)]
    return np.concatenate(tiles, 0) if tiles else np.zeros((16, 16, 3))


def _token_column(ax, segments, gap=2):
    """segments: list of (mask_vector, colour); drawn top to bottom as a (N,1) colour
    image, visible cells in the stream colour and hidden ones grey, `gap` white rows between streams."""
    rows = []
    for mv, col in segments:
        c = np.array(matplotlib.colors.to_rgb(col)); gr = np.array(matplotlib.colors.to_rgb(MASK_TOKEN))
        rows.append(np.where(np.asarray(mv)[:, None] > 0.5, gr, c))
        rows.append(np.ones((gap, 3)))
    im = np.concatenate(rows[:-1], 0)[:, None, :]
    ax.imshow(im, aspect='auto', interpolation='nearest'); ax.axis('off')


ROW = {'rgb': 'RGB', 'depth': 'Depth', 'pc': 'Point cloud', 'text': 'Recipe'}
NAVY = '#2b3a67'


def architecture(s, path):
    fig = plt.figure(figsize=(6.9, 3.4))
    ax = overlay(fig)
    m = s['masked']
    rows = ['rgb', 'depth', 'pc', 'text']
    n_vis = {k: int((m[f'm_{k}'] < 0.5).sum()) for k in rows}
    L = {'rgb': 196, 'depth': 196, 'pc': 196, 'text': 25}
    rgb_gt = np.clip(s['rgb'], 0, 1)
    rgb_rec = _composite(rgb_gt, np.clip(m['rgb'], 0, 1), m['m_rgb'])
    dep_gt = _depth_rgb(s['depth'], s['bg'])
    dep_pred = matplotlib.colormaps['viridis'](np.clip(m['depth'], 0, 1))[..., :3]; dep_pred[s['bg']] = 1.0
    dep_rec = _composite(dep_gt, dep_pred, m['m_depth'])
    fg = (~s['bg']).astype(float)
    vis_pts = pc_visible_points(s)
    short = lambda rows: [(k.replace('leaf1 ', 'leaf '), v) for k, v in rows]   # one leaf shown: drop its index
    rows_gt = short(recipe_rows(s['sp'], s['par'], s['tv'], (1,)))
    rows_pred = short(recipe_rows(s['sp'], m['par'], s['tv'], (1,)))

    # ---------- geometry (figure fractions) ----------
    TOP, BOT = 0.865, 0.115
    GAP = 0.02
    rh = (TOP - BOT - 3 * GAP) / 4
    ry = {r: TOP - (i + 1) * rh - i * GAP for i, r in enumerate(rows)}
    iw = rh * 3.4 / 6.9                        # square image width
    PW, TW, ENC_W, DEC_W = 0.019, 0.011, 0.128, 0.086
    x = 0.03
    X = {'inp': x}; x += iw + 0.006
    X['p1'] = x; x += PW + 0.007
    X['msk'] = x; x += iw + 0.008
    X['vis'] = x; x += TW + 0.007
    X['p2'] = x; x += PW + 0.009
    X['tin'] = x; x += TW + 0.009
    X['enc'] = x; x += ENC_W + 0.009
    X['tout'] = x; x += TW + 0.009
    X['dec'] = x; x += DEC_W + 0.009
    X['out'] = x; LEFT_END = x + iw + 0.006
    HY = 0.9                                   # column headers

    label(ax, (X['inp'] + LEFT_END) / 2, 0.985, 'Training', fs=8, bold=True, va='top')
    for k, t in (('inp', 'Input'), ('msk', 'Masked input'), ('out', 'Reconstruction')):
        label(ax, X[k] + iw / 2, HY, t, fs=6, bold=True, color=GREY, va='bottom')
    label(ax, X['dec'] + DEC_W / 2, HY, 'Decoders', fs=6, bold=True, color=GREY, va='bottom')
    label(ax, X['tin'] + TW / 2, HY, f'{1 + sum(n_vis[r] for r in ("rgb", "depth", "pc"))}\ntokens', fs=4.2, color=GREY, va='bottom')
    label(ax, X['tout'] + TW / 2, HY, '613\npositions', fs=4.2, color=GREY, va='bottom')

    pill1 = {'rgb': 'Patchify', 'depth': 'Patchify', 'pc': 'FPS & kNN', 'text': 'Tokenise'}
    pill2 = {'rgb': 'Linear', 'depth': 'Linear', 'pc': 'PointNet', 'text': 'Linear'}
    head = {'rgb': 'Decoder\n16²·3 / patch', 'depth': 'Decoder\n16² / patch', 'pc': 'Decoder\n41 pts / token', 'text': 'Decoder\n9 floats / token'}
    for r in rows:
        y, c = ry[r], C[r]
        ax.text(0.014, y + rh / 2, ROW[r], rotation=90, ha='center', va='center', fontsize=6, weight='bold', color=c, zorder=4)
        for xx in (X['inp'], X['msk'], X['out']):
            box(ax, xx, y, iw, rh, fill='white', edge=c, lw=0.9, rounding=0.004)
        pad = 0.004
        panels = (X['inp'] + pad, X['msk'] + pad, X['out'] + pad)
        count = f'{n_vis[r]} / {L[r]} visible'
        bb = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.9)
        if r in ('rgb', 'depth'):
            gt_, rec_, mk_ = (rgb_gt, rgb_rec, m['m_rgb']) if r == 'rgb' else (dep_gt, dep_rec, m['m_depth'])
            for j, (xx, im_) in enumerate(zip(panels, (gt_, _mask_fill(gt_, mk_), rec_))):
                a = img_axes(fig, xx, y + pad, iw - 2 * pad, rh - 2 * pad); a.imshow(im_); _grid(a, c)
                if j == 1:
                    a.text(0.96, 0.04, count, fontsize=4.0, color=c, ha='right', va='bottom', transform=a.transAxes, bbox=bb, zorder=6)
            st = _patch_stack(gt_, mk_, fg)
        elif r == 'pc':
            for xx, pts in ((X['inp'], s['pc']), (X['msk'], vis_pts), (X['out'], m['pc'])):
                show_pc(img_axes(fig, xx - 0.004, y - 0.004, iw + 0.008, rh + 0.008, projection='3d'), pts, s=0.3, lim=1.0, zoom=1.35)
            st = None
        else:
            show_recipe(img_axes(fig, X['inp'] + 0.003, y + 0.003, iw - 0.006, rh - 0.016), rows_gt, fs=3.7)
            show_recipe(img_axes(fig, X['msk'] + 0.003, y + 0.003, iw - 0.006, rh - 0.016), rows_gt, mask=[True] * len(rows_gt), fs=3.7)
            show_recipe(img_axes(fig, X['out'] + 0.003, y + 0.003, iw - 0.006, rh - 0.016), rows_pred, fs=3.7)
            st = None
        if r in ('pc', 'text'):
            ax.text(X['msk'] + iw - 0.006, y + 0.006, count, fontsize=4.0, color=c, ha='right', va='bottom', zorder=6, bbox=bb)
        # pills, the visible-patch stack, arrows
        _pill(ax, X['p1'], y + 0.01, PW, rh - 0.02, pill1[r], c)
        if r != 'text':
            _pill(ax, X['p2'], y + 0.01, PW, rh - 0.02, pill2[r], c)
            arrow(ax, (X['p2'] + PW + 0.001, y + rh / 2), (X['tin'] - 0.001, y + rh / 2), color=c, lw=0.7)
        if st is not None:
            sh = (rh - 0.02) * st.shape[0] / (16 * 6)
            a = img_axes(fig, X['vis'], y + rh / 2 - sh / 2, TW, sh); a.imshow(st, aspect='auto')
            for sp_ in a.spines.values():
                sp_.set_visible(True); sp_.set_edgecolor(c); sp_.set_linewidth(0.5)
            a.set_xticks([]); a.set_yticks([])
        elif r == 'pc':
            a = img_axes(fig, X['vis'], y + 0.01, TW, rh - 0.02); _token_column(a, [(np.zeros(n_vis['pc']), c)])
        else:
            label(ax, (X['vis'] + X['tin'] + TW) / 2, y + rh / 2, 'target\nonly:\nnever\nencoded', fs=4.0, color=c, style='italic')
        box(ax, X['dec'], y + 0.012, DEC_W, rh - 0.024, head[r], fill=c, edge=c, lw=0, fs=4.6, bold=True, color='white', rounding=0.008)
        arrow(ax, (X['tout'] + TW + 0.001, y + rh / 2), (X['dec'] - 0.001, y + rh / 2), color=c, lw=0.7)
        arrow(ax, (X['dec'] + DEC_W + 0.001, y + rh / 2), (X['out'] - 0.001, y + rh / 2), color=c, lw=0.7)

    # encoder-input token column: CLS + the visible tokens of the three sensor streams
    a = img_axes(fig, X['tin'], BOT, TW, TOP - BOT)
    _token_column(a, [(np.zeros(1), '#111111')] + [(np.zeros(n_vis[r]), C[r]) for r in ('rgb', 'depth', 'pc')])
    # encoder
    box(ax, X['enc'], BOT, ENC_W, TOP - BOT, fill='white', edge=NAVY, lw=1.4, rounding=0.02)
    label(ax, X['enc'] + ENC_W / 2, TOP - 0.03, 'Transformer\nEncoder', fs=7.2, bold=True, color=NAVY, va='top')
    label(ax, X['enc'] + ENC_W / 2, TOP - 0.145, 'ViT-B, 12 blocks, d = 768\nCLS + visible tokens only', fs=4.4, color=DARK, va='top')
    cw_ = (ENC_W - 0.028) / 3
    for i, (im_, col) in enumerate(((rgb_gt, C['rgb']), (dep_gt, C['depth']), (None, C['pc']))):
        xx = X['enc'] + 0.01 + i * (cw_ + 0.004); yy = BOT + 0.17
        if im_ is None:
            show_pc(img_axes(fig, xx - 0.004, yy - 0.012, cw_ + 0.008, cw_ * 6.9 / 3.4 + 0.024, projection='3d'), s['pc'], s=0.12, lim=1.0, zoom=1.4)
        else:
            a = img_axes(fig, xx, yy, cw_, cw_ * 6.9 / 3.4); a.imshow(im_)
            for sp_ in a.spines.values():
                sp_.set_visible(True); sp_.set_edgecolor(col); sp_.set_linewidth(0.5)
    label(ax, X['enc'] + ENC_W / 2, BOT + 0.15, 'mask ratio 0.8;\nDirichlet(α = 1) splits the\nvisible budget across the\nthree sensor streams', fs=4.2, color=DARK, va='top')
    arrow(ax, (X['tin'] + TW + 0.001, (TOP + BOT) / 2), (X['enc'] - 0.001, (TOP + BOT) / 2), color=NAVY, lw=0.9)
    arrow(ax, (X['enc'] + ENC_W + 0.001, (TOP + BOT) / 2), (X['tout'] - 0.001, (TOP + BOT) / 2), color=NAVY, lw=0.9)
    # decoder-input column: every position, hidden ones as mask tokens
    a = img_axes(fig, X['tout'], BOT, TW, TOP - BOT)
    _token_column(a, [(np.zeros(1), '#111111')] + [(m[f'm_{r}'], C[r]) for r in rows], gap=3)
    # token legend
    ly = 0.072
    for i, (t, col) in enumerate((('CLS', '#111111'), ('RGB', C['rgb']), ('depth', C['depth']), ('point cloud', C['pc']), ('recipe', C['text']), ('mask token', MASK_TOKEN))):
        xx = 0.03 + i * 0.072
        ax.add_patch(Rectangle((xx, ly - 0.008), 0.012, 0.016, fc=col, ec='none', zorder=4))
        label(ax, xx + 0.016, ly, t, fs=4.5, color=DARK, ha='left')
    label(ax, 0.03, 0.04, 'decoders: 8 shared blocks, d = 512, then one head per stream', fs=4.5, color=GREY, ha='left')

    # ---------- right: distillation ----------
    ax.plot([LEFT_END + 0.012] * 2, [0.03, 0.97], color='#adb5bd', lw=0.8, ls=(0, (4, 3)))
    RX0 = LEFT_END + 0.026; RW = 0.995 - RX0
    label(ax, RX0 + RW / 2, 0.985, 'Distillation', fs=8, bold=True, va='top')
    GAPB = 0.034
    bw = (RW - GAPB) / 2; bx = {'T': RX0, 'S': RX0 + bw + GAPB}
    by0, by1 = 0.22, 0.9
    for k, title in (('T', 'Teacher (frozen)'), ('S', 'Student')):
        box(ax, bx[k], by0, bw, by1 - by0, fill='white', edge=DARK, lw=1.1, rounding=0.015)
        label(ax, bx[k] + bw / 2, by1 - 0.012, title, fs=6, bold=True, va='top')
    blocks = [('Modality\nEncoders', C['depth'], 0.255, 0.085), ('Transformer\nEncoder', C['rgb'], 0.395, 0.2), ('Decoders', C['depth'], 0.665, 0.085)]
    inner = 0.016
    full = [(np.zeros(6), C['rgb']), (np.zeros(6), C['depth']), (np.zeros(6), C['pc']), (np.zeros(3), C['text'])]
    for k in ('T', 'S'):
        x0 = bx[k] + inner; w0 = bw - 2 * inner
        for name, col, yb, hb in blocks:
            box(ax, x0, yb, w0, hb, name, fill=col, edge=col, lw=0, fs=5.2, bold=True, color='white', rounding=0.008)
        arrow(ax, (x0 + w0 / 2, 0.34), (x0 + w0 / 2, 0.395), color=GREY, lw=0.7, ls=(0, (2, 1.5)))
        arrow(ax, (x0 + w0 / 2, 0.595), (x0 + w0 / 2, 0.665), color=GREY, lw=0.7, ls=(0, (2, 1.5)))
        a = img_axes(fig, x0, 0.61, w0, 0.016)      # the full sequence entering the decoders
        _token_strip_h(a, full if k == 'T' else [(np.zeros(6), C['rgb']), (np.ones(6), C['depth']), (np.ones(6), C['pc']), (np.ones(3), C['text'])])
        a = img_axes(fig, x0, by0 - 0.045, w0, 0.02)  # the input
        _token_strip_h(a, full if k == 'T' else [(np.zeros(6), C['rgb'])])
        arrow(ax, (x0 + w0 / 2, by0 - 0.024), (x0 + w0 / 2, 0.255), color=GREY, lw=0.7, ls=(0, (2, 1.5)))
    label(ax, bx['T'] + bw / 2, by0 - 0.05, 'all four streams,\nrecipe included', fs=4.2, color=DARK, va='top')
    label(ax, bx['S'] + bw / 2, by0 - 0.05, 'one sensor stream\n(here RGB), rest absent', fs=4.2, color=DARK, va='top')
    # student output: all four reconstructed
    a = img_axes(fig, bx['S'] + inner, 0.775, bw - 2 * inner, 0.016)
    _token_strip_h(a, full)
    arrow(ax, (bx['S'] + bw / 2, 0.75), (bx['S'] + bw / 2, 0.775), color=GREY, lw=0.7)
    label(ax, bx['S'] + bw / 2, 0.8, 'reconstructs all four', fs=4.3, color=DARK, va='bottom')
    label(ax, bx['T'] + bw / 2, 0.8, 'initialises the student', fs=4.3, color=GREY, va='bottom', style='italic')
    # matching losses
    gx0, gx1 = bx['T'] + bw - inner, bx['S'] + inner
    for yy, t in ((0.705, 'F'), (0.49, 'c')):
        ax.annotate('', (gx1, yy), (gx0, yy), arrowprops=dict(arrowstyle='<->', color=DARK, lw=0.9, ls=(0, (1.5, 1.5)), shrinkA=0, shrinkB=0), zorder=5)
        label(ax, (gx0 + gx1) / 2, yy + 0.008, t, fs=5.5, bold=True, color=DARK, va='bottom', style='italic')
    ax.annotate('', (RX0 + 0.035, 0.092), (RX0 + 0.005, 0.092), arrowprops=dict(arrowstyle='<->', color=DARK, lw=0.9, ls=(0, (1.5, 1.5)), shrinkA=0, shrinkB=0), zorder=5)
    label(ax, RX0 + 0.042, 0.092, 'MSE:  F = decoder features at the 417 generated\npositions (×1);  c = CLS (×0.5)', fs=4.2, color=DARK, ha='left')
    label(ax, RX0 + RW / 2, 0.04, 'init = teacher weights; 80 % of steps distil, 20 % are plain training steps', fs=4.1, color=GREY)

    label(ax, 0.5, 0.003, f'all panels: held-out sorghum plant {s["name"]} (data v2): real inputs, the sampled mask, and model outputs at epoch {s["epoch"]}; '
          'the maize model differs only in its recipe tokens (1 + 28 tokens of 14 floats)', fs=4.3, color=GREY, va='bottom', style='italic')
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    print('wrote', path)


def _token_strip_h(ax, segments, gap=1):
    """Horizontal version of _token_column."""
    cols = []
    for mv, col in segments:
        c = np.array(matplotlib.colors.to_rgb(col)); gr = np.array(matplotlib.colors.to_rgb(MASK_TOKEN))
        cols.append(np.where(np.asarray(mv)[None, :, None] > 0.5, gr, c)[0])
        cols.append(np.ones((gap, 3)))
    im = np.concatenate(cols[:-1], 0)[None, :, :]
    ax.imshow(im, aspect='auto', interpolation='nearest'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sorghum_ckpt', default=None); ap.add_argument('--maize_ckpt', default=None)
    ap.add_argument('--sorghum_dir', default=None, help='raw sample folder (default paper/data/Sorghum_10_04)')
    ap.add_argument('--maize_dir', default=None, help='raw sample folder (default paper/data/Maize_1_plant_0004_04)')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--style', default='embodiedmae', choices=['embodiedmae', 'columns'],
                    help='Fig. 2 layout: image-led rows like EmbodiedMAE Fig. 1 (default) or the older text-led columns')
    ap.add_argument('--only', default=None, choices=['teaser', 'architecture'])
    ap.add_argument('--cache', default=None, help='pickle of the loaded species dicts (skips the model forward passes)')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    import pickle
    if args.cache and Path(args.cache).exists():
        so, ma = pickle.load(open(args.cache, 'rb'))
        for s in (so, ma):                       # the species table holds lambdas: restore it by name
            s['sp'] = dict(SPECIES[s['species']], folder=s['sp_folder'])
    else:
        so = load('sorghum', args.sorghum_ckpt, args.sorghum_dir)
        ma = load('maize', args.maize_ckpt, args.maize_dir)
        if args.cache:
            slim = [dict(s, sp=None, sp_folder=s['sp']['folder']) for s in (so, ma)]
            pickle.dump(slim, open(args.cache, 'wb'))
    if args.only != 'architecture':
        teaser([so, ma], out / 'fig_teaser.pdf')
    if args.only != 'teaser':
        (architecture if args.style == 'embodiedmae' else architecture_columns)(so, out / 'fig_architecture.pdf')


if __name__ == '__main__':
    main()
