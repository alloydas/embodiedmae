"""Shared pieces of Figures 1 and 2 (second version): palette, type scale, tiles drawn from the dump of
figures/make_fig12_dump.py, token rows, encoder / decoder shapes, frozen / trained marks and the small plots.
Everything here is light (numpy, matplotlib, PIL): no model, no torch."""
import io, pathlib, pickle
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from _scene import Scene, png_array, png_points, png_icon, png_fig, cloud_frame, inset, text_size, PLANT_CMAP, SANS

R = pathlib.Path(__file__).resolve().parents[1]
DUMP = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/v2_dump.pkl')
A = R / 'paper/figs/fig12_assets'

# ---- palette (one colour per stream everywhere; the model is navy; losses and warm starts are red)
INK, MUTED, FAINT, NAVY, LOSS = '#1d221c', '#5f685c', '#b9beb4', '#2b3a67', '#c0392b'
CS = {'rgb': '#c0392b', 'depth': '#b8860b', 'pc': '#2a9d8f', 'text': '#30638e'}
CS_FILL = {'rgb': '#f6dcd8', 'depth': '#f4e6c3', 'pc': '#d3ece8', 'text': '#d6e3ee'}
NAME = {'rgb': 'Image', 'depth': 'Depth image', 'pc': 'Point cloud', 'text': 'Procedural parameters'}
ENC_FILL, DEC_FILL, BOX_FILL = '#d9e2f3', '#fbe3c4', '#eef1f7'
STAGE = {1: ('#e0a82e', '#fdf8ea'), 2: ('#6aa84f', '#f1f8ee'), 3: ('#4a78b5', '#eef3fa')}
# ---- type scale, pt at print size (nothing smaller than T_MIN anywhere, plots included)
T_TITLE, T_HEAD, T_LAB, T_MIN = 8.0, 7.0, 6.5, 6.0
_ICON = {}


def load_dump():
    return pickle.load(open(DUMP, 'rb'))


# ---------------------------------------------------------------- tiles (PNG bytes)
def depth_rgb(d, lo=None, hi=None, cmap='viridis'):
    fg = d > 0
    if not fg.any(): return np.ones(d.shape + (3,))
    lo = np.percentile(d[fg], 1) if lo is None else lo; hi = np.percentile(d[fg], 99) if hi is None else hi
    rgb = matplotlib.colormaps[cmap](np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1))[..., :3]
    rgb[~fg] = 1.0
    return rgb


def tile_rgba(rgba, gain=None, bg=0.0):
    """An RGBA render composited over black (as the models read it) or another grey, at the run's brighten gain."""
    a = rgba.astype(np.float32) / 255; rgb = a[..., :3] * (gain or 1.0)
    return png_array(np.clip(rgb, 0, 1) * a[..., 3:] + bg * (1 - a[..., 3:]))


def tile_img(arr):
    return png_array(arr)


def tile_depth(d):
    return png_array(depth_rgb(np.asarray(d, float)))


def masked(img, mask, grey=0.86):
    """img (H, W, 3) with the hidden 16 px patches (mask: 196 values, 1 = hidden) painted light grey."""
    out = np.array(img, dtype=np.float32, copy=True); hid = np.asarray(mask).reshape(14, 14) > 0.5; p = out.shape[0] // 14
    for i in range(14):
        for j in range(14):
            if hid[i, j]: out[i * p:(i + 1) * p, j * p:(j + 1) * p] = grey
    return out


def tile_cloud(P, frame=None, side=0.6, elev=8.0, azim=20.0, s=0.55, **kw):
    return png_points(P, side=side, elev=elev, azim=azim, frame=frame, s=s, **kw)


def tile_masked_cloud(P, centers, mask, frame=None, side=0.6, elev=8.0, azim=20.0, s=0.55):
    """The cloud with the point groups of hidden tokens in light grey and those of visible tokens in leaf green."""
    P = np.asarray(P, float); grp = np.linalg.norm(P[:, None] - np.asarray(centers, float)[None], axis=2).argmin(1)
    vis = np.asarray(mask)[grp] < 0.5
    return png_points(P[vis], side=side, elev=elev, azim=azim, frame=frame, s=s, under=P[~vis], under_color='#d3d6cf')


def tile_hidden(color, side=0.5, px=200):
    """A stream that the model does not see: a pale tile with diagonal hatching in the stream's colour."""
    fig = plt.figure(figsize=(1, 1), dpi=px); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#f4f5f2', edgecolor='none'))
    for k in np.arange(-1, 1, 0.16):
        ax.plot([k, k + 1], [0, 1], color=color, lw=1.6, alpha=0.38)
    return png_fig(fig, px, transparent=False)


def glyph(ax, species, rec, color, stem='#4a3b2a', lw_scale=1.0, alpha=0.95):
    """The procedural parameters drawn as a stem-and-leaves schematic (figures/_plantviz.draw_recipe_glyph):
    insertion height, length, width and side of each leaf, and for maize its angle and droop. Not the generator's mesh."""
    H, L = rec['height'], rec['leaves']
    ax.plot([0, 0], [0, H], color=stem, lw=1.5 * lw_scale, solid_capstyle='round', zorder=2)
    xs = [0.05]; t = np.linspace(0, 1, 24)[:, None]; y = 0.0
    for i, l in enumerate(L):
        if species == 'sorghum':
            y0 = float(l['starting_point']) * H; ln = float(l['length']); b = np.radians(float(l['branching_angle']))
            side = 1.0 if np.cos(np.radians(float(l['roll_angle']))) >= 0 else -1.0
            phi0, phi1 = b, b + np.radians(75.0); w = 0.45 + 0.45 * 35.0 * float(l['width'])
        else:
            y += float(l['distance']); y0 = y; ln = float(l['leafLength'])
            side = 1.0 if np.cos(np.radians(180.0 * i + float(l['azJitterDeg']))) >= 0 else -1.0
            phi0 = np.radians(35.0 + float(l['leafAngle'])); phi1 = phi0 - np.radians(float(l['droopiness'])); w = 0.45 + 0.75 * 35.0 * float(l['leafWidth'])
        p0 = np.array([0.0, y0]); p1 = p0 + 0.5 * ln * np.array([side * np.sin(phi0), np.cos(phi0)])
        p2 = p1 + 0.5 * ln * np.array([side * np.sin(phi1), np.cos(phi1)])
        q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        ax.plot(q[:, 0], q[:, 1], color=color, lw=w * lw_scale, solid_capstyle='round', alpha=alpha, zorder=3)
        xs.append(abs(q[:, 0]).max())
    return H, max(xs)


def tile_glyph(species, rec, color=CS['text'], side=0.6, px=600, ref_h=None, lw_scale=1.0, under=None, under_color='#c3c7be'):
    """The schematic of a recipe; `under`: a second recipe (the truth) drawn first in light grey at the same scale."""
    fig = plt.figure(figsize=(side, side), dpi=px); ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    Hu = wu = 0.0
    if under is not None: Hu, wu = glyph(ax, species, under, under_color, stem=under_color, lw_scale=lw_scale * 1.9, alpha=1.0)
    H, w = glyph(ax, species, rec, color, lw_scale=lw_scale)
    top = max(ref_h or 0, H, Hu); hw = max(w, wu, 0.5 * top * 1.04) * 1.04; y0 = -0.02 * top if 2 * hw > top * 1.04 else (top - 2 * hw) / 2
    ax.set_xlim(-hw, hw); ax.set_ylim(y0, y0 + 2 * hw); ax.set_aspect('equal'); ax.set_axis_off()
    return png_fig(fig, px)


# ---------------------------------------------------------------- scene helpers
def tile(S, x, y, t, png, color, lw=0.8, dash=False, name='tile', h=None, pad=0.0):
    """An image in a coloured frame; pad > 0 lets a transparent cloud bleed past the frame."""
    h = t if h is None else h
    S.image(x - pad, y - pad, t + 2 * pad, h + 2 * pad, png, name)
    S.rect(x, y, t, h, color, lw=lw, dash=dash, r=0.03)


def icon(S, kind, x, y, size=0.13):
    if kind not in _ICON: _ICON[kind] = png_icon(kind)
    S.image(x - size / 2, y - size / 2, size, size, _ICON[kind], kind)


def tok_row(S, x, y, n, a=0.07, gap=0.022, color=NAVY, hidden=(), lw=0.5, vertical=False):
    """n token squares of side a starting at (x, y) (lower left): filled = visible, hollow = hidden. Returns the far end."""
    for i in range(n):
        xx, yy = (x, y + i * (a + gap)) if vertical else (x + i * (a + gap), y)
        S.rect(xx, yy, a, a, color, lw=lw, fill='#ffffff' if i in hidden else color, r=0.012, z=0)
    return (y if vertical else x) + n * (a + gap) - gap


def trapezoid(S, x, y, w, h, narrow='right', k=0.24, stroke=NAVY, fill=ENC_FILL, lw=0.9):
    """A module that narrows (an encoder) or widens (a decoder): the short side is inset by k * h at top and bottom."""
    d = k * h
    pts = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)] if narrow == 'right' else [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
    S.poly(pts, stroke, lw=lw, fill=fill)


def model(S, x, y, w, h, mark=None, labels=True, size=T_LAB, gap=0.035, lw=0.9, names=('Enc', 'Dec')):
    """CropCode as it appears in every stage: an encoder trapezoid and a decoder trapezoid; mark 'fire' / 'snow' on top."""
    we = (w - gap) * 0.54; wd = w - gap - we
    trapezoid(S, x, y, we, h, 'right', fill=ENC_FILL, lw=lw); trapezoid(S, x + we + gap, y + 0.1 * h, wd, 0.8 * h, 'left', k=0.2, fill=DEC_FILL, lw=lw)
    if labels:
        S.text(x + we * 0.46, y + h / 2, names[0], size, color=NAVY, ha='center', bold=True)
        S.text(x + we + gap + wd * 0.5, y + h / 2, names[1], size, color=NAVY, ha='center', bold=True)
    if mark: icon(S, mark, x + w - 0.02, y + h - 0.02, 0.15)


def stage_panel(S, x, y, w, h, k, title, tab_w=None):
    stroke, fill = STAGE[k]
    S.rect(x, y, w, h, stroke, lw=1.2, fill=fill, r=0.09)
    tw = (tab_w or text_size(title, T_HEAD + 0.5, bold=True)[0] + 0.16)
    S.rect(x + 0.14, y + h - 0.085, tw, 0.17, stroke, lw=0.9, fill='#ffffff', r=0.05, z=0)
    S.text(x + 0.14 + 0.08, y + h - 0.003, title, T_HEAD + 0.5, color=INK, bold=True)


# ---------------------------------------------------------------- plots (vector in the PDF)
RADAR_AXES = [('height', 's'), ('leaf count', 's'), ('branch angle', 's'), ('roll', 's'), ('leaf length', 's'),
              ('height', 'm'), ('leaf angle', 'm'), ('curl', 'm'), ('stem radius', 'm')]
RADAR = {'DINOv2': [0.924, 0.928, 0.840, 0.573, 0.235, 0.561, 0.816, 0.072, 0.588],
         'MultiMAE': [0.922, 0.925, 0.833, 0.679, 0.232, 0.570, 0.783, 0.069, 0.442],
         'Point-MAE': [0.958, 0.963, 0.863, 0.716, 0.188, 0.509, 0.863, 0.199, 0.426],
         'EmbodiedMAE': [0.976, 0.979, 0.871, 0.765, 0.265, 0.528, 0.823, 0.139, 0.428],
         'Supervised': [0.973, 0.978, 0.924, 0.634, 0.271, 0.785, 0.935, 0.618, 0.860],
         'CropCode': [0.983, 0.988, 0.912, 0.798, 0.371, 0.747, 0.967, 0.739, 0.841]}
RADAR_STY = {'DINOv2': ('#8d99a6', '-'), 'MultiMAE': ('#c08f57', '-'), 'Point-MAE': ('#6fa77c', '-'), 'EmbodiedMAE': ('#a77aa0', '-'),
             'Supervised': ('#4d4d4d', (0, (2.5, 1.5))), 'CropCode': (NAVY, '-')}
SP_COL = {'s': '#6b4f1d', 'm': '#1f5f3a'}


def radar_draw(fs=T_MIN, rel=(0.22, 0.33, 0.56, 0.50), legend=True, tab=None, key=True):
    tab = tab or RADAR

    def draw(fig, box):
        M = np.array(list(tab.values())); lo, hi = M.min(0), M.max(0); n = len(RADAR_AXES)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False); angc = np.r_[ang, ang[:1]]
        fw, fh = fig.get_size_inches(); wi, hi_ = box[2] * fw, box[3] * fh             # keep the polar axes square in inches
        side = min(rel[2] * wi, rel[3] * hi_)
        ax = fig.add_axes([box[0] + (rel[0] * wi + (rel[2] * wi - side) / 2) / fw, box[1] + (rel[1] * hi_ + (rel[3] * hi_ - side) / 2) / fh,
                           side / fw, side / fh], projection='polar')
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.patch.set_alpha(0)
        for name, v in tab.items():
            r = 0.22 + 0.78 * (np.array(v) - lo) / np.maximum(hi - lo, 1e-9); r = np.r_[r, r[:1]]
            c, ls = RADAR_STY[name]; ours = name == 'CropCode'
            ax.plot(angc, r, color=c, lw=1.6 if ours else 0.75, ls=ls, zorder=5 if ours else 2, label=name, solid_joinstyle='round')
            if ours: ax.fill(angc, r, color=c, alpha=0.14, zorder=1)
        ax.set_xticks(ang); ax.set_xticklabels([a.replace(' ', '\n') for a, _ in RADAR_AXES], fontsize=fs, linespacing=0.9)
        for lab, (_, sp), a in zip(ax.get_xticklabels(), RADAR_AXES, ang):
            lab.set_color(SP_COL[sp]); d = np.degrees(a) % 360
            lab.set_horizontalalignment('center' if min(d, 360 - d) < 25 or abs(d - 180) < 25 else ('left' if d < 180 else 'right'))
        ax.set_yticks([0.22, 0.48, 0.74, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.04)
        ax.grid(color='#d9dcd5', lw=0.4); ax.spines['polar'].set_color('#c4c8bf'); ax.spines['polar'].set_linewidth(0.5); ax.tick_params(axis='x', pad=-2.5)
        if key:
            fig.text(box[0] + 0.01 * box[2], box[1] + 0.995 * box[3], 'maize traits', fontsize=fs, color=SP_COL['m'], ha='left', va='top', fontweight='bold')
            fig.text(box[0] + 0.99 * box[2], box[1] + 0.995 * box[3], 'sorghum traits', fontsize=fs, color=SP_COL['s'], ha='right', va='top', fontweight='bold')
        if legend:
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, loc='lower center', ncol=2, fontsize=fs, frameon=False, handlelength=1.3, columnspacing=1.0, handletextpad=0.4,
                       labelspacing=0.22, borderaxespad=0, bbox_to_anchor=(box[0] + box[2] / 2, box[1]))
    return draw


def _trunc(name, lo=0.28, hi=1.0):
    return matplotlib.colors.LinearSegmentedColormap.from_list(name + '_t', matplotlib.colormaps[name](np.linspace(lo, hi, 256)))


def tsne_draw(sp, label, cmap, fs=T_MIN, title=None, fmt='{:.0f}'):
    z = np.load(A / f'tsne_{sp}.npz'); E, y = z['E'], z['y']; cm = _trunc(cmap)

    def draw(fig, box):
        ax = inset(fig, box, (0.03, 0.30, 0.94, 0.58))
        o = np.argsort(y); lo, hi = np.percentile(y, 2), np.percentile(y, 98)
        sc = ax.scatter(E[o, 0], E[o, 1], c=y[o], cmap=cm, s=0.9, linewidths=0, vmin=lo, vmax=hi, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal', adjustable='datalim'); ax.patch.set_alpha(0)
        for s_ in ax.spines.values(): s_.set_visible(False)
        if title: fig.text(box[0] + box[2] / 2, box[1] + 0.985 * box[3], title, fontsize=fs + 0.5, color=INK, ha='center', va='top', fontweight='bold')
        cax = inset(fig, box, (0.27, 0.05, 0.46, 0.06)); cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
        cb.set_ticks([]); cb.outline.set_linewidth(0.4)
        yy = box[1] + 0.08 * box[3]
        fig.text(box[0] + 0.24 * box[2], yy, fmt.format(lo), fontsize=fs, ha='right', va='center', color=INK)
        fig.text(box[0] + 0.76 * box[2], yy, fmt.format(hi), fontsize=fs, ha='left', va='center', color=INK)
        fig.text(box[0] + box[2] / 2, box[1] + 0.135 * box[3], label, fontsize=fs, ha='center', va='bottom', color=MUTED)
    return draw
