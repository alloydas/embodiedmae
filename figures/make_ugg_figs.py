"""Paper figures in the style of the UGG paper (Unified Generative Geometry Model, arXiv:2611.14440), for comparison only.

That style: white ground, Times-like serif text, one pastel colour per modality (RGB blue, depth yellow, ray map pink, text
purple), each item in a rounded box filled with its modality's pastel and edged with its colour, inputs dashed and outputs
solid, grey block arrows from inputs to outputs. Our four streams map onto it: image = blue, depth image = yellow,
point cloud = pink (Okabe-Ito reddish purple for the predicted points), procedural parameters = purple.
Same data, plants and numbers as the paper's figures; 9 pt text (Fig. 2 keeps its own sizes). The type is the
paper figures' own Liberation Sans (set by _scene; user, 2026-11-13: "for previous diagrams the font was good so keep
that"): only UGG's layout ideas are taken. `--serif` gives the earlier Liberation Serif variant (*_ugg_serif).
  Fig. 2      figures/make_fig12_saga.py figure2()   colours and serif only
  Fig. 9      figures/make_row_gallery_v2.py         dump row_gallery_dump.pkl
  Figs. 11/12 figures/make_any2all_fig.py            dump any2all4_dump.pkl
Output: paper/figs/ugg_preview/. CPU only:
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=4 --mem=24G --output=logs/%x_%j.out \
         --wrap "python figures/make_ugg_figs.py [--serif]"
"""
import sys, pathlib, pickle
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrow
from matplotlib import font_manager
import _scene
import _fig12_kit as K
import make_row_gallery_v2 as RG
import make_any2all_fig as A
import make_fig12_saga as SAGA

OUT = R / 'paper/figs/ugg_preview'
USE_SERIF = '--serif' in sys.argv
SUF = '_ugg_serif' if USE_SERIF else '_ugg'
if USE_SERIF:
    for f in sorted((R / 'figures/fonts').glob('LiberationSerif*.ttf')): font_manager.fontManager.addfont(str(f))
    SERIF = ['Liberation Serif', 'DejaVu Serif']
    _scene.SANS[:] = SERIF                               # every module that imported the list sees serif
    matplotlib.rcParams.update({'font.family': SERIF, 'mathtext.fontset': 'stix'})

# ---- UGG palette (sampled from its Fig. 2), mapped onto our streams
EDGE = {'rgb': '#809ec2', 'depth': '#e0b020', 'pc': '#d091a7', 'text': '#9d8cc4'}
FILL = {'rgb': '#e6ecf3', 'depth': '#faf2d4', 'pc': '#f6e9ed', 'text': '#eae7f2'}
INK_C = {'rgb': '#3f5f8f', 'depth': '#8a6a00', 'pc': '#a8456e', 'text': '#5e4a96'}   # darker, for text and points
PRED = '#CC79A7'                                          # Okabe-Ito reddish purple: predicted points
INK, ARROW, TRUTH, NEUTRAL = '#1a1a1a', '#8c8c8c', '#c9c9c9', ('#f4f4f5', '#9a9a9a')


def box(target, x, y, w, h, stream=None, dashed=False, transform=None, z=-1):
    fc, ec = (FILL[stream], EDGE[stream]) if stream else NEUTRAL
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=fc, ec=ec, lw=1.0,
                       ls=(0, (3.2, 2.0)) if dashed else '-', zorder=z, **({'transform': transform} if transform else {}))
    (target.add_artist if transform else target.add_patch)(p)
    return p


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.05, head_width=0.13, head_length=0.07, color=ARROW, lw=0,
                            length_includes_head=True))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.pdf', dpi=300); fig.savefig(OUT / f'{stem}.png', dpi=220); plt.close(fig)
    print('wrote', OUT / f'{stem}.pdf', flush=True)


# ====================================================================== Figs. 11 / 12
A.INK, A.MUTED, A.GREY, A.C_OUT = INK, '#5c5c5c', TRUTH, INK_C['text']


def render4_ugg(sp, nm, D):
    e = D[sp][nm]
    W = 6.875; t = 0.72; g = 0.06; gd = 0.24; gb = 0.24; tag = 0.24; lab = 0.31; num = 0.19; gr = 0.10; m = 0.035
    block = t + gd + 3 * t + 2 * g
    x0 = (W - (2 * block + gb)) / 2
    rowh = lab + t + num
    H = tag + 2 * rowh + gr + 0.01
    fig = plt.figure(figsize=(W, H))
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.set_axis_off(); ov.patch.set_alpha(0)
    for k, src in enumerate(A.SRCS):
        r, c = divmod(k, 2)
        bx = x0 + c * (block + gb); ytop = H - tag - r * (rowh + gr); ty = ytop - lab - t
        outs = [st for st in A.SRCS if st != src]
        xs = [bx] + [bx + t + gd + j * (t + g) for j in range(3)]
        if r == 0:
            ov.text(bx + t / 2, H - tag / 2, 'Input', fontsize=A.FS + 1, fontweight='bold', ha='center', va='center')
            ov.text(xs[1] + (3 * t + 2 * g) / 2, H - tag / 2, 'Output', fontsize=A.FS + 1, fontweight='bold', ha='center', va='center')
        for j, (st, x) in enumerate(zip([src] + outs, xs)):
            box(ov, x, ty, t, t + lab, st, dashed=j == 0)
            ov.text(x + t / 2, ty + t + lab / 2 - 0.01, A.NAME2[st], fontsize=A.FS, ha='center', va='center', linespacing=0.95)
            ax = A.tile(fig, sp, e, st, None if j == 0 else src, (x + m, ty + m, t - 2 * m, t - 2 * m), W, H)
            ax.patch.set_alpha(0)
            for s_ in ax.spines.values(): s_.set_visible(False)
            if j and st == 'pc':
                ov.text(x + t / 2, ty - 0.03, f"{e['out'][src]['chamfer'] * 1e3:.2f}", fontsize=A.FS, ha='center', va='top', color=INK)
            if j and st == 'text':
                rp = e['out'][src]['rec']
                ov.text(x + t / 2, ty - 0.03, f"{rp['height']:.2f} m, {rp['n']}", fontsize=A.FS, ha='center', va='top', color=INK)
        arrow(ov, bx + t + 0.045, bx + t + gd - 0.045, ty + t / 2)
    return fig


# ====================================================================== Fig. 9
def draw_tiles_ugg(fig, x, y, s, it, gap=0.04, W=1.0, H=1.0, numbers=True, ncols=3):
    gt = it['gt']; lim = np.abs(gt[:, :2]).max() * 1.12; m = 0.03
    cx = (gt[:, 0].max() + gt[:, 0].min()) / 2; cy = (gt[:, 1].max() + gt[:, 1].min()) / 2
    kinds = [('rgb', True), (None, False), ('pc', False)]
    axs = []
    for k in range(ncols):
        xk = x + k * (s + gap)
        box(fig, xk, y, s, s, kinds[k][0], dashed=kinds[k][1], transform=fig.dpi_scale_trans)
        ax = fig.add_axes([(xk + m) / W, (y + m) / H, (s - 2 * m) / W, (s - 2 * m) / H]); ax.patch.set_alpha(0); axs.append(ax)
    axs[0].imshow(it['img'])
    ps = 0.10 * (s / 0.72) ** 2
    axs[1].scatter(gt[:, 0], gt[:, 1], s=ps * 1.2, c='#3c3b38', lw=0, rasterized=True)
    p, cd = it['out']['ft']
    axs[2].scatter(gt[:, 0], gt[:, 1], s=ps * 1.2, c=TRUTH, lw=0, rasterized=True)
    axs[2].scatter(p[:, 0], p[:, 1], s=ps, c=PRED, lw=0, rasterized=True)
    if numbers:
        axs[2].text(0.96, 0.04, f'{cd:.1f}' if cd < 100 else f'{cd:.0f}', transform=axs[2].transAxes, fontsize=RG.FS, color=INK_C['pc'],
                    ha='right', va='bottom', path_effects=[pe.withStroke(linewidth=2.2, foreground=FILL['pc'])])
    for ax in axs[1:]:
        ax.set_xlim(cx - lim, cx + lim); ax.set_ylim(cy - lim, cy + lim); ax.set_aspect('equal')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
    return axs


RG.draw_tiles = draw_tiles_ugg
RG.HEAD = ['Row\nimage', 'Truth', 'Row-\ntrained']
RG.HEAD_C = [INK, INK, INK_C['pc']]


# ====================================================================== Fig. 2 (colours and serif only)
def figure2_ugg():
    for d in (SAGA.CS, K.CS): d.update({k: INK_C[k] for k in INK_C})
    K.CS_FILL.update(FILL)
    return SAGA.figure2()


def figure1_ugg():
    for d in (SAGA.CS, K.CS): d.update({k: INK_C[k] for k in INK_C})
    K.CS_FILL.update(FILL)
    return SAGA.figure1()


# ====================================================================== Fig. 2, full UGG treatment (2026-11-13)
def _band_png(stops=('#e6ecf3', '#faf2d4', '#f6e9ed', '#eae7f2'), px=(400, 60)):
    """UGG model-block fill: a soft horizontal gradient through the four stream pastels."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    cm = LinearSegmentedColormap.from_list('band', [to_rgb(c) for c in stops])
    fig = plt.figure(figsize=(px[0] / 100, px[1] / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=cm, extent=(0, 1, 0, 1))
    return K.png_fig(fig, 100, transparent=False)


def _hatch_png(fill, line, px=120):
    """UGG 'noisy token': a pastel square with diagonal hatching in the stream colour."""
    fig = plt.figure(figsize=(1, 1), dpi=px); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor='none'))
    for k in np.arange(-1, 1, 0.25):
        ax.plot([k, k + 1], [0, 1], color=line, lw=2.2, alpha=0.75, solid_capstyle='butt')
    return K.png_fig(fig, px, transparent=False)


def figure2_ugg2():
    """Fig. 2 (make_fig12_saga.figure2) with the UGG paper's diagram conventions, same layout, data, numbers and font:
    stream colours (image blue, depth yellow, cloud pink, parameters purple); inputs in dashed frames, outputs solid;
    neutral grey stage panels (a coloured panel would read as a stream); hidden tokens HATCHED, visible tokens plain
    pastel, with a clean / hidden legend; the encoder and decoder blocks filled with UGG's stream gradient; grey block
    arrows; our trained model ('Student', 'after') in UGG's 'Ours' violet; losses stay red."""
    from _scene import cloud_frame, Scene, text_size
    LOSS, MUTED, DARK, OURS, ARROWC = '#c0392b', '#5f685c', '#2b2f3a', '#6a2fd6', '#8c8c8c'
    CSk, CSE, CSF = INK_C, EDGE, FILL
    PANEL_E, PANEL_F = '#a9a9ae', '#f7f7f9'
    ENC_F, DEC_F = '#eef0f6', '#f5eff4'
    F = 9.0
    D = K.load_dump(); d = D['stage']['sorghum']['Sorghum_10036_00']; r = D['rows']['sorghum']['Sorghum_10113']
    W, H = 6.875, 4.42
    S = Scene(W, H, 'Fig 2 stages (UGG)')
    V = dict(elev=10.0, azim=20.0)
    fr = cloud_frame(d['pc'], **V); frr = cloud_frame(r['gt_pc'], **V)
    hidden_c = '#d3d6cf'; BAND = _band_png(); HATCH = {k: _hatch_png(CSF[k], CSE[k]) for k in CSF}

    def arrow(p, q):                                             # UGG grey block arrow
        S.arrow(p, q, color=ARROWC, lw=2.2, ms=9)

    def glyph_png(rec, side, vis=None):
        fig = plt.figure(figsize=(side, side), dpi=600); ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        if vis is not None:
            K.glyph(ax, 'sorghum', rec, hidden_c, stem=hidden_c, lw_scale=0.55)
            keep = dict(rec, leaves=[l for i, l in enumerate(rec['leaves']) if vis[1 + i]])
            if keep['leaves'] or vis[0]: K.glyph(ax, 'sorghum', keep, CSk['text'], stem='#4a3b2a' if vis[0] else hidden_c, lw_scale=0.55)
        else:
            K.glyph(ax, 'sorghum', rec, CSk['text'], lw_scale=0.55)
        Hh = rec['height']; ax.set_xlim(-0.55 * Hh, 0.55 * Hh); ax.set_ylim(-0.02 * Hh, 1.08 * Hh); ax.set_aspect('equal'); ax.set_axis_off()
        return K.png_fig(fig, 600)

    def pred_depth(pn):
        z = pn * (d['depth'].max() - d['depth'].min()) + d['depth'].min(); z = z.copy(); z[pn < 0.05] = 0; return z

    def grid(x, y, t, tiles, dash, g=0.03):
        for i, (png, c, nm) in enumerate(tiles):
            K.tile(S, x + (i % 2) * (t + g), y + (1 - i // 2) * (t + g), t, png, c, lw=0.9, dash=dash, name=nm)
        return 2 * t + g

    def model(x, y, w, h, names):                                # encoder / decoder trapezoids, neutral pastel
        gap = 0.035; we = (w - gap) * 0.54; wd = w - gap - we
        K.trapezoid(S, x, y, we, h, 'right', stroke='#6b6f7a', fill=ENC_F, lw=0.9)
        K.trapezoid(S, x + we + gap, y + 0.1 * h, wd, 0.8 * h, 'left', k=0.2, stroke='#6b6f7a', fill=DEC_F, lw=0.9)
        S.text(x + we * 0.46, y + h / 2, names[0], F, color=DARK, ha='center', bold=True)
        S.text(x + we + gap + wd * 0.5, y + h / 2, names[1], F, color=DARK, ha='center', bold=True)

    def tokrow(x, y, n, k, hidden, a=0.12, gp=0.035):
        for i in range(n):
            xx = x + i * (a + gp)
            if i in hidden:
                S.image(xx, y, a, a, HATCH[k], f'hidden {k} token'); S.rect(xx, y, a, a, CSE[k], lw=0.7, r=0.012)
            else:
                S.rect(xx, y, a, a, CSE[k], lw=0.7, fill=CSF[k], r=0.012, z=0)

    # ---------------------------------------------------------------- stage panels (neutral)
    XL, XR = 0.40, 4.98
    bands = {1: (2.72, 1.56), 2: (1.40, 1.16), 3: (0.04, 1.20)}
    titles = {1: 'Stage 1: masked pretraining', 2: 'Stage 2: cross-modal distillation', 3: 'Stage 3: row fine-tuning'}
    for k, (y, h) in bands.items():
        S.rect(XL, y, XR - XL, h, PANEL_E, lw=1.0, fill=PANEL_F, r=0.09)
        tw = text_size(titles[k], F, bold=True)[0] + 0.18
        S.rect(XL + 0.14, y + h - 0.11, tw, 0.22, PANEL_E, lw=0.9, fill='#ffffff', r=0.05, z=0)
        S.text(XL + 0.23, y + h, titles[k], F, bold=True)
    S.arrow((XL, 3.40), (XL, 2.00), color='#6b6f7a', lw=1.1, curve=0.22)
    S.arrow((XL, 3.20), (XL, 0.66), color='#6b6f7a', lw=1.1, curve=0.11)
    S.text(0.10, 2.00, 'warm start', F, color='#6b6f7a', rot=90, ha='center', bold=True)

    # ---------------------------------------------------------------- Stage 1
    t, y1 = 0.40, 3.03
    full = [(K.tile_img(d['rgb']), CSk['rgb'], 'image'), (K.tile_depth(d['depth']), CSk['depth'], 'depth image'),
            (K.tile_cloud(d['pc'], fr, t, s=0.35, **V), CSk['pc'], 'point cloud'), (glyph_png(d['recipe_true'], t), CSk['text'], 'parameters')]
    gw = grid(0.52, y1, t, full, dash=True)
    vis_t = [m < 0.5 for m in d['mask']['text']]
    masked = [(K.tile_img(K.masked(d['rgb'], d['mask']['rgb'])), CSk['rgb'], 'image masked'),
              (K.tile_img(K.masked(K.depth_rgb(d['depth']), d['mask']['depth'])), CSk['depth'], 'depth masked'),
              (K.tile_masked_cloud(d['pc'], d['pc_centers'], d['mask']['pc'], fr, t, s=0.35, **V), CSk['pc'], 'cloud masked'),
              (glyph_png(d['recipe_true'], t, vis_t), CSk['text'], 'parameters masked')]
    grid(1.55, y1, t, masked, dash=True)
    rec = [(K.tile_img(d['rec']['rgb']), CSk['rgb'], 'image output'), (K.tile_depth(pred_depth(d['rec']['depth'])), CSk['depth'], 'depth output'),
           (K.tile_cloud(d['rec']['pc'], fr, t, s=0.35, **V), CSk['pc'], 'cloud output'), (glyph_png(d['rec']['recipe'], t), CSk['text'], 'parameters output')]
    grid(4.06, y1, t, rec, dash=False)
    yl = y1 + gw + 0.10
    S.text(0.52 + gw / 2, yl, 'Input', F, ha='center', va='bottom', bold=True)
    S.text(1.55 + gw / 2, yl, '80 % hidden', F, ha='center', va='bottom', bold=True)
    S.text(4.06 + gw / 2, yl, 'Output', F, ha='center', va='bottom', bold=True)
    arrow((0.52 + gw + 0.04, y1 + gw / 2), (1.55 - 0.04, y1 + gw / 2))
    arrow((1.55 + gw + 0.04, y1 + gw / 2), (2.62 - 0.04, y1 + gw / 2))
    model(2.62, y1, 1.28, gw, ('Encoder', 'Decoder'))
    K.icon(S, 'fire', 2.62 + 0.66, y1 + gw - 0.05, 0.17)
    arrow((2.62 + 1.28 + 0.04, y1 + gw / 2), (4.06 - 0.04, y1 + gw / 2))
    S.arrow((4.06 + gw / 2, y1 - 0.08), (0.52 + gw / 2, y1 - 0.08), color=LOSS, lw=0.9, dash=True)
    S.text(2.69, y1 - 0.13, 'loss on the hidden tokens', F, color=LOSS, ha='center', va='top')

    # ---------------------------------------------------------------- Stage 2
    ts, y2 = 0.26, 1.62
    S.rect(0.52, y2 - 0.07, 1.25, 0.72, '#8c8c8c', lw=0.9, fill='#ffffff', r=0.05)
    K.icon(S, 'snow', 0.68, y2 + 0.49, 0.17)
    S.text(0.80, y2 + 0.49, 'Teacher', F, bold=True)
    for i, (png, c, nm) in enumerate(full):
        K.tile(S, 0.58 + i * (ts + 0.04), y2, ts, png, c, lw=0.7, dash=True, name=f'teacher {nm}')
    S.arrow((1.80, y2 + 0.26), (2.27, y2 + 0.26), color=LOSS, lw=0.9, dash=True)
    S.text(2.035, y2 + 0.31, 'features', F, color=LOSS, ha='center', va='bottom')
    S.rect(2.30, y2 - 0.07, 1.25, 0.72, OURS, lw=1.0, fill='#ffffff', r=0.05)
    K.icon(S, 'fire', 2.46, y2 + 0.49, 0.17)
    S.text(2.58, y2 + 0.49, 'Student', F, color=OURS, bold=True)
    K.tile(S, 2.36, y2, ts, full[0][0], CSk['rgb'], lw=0.9, dash=True, name='student image')
    for i, k in enumerate(('depth', 'pc', 'text')):
        S.image(2.36 + (i + 1) * (ts + 0.04), y2, ts, ts, HATCH[k], f'student hidden {k}')
        S.rect(2.36 + (i + 1) * (ts + 0.04), y2, ts, ts, CSE[k], lw=0.7, dash=True, r=0.03)
    arrow((3.58, y2 + 0.29), (3.72, y2 + 0.29))
    to = 0.58; xo = (3.75, 4.37)
    for x, key, lab, c in ((xo[0], 'rgb:before', 'before', MUTED), (xo[1], 'rgb:after', 'after', OURS)):
        K.tile(S, x, y2 + 0.02, to, K.tile_cloud(d['one'][key]['pc'], fr, to, s=0.4, **V), CSk['pc'] if lab == 'after' else MUTED,
               lw=0.9, dash=lab == 'before', name=f'stage 2 {lab}')
        S.text(x + to / 2, y2 + 0.02 + to + 0.05, lab, F, color=c, ha='center', va='bottom', bold=lab == 'after')
    S.text((xo[0] + xo[1] + to) / 2, y2 - 0.04, 'Chamfer 2.27 → 0.88', F, ha='center', va='top')

    # ---------------------------------------------------------------- Stage 3
    tr, y3 = 0.40, 0.44
    for i, (png, c, nm) in enumerate(((K.tile_rgba(r['rgba'], r['gain']), CSk['rgb'], 'row image'), (K.tile_depth(r['depth448']), CSk['depth'], 'row depth'),
                                      (K.tile_cloud(r['scene'], cloud_frame(r['scene'], **V), tr, s=0.18, **V), CSk['pc'], 'row cloud'))):
        K.tile(S, 0.52 + i * (tr + 0.04), y3, tr, png, c, lw=0.9, dash=True, name=nm)
    S.text(0.52 + (3 * tr + 0.08) / 2, y3 + tr + 0.08, 'Crop row', F, ha='center', va='bottom', bold=True)
    arrow((1.86, y3 + tr / 2), (2.20, y3 + tr / 2))
    model(2.23, y3 - 0.02, 1.20, tr + 0.04, ('Enc', 'Dec'))
    K.icon(S, 'fire', 2.23 + 0.62, y3 + tr - 0.03, 0.17)
    S.text(2.83, y3 - 0.09, 'loss: center plant only', F, color=LOSS, ha='center', va='top')
    arrow((3.46, y3 + tr / 2), (3.72, y3 + tr / 2))
    for x, key, lab, c in ((xo[0], 'head:all', 'before', MUTED), (xo[1], 'ft:all', 'after', OURS)):
        K.tile(S, x, y3 - 0.06, to, K.tile_cloud(r['pred'][key]['pc'], frr, to, s=0.4, **V), CSk['pc'] if lab == 'after' else MUTED,
               lw=0.9, dash=lab == 'before', name=f'stage 3 {lab}')
        S.text(x + to / 2, y3 - 0.06 + to + 0.05, lab, F, color=c, ha='center', va='bottom', bold=lab == 'after')
    S.text((xo[0] + xo[1] + to) / 2, y3 - 0.12, 'Chamfer 230 → 31', F, ha='center', va='top')

    # ---------------------------------------------------------------- Inside CropCode
    XZ, WZ = 5.10, W - 0.02 - 5.10
    S.rect(XZ, 0.04, WZ, 4.24, '#8a8a8a', lw=1.2, fill='#ffffff', r=0.09)
    S.text(XZ + WZ / 2, 4.06, 'Inside CropCode', F, ha='center', bold=True)
    a, gp = 0.12, 0.035
    for i, (nm, k, hid) in enumerate((('Image  196', 'rgb', (1, 3)), ('Depth  196', 'depth', (0, 2, 3)),
                                      ('Cloud  196', 'pc', (2,)), ('Params  1+K', 'text', (1, 2)))):
        yy = 3.65 - i * 0.30
        tokrow(XZ + 0.14, yy, 4, k, hid, a, gp)
        S.text(XZ + 0.14 + 4 * a + 3 * gp + 0.10, yy + a / 2, nm, F, color=CSk[k], va='center', bold=True)
    # legend, UGG style: clean token / hidden token
    yl2 = 2.46
    S.rect(XZ + 0.14, yl2, a, a, '#8c8c8c', lw=0.7, fill='#ffffff', r=0.012, z=0)
    S.text(XZ + 0.14 + a + 0.06, yl2 + a / 2, 'visible', F, color=MUTED, va='center')
    xh = XZ + 0.14 + a + 0.06 + text_size('visible', F)[0] + 0.14
    S.image(xh, yl2, a, a, _hatch_png('#f4f4f5', '#8c8c8c'), 'legend hidden'); S.rect(xh, yl2, a, a, '#8c8c8c', lw=0.7, r=0.012)
    S.text(xh + a + 0.06, yl2 + a / 2, 'hidden', F, color=MUTED, va='center')
    S.text(XZ + 0.14, 2.20, 'Dirichlet split', F, va='center', bold=True)
    x = XZ + 0.14; tot = sum(d['visible'].values()); bw = WZ - 0.28
    for kk in ('rgb', 'depth', 'pc', 'text'):
        w_ = bw * d['visible'][kk] / tot; S.rect(x, 1.92, w_, 0.13, CSE[kk], lw=0.4, fill=CSE[kk], r=0.01, z=0); x += w_
    S.text(XZ + 0.14, 1.76, 'of the visible tokens', F, color=MUTED, va='center')
    for (lab, sub, y0) in (('Encoder', '12 blocks', 1.06), ('Decoder', '8 blocks', 0.50)):
        S.image(XZ + 0.14, y0, WZ - 0.28, 0.42, BAND, f'{lab} band')
        S.rect(XZ + 0.14, y0, WZ - 0.28, 0.42, '#8c8c8c', lw=0.9, r=0.04)
        S.text(XZ + WZ / 2, y0 + 0.21, f'{lab}, {sub}', F, color=DARK, ha='center', bold=True)
    for (p0, p1) in (((XZ + WZ / 2, 1.64), (XZ + WZ / 2, 1.49)), ((XZ + WZ / 2, 1.05), (XZ + WZ / 2, 0.93)), ((XZ + WZ / 2, 0.49), (XZ + WZ / 2, 0.40))):
        S.arrow(p0, p1, color=ARROWC, lw=1.6, ms=7)
    for i, k in enumerate(('rgb', 'depth', 'pc', 'text')):
        wq = (WZ - 0.28 - 0.09) / 4; xx = XZ + 0.14 + i * (wq + 0.03)
        S.rect(xx, 0.13, wq, 0.10, CSE[k], lw=0.6, fill=CSF[k], r=0.02, z=0)
    S.text(XZ + WZ / 2, 0.30, 'one head per stream', F, color=DARK, ha='center')
    return S


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if '--fig2' in sys.argv:                     # only the full-UGG Fig. 2: preview + TikZ twin for the paper
        S = figure2_ugg2(); S.render(OUT / f'fig2_stages{SUF}2')
        S.render_tikz(R / 'paper/overleaf/Figures/fig2_stages_tikz_ugg.tex', R / 'paper/overleaf/Figures/fig2_ugg', 'Figures/fig2_ugg')
        print('wrote', OUT / f'fig2_stages{SUF}2.pdf', 'and paper/overleaf/Figures/fig2_stages_tikz_ugg.tex', flush=True); sys.exit(0)
    D4 = pickle.load(open(A.OUT4, 'rb'))
    for sp, nm in (('maize', 'plant_1875_09'), ('sorghum', 'Sorghum_7712_05')):
        save(render4_ugg(sp, nm, D4), f'fig_any2all_{sp}{SUF}')
    D = pickle.load(open(RG.OUT, 'rb'))
    picks = {'sorghum': [('Sorghum_10756', 90), ('Sorghum_14816', 0), ('Sorghum_4815', 0)],
             'maize': [('plant_0320', 90), ('plant_6665', 0), ('plant_4604', 0)]}
    save(RG.render_main(D, picks), f'fig_rows_examples{SUF}')
    S = figure1_ugg(); S.render(OUT / f'fig1_overview{SUF}'); print('wrote', OUT / f'fig1_overview{SUF}.pdf', flush=True)
    S = figure2_ugg(); S.render(OUT / f'fig2_stages{SUF}'); print('wrote', OUT / f'fig2_stages{SUF}.pdf', flush=True)
