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


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    D4 = pickle.load(open(A.OUT4, 'rb'))
    for sp, nm in (('maize', 'plant_1875_09'), ('sorghum', 'Sorghum_7712_05')):
        save(render4_ugg(sp, nm, D4), f'fig_any2all_{sp}{SUF}')
    D = pickle.load(open(RG.OUT, 'rb'))
    picks = {'sorghum': [('Sorghum_10756', 90), ('Sorghum_14816', 0), ('Sorghum_4815', 0)],
             'maize': [('plant_0320', 90), ('plant_6665', 0), ('plant_4604', 0)]}
    save(RG.render_main(D, picks), f'fig_rows_examples{SUF}')
    S = figure1_ugg(); S.render(OUT / f'fig1_overview{SUF}'); print('wrote', OUT / f'fig1_overview{SUF}.pdf', flush=True)
    S = figure2_ugg(); S.render(OUT / f'fig2_stages{SUF}'); print('wrote', OUT / f'fig2_stages{SUF}.pdf', flush=True)
