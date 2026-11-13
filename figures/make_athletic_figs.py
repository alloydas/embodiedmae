"""Athletic-editorial variants of paper figures, for side-by-side comparison only (the paper's own figures are untouched).

Style: black ground, Barlow Condensed (figures/fonts, SIL OFL) with all-caps labels, no frames, 3D clouds as product shots.
Kept from the paper style: >= 9 pt text, colour-blind-safe colours (Okabe-Ito sky blue / orange / vermillion, viridis depth).
It re-uses the two figure modules and overrides only colours, fonts and text case, so the data and layout are identical:
  Fig. 9   figures/make_row_gallery_v2.py   (row_gallery_dump.pkl, same picks as the paper)
  Fig. 11/12 figures/make_any2all_fig.py    (any2all4_dump.pkl, same plants as the paper)
Output: paper/figs/athletic_preview/*.{pdf,png}. CPU only (reads the dumps):
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=4 --mem=24G --output=logs/%x_%j.out \
         --wrap "python figures/make_athletic_figs.py"
"""
import sys, pathlib, pickle, shutil, re
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patheffects as pe, matplotlib.text
from matplotlib import font_manager
import _scene
import _fig12_kit as K
import make_row_gallery_v2 as RG
import make_any2all_fig as A

OUT = R / 'paper/figs/athletic_preview'
for f in sorted((R / 'figures/fonts').glob('Barlow*.ttf')): font_manager.fontManager.addfont(str(f))

SKY, ORANGE, VERM = '#56B4E9', '#E69F00', '#D55E00'          # Okabe-Ito, readable on black
WHITE, DIM, UNDER = '#f2f2f2', '#9a9a9a', '#4a4a4a'
matplotlib.rcParams.update({
    'font.family': ['Barlow Condensed', 'DejaVu Sans'], 'font.weight': 'semibold',
    'figure.facecolor': 'black', 'savefig.facecolor': 'black', 'axes.facecolor': 'black',
    'text.color': WHITE, 'axes.edgecolor': '#2a2a2a', 'axes.labelcolor': WHITE,
})

# all-caps for words; anything with a digit (values, units, plant ids) keeps its case
_set_text = matplotlib.text.Text.set_text
def _caps(self, s):
    s = '' if s is None else str(s)
    return _set_text(self, s if re.search(r'\d', s) else s.upper())
matplotlib.text.Text.set_text = _caps

# depth images: black background instead of white
_depth_rgb = K.depth_rgb
def depth_rgb_black(d, lo=None, hi=None, cmap='viridis'):
    rgb = _depth_rgb(d, lo, hi, cmap); rgb[~(d > 0)] = 0.0; return rgb
K.depth_rgb = depth_rgb_black


# ---------------------------------------------------------------- Fig. 9 (crop rows)
def draw_tiles(fig, x, y, s, it, gap=0.04, W=1.0, H=1.0, numbers=True, ncols=3):
    gt = it['gt']; lim = np.abs(gt[:, :2]).max() * 1.12
    cx = (gt[:, 0].max() + gt[:, 0].min()) / 2; cy = (gt[:, 1].max() + gt[:, 1].min()) / 2
    axs = [fig.add_axes([(x + k * (s + gap)) / W, y / H, s / W, s / H]) for k in range(ncols)]
    axs[0].imshow(it['img'])
    ps = 0.10 * (s / 0.72) ** 2
    axs[1].scatter(gt[:, 0], gt[:, 1], s=ps * 1.2, c=WHITE, lw=0, rasterized=True)
    for ax, (tag, col) in zip(axs[2:], (('ft', SKY), ('iso', VERM))[:ncols - 2]):
        p, cd = it['out'][tag]
        ax.scatter(gt[:, 0], gt[:, 1], s=ps * 1.2, c=UNDER, lw=0, rasterized=True)
        ax.scatter(p[:, 0], p[:, 1], s=ps, c=col, lw=0, rasterized=True)
        if numbers:
            ax.text(0.97, 0.03, f'{cd:.1f}' if cd < 100 else f'{cd:.0f}', transform=ax.transAxes, fontsize=RG.FS, color=col, ha='right',
                    va='bottom', fontweight='bold', path_effects=[pe.withStroke(linewidth=2.2, foreground='black')])
    for ax in axs[1:]:
        ax.set_xlim(cx - lim, cx + lim); ax.set_ylim(cy - lim, cy + lim); ax.set_aspect('equal')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
    return axs


RG.draw_tiles = draw_tiles
RG.HEAD_C = [WHITE, WHITE, SKY, VERM]
RG.INK, RG.BLUE, RG.ORANGE = WHITE, SKY, VERM


# ---------------------------------------------------------------- Figs. 11/12 (any to all)
def glyph_dark(ax, sp, rec, truth=None, color=None):
    color = color or A.C_OUT
    if truth is not None: K.glyph(ax, sp, truth, UNDER, stem=UNDER, lw_scale=0.75)
    K.glyph(ax, sp, rec, color, stem='#b9ab93', lw_scale=0.42)
    bb = ax.dataLim; w = max(abs(bb.x0), abs(bb.x1)); y0 = min(bb.y0, 0.0); st = max(2 * w, bb.y1 - y0) * 1.06
    ax.set_xlim(-st / 2, st / 2); ax.set_ylim(y0 - 0.02 * st, y0 + 0.98 * st); ax.set_aspect('equal')


A._glyph = glyph_dark
A.INK, A.MUTED, A.GREY, A.C_IN, A.C_OUT = WHITE, DIM, UNDER, ORANGE, SKY


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.pdf', dpi=300); fig.savefig(OUT / f'{stem}.png', dpi=200); plt.close(fig)
    print('wrote', OUT / f'{stem}.pdf', flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    D = pickle.load(open(RG.OUT, 'rb'))
    picks = {'sorghum': [('Sorghum_10756', 90), ('Sorghum_14816', 0), ('Sorghum_4815', 0)],
             'maize': [('plant_0320', 90), ('plant_6665', 0), ('plant_4604', 0)]}
    save(RG.render_main(D, picks), 'fig_rows_examples_athletic')
    # render4 writes under its module's R (paper/figs + a copy into overleaf/Figures): point it at a scratch root here
    root = OUT / '_root'; (root / 'paper/figs').mkdir(parents=True, exist_ok=True); (root / 'paper/overleaf/Figures').mkdir(parents=True, exist_ok=True)
    A.R = root
    D4 = pickle.load(open(A.OUT4, 'rb'))
    for sp, nm in (('maize', 'plant_1875_09'), ('sorghum', 'Sorghum_7712_05')):
        A.render4(sp, nm, f'fig_any2all_{sp}_athletic', D4)
        for ext in ('pdf', 'png'): shutil.move(str(root / 'paper/figs' / f'fig_any2all_{sp}_athletic.{ext}'), OUT / f'fig_any2all_{sp}_athletic.{ext}')
    shutil.rmtree(root)
