"""Paper Fig. 5 (crop-row examples, procedural models) as TikZ (user, 2026-11-13: "can you make fig 5 as tikz?").

Same figure as make_ugg_figs.py --fig5 (fig_rows_examples_ugg5_proc): sorghum left, maize right, three test rows each;
columns row image | ground-truth procedural model | the row-trained model's written procedural model (purple) over the
truth (grey) with its leaf-parameter MAE. Same plants, yaws, predictions and numbers (row_gallery_dump.pkl for the images,
row_gallery_recipes.pkl from make_rows_recipe_dump.py for the recipes). Built on _scene.Scene like Figs. 2 and 3: boxes and
text are TikZ, each tile's content is an embedded PNG (Figures/fig5/).
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=2 --mem=8G --output=logs/%x_%j.out \
         --wrap "python figures/make_fig5_tikz.py"
-> paper/overleaf/Figures/fig_rows_examples_tikz.tex + Figures/fig5/, preview paper/figs/ugg_preview/fig_rows_examples_tikz.{pdf,png}
"""
import sys, pathlib, pickle
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _scene import Scene
import _fig12_kit as K
import make_row_gallery_v2 as RG
from ugg_style import STREAM, STREAM_INK, STREAM_FILL

PICKS = {'sorghum': [('Sorghum_10756', 90), ('Sorghum_14816', 0), ('Sorghum_4815', 0)],     # = make_ugg_figs.py --fig5
         'maize': [('plant_0320', 90), ('plant_6665', 0), ('plant_4604', 0)]}
INK, TRUTH_INK, TRUTH_UNDER, STEM = '#1a1a1a', '#3c3b38', '#b4b4b4', '#4a3b2a'
NEUTRAL_E, NEUTRAL_F = '#9a9a9a', '#f4f4f5'
F = 9.0
W, s, g, gy, lab, nc, m = 6.875, 0.90, 0.05, 0.07, 0.62, 3, 0.03
TEX = R / 'paper/overleaf/Figures/fig_rows_examples_tikz.tex'


def lims(recs, sp):
    """make_ugg_figs._rec_lims: one square frame for truth and prediction, room at the bottom for the MAE label."""
    f = plt.figure(); a = f.add_axes([0, 0, 1, 1])
    for r in recs: K.glyph(a, sp, r, '#000000')
    bb = a.dataLim; plt.close(f)
    w = max(abs(bb.x0), abs(bb.x1)); y0 = min(bb.y0, 0.0); st = max(2 * w, bb.y1 - y0) * 1.06 * 1.18
    return (-st / 2, st / 2), (y0 - 0.17 * st, y0 + 0.83 * st)


def glyph_png(sp, layers, lm, side):
    fig = plt.figure(figsize=(side, side), dpi=600); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    lw = 0.42 * side / 0.735
    for rec, col, stem, k in layers:
        K.glyph(ax, sp, rec, col, stem=stem, lw_scale=lw * k, alpha=1.0 if k > 1 else 0.95)
    ax.set_xlim(*lm[0]); ax.set_ylim(*lm[1]); ax.set_aspect('equal')
    return K.png_fig(fig, 600)


if __name__ == '__main__':
    D = pickle.load(open(RG.OUT, 'rb')); REC = pickle.load(open(RG.OUT.with_name('row_gallery_recipes.pkl'), 'rb'))
    half = lab + nc * s + (nc - 1) * g; hdr = 0.48; n = 3
    H = hdr + n * s + (n - 1) * gy + 0.02
    S = Scene(W, H, 'Fig 5 crop-row examples (procedural)')
    head = [('Row image', INK), ('Ground truth', INK), ('Row-trained', STREAM_INK['text'])]
    box = {0: (STREAM['rgb'], STREAM_FILL['rgb'], True), 1: (NEUTRAL_E, NEUTRAL_F, False), 2: (STREAM['text'], STREAM_FILL['text'], False)}
    inner = s - 2 * m
    for c, sp in enumerate(('sorghum', 'maize')):
        x0 = c * (W - half - 0.08)
        S.text(x0 + lab + (nc * s + (nc - 1) * g) / 2, H - 0.01, sp.capitalize(), F, ha='center', va='top', bold=True)
        for k, (t, col) in enumerate(head):
            S.text(x0 + lab + k * (s + g) + s / 2, H - 0.20, t, F, color=col, ha='center', va='top')
        y = H - hdr
        for pl, th in PICKS[sp]:
            y -= s
            it = D[sp]['items'][(pl, th)]; r = REC[sp][(pl, th)]; lm = lims([r['true'], r['ft']], sp)
            tiles = [K.png_array(it['img'] / 255.0),
                     glyph_png(sp, [(r['true'], TRUTH_INK, STEM, 1.0)], lm, inner),
                     glyph_png(sp, [(r['true'], TRUTH_UNDER, TRUTH_UNDER, 1.8), (r['ft'], STREAM_INK['text'], STEM, 1.0)], lm, inner)]
            for k, png in enumerate(tiles):
                xk = x0 + lab + k * (s + g); e, f_, dash = box[k]
                S.rect(xk, y, s, s, e, lw=1.0, fill=f_, r=0.05, dash=dash)
                S.image(xk + m, y + m, inner, inner, png, f'{sp} {pl} {th} {head[k][0]}')
            S.text(x0 + lab + 2 * (s + g) + s - 0.04, y + 0.04, f"MAE = {r['ft_leaf_mae']:.3f}", F, color=STREAM_INK['text'],
                   ha='right', va='bottom')
            S.text(x0 + lab - 0.06, y + s / 2, f"plant\n{pl.split('_')[-1]}", F, ha='right', va='center', ls=1.1)
            y -= gy
    out = R / 'paper/figs/ugg_preview/fig_rows_examples_tikz'
    S.render(out)
    S.render_tikz(TEX, TEX.parent / 'fig5', 'Figures/fig5')
    print('wrote', TEX, flush=True)
