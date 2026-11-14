"""Fig. 2 (paper/overleaf/Figures/fig2_stages_tikz.tex): procedural-model tiles for the Stage 2 and Stage 3 before/after panels.

The paper's Fig. 2 showed point clouds there; since 2026-11-13 it shows the procedural model each model writes (user: "instead of
point cloud focus on procedural model in fig 2"). Same plant and row as before (figure dump of _fig12_kit, sorghum test plant
Sorghum_10036 view 00 from the photograph alone; test row Sorghum_10113, all streams), drawn as the stem-and-leaf schematic of
Figs. 1 and 4: the true recipe in light grey underneath, the written recipe in the figure's parameter navy on top. Before and
after of one stage share one frame so the two tiles compare directly.
  Stage 2: 'rgb:before' (headline model, zero-shot) vs 'rgb:after' (distilled generalist)
  Stage 3: 'head:all' (headline model on the row, zero-shot) vs 'ft:all' (row fine-tuned)
Writes Figures/fig2/stage_{2,3}_{before,after}_recipe.png. CPU only, seconds:
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=2 --mem=8G --output=logs/%x_%j.out \
         --wrap "python figures/make_fig2_recipe_tiles.py"
"""
import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _fig12_kit as K

OUT = R / 'paper/overleaf/Figures/fig2'
NAVY, GREY, STEM = '#30638E', '#c3c7be', '#4a3b2a'      # the figure's parameter colour (c30638E), Fig. 4's truth grey
SIDE, PX = 0.58, 600                                    # tile size in the TikZ (in)


def frame(recs):
    """One square frame holding every recipe of a stage (drawn once off-screen to measure)."""
    fig = plt.figure(figsize=(SIDE, SIDE)); ax = fig.add_axes([0, 0, 1, 1])
    for r in recs: K.glyph(ax, 'sorghum', r, GREY, stem=GREY)
    bb = ax.dataLim; plt.close(fig)
    w = max(abs(bb.x0), abs(bb.x1)); y0 = min(bb.y0, 0.0); st = max(2 * w, bb.y1 - y0) * 1.06
    return (-st / 2, st / 2), (y0 - 0.02 * st, y0 + 0.98 * st)


def tile(rec, truth, lims, path):
    fig = plt.figure(figsize=(SIDE, SIDE), dpi=PX); ax = fig.add_axes([0.05, 0.05, 0.90, 0.90]); ax.set_axis_off()
    K.glyph(ax, 'sorghum', truth, GREY, stem=GREY, lw_scale=0.75 * SIDE / 0.735)
    K.glyph(ax, 'sorghum', rec, NAVY, stem=STEM, lw_scale=0.42 * SIDE / 0.735)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect('equal')
    path.write_bytes(K.png_fig(fig, PX)); print('wrote', path, flush=True)


if __name__ == '__main__':
    D = K.load_dump()
    d = D['stage']['sorghum']['Sorghum_10036_00']; r = D['rows']['sorghum']['Sorghum_10113']
    for stage, truth, pre, post in ((2, d['recipe_true'], d['one']['rgb:before']['recipe'], d['one']['rgb:after']['recipe']),
                                    (3, r['recipe_true'], r['pred']['head:all']['recipe'], r['pred']['ft:all']['recipe'])):
        lims = frame([truth, pre, post])
        tile(pre, truth, lims, OUT / f'stage_{stage}_before_recipe.png')
        tile(post, truth, lims, OUT / f'stage_{stage}_after_recipe.png')
