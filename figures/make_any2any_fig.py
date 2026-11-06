"""Paper figure: any source in, every stream out (one val plant per species, the distilled generalist).
Rows: ground truth, then the generalist given ONE stream alone (photograph / depth / point cloud / recipe).
Columns: photograph, depth, point cloud, recipe (as text). A row's own source is shown in its input column
(framed in orange). Writes paper/figs/fig_any2any.{pdf,png} + the Overleaf copy. CPU (8 forwards), ~3 min.
usage: python figures/make_any2any_fig.py [--sorghum Sorghum_10001] [--maize plant_0003]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import argparse, shutil
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import _plantviz as V

ap = argparse.ArgumentParser()
ap.add_argument('--sorghum', default='Sorghum_10001'); ap.add_argument('--maize', default='plant_0003')
ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
PL = {'sorghum': a.sorghum, 'maize': a.maize}
SRC = [('rgb', 'photograph'), ('depth', 'depth'), ('pc', 'point cloud'), ('text', 'recipe')]
FS = 6.5
FIELDS = {'sorghum': None, 'maize': ('dist', 'len', 'wid', 'angle', 'droop')}   # maize: five of the seven leaf fields fit the column
fig = plt.figure(figsize=(7.1, 7.0))
outer = GridSpec(1, 2, figure=fig, wspace=0.07, left=0.07, right=0.998, top=0.93, bottom=0.01)
for j, sp in enumerate(('sorghum', 'maize')):
    ds = V.dataset(sp); it = V.item_by_name(ds, f'{PL[sp]}_00', seed=a.seed)
    m, cfg, ep = V.load_model(sp, 'generalist', a.device)
    gt_cloud = it[2].numpy(); frame_ = V.frame_of(gt_cloud); gt_rec = V.decode_recipe(sp, it[3], it[4])
    sub = outer[0, j].subgridspec(5, 4, width_ratios=[1, 1, 1, 2.7], wspace=0.05, hspace=0.3)
    rows = [('ground truth', None)] + [(f'from the {lab} alone', src) for src, lab in SRC]

    def frame(ax):
        if hasattr(ax, 'get_zlim'):   # 3-D axes: a figure-level rectangle in axes coordinates (set_axis_on would draw the box)
            fig.add_artist(mpatches.Rectangle((0.08, 0.02), 0.84, 0.96, transform=ax.transAxes, fill=False, edgecolor=V.C['hidden'], lw=1.6)); return
        ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_edgecolor(V.C['hidden']); s_.set_linewidth(1.6)
    gt_rgb, gt_dep = V.tensor_rgb(it[0]), it[1][0].numpy()
    for r, (rlab, src) in enumerate(rows):
        if src is None:
            rgb, dep, pc, rec = gt_rgb, gt_dep, gt_cloud, gt_rec
        else:
            g = V.generate(m, it, [src], a.device, seed=a.seed)
            rgb, dep, pc = g['rgb'], g['depth'], g['pc']; rec = V.decode_recipe(sp, g['par'], it[4], predicted=True)
        # the source column shows the INPUT itself (framed); the other columns the generated streams
        if src == 'rgb': rgb = gt_rgb
        if src == 'depth': dep = gt_dep
        if src == 'pc': pc = gt_cloud
        if src == 'text': rec = gt_rec
        col = V.C['gt'] if src in (None, 'text') else V.C['pred']
        ax = fig.add_subplot(sub[r, 0]); ax.imshow(rgb); ax.set_axis_off()
        if r == 0: ax.set_title('photograph', fontsize=FS, pad=2)
        if src == 'rgb': frame(ax)
        # depth: the model predicts per-image normalised depth, so each map is shown on its own range
        ax = fig.add_subplot(sub[r, 1]); fg = dep > 0.02
        ax.imshow(V.depth_rgb(np.where(fg, dep, 0), lo=np.percentile(dep[fg], 1) if fg.any() else 0, hi=np.percentile(dep[fg], 99) if fg.any() else 1)); ax.set_axis_off()
        if r == 0: ax.set_title('depth', fontsize=FS, pad=2)
        if src == 'depth': frame(ax)
        ax = fig.add_subplot(sub[r, 2], projection='3d')
        ttl = 'point cloud' if r == 0 else ('input cloud' if src == 'pc' else f'Chamfer\n{V.chamfer(pc, gt_cloud) * 1e3:.2f}e-3')
        V.scatter_cloud(ax, pc, c=-np.asarray(pc)[:, 1], cmap='viridis', frame=frame_, s=0.45, title=ttl, fs=FS, elev=12, azim=-35)
        if src == 'pc': frame(ax)
        ax = fig.add_subplot(sub[r, 3])
        V.draw_recipe_text(ax, V.recipe_lines(sp, rec, max_leaves=6, fields=FIELDS[sp]), col, fs=FS, tfs=4.8,
                           title=('recipe' if r == 0 else ('input recipe' if src == 'text' else 'generated recipe')))
        if src == 'text': frame(ax)
        if j == 0: fig.text(0.005, 0.93 - (r + 0.5) * 0.92 / 5, rlab, fontsize=FS, rotation=90, va='center')
    fig.text(0.075 + 0.46 * j + 0.23, 0.955, f'{sp} ({PL[sp]}, val), distilled generalist', fontsize=FS + 0.5, ha='center')
    print(sp, 'done')
out = pathlib.Path('paper/figs')
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_any2any.{ext}', dpi=200)
shutil.copy(out / 'fig_any2any.pdf', 'paper/overleaf/Figures/fig_any2any.pdf'); print('wrote', out / 'fig_any2any.pdf')
