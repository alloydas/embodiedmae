"""Paper figure: FieldProc at a glance (both species, final data releases).
  a. the four streams of one view (photograph, depth, point cloud, procedural recipe as a glyph),
  b. the ten cameras of one plant (sorghum: the fixed elevation ladder; maize: random cameras, sorted),
  c. rows: the row photograph at yaw 0 and 90 with neighbour pixels (red) and hidden target pixels
     (orange) marked, the target alone, and the scene cloud with the target's points coloured.
Writes paper/figs/fig_dataset.{pdf,png} + the Overleaf copy. CPU, ~2 min.
usage: python figures/make_dataset_fig.py [--sorghum Sorghum_10001] [--maize plant_0003] [--row-sorghum Sorghum_10007] [--row-maize plant_0004]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import argparse, shutil
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
import _plantviz as V

ap = argparse.ArgumentParser()
ap.add_argument('--sorghum', default='Sorghum_10001'); ap.add_argument('--maize', default='plant_0003')
ap.add_argument('--row-sorghum', default='Sorghum_10007'); ap.add_argument('--row-maize', default='plant_0004')
a = ap.parse_args()
PL = {'sorghum': a.sorghum, 'maize': a.maize}; RP = {'sorghum': a.row_sorghum, 'maize': a.row_maize}
elev_tab = pd.read_csv('data_split/maize_view_elevation.csv.gz')

fig = plt.figure(figsize=(7.1, 5.9))
gs = GridSpec(4, 8, figure=fig, height_ratios=[1.0, 0.66, 0.66, 1.0], hspace=0.42, wspace=0.08, left=0.035, right=0.99, top=0.95, bottom=0.01)
FS = 6.2


def label(ax, s): ax.set_title(s, fontsize=FS, pad=2)


# ---- a. four streams
for j, sp in enumerate(('sorghum', 'maize')):
    f = V.folder(sp, 'val', PL[sp], 0)
    ds = V.dataset(sp); it = V.item_by_name(ds, f'{PL[sp]}_00')
    ax = fig.add_subplot(gs[0, 4 * j]); ax.imshow(V.rgba_on_white(f / V.RGB_FILE[sp])); ax.set_axis_off(); label(ax, f'{sp}: photograph')
    ax = fig.add_subplot(gs[0, 4 * j + 1]); ax.imshow(V.depth_rgb(V.decode_depth_png(f / 'depth.png'))); ax.set_axis_off(); label(ax, 'depth')
    ax = fig.add_subplot(gs[0, 4 * j + 2], projection='3d'); P = it[2].numpy()
    V.scatter_cloud(ax, P, c=-P[:, 1], cmap='viridis', s=0.5, title='point cloud', fs=FS, elev=12, azim=-35)
    ax = fig.add_subplot(gs[0, 4 * j + 3]); rec = V.decode_recipe(sp, it[3], it[4])
    V.draw_recipe_glyph(ax, sp, rec, V.C['gt'], title=f'recipe ({rec["n"]} leaves)', fs=FS)
fig.text(0.005, 0.975, 'a', fontsize=9, fontweight='bold')
fig.text(0.03, 0.975, 'Four streams per view: photograph, depth frame, camera-frame point cloud, procedural recipe (drawn as a stem-and-leaves glyph)', fontsize=FS)

# ---- b. ten cameras
for r, sp in enumerate(('sorghum', 'maize')):
    if sp == 'sorghum':
        order = list(range(10)); el = V.SORGHUM_ELEV
    else:
        t = elev_tab[(elev_tab.split == 'val') & (elev_tab.plant == PL[sp])].sort_values('elev_deg', ascending=False)
        order = t.view.tolist(); el = t.elev_deg.values
    sub = gs[1 + r, :].subgridspec(1, 10, wspace=0.04)
    for k, (v, e) in enumerate(zip(order, el)):
        ax = fig.add_subplot(sub[0, k]); ax.imshow(V.rgba_on_white(V.folder(sp, 'val', PL[sp], v) / V.RGB_FILE[sp], 256)); ax.set_axis_off()
        ax.set_title(f'{e:+.0f}°' if sp == 'sorghum' else f'{e:+.0f}°', fontsize=FS - 0.5, pad=1.5)
    fig.text(0.005, 0.64 - 0.2 * r, sp, fontsize=FS, rotation=90, va='center')
fig.text(0.005, 0.73, 'b', fontsize=9, fontweight='bold')
fig.text(0.03, 0.73, 'Ten cameras per plant, labelled by camera elevation above the plant: sorghum on a fixed ladder, maize at random elevations and azimuths (sorted here)', fontsize=FS)

# ---- c. rows
for j, sp in enumerate(('sorghum', 'maize')):
    for k, th in enumerate((0, 90)):
        d = V.row_shot_dir(sp, 'test', RP[sp], th)
        img, st = V.row_overlay(sp, d)
        ax = fig.add_subplot(gs[3, 4 * j + 2 * k]); ax.imshow(img); ax.set_axis_off()
        label(ax, f'row, yaw {th}°\n{st["occ"] * 100:.0f} % of target hidden')
        if k == 0:
            ax = fig.add_subplot(gs[3, 4 * j + 1]); ax.imshow(V.rgba_on_white(d / 'gt_rgba.png')); ax.set_axis_off(); label(ax, 'target alone\n(same camera)')
        else:
            scene = V.read_cloud(d / 'pointcloud_cam.ply'); tgt = V.read_cloud(d / 'gt_pointcloud_cam.ply')
            is_t = cKDTree(tgt).query(scene, workers=1)[0] < 3e-3
            ax = fig.add_subplot(gs[3, 4 * j + 3], projection='3d')
            col = np.where(is_t[:, None], np.array([[0.18, 0.42, 0.16]]), np.array([[0.75, 0.75, 0.75]]))
            V.scatter_cloud(ax, scene, c=col, s=0.4, title=f'scene cloud, yaw 90°\n{100 * (~is_t).mean():.0f} % neighbour pts', fs=FS, elev=10, azim=-30)
for j_, sp_ in enumerate(('sorghum', 'maize')): fig.text(0.25 + 0.49 * j_, 0.255, sp_, fontsize=FS + 0.5, ha='center')
fig.text(0.005, 0.285, 'c', fontsize=9, fontweight='bold')
fig.text(0.03, 0.285, 'Rows: the target between two neighbours. Red = neighbour pixels, orange = target pixels hidden behind a neighbour, grey points = neighbours', fontsize=FS)
out = pathlib.Path('paper/figs')
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_dataset.{ext}', dpi=220)
shutil.copy(out / 'fig_dataset.pdf', 'paper/overleaf/Figures/fig_dataset.pdf'); print('wrote', out / 'fig_dataset.pdf')
