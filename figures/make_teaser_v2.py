"""Fig. 1 (teaser), final data: any field sensor in, the procedural plant out.
Two rows (sorghum, maize; one held-out TEST plant each). Left: the three sensor streams of the view
(photograph, depth frame, point cloud), any one of which is an input. Middle: the CropCode encoder.
Right: the recipe the model writes from the PHOTOGRAPH ALONE (the distilled generalist), and what the
recipe gives: the re-rendered plant (the generated cloud, Chamfer against the true cloud) and the
traits read from it, each beside its true value.
Writes paper/figs/fig_teaser.{pdf,png} + paper/overleaf/Figures/fig_teaser.pdf. CPU, ~3 min.
usage: python figures/make_teaser_v2.py [--sorghum Sorghum_10036] [--maize plant_0043]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import argparse, shutil
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import _plantviz as V

ap = argparse.ArgumentParser()
ap.add_argument('--sorghum', default='Sorghum_10036'); ap.add_argument('--maize', default='plant_0043')
ap.add_argument('--split', default='test'); ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
PL = {'sorghum': a.sorghum, 'maize': a.maize}
FIELDS = {'sorghum': None, 'maize': ('dist', 'len', 'wid', 'angle', 'droop')}
W, H = 7.05, 3.15
fig = plt.figure(figsize=(W, H))
ax0 = fig.add_axes([0, 0, 1, 1]); ax0.set_xlim(0, W); ax0.set_ylim(0, H); ax0.set_axis_off()
INK, MUTED, NAVY = '#1d221c', '#5f685c', '#2b3a67'
CS = {'rgb': '#c0392b', 'depth': '#b8860b', 'pc': '#2a9d8f', 'text': '#30638e'}
# geometry (inches)
T = 0.74                          # tile side
ROW_Y = {'sorghum': 1.72, 'maize': 0.34}
X_IN = [0.30, 0.30 + T + 0.05, 0.30 + 2 * (T + 0.05)]
X_BOX, W_BOX = 2.80, 0.88
X_REC, W_REC = 3.92, 1.40
X_CLOUD = 5.50; X_TRAIT = 6.32; W_TRAIT = 0.72
HDR_Y = 2.72


def ax_in(x, y, w, h, **kw): return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def frame(x, y, w, h, color, lw=1.0):
    ax0.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.04', fill=False, edgecolor=color, lw=lw))


def arrow(p, q, color=INK, lw=0.9):
    ax0.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, shrinkA=0, shrinkB=0))


def txt(x, y, s, size=6.5, color=INK, ha='left', va='center', weight='normal', family=None, style='normal'):
    ax0.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, family=family, style=style, linespacing=1.25)


# column headers
txt(0.30, HDR_Y + 0.22, 'Field sensing: any one of these', 7.5, weight='bold')
txt(X_BOX + W_BOX / 2, HDR_Y + 0.22, 'One model', 7.5, weight='bold', ha='center')
txt(X_REC, HDR_Y + 0.22, 'Procedural plant out, and what it gives', 7.5, weight='bold')
for lab, x, c in (('photograph', X_IN[0], CS['rgb']), ('depth frame', X_IN[1], CS['depth']), ('point cloud', X_IN[2], CS['pc'])):
    txt(x + T / 2, HDR_Y, lab, 6.2, color=c, ha='center', weight='bold')
txt(X_REC + W_REC / 2, HDR_Y, 'recipe (written by the model)', 6.2, color=CS['text'], ha='center', weight='bold')
txt(X_CLOUD + T / 2, HDR_Y, 're-rendered plant', 6.2, color=CS['pc'], ha='center', weight='bold')
txt(X_TRAIT + W_TRAIT / 2, HDR_Y, 'traits', 6.2, color=INK, ha='center', weight='bold')

# the model box, spanning both rows
by0, by1 = ROW_Y['maize'], ROW_Y['sorghum'] + T
ax0.add_patch(mpatches.FancyBboxPatch((X_BOX, by0), W_BOX, by1 - by0, boxstyle='round,pad=0,rounding_size=0.06', fill=True, facecolor='#eef1f7', edgecolor=NAVY, lw=1.3))
txt(X_BOX + W_BOX / 2, (by0 + by1) / 2 + 0.22, 'CropCode', 8, color=NAVY, ha='center', weight='bold')
txt(X_BOX + W_BOX / 2, (by0 + by1) / 2 - 0.16, 'one encoder for\nphotographs, depth,\nclouds and recipes;\npretrained with 80 %\nof tokens hidden,\nthen distilled so\none sensor completes\nthe rest', 5.0, color=NAVY, ha='center')

TRAITS = {'sorghum': [('height', lambda r: r['height'], 'm'), ('leaves', lambda r: r['n'], ''), ('leaf length', lambda r: np.mean([l['length'] for l in r['leaves']]), 'm'), ('leaf width', lambda r: np.mean([l['width'] for l in r['leaves']]), 'm')],
          'maize': [('height', lambda r: r['height'], 'm'), ('leaves', lambda r: r['n'], ''), ('leaf length', lambda r: np.mean([l['leafLength'] for l in r['leaves']]), 'm'), ('leaf angle', lambda r: np.mean([l['leafAngle'] for l in r['leaves']]), '°')]}
for sp in ('sorghum', 'maize'):
    y = ROW_Y[sp]
    ds = V.dataset(sp, a.split); it = V.item_by_name(ds, f'{PL[sp]}_00', seed=a.seed)
    m, cfg, ep = V.load_model(sp, 'generalist', a.device)
    g = V.generate(m, it, ['rgb'], a.device, seed=a.seed)
    gt_cloud = it[2].numpy(); fr = V.frame_of(gt_cloud)
    rec_t = V.decode_recipe(sp, it[3], it[4]); rec_p = V.decode_recipe(sp, g['par'], it[4], predicted=True)
    ch = V.chamfer(g['pc'], gt_cloud)
    ax0.text(0.13, y + T / 2, sp, fontsize=7, fontweight='bold', color=INK, ha='center', va='center', rotation=90)
    # inputs
    dep = it[1][0].numpy(); ys_, xs_ = np.where(dep > 0.005)
    c0, c1 = max(0, ys_.min() - 12), min(224, ys_.max() + 12); d0, d1 = max(0, xs_.min() - 12), min(224, xs_.max() + 12)
    side = max(c1 - c0, d1 - d0); cy, cx = (c0 + c1) // 2, (d0 + d1) // 2
    c0, c1 = max(0, cy - side // 2), min(224, cy + side // 2); d0, d1 = max(0, cx - side // 2), min(224, cx + side // 2)   # square crop on the plant
    ax = ax_in(X_IN[0], y, T, T); ax.imshow(V.tensor_rgb(it[0])[c0:c1, d0:d1]); ax.set_axis_off(); frame(X_IN[0], y, T, T, CS['rgb'], 1.6)
    ax = ax_in(X_IN[1], y, T, T); ax.imshow(V.depth_rgb(dep)[c0:c1, d0:d1]); ax.set_axis_off(); frame(X_IN[1], y, T, T, CS['depth'])
    ax = ax_in(X_IN[2] - 0.03, y - 0.03, T + 0.06, T + 0.06, projection='3d'); V.scatter_cloud(ax, gt_cloud, c=-gt_cloud[:, 1], cmap='viridis', frame=fr, s=0.35, elev=12, azim=-35); frame(X_IN[2], y, T, T, CS['pc'])
    # arrows into the model: the photograph is the one used
    arrow((X_IN[2] + T + 0.02, y + T / 2), (X_BOX - 0.02, y + T / 2), color=INK)
    txt(X_IN[0], y - 0.07, 'any one of these is enough; this example uses the photograph (thick frame)', 4.6, color=MUTED, va='top')
    # recipe
    arrow((X_BOX + W_BOX + 0.02, y + T / 2), (X_REC - 0.02, y + T / 2), color=CS['text'])
    frame(X_REC, y, W_REC, T, CS['text'])
    lines = V.recipe_lines(sp, rec_p, max_leaves=5, fields=FIELDS[sp])
    ax0.text(X_REC + 0.04, y + T - 0.04, '\n'.join(lines), fontsize=4.4, family='monospace', color=CS['text'], va='top', ha='left', linespacing=1.3)
    # re-rendered plant
    arrow((X_REC + W_REC + 0.02, y + T / 2), (X_CLOUD - 0.02, y + T / 2), color=CS['pc'])
    ax = ax_in(X_CLOUD - 0.03, y - 0.03, T + 0.06, T + 0.06, projection='3d'); V.scatter_cloud(ax, g['pc'], c=-np.asarray(g['pc'])[:, 1], cmap='viridis', frame=fr, s=0.35, elev=12, azim=-35); frame(X_CLOUD, y, T, T, CS['pc'])
    txt(X_CLOUD + T / 2, y - 0.07, f'Chamfer {ch * 1e3:.2f}e-3 vs the true plant', 4.6, color=MUTED, ha='center', va='top')
    # traits
    frame(X_TRAIT, y, W_TRAIT, T, INK, 0.6)
    rows = []
    for name, f, unit in TRAITS[sp]:
        vp, vt = f(rec_p), f(rec_t)
        fmt = (lambda v: f'{v:.0f}') if unit == '' else ((lambda v: f'{v:.0f}') if unit == '°' else (lambda v: f'{v:.2f}' if v >= 0.1 else f'{v:.3f}'))
        rows.append(f'{name}\n  {fmt(vp)}{unit}  (true {fmt(vt)}{unit})')
    ax0.text(X_TRAIT + 0.04, y + T - 0.04, '\n'.join(rows), fontsize=4.4, color=INK, va='top', ha='left', linespacing=1.2)
    print(sp, PL[sp], 'generalist ep', ep, f'chamfer {ch*1e3:.2f}e-3', 'pred', rec_p['n'], f"{rec_p['height']:.2f}", 'true', rec_t['n'], f"{rec_t['height']:.2f}", flush=True)
txt(0.30, 0.12, f"Held-out {a.split} plants, view 0. The recipe and the cloud are generated from the photograph alone by the distilled generalist; the true cloud and the true traits are shown for comparison only.", 4.6, color=MUTED, style='italic')
out = pathlib.Path('paper/figs')
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_teaser.{ext}', dpi=220)
shutil.copy(out / 'fig_teaser.pdf', 'paper/overleaf/Figures/fig_teaser.pdf'); print('wrote', out / 'fig_teaser.pdf')
