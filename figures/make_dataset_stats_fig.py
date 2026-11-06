"""Paper figure: what the two generators span and how the splits are built.
  a. eight val plants per species along the leaf-count range (photograph near the horizon), labelled n leaves / height,
  b. leaf-count distribution per split (the extreme-enriched val / test tails),
  c. height against leaf count for all 15,000 plants per species (sorghum r = 0.99: one variable; maize r = 0.2),
  d. row occlusion by yaw (test rows): share of the target hidden and share of plant pixels belonging to neighbours.
Writes paper/figs/fig_dataset_stats.{pdf,png} + the Overleaf copy. CPU.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glob, shutil
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import _plantviz as V

FS = 6.5
sf = pd.read_csv(V.ROOTS['sorghum'] / 'features.csv').merge(pd.read_csv(V.ROOTS['sorghum'] / 'assignment.csv')[['plant', 'split']], on='plant')
sf['height'] = sf['stem_length']; sf['leaves'] = sf['n_leaves']; sf['name'] = 'Sorghum_' + sf['plant'].astype(str)
mf = pd.read_csv(V.ROOTS['maize'] / 'plant_scores.csv'); mf['height'] = mf['stem_internodeSum']; mf['leaves'] = mf['leaf_count']; mf['name'] = mf['plant']
T = {'sorghum': sf, 'maize': mf}
elev_tab = pd.read_csv('data_split/maize_view_elevation.csv.gz')
SPLITC = {'train': '#9aa5a0', 'val': '#1f6f8b', 'test': '#b8860b'}

fig = plt.figure(figsize=(7.1, 4.9))
gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 0.8, 1.25], width_ratios=[1, 1, 1, 1.25], hspace=0.75, wspace=0.42, left=0.065, right=0.99, top=0.95, bottom=0.1)

# ---- a. the range
for r, sp in enumerate(('sorghum', 'maize')):
    t = T[sp][T[sp].split == 'val'].sort_values('leaves')
    picks = [t.iloc[int(q * (len(t) - 1))] for q in np.linspace(0.02, 0.98, 8)]
    sub = gs[r, :].subgridspec(1, 8, wspace=0.04)
    for k, p in enumerate(picks):
        if sp == 'sorghum': v = 4                                    # +6 deg above the horizon on the ladder
        else:
            e = elev_tab[(elev_tab.split == 'val') & (elev_tab.plant == p['name'])]; v = int(e.iloc[(e.elev_deg - 10).abs().argmin()]['view'])
        ax = fig.add_subplot(sub[0, k]); ax.imshow(V.rgba_on_white(V.folder(sp, 'val', p['name'], v) / V.RGB_FILE[sp], 256)); ax.set_axis_off()
        ax.set_title(f'{int(p["leaves"])} leaves, {p["height"]:.2f} m', fontsize=FS - 0.8, pad=1.5)
    fig.text(0.005, 0.84 - 0.26 * r, sp, fontsize=FS, rotation=90, va='center')
fig.text(0.005, 0.975, 'a', fontsize=9, fontweight='bold')
fig.text(0.03, 0.975, 'The morphological range: eight val plants per species along the leaf-count range (sorghum 4-35 leaves, maize 3-22)', fontsize=FS)

# ---- b. splits
for j, sp in enumerate(('sorghum', 'maize')):
    ax = fig.add_subplot(gs[2, j]); t = T[sp]
    bins = np.arange(t.leaves.min() - 0.5, t.leaves.max() + 1.5, 1)
    for split in ('train', 'val', 'test'):
        ax.hist(t[t.split == split].leaves, bins=bins, density=True, histtype='step', lw=1.2, color=SPLITC[split], label=f'{split} ({(t.split == split).sum():,})')
    ax.set_xlabel(f'{sp} leaf count', fontsize=FS); ax.set_ylabel('density', fontsize=FS); ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 1.5, frameon=False, loc='upper right', handlelength=1.2, borderaxespad=0.1); ax.grid(alpha=.3, lw=.5); ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
fig.text(0.005, 0.40, 'b', fontsize=9, fontweight='bold')
fig.text(0.03, 0.40, 'Splits enriched with extreme plants', fontsize=FS)

# ---- c. the trait structure
ax = fig.add_subplot(gs[2, 2])
for sp, col in (('sorghum', V.C['gt']), ('maize', V.C['pred'])):
    t = T[sp]; r = np.corrcoef(t.leaves, t.height)[0, 1]
    ax.scatter(t.leaves + np.random.default_rng(0).uniform(-0.3, 0.3, len(t)), t.height, s=1.2, alpha=0.25, color=col, linewidths=0, label=f'{sp}: r = {r:.2f}', rasterized=True)
ax.set_xlabel('leaf count', fontsize=FS); ax.set_ylabel('height (m)', fontsize=FS); ax.tick_params(labelsize=FS - 1)
ax.legend(fontsize=FS - 1.5, frameon=False, markerscale=4, loc='upper left', handlelength=1.0); ax.grid(alpha=.3, lw=.5)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
fig.text(0.49, 0.40, 'c', fontsize=9, fontweight='bold')
fig.text(0.51, 0.40, 'Height vs leaf count (15,000 plants)', fontsize=FS)

# ---- d. occlusion by yaw (test rows, row mode, all streams)
ax = fig.add_subplot(gs[2, 3])
for sp, pat, col in (('sorghum', 'reports/row_sorghum_sorghum_row_ft_checkpoint_epoch_100_test_*.csv', V.C['gt']), ('maize', 'reports/row_maize_maize_row_ft_e1_checkpoint_epoch_100_test_*.csv', V.C['pred'])):
    d = pd.read_csv(sorted(glob.glob(pat))[-1]); d = d[(d['mode'] == 'row') & (d['source'] == 'all')]
    g = d.groupby('theta')
    for key, ls, lab in (('occ_frac', '-', 'target hidden'), ('nbr_frac', '--', 'neighbour share')):
        med = g[key].median() * 100; lo = g[key].quantile(.25) * 100; hi = g[key].quantile(.75) * 100
        ax.plot(med.index, med.values, ls, color=col, lw=1.3, label=f'{sp}: {lab}')
        ax.fill_between(med.index, lo.values, hi.values, color=col, alpha=0.12, lw=0)
ax.set_xlabel('row yaw (°): 0 broadside, 90 end-on', fontsize=FS); ax.set_ylabel('% of pixels (median, IQR)', fontsize=FS); ax.tick_params(labelsize=FS - 1)
ax.legend(fontsize=FS - 1.8, frameon=False, loc='upper center', ncol=2, handlelength=1.3, columnspacing=0.8, bbox_to_anchor=(0.5, 1.02)); ax.grid(alpha=.3, lw=.5); ax.set_xticks(range(0, 91, 30)); ax.set_ylim(-3, 100)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
fig.text(0.745, 0.40, 'd', fontsize=9, fontweight='bold')
fig.text(0.765, 0.40, 'Row occlusion by yaw (test rows)', fontsize=FS)
out = pathlib.Path('paper/figs')
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_dataset_stats.{ext}', dpi=220)
shutil.copy(out / 'fig_dataset_stats.pdf', 'paper/overleaf/Figures/fig_dataset_stats.pdf'); print('wrote', out / 'fig_dataset_stats.pdf')
for sp in ('sorghum', 'maize'):
    t = T[sp]; print(sp, 'r(leaves, height) = %.3f' % np.corrcoef(t.leaves, t.height)[0, 1], 'leaves', t.leaves.min(), t.leaves.max(), 'height %.2f-%.2f' % (t.height.min(), t.height.max()))
