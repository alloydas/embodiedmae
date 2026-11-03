"""E5 held-out-view probe figure: test R2 of roll and height per probe view, one line per
training-view band; filled markers = views the arm trained on.
Reads reports/probe_pr_e5_v<k>_<job>.csv (eval/linear_probe.py --probe-view k).
Writes paper/figs/fig_e5_views.{pdf,png} and the Overleaf copy.
usage: python figures/make_e5_view_fig.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import glob, re, shutil
import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob('reports/probe_pr_e5_v*_*.csv')):
    k = int(re.search(r'_v(\d)_', f).group(1)); d = pd.read_csv(f); d = d[d.feature == 'cls']; d['view'] = k; rows.append(d)
d = pd.concat(rows)
ARMS = [('e2_pcrgbdt_d2', 'all ten views', '#2a6f97', list(range(10))),
        ('e5_topdown_d2', 'top-down (0-1)', '#c46b2d', [0, 1]),
        ('e5_single_d2', 'single (0)', '#7a5195', [0]),
        ('e5_horizon_d2', 'horizon (4-5)', '#3a8d55', [4, 5])]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, (t, lab) in zip(axs, [('leaf_angle_roll', 'roll'), ('height', 'height')]):
    p = d[d.target == t].pivot_table(index='view', columns='run', values='test_r2')
    for r, name, c, tr in ARMS:
        y = p[r]; ax.plot(y.index, y.values, color=c, lw=1.6, label=name)
        for v, val in y.items():
            ax.plot(v, val, 'o', ms=4.5, mec=c, mfc=c if v in tr else 'white', mew=1.4)
    ax.set_xticks(range(10)); ax.set_xlabel('probe view (0 = 64° down, 4-5 horizon, 9 = 64° up)', fontsize=7)
    ax.set_ylabel(f'test $R^2$ ({lab})', fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=.3, lw=.5)
    ax.set_xlim(-0.4, 9.4)
axs[0].legend(fontsize=6.5, frameon=False, loc='lower left')
fig.tight_layout()
out = pathlib.Path('paper/figs'); out.mkdir(exist_ok=True)
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_e5_views.{ext}', dpi=200)
shutil.copy(out / 'fig_e5_views.pdf', 'paper/overleaf/Figures/fig_e5_views.pdf')
print('views', sorted(d.view.unique()))
