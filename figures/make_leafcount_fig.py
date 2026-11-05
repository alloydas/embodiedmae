"""Fig. 4 companion: photograph -> cloud Chamfer (test, 2,250 plants per species) by leaf-count decile,
for the headline model used as is and for the distilled generalist (from RGB alone). Reads
reports/quant_{teacher_undistilled,generalist_rgb}_d2_test.json (sorghum) and
reports/quant_maize_{e1,generalist_rgb}_d2_test.json (maize); leaf counts from Sorghum_15K/features.csv
and Maize_1/plant_scores.csv. Writes paper/figs/fig_leafcount.{pdf,png} + Overleaf copy and prints
Spearman rho of error vs leaf count.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, re, shutil, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = pathlib.Path('reports')
sf = pd.read_csv('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/features.csv').set_index('plant')['n_leaves']
mroot = '/work/mech-ai-scratch/alloy/Maize_1' if os.path.exists('/work/mech-ai-scratch/alloy/Maize_1/plant_scores.csv') else '/work/mech-ai-scratch/alloy/Maize'
mf = pd.read_csv(f'{mroot}/plant_scores.csv'); mf = mf.set_index(mf.columns[0])['leaf_count']
SP = {'sorghum': (['quant_teacher_undistilled_d2_test', 'quant_generalist_rgb_d2_test'], lambda n: sf.get(int(re.match(r'Sorghum_(\d+)_', n).group(1)))),
      'maize': (['quant_maize_e1_d2_test', 'quant_maize_generalist_rgb_d2_test'], lambda n: mf.get(re.match(r'(plant_\d+)_', n).group(1)))}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, (sp, (files, leaves)) in zip(axs, SP.items()):
    for f, lab, c, mk in zip(files, ['headline model (before)', 'distilled generalist (after)'], ['#b8860b', '#1f6f8b'], ['o', 's']):
        j = json.load(open(R / f'{f}.json'))
        df = pd.DataFrame({'n': [leaves(p) for p in j['plants']], 'ch': np.asarray(j['per_sample']['model']) * 1e3}).dropna()
        df['dec'] = pd.qcut(df.n.rank(method='first'), 10, labels=False)
        g = df.groupby('dec').agg(n=('n', 'mean'), med=('ch', 'median'), lo=('ch', lambda x: x.quantile(.25)), hi=('ch', lambda x: x.quantile(.75)))
        ax.plot(g.n, g.med, marker=mk, ms=3.5, lw=1.4, color=c, label=lab)
        ax.fill_between(g.n, g.lo, g.hi, color=c, alpha=0.15, lw=0)
        rho = spearmanr(df.n, df.ch).correlation
        print(f'{sp:8s} {lab:30s} median {df.ch.median():.2f}e-3  spearman(leaves, chamfer) {rho:+.3f}  n={len(df)}')
    ax.set_yscale('log'); ax.set_title(sp, fontsize=9)
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5))); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.set_xlabel('leaf count (decile mean)', fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.3, lw=.5)
axs[0].set_ylabel('Chamfer from the photograph\n($\\times10^{-3}$, median and IQR)', fontsize=8); axs[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout()
out = pathlib.Path('paper/figs')
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_leafcount.{ext}', dpi=220)
shutil.copy(out / 'fig_leafcount.pdf', 'paper/overleaf/Figures/fig_leafcount.pdf')
print('wrote', out / 'fig_leafcount.pdf')
