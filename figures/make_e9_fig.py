"""E9 paper figure: t-SNE of the frozen CLS latents (val, sorghum final release) of the three-stream
(PC+RGB+D) and four-stream arms, coloured by leaf count (the size axis) and by mean roll (the axis
independent of size). Reads the probe caches (outputs/_probe_cache/<run>__checkpoint_epoch_600__val__cls__
seed0__rep1__g1.npz) and eval/linear_probe.load_targets. Writes paper/figs/fig_e9_tsne.{pdf,png} + the Overleaf copy.
usage: python figures/make_e9_fig.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'eval'))
import shutil
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import linear_probe as LP

T = LP.load_targets(LP.DATA_ROOT)
ARMS = [('e2_pcrgbd_d2', 'three streams (PC+RGB+D)'), ('e2_pcrgbdt_d2', 'four streams (+ recipe)')]
COLS = [('n_leaves', 'leaf count', 'Blues'), ('roll_mean', 'mean leaf roll (deg)', 'viridis')]
fig, axs = plt.subplots(2, 2, figsize=(6.8, 6.2))
for i, (run, lab) in enumerate(ARMS):
    z = np.load(f'outputs/_probe_cache/{run}__checkpoint_epoch_600__val__cls__seed0__rep1__g1.npz', allow_pickle=True)
    X = StandardScaler().fit_transform(z['feats']); Y = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(X)
    t = T.reindex(z['plants'].astype(int))
    for j, (col, clab, cmap) in enumerate(COLS):
        ax = axs[i, j]; v = t[col].values
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=v, s=3, cmap=cmap, vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98), linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if i == 0: ax.set_title(clab, fontsize=9)
        if j == 0: ax.set_ylabel(lab, fontsize=9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
fig.tight_layout()
out = pathlib.Path('paper/figs'); out.mkdir(exist_ok=True)
for ext in ('pdf', 'png'): fig.savefig(out / f'fig_e9_tsne.{ext}', dpi=220)
shutil.copy(out / 'fig_e9_tsne.pdf', 'paper/overleaf/Figures/fig_e9_tsne.pdf')
print('wrote', out / 'fig_e9_tsne.pdf')
