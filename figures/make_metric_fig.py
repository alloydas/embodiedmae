"""Fig. 6: Chamfer distance against the frozen-probe R^2 across every sorghum arm.

The twelve points are the E2 arms, the three token-budget controls, the E3 data
arms and the E4 model sizes (val, first sorghum release), read from the tables
in paper/results.tex.  The metrics agree on the gross ordering (PC-only is worst
on both) but disagree on the decisions that matter: depth, the parameter
stream, and more data all move Chamfer and leave the probe alone.

    python figures/make_metric_fig.py        # writes paper/fig_metrics.{pdf,png}
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'paper'
# name: (chamfer x1e-3, height R2, roll R2, leaf-length R2), group
ARMS = {
    'PC':            ((4.96, 0.899, 0.452, 0.102), 'E2'),
    'PC+RGB':        ((2.50, 0.978, 0.796, 0.273), 'E2'),
    'PC+RGB+D':      ((1.71, 0.981, 0.806, 0.362), 'E2'),
    r'+$\theta$':    ((2.50, 0.966, 0.793, 0.280), 'E2'),
    r'+$\theta$ 20%':((2.41, 0.958, 0.764, 0.288), 'ctrl'),
    r'+$\theta$ 60%':((2.72, 0.922, 0.676, 0.202), 'ctrl'),
    r'+$\theta$ target':((2.54, 0.970, 0.737, 0.259), 'ctrl'),
    '1k':            ((3.02, 0.963, 0.770, 0.269), 'E3'),
    '3k':            ((2.61, 0.963, 0.764, 0.255), 'E3'),
    '10k':           ((2.42, 0.962, 0.789, 0.301), 'E3'),
    'small':         ((3.02, 0.946, 0.761, 0.252), 'E4'),
    'large':         ((2.53, 0.967, 0.799, 0.298), 'E4'),
}
COL = {'E2': '#1f4e79', 'ctrl': '#c0504d', 'E3': '#4f9d69', 'E4': '#8064a2'}
MK = {'E2': 'o', 'ctrl': 's', 'E3': '^', 'E4': 'D'}
LAB = {'E2': 'streams (E2)', 'ctrl': r'$\theta$ controls', 'E3': 'training plants (E3)', 'E4': 'model size (E4)'}

def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
    cols = [(1, 'height $R^2$'), (2, 'leaf roll $R^2$'), (3, 'leaf length $R^2$')]
    for ax, (j, yl) in zip(axes, cols):
        xs = np.array([v[0][0] for v in ARMS.values()]); ys = np.array([v[0][j] for v in ARMS.values()])
        rho = stats.spearmanr(xs, ys)[0]
        for name, (v, g) in ARMS.items():
            ax.scatter(v[0], v[j], s=22, c=COL[g], marker=MK[g], zorder=3, label=LAB[g] if name in ('PC', r'+$\theta$ 20%', '1k', 'small') else None)
            dx = 0.08
            ha = 'left'
            if name in ('PC+RGB', '10k', '3k'):
                dx, ha = -0.08, 'right'
            ax.annotate(name, (v[0], v[j]), xytext=(dx * 10, 0), textcoords='offset points', fontsize=5.2, ha=ha, va='center', color='#333')
        ax.set_xlabel(r'Chamfer $\times10^{-3}$ (lower is better)', fontsize=6.5)
        ax.set_ylabel(yl, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlim(1.4, 5.3)
        ax.invert_xaxis()
        ax.text(0.02, 0.93, r'Spearman $\rho$ = %+.2f' % rho, transform=ax.transAxes, fontsize=6, color='#555')
        ax.grid(True, lw=0.3, alpha=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=5.5, loc='lower right', frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_metrics.{ext}', dpi=300)
    print('wrote', OUT / 'fig_metrics.pdf')

if __name__ == '__main__':
    main()
