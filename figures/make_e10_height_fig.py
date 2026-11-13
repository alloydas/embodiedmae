"""E10: predicted vs true plant height on the held-out real maize plants, before and after the real-row fine-tune.

One point per held-out plant (130 plants, both layouts): the median over its 36 camera stops of the height the model
writes into the recipe (all three streams in), against the plant's true height. Row fine-tune (synthetic rows) vs the
real-row fine-tune (75 % of the real plants). The shaded band is the training height range; the dashed line at 2.0 m
is the top of the recipe head's height range.
usage: python figures/make_e10_height_fig.py  ->  paper/figs/fig_e10_height.{pdf,png} (+ the Overleaf copy)
`--ugg`: the UGG-style preview (figures/ugg_style.py: the synthetic-row fine-tune in grey, the real-row fine-tune in
OURS violet, hollow markers, y = x and the head-range limit as dashed grey references, legend above the panel, light
horizontal grid; same data, fonts and figure size), written ONLY to paper/figs/ugg_preview/fig_e10_height_ugg.{pdf,png}.
"""
import pathlib, shutil, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
UGG = '--ugg' in sys.argv
if UGG:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import ugg_style as U

R = pathlib.Path(__file__).resolve().parents[1]
d = pd.read_csv(R / 'reports/e10_real_ft_test_17383192.csv', dtype={'row': str})
d = d[d.source == 'all']
TRAIN_MIN, TRAIN_MAX, HEAD_MAX = 0.66, 1.326, 2.0
INK, MUTED = '#1d221c', '#5f685c'
S = {'maize_row_ft_e1': ('Row fine-tune (synthetic rows)', '#b7ae9a', 'o'), 'maize_real_ft': ('Real-row fine-tune (75 % of real plants)', '#2b3a67', 'o')}
if UGG:
    S = {'maize_row_ft_e1': (S['maize_row_ft_e1'][0], U.GREYS[1], 'o'), 'maize_real_ft': (S['maize_real_ft'][0], U.OURS, 'o')}
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    ax.axvspan(TRAIN_MIN, TRAIN_MAX, color='#f1f1f1', zorder=0, lw=0)
    ax.text((TRAIN_MIN + TRAIN_MAX) / 2, 2.42, 'training\nheights', ha='center', va='top', fontsize=6, color=MUTED)
    ax.plot([0.6, 2.55], [0.6, 2.55], color=U.REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
    U.reference(ax, y=HEAD_MAX)
    ax.text(2.53, HEAD_MAX + 0.03, 'top of the head range', ha='right', va='bottom', fontsize=5.6, color=MUTED, zorder=2,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
    for m, (lab, c, mk) in S.items():
        g = d[d.model == m].groupby('row').agg(t=('height', 'first'), p=('height_pred', 'median'), e=('height_rel_err', 'median'))
        ax.scatter(g.t, g.p, s=9, marker=mk, facecolors='white', edgecolors=c, linewidths=0.7, zorder=3,
                   label=f'{lab}: median error {100 * d[d.model == m].height_rel_err.median():.1f} %')
    ax.set_xlim(0.6, 2.55); ax.set_ylim(0.6, 2.55); ax.set_aspect('equal')
    ax.set_xlabel('true plant height (m)', fontsize=7); ax.set_ylabel('predicted height (m)', fontsize=7)
    ax.tick_params(labelsize=6, length=2)
    U.style_axes(ax, grid='y'); ax.tick_params(length=2)
    ax.legend(fontsize=5.6, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), handletextpad=0.3,
              borderaxespad=0, ncol=1, markerscale=1.3)
    fig.tight_layout(pad=0.3)
    out = R / 'paper/figs/ugg_preview'; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig_e10_height_ugg.pdf'); fig.savefig(out / 'fig_e10_height_ugg.png', dpi=300); plt.close(fig)
    print('wrote', out / 'fig_e10_height_ugg.pdf'); sys.exit(0)
fig, ax = plt.subplots(figsize=(3.3, 2.75))
ax.axvspan(TRAIN_MIN, TRAIN_MAX, color='#e8ede4', zorder=0, lw=0)
ax.text((TRAIN_MIN + TRAIN_MAX) / 2, 2.42, 'training\nheights', ha='center', va='top', fontsize=6, color=MUTED)
ax.plot([0.6, 2.55], [0.6, 2.55], color='#999999', lw=0.7, zorder=1)
ax.axhline(HEAD_MAX, color='#999999', lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(2.53, HEAD_MAX + 0.03, 'top of the head range', ha='right', va='bottom', fontsize=5.6, color=MUTED)
for m, (lab, c, mk) in S.items():
    g = d[d.model == m].groupby('row').agg(t=('height', 'first'), p=('height_pred', 'median'), e=('height_rel_err', 'median'))
    ax.scatter(g.t, g.p, s=9, color=c, marker=mk, linewidths=0, alpha=0.9, zorder=3,
               label=f'{lab}: median error {100 * d[d.model == m].height_rel_err.median():.1f} %')
ax.set_xlim(0.6, 2.55); ax.set_ylim(0.6, 2.55); ax.set_aspect('equal')
ax.set_xlabel('true plant height (m)', fontsize=7); ax.set_ylabel('predicted height (m)', fontsize=7)
ax.tick_params(labelsize=6, length=2)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(fontsize=5.6, frameon=False, loc='upper left', bbox_to_anchor=(0.0, -0.2), handletextpad=0.3, borderaxespad=0)
fig.tight_layout(pad=0.3)
out = R / 'paper/figs/fig_e10_height'
fig.savefig(f'{out}.pdf'); fig.savefig(f'{out}.png', dpi=300); plt.close(fig)
shutil.copy(f'{out}.pdf', R / 'paper/overleaf/Figures/fig_e10_height.pdf')
print('wrote', f'{out}.pdf', f'{out}.png')
