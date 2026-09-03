import csv, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SP = "/tmp/claude-490224/-work-mech-ai-scratch-alloy-embodiedmae/36e07a10-6288-4299-a7ff-576ed7a36a14/scratchpad"
OUT = "/work/mech-ai-scratch/alloy/embodiedmae/data_split_distribution.png"

# ── load + merge ────────────────────────────────────────────────────────────
assign = {int(r['plant']): r for r in csv.DictReader(open(f"{SP}/assignment.csv"))}
feats  = {int(r['plant']): r for r in csv.DictReader(open(f"{SP}/features.csv"))}
plants = sorted(assign)

split = np.array([assign[p]['split'] for p in plants])
ext   = np.array([float(assign[p]['extremeness']) for p in plants])
F = {k: np.array([float(feats[p][k]) for p in plants]) for k in
     ['n_leaves','stem_length','leaf_len_mean','leaf_len_max','roll_mean','roll_std']}
F['extremeness'] = ext

SPLITS = ['train', 'val', 'test']
# Okabe-Ito colorblind-safe, fixed order
COLOR = {'train': '#0072B2', 'val': '#E69F00', 'test': '#009E73'}
masks = {s: (split == s) for s in SPLITS}
N = {s: int(masks[s].sum()) for s in SPLITS}

def kde(x, grid, bw=None):
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std() == 0:
        return np.zeros_like(grid)
    from scipy.stats import gaussian_kde
    k = gaussian_kde(x, bw_method=bw)
    return k(grid)

PANELS = [
    ('extremeness',   'Extremeness (Mahalanobis dist.)', False),
    ('n_leaves',      'Number of leaves',                 True),
    ('stem_length',   'Stem length',                      False),
    ('leaf_len_mean', 'Mean leaf length',                 False),
    ('leaf_len_max',  'Max leaf length',                  False),
    ('roll_mean',     'Mean leaf roll angle (deg)',       False),
    ('roll_std',      'Leaf roll angle spread (std, deg)',False),
]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#888',
                     'axes.linewidth': 0.8, 'axes.grid': True,
                     'grid.color': '#e8e8e8', 'grid.linewidth': 0.7})

fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
axes = axes.ravel()

for ax, (key, label, discrete) in zip(axes, PANELS):
    x = F[key]
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    for s in SPLITS:
        xs = x[masks[s]]
        c = COLOR[s]
        if discrete:
            vals = np.arange(int(x.min()), int(x.max()) + 2)
            h, edges = np.histogram(xs, bins=vals - 0.5, density=True)
            ax.step(vals[:-1], h, where='mid', color=c, lw=2, alpha=0.9)
            ax.fill_between(vals[:-1], h, step='mid', color=c, alpha=0.12)
        else:
            grid = np.linspace(lo, hi, 300)
            d = kde(xs, grid)
            ax.plot(grid, d, color=c, lw=2, alpha=0.95)
            ax.fill_between(grid, d, color=c, alpha=0.12)
        # median tick
        med = np.median(xs)
        ax.axvline(med, color=c, lw=1.1, ls=(0, (4, 3)), alpha=0.8)
    ax.set_title(label, fontsize=11, pad=6)
    ax.set_xlim(lo, hi)
    ax.set_ylabel('density', color='#555', fontsize=9)
    ax.margins(y=0.02)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    if key == 'extremeness':
        ax.set_facecolor('#fbfbf7')

# ── 8th panel: enrichment summary (% top-decile-extreme per split) ──────────
ax = axes[7]
rank = np.array([float(assign[p]['rank']) for p in plants])
pct = {s: 100 * np.mean(rank[masks[s]] > 0.9) for s in SPLITS}
medext = {s: np.median(ext[masks[s]]) for s in SPLITS}
xpos = np.arange(3)
bars = ax.bar(xpos, [pct[s] for s in SPLITS],
              color=[COLOR[s] for s in SPLITS], width=0.62,
              edgecolor='white', linewidth=1.5)
for i, s in enumerate(SPLITS):
    ax.text(i, pct[s] + 0.6, f"{pct[s]:.1f}%", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#333')
    ax.text(i, -2.4, f"med ext\n{medext[s]:.2f}", ha='center', va='top',
            fontsize=8.5, color='#555')
ax.set_xticks(xpos); ax.set_xticklabels([s for s in SPLITS])
ax.set_title('Extreme-plant enrichment\n(% in top-decile of extremeness)',
             fontsize=11, pad=6)
ax.set_ylabel('% of split', color='#555', fontsize=9)
ax.set_ylim(0, max(pct.values()) * 1.25)
for sp_ in ('top', 'right'):
    ax.spines[sp_].set_visible(False)
ax.grid(axis='x', visible=False)

# ── legend + title ──────────────────────────────────────────────────────────
handles = [Line2D([0], [0], color=COLOR[s], lw=3,
                  label=f"{s}  (n={N[s]:,} plants)") for s in SPLITS]
handles.append(Line2D([0], [0], color='#888', lw=1.1, ls=(0, (4, 3)),
                      label='per-split median'))
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.99), fontsize=10.5)

fig.suptitle("Sorghum_15K data-split distributions  —  extreme-enriched split "
             "(density-normalized; all 10 views per plant kept in one split)",
             fontsize=13, fontweight='bold', y=1.04)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print("saved", OUT)

# print the numbers too
print("\nsplit  n_plants  med_extremeness  %top-decile-extreme")
for s in SPLITS:
    print(f"{s:6s} {N[s]:8d}  {medext[s]:.3f}          {pct[s]:.1f}%")
