"""Degradation curves for the three RGB -> {PC, spline} specialist attempts.

All three warm-start a student from the v2 pretrain teacher and fine-tune it to
generate point cloud + growth params from RGB alone. All three reach their best
validation within 10 epochs and get worse from there (two bottom out at epoch 1,
the 1e-4 run at epoch 10). This plots that, and separates the two things that
could explain the differing rates: learning rate, and whether any full
reconstruction steps were mixed in (`crossmodal_prob`).

Reads the SLURM logs directly rather than training_history.json, because the
1e-4 and 3e-5 runs were cancelled mid-flight and their histories were never
finalised.

    python figures/plot_specialist_runs.py            # -> figures_fixed/specialist_degradation.png
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('figures_fixed'); OUT.mkdir(exist_ok=True)

# label, log, lr, crossmodal_prob, global batch, colour
RUNS = [
    ('lr 1e-4, xm 1.0, batch 128', 'logs/r2ps500_12086208.out',   '#c2453a', '-'),
    ('lr 3e-5, xm 1.0, batch 128', 'logs/r2ps3e5_12087915.out',   '#d98c2b', '-'),
    ('lr 1.5e-5, xm 0.8, batch 64', 'logs/r2psmix4g_12099830.out', '#1baf7a', '-'),
]

EPOCH_RE = re.compile(r'^Epoch (\d+)/')
SRC_RE = re.compile(
    r'src=rgb\s+→\s+total\s+([\d.]+)\s+rgb\s+([\d.]+)\s+depth\s+([\d.]+)\s+'
    r'pc_chamfer\s+([\d.]+)\s+param_mae\(masked\)\s+([\d.]+)')
MEAN_RE = re.compile(r'MEAN generation metric over sources .*?: ([\d.]+)')


def parse(path):
    """-> dict of epoch-aligned lists. Validation lines follow their epoch header."""
    ep = None
    out = {k: [] for k in ('epoch', 'total', 'depth', 'pc', 'param', 'mean')}
    for line in Path(path).read_text(errors='replace').splitlines():
        m = EPOCH_RE.match(line)
        if m:
            ep = int(m.group(1)); continue
        m = SRC_RE.search(line)
        if m and ep is not None:
            out['epoch'].append(ep)
            out['total'].append(float(m.group(1)))
            out['depth'].append(float(m.group(3)))
            out['pc'].append(float(m.group(4)))
            out['param'].append(float(m.group(5)))
            continue
        m = MEAN_RE.search(line)
        if m and ep is not None:
            out['mean'].append(float(m.group(1)))
    # A cancelled job can die between the per-source line and the MEAN line.
    n = min(len(out['epoch']), len(out['mean']))
    return {k: v[:n] for k, v in out.items()}


PANELS = [
    ('mean',  'mean_gen  (pc + text)', 'the best-model metric'),
    ('pc',    'pc_chamfer',            'target 1 of 2'),
    ('param', 'param_mae (masked)',    'target 2 of 2'),
    ('depth', 'depth recon',           'NOT a target — free-rides on mixing'),
]

# True epoch-0 baseline: the v2 pretrain teacher evaluated on this exact task with
# NO distillation (eval/eval_warmstart.py, job 12086899, same val_max_batches=100 subset).
# Every run's first validation is already one epoch in, so without this line the
# figure cannot show how much distillation actually bought. Depth was not reported
# by that eval, hence the None.
BASELINE = {'mean': 0.03923, 'pc': 0.00104, 'param': 0.0337, 'depth': None}

fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
data = {label: parse(p) for label, p, _, _ in RUNS}

for ax, (key, title, sub) in zip(axes, PANELS):
    for (label, _, colour, ls) in RUNS:
        d = data[label]
        if not d['epoch']:
            continue
        ax.plot(d['epoch'], d[key], ls, color=colour, marker='o', ms=4,
                lw=1.8, label=label)
        # Mark each run's best (lowest) point.
        i = min(range(len(d[key])), key=lambda j: d[key][j])
        ax.scatter([d['epoch'][i]], [d[key][i]], s=90, facecolors='none',
                   edgecolors=colour, lw=1.8, zorder=5)
    b = BASELINE.get(key)
    if b is not None:
        ax.axhline(b, color='#555', ls='--', lw=1.2, zorder=1)
        ax.annotate('epoch 0: no distillation', xy=(0.98, b), xycoords=('axes fraction', 'data'),
                    ha='right', va='bottom', fontsize=7.5, color='#555')
    ax.set_title(f'{title}\n{sub}', fontsize=10)
    ax.set_xlabel('epoch')
    ax.grid(alpha=.25, lw=.6)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('validation')
axes[0].legend(fontsize=8, frameon=False)

fig.suptitle(
    'RGB → {point cloud, spline params} specialist distillation: the gain is real but saturates '
    'within ~10 epochs, then reverses\n'
    'Dashed line = the undistilled teacher. Rings mark each run\'s best epoch. Each lr drop bought '
    'an immediate step down, then degradation resumed.',
    fontsize=11, y=1.06)
fig.tight_layout()
fig.savefig(OUT / 'specialist_degradation.png', dpi=170, bbox_inches='tight')
print('wrote', OUT / 'specialist_degradation.png')

# ── Table to stdout, so the numbers are quotable without reopening the figure ──
for label, _, _, _ in RUNS:
    d = data[label]
    if not d['epoch']:
        print(f'{label}: no validation points parsed'); continue
    i = min(range(len(d['mean'])), key=lambda j: d['mean'][j])
    last = len(d['mean']) - 1
    print(f'\n{label}')
    print(f'  epochs validated : {d["epoch"][0]} .. {d["epoch"][last]}  ({len(d["mean"])} points)')
    print(f'  best mean_gen    : {d["mean"][i]:.5f} @ ep{d["epoch"][i]}'
          f'   ({(d["mean"][i]/BASELINE["mean"]-1)*100:+.1f}% vs undistilled teacher)')
    print(f'  final mean_gen   : {d["mean"][last]:.5f} @ ep{d["epoch"][last]}'
          f'   ({(d["mean"][last]/d["mean"][i]-1)*100:+.1f}% vs best)')
    print(f'  pc_chamfer       : {d["pc"][0]:.5f} -> {d["pc"][last]:.5f}'
          f'   ({(d["pc"][last]/d["pc"][0]-1)*100:+.1f}%)')
    print(f'  param_mae        : {d["param"][0]:.4f} -> {d["param"][last]:.4f}'
          f'   ({(d["param"][last]/d["param"][0]-1)*100:+.1f}%)')
    print(f'  depth            : {d["depth"][0]:.4f} -> {d["depth"][last]:.4f}'
          f'   ({(d["depth"][last]/d["depth"][0]-1)*100:+.1f}%)')
