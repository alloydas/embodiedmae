#!/usr/bin/env python3
"""Render reports/pc_fscore_maize.csv (eval/pc_fscore_maize.py) as a Markdown report.

Re-run after every new run is scored; it rewrites the whole report from the CSV.

    python eval/pc_fscore_report.py                       # -> reports/pc_fscore_maize.md
    python eval/pc_fscore_report.py --tau 0.02
"""

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Display order and what each arm is. Unknown runs are appended alphabetically.
ARMS = [
    ('maize_e2_pc_d2',     'E2', 'PC only'),
    ('maize_e2_pcrgb_d2',  'E2', 'PC + RGB'),
    ('maize_e2_pcrgbd_d2', 'E2', 'PC + RGB + depth'),
    ('maize_4m_d2',        'E2 / E3 / E4 reference', 'all four, 10,500 plants'),
    ('maize_e3_1k_d2',     'E3', '1,000 train plants'),
    ('maize_e3_3k_d2',     'E3', '3,000 train plants'),
    ('maize_e3_10k_d2',    'E3', '10,000 train plants'),
]


def planned_epochs(run):
    try:
        return int(json.loads((REPO / 'outputs' / run / 'config.json').read_text())['epochs'])
    except Exception:
        return None


def fmt(v, nd=3):
    return '—' if pd.isna(v) else f'{v:.{nd}f}'


def table(df, split, tau):
    d = df[(df.split == split) & (df.tau == tau)].set_index('run')
    order = [r for r, _, _ in ARMS if r in d.index] + sorted(set(d.index) - {r for r, _, _ in ARMS})
    lines = ['| run | arm | epoch | plants | precision | recall | **F1** | F1 median | F1 p10 | chamfer |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in order:
        x = d.loc[r]
        label = next((f'{e}: {w}' for rr, e, w in ARMS if rr == r), '')
        ep = planned_epochs(r)
        epoch = f'{int(x.epoch)}' + (f' / {ep}' if ep else '')
        lines.append(f'| `{r}` | {label} | {epoch} | {int(x.n_plants)} | {fmt(x.precision)} | '
                     f'{fmt(x.recall)} | **{fmt(x.f1)}** | {fmt(x.f1_median)} | {fmt(x.f1_p10)} | '
                     f'{x.chamfer:.5f} |')
    return '\n'.join(lines)


def tau_sweep(df, split):
    d = df[df.split == split]
    taus = sorted(d.tau.unique())
    runs = [r for r, _, _ in ARMS if r in set(d.run)] + sorted(set(d.run) - {r for r, _, _ in ARMS})
    lines = ['| run | ' + ' | '.join(f'F1 @ {t:g}' for t in taus) + ' |',
             '|---|' + '---:|' * len(taus)]
    for r in runs:
        x = d[d.run == r].set_index('tau')
        lines.append(f'| `{r}` | ' + ' | '.join(fmt(x.f1.get(t, float("nan"))) for t in taus) + ' |')
    return '\n'.join(lines)


def anchor(df):
    d = df[(df.split == 'val') & (df.tau == df.tau.min())].set_index('run')
    lines = ['| run | chamfer here (val) | run\'s own last val chamfer | ratio |', '|---|---:|---:|---:|']
    for r in [r for r, _, _ in ARMS if r in d.index] + sorted(set(d.index) - {r for r, _, _ in ARMS}):
        x = d.loc[r]
        ratio = x.chamfer / x.published_val_chamfer if x.published_val_chamfer else float('nan')
        lines.append(f'| `{r}` | {x.chamfer:.5f} | {x.published_val_chamfer:.5f} | {fmt(ratio, 2)} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(REPO / 'reports' / 'pc_fscore_maize.csv'))
    ap.add_argument('--out', default=str(REPO / 'reports' / 'pc_fscore_maize.md'))
    ap.add_argument('--tau', type=float, default=0.01, help='headline threshold')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.tau not in set(df.tau.round(6)):
        raise SystemExit(f'tau {args.tau} not in CSV ({sorted(df.tau.unique())})')
    spacing = df.gt_nn_spacing_median.median()
    scored = sorted(df.run.unique())
    pending = [r for r, _, _ in ARMS if r not in scored]

    parts = [
        f'# Maize data v2 — point-cloud precision / recall / F1',
        f'_Generated {datetime.now():%Y-%m-%d %H:%M} by `eval/pc_fscore_report.py` from `{Path(args.csv).name}`._',
        '',
        '## How to read this',
        '',
        f'- Each run reconstructs the full 8,192-point cloud of every plant (one view per plant, view 00) '
        f'in **its own training regime**: Dirichlet token masking at the run\'s mask ratio (0.80). '
        f'All-visible input makes the decoder collapse, so it is not used.',
        f'- Clouds are in the loader\'s frame: centred, scaled to the unit sphere. A threshold τ is in those units; '
        f'the GT clouds\' own median nearest-neighbour spacing is **{spacing:.4f}**, so τ below that mostly '
        f'measures sampling jitter, not shape.',
        '- **Precision** = share of predicted points within τ of the GT surface (no spurious points). '
        '**Recall** = share of GT points within τ of a predicted point (nothing missed). '
        '**F1** = their harmonic mean. All three are computed per plant and then averaged; '
        'F1 median and 10th percentile show the spread across plants.',
        '- `chamfer` is the trainer\'s metric (mean squared NN distance, both directions summed), '
        'reported so the harness can be checked against each run\'s own validation number (next section).',
        f'- Headline τ = **{args.tau:g}**. The sweep over τ is further down.',
        '',
        '## Harness check — chamfer here vs the run\'s own last val chamfer',
        '',
        'A ratio near 1 means this script reproduces the training-time validation pass. '
        'Differences come from the masking draw and from scoring the final checkpoint rather than the last '
        'validated epoch.',
        '',
        anchor(df),
        '',
    ]
    for split in ['val', 'test']:
        if split not in set(df.split):
            continue
        parts += [f'## {split.upper()} — τ = {args.tau:g}', '', table(df, split, args.tau), '',
                  f'### {split.upper()} — F1 across thresholds', '', tau_sweep(df, split), '']
    parts += ['## Status', '',
              f'- Scored: {", ".join(f"`{r}`" for r in scored)}',
              f'- Not yet scored (still training or not run): '
              + (', '.join(f'`{r}`' for r in pending) if pending else 'none'),
              '',
              'To add a run once it finishes: '
              '`sbatch --account=bepk-delta-gpu --partition=gpuA40x4 slurm/delta/pc_fscore_maize.sbatch <run> [run...]`, '
              'then `python eval/pc_fscore_report.py`.',
              '']
    Path(args.out).write_text('\n'.join(parts))
    print(f'📄 {args.out}')


if __name__ == '__main__':
    main()
