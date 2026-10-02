#!/usr/bin/env python3
"""Per-plant leaf-width probe targets for Sorghum_15K -> data_split/leaf_width_targets.csv.

Since 2026-09-30 every *_spline.yml carries a per-leaf blade `width` (metres;
shorgum_data/add_leaf_width.py measured it on the mesh and wrote it into the
files). features.csv predates that and has no width column, so the probe's two
width targets come from here:

    width_mean, width_max  over the plant's leaves WITH procedural parameters,

which are the leaves features.csv's n_leaves / leaf_len_mean / leaf_len_max
count, and the leaves load_spline_params turns into tokens. The mesh can hold
more: plant 0 has 13 mesh leaves, of which 12 and 13 are geometry-only (no
length, no starting point), so it counts 11 here as in features.csv.

Checks, any failure is fatal: every plant present with status ok; n_leaves equal
to features.csv for all 15,000 plants; and for --check-plants plants the widths
parsed from the dataset's own view-00 spline file, through the probe's loader
rules, equal the table to 1e-6.

    python data_split/make_leaf_width_targets.py
"""
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from embodied_mae_4m import LEAF_FIELDS

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
WIDTHS = '/work/mech-ai-scratch/alloy/shorgum_data/leaf_widths.csv'
OUT = REPO / 'data_split' / 'leaf_width_targets.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--widths', default=WIDTHS)
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--check-plants', type=int, default=200)
    args = ap.parse_args()

    w = pd.read_csv(args.widths)
    if (w.status != 'ok').any():
        raise SystemExit(f'{(w.status != "ok").sum()} rows of {args.widths} are not ok')
    feats = pd.read_csv(Path(args.data_root) / 'features.csv').set_index('plant')
    assign = pd.read_csv(Path(args.data_root) / 'assignment.csv').set_index('plant')

    # leaves with procedural parameters: length_yml is empty for geometry-only leaves
    w = w[w.length_yml.notna()]
    t = w.groupby('plant').agg(n_leaves=('leaf', 'size'), width_mean=('width', 'mean'),
                               width_max=('width', 'max'))
    if len(t) != len(feats) or not t.index.equals(feats.index.sort_values()):
        raise SystemExit(f'plants: {len(t)} in widths vs {len(feats)} in features.csv')
    bad = t.n_leaves != feats.n_leaves.reindex(t.index)
    if bad.any():
        raise SystemExit(f'n_leaves differs from features.csv for {int(bad.sum())} plants, '
                         f'e.g. {list(t.index[bad][:5])}')

    # the dataset's own files, read with load_spline_params' leaf rule
    rng = np.random.default_rng(0)
    worst = 0.0
    for pid in rng.choice(t.index.to_numpy(), size=min(args.check_plants, len(t)), replace=False):
        yml = Path(args.data_root) / assign.loc[pid, 'split'] / f'Sorghum_{pid}_00' / f'Sorghum_{pid}_spline.yml'
        with open(yml) as f:
            leaves = yaml.load(f, Loader=yaml.CSafeLoader)['Sorghums'][0]['Leaves']
        ws = np.array([lf['width'] for lf in leaves if all(k in lf for k in LEAF_FIELDS)], float)
        if len(ws) != t.loc[pid, 'n_leaves']:
            raise SystemExit(f'plant {pid}: {len(ws)} parameterised leaves in {yml}, '
                             f'{t.loc[pid, "n_leaves"]} in the table')
        worst = max(worst, abs(ws.mean() - t.loc[pid, 'width_mean']),
                    abs(ws.max() - t.loc[pid, 'width_max']))
    if worst > 1e-6:
        raise SystemExit(f'spline-file widths differ from {args.widths} by {worst:.2e}')

    t = t.reset_index()
    t.to_csv(args.out, index=False, float_format='%.6f')
    print(f'wrote {args.out}: {len(t)} plants; width_mean {t.width_mean.mean():.4f} '
          f'+- {t.width_mean.std():.4f}, width_max {t.width_max.mean():.4f} '
          f'+- {t.width_max.std():.4f} m; {args.check_plants} spline files agree to {worst:.1e}')


if __name__ == '__main__':
    main()
