"""Paired row-vs-isolated comparison for Regime III (any species).

Reads two per-item CSVs written by eval/eval_row.py, one for the row fine-tune and one for the
isolated control, both on the same test split, and writes the paired deltas the paper's Tab. rows
and the ledger use:
  ft_row_vs_iso_isolated   row-trained on rows      minus isolated-trained on isolated inputs
  ft_row_vs_iso_row        row-trained on rows      minus isolated-trained on rows
  iso_row_vs_iso_isolated  isolated-trained on rows minus isolated-trained on isolated inputs
Each item is first averaged per plant (over yaws and shots), the delta is taken per plant, and
the 95 % interval is a bootstrap over plants (2,000 resamples, seed 0).

  python eval/row_pair_bootstrap.py --ft <ft.csv> --iso <iso.csv> --out reports/row_<species>_pair_ft<job>_iso<job>.csv
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import numpy as np
import pandas as pd

METRICS = ('chamfer_gt_m2', 'miss_5mm', 'plant_mae', 'leaf_mae')
PAIRS = (('ft_row_vs_iso_isolated', ('ft', 'row'), ('iso', 'isolated')),
         ('ft_row_vs_iso_row', ('ft', 'row'), ('iso', 'row')),
         ('iso_row_vs_iso_isolated', ('iso', 'row'), ('iso', 'isolated')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ft', required=True); ap.add_argument('--iso', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n-boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    per = {}
    for tag, f in (('ft', a.ft), ('iso', a.iso)):
        d = pd.read_csv(f)
        for (mode, src), g in d.groupby(['mode', 'source']):
            per[(tag, mode, src)] = g.groupby('plant')[[m for m in METRICS if m in g.columns]].mean()
    rng = np.random.default_rng(a.seed); rows = []
    sources = sorted({k[2] for k in per})
    for name, (ta, ma), (tb, mb) in PAIRS:
        for src in sources:
            A, B = per.get((ta, ma, src)), per.get((tb, mb, src))
            if A is None or B is None: continue
            plants = A.index.intersection(B.index)
            idx = rng.integers(0, len(plants), size=(a.n_boot, len(plants)))
            for m in METRICS:
                if m not in A.columns or m not in B.columns: continue
                dlt = (A.loc[plants, m] - B.loc[plants, m]).to_numpy()
                if np.isnan(dlt).all(): continue
                boots = np.nanmean(dlt[idx], axis=1)
                rows.append({'comparison': name, 'source': src, 'metric': m, 'delta': float(np.nanmean(dlt)),
                             'lo': float(np.percentile(boots, 2.5)), 'hi': float(np.percentile(boots, 97.5)), 'n_plants': len(plants)})
    pd.DataFrame(rows).to_csv(a.out, index=False); print('wrote', a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
