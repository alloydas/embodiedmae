"""Paired bootstrap CIs for the 1-view / 3-view probe drops (eval/pc_view_eval.py).

Refits the matched probe exactly as pc_view_eval.py does (ridge on the cached CLS
features, train-only standardiser, alpha by 5-fold CV), keeps the per-plant val and
test predictions, and resamples PLANTS: each draw scores R2(condition) - R2(full)
on the same plants, so the model, the plants and the probe protocol are held fixed
and only the input cloud differs. Reads outputs/_probe_cache_pcview/ only; no GPU.

    python eval/pc_view_bootstrap.py --runs e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt \
        4m_pretrain_15k_v2_depthfix_qal:checkpoints/checkpoint_epoch_1000.pth \
        --out reports/pcview_16676994_bootstrap.csv
"""
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import numpy as np
import pandas as pd

import importlib.util
def _import_file(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

PV = _import_file('pc_view_eval', REPO / 'eval' / 'pc_view_eval.py')
LP = PV.LP


def fit_predict(Xtr, ytr, evals, alphas):
    """LP.fit_probe's fit, returning the predictions instead of the metrics."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(Xtr)
    ym, ysd = float(ytr.mean()), float(ytr.std())
    ysd = ysd if ysd > 0 else 1.0
    gs = GridSearchCV(Ridge(), {'alpha': alphas}, cv=KFold(5, shuffle=True, random_state=0),
                      scoring='r2', n_jobs=-1)
    gs.fit(xs.transform(Xtr), (ytr - ym) / ysd)
    return {k: gs.best_estimator_.predict(xs.transform(X)) * ysd + ym for k, X in evals.items()}


def r2_rows(y, P, idx):
    """R2 of each prediction column over the plants in each row of idx: (B, k)."""
    yy = y[idx]                                               # (B, n)
    ss_tot = ((yy - yy.mean(1, keepdims=True)) ** 2).sum(1)
    return np.stack([1 - ((yy - p[idx]) ** 2).sum(1) / ss_tot for p in P], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--data-root', default=PV.PM.DATA_ROOT)
    ap.add_argument('--cache-dir', default=str(REPO / 'outputs' / '_probe_cache_pcview'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--targets', nargs='+', default=None, help='probe target names (default: all)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.limit = None                                         # cache_path's key

    tgt = LP.load_targets(args.data_root)
    conds = list(PV.CONDITIONS)                               # full, 3view, 1view
    rng = np.random.default_rng(12345)
    rows = []
    for spec in args.runs:
        run, ckpt = PV.parse_run(spec)
        z = {s: np.load(PV.cache_path(args, run, ckpt, s)) for s in ('train', 'val', 'test')}
        y = {s: tgt.reindex(z[s]['plants']) for s in z}
        for name, src, prov, unit in LP.TARGETS:
            if args.targets and name not in args.targets:
                continue
            ytr = y['train'][src].to_numpy(np.float64)
            preds = {}
            for c in conds:
                p = fit_predict(z['train'][f'feats_{c}'], ytr,
                                {s: z[s][f'feats_{c}'] for s in ('val', 'test')}, args.alphas)
                for s in ('val', 'test'):
                    preds[(s, c)] = p[s]
            for s in ('val', 'test'):
                ys = y[s][src].to_numpy(np.float64)
                n = len(ys)
                idx = rng.integers(0, n, size=(args.n_boot, n))
                P = [preds[(s, c)] for c in conds]
                point = r2_rows(ys, P, np.arange(n)[None])[0]
                boot = r2_rows(ys, P, idx)
                for j, c in enumerate(conds):
                    d = boot[:, j] - boot[:, 0]
                    rows.append({'run': run, 'ckpt': ckpt, 'target': name, 'split': s,
                                 'condition': c, 'n': n, 'r2': point[j],
                                 'delta_vs_full': point[j] - point[0],
                                 'delta_lo': np.percentile(d, 2.5) if j else 0.0,
                                 'delta_hi': np.percentile(d, 97.5) if j else 0.0})
            print(f'  {run:34s} {name:16s} done', flush=True)
    df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'📄 {args.out}')
    show = df[df.condition != 'full'].copy()
    show['ci'] = show.apply(lambda r: f"{r.delta_vs_full:+.3f} [{r.delta_lo:+.3f},{r.delta_hi:+.3f}]", axis=1)
    pd.set_option('display.width', 250)
    print(show.pivot_table(index=['run', 'condition'], columns=['split', 'target'], values='ci',
                           aggfunc='first').to_string())


if __name__ == '__main__':
    main()
