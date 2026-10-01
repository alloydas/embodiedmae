"""Paired bootstrap of the probe R2 gap between two runs, from the probe caches.

    python eval/probe_pair_bootstrap.py --a e2_pcrgbd --b e8_supervised --feature cls
    python eval/probe_pair_bootstrap.py --a e2_pcrgbd e2_pcrgbdt --b e8_supervised \
        --feature cls mean --out reports/e8_supervised_delta.csv
    python eval/probe_pair_bootstrap.py --species maize --a maize_e2_pcrgbd --b maize_4m

--species picks the probe module (eval/linear_probe.py or eval/linear_probe_maize.py),
its cache directory and naming, and its data root.

Refits eval/linear_probe.py's ridge (train-only standardiser, alpha by 5-fold CV)
on each run's cached features in outputs/_probe_cache/, keeps the per-plant val and
test predictions, and resamples PLANTS: each draw scores R2(a) - R2(b) on the same
plants. Delta is a over b, with a 95 % percentile CI from --n-boot resamples. The two
runs' caches must list the same plants in the same order (asserted). No GPU; the
caches must already exist (eval/linear_probe.py writes them).
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import importlib.util
import numpy as np
import pandas as pd


def _import_file(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

SPECIES = {
    # probe module, cache dir, cache-name prefix, data root
    'sorghum': ('linear_probe.py', '_probe_cache', '',
                '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'),
    'maize':   ('linear_probe_maize.py', '_probe_cache_maize', 'maize_',
                '/work/mech-ai-scratch/alloy/Maize'),
}
LP = None      # set in main() from --species


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


def r2_rows(y, p, idx):
    yy = y[idx]
    return 1 - ((yy - p[idx]) ** 2).sum(1) / ((yy - yy.mean(1, keepdims=True)) ** 2).sum(1)


def load(args, run, feature, split):
    ck = Path(args.ckpt).stem
    f = Path(args.cache_dir) / (f'{args.prefix}{run}__{ck}__{split}__{feature}'
                                f'__seed{args.seed}__rep{args.repeats}.npz')
    z = np.load(f, allow_pickle=True)   # maize plant ids are strings ('plant_0000')
    return z['plants'], z['feats']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--species', choices=sorted(SPECIES), default='sorghum')
    ap.add_argument('--a', nargs='+', required=True, help='run(s) whose R2 is the minuend')
    ap.add_argument('--b', required=True, help='the run every --a is compared with')
    ap.add_argument('--feature', nargs='+', default=['cls'])
    ap.add_argument('--ckpt', default='checkpoints/checkpoint_epoch_600.pth')
    ap.add_argument('--data-root', default=None, help='default: the species\' data root')
    ap.add_argument('--cache-dir', default=None, help='default: the species\' probe cache')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--targets', nargs='+', default=None, help='probe target names (default: all)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    global LP
    mod, cache, args.prefix, root = SPECIES[args.species]
    LP = _import_file(mod[:-3], REPO / 'eval' / mod)
    args.cache_dir = args.cache_dir or str(REPO / 'outputs' / cache)
    args.data_root = args.data_root or root

    tgt = LP.load_targets(args.data_root)
    rows = []
    for feature in args.feature:
        preds = {}
        for run in [*args.a, args.b]:
            data = {s: load(args, run, feature, s) for s in ('train', 'val', 'test')}
            if run == args.b:
                ref_plants = {s: data[s][0] for s in data}
            preds[run] = (data, {})
        for run in args.a:
            for s in ('train', 'val', 'test'):
                assert np.array_equal(preds[run][0][s][0], ref_plants[s]), (run, feature, s)
        for name, src, prov, unit in LP.TARGETS:
            if args.targets and name not in args.targets:
                continue
            y = {s: tgt.reindex(ref_plants[s])[src].to_numpy(np.float64) for s in ref_plants}
            P = {}
            for run, (data, _) in preds.items():
                P[run] = fit_predict(data['train'][1], y['train'],
                                     {s: data[s][1] for s in ('val', 'test')}, args.alphas)
            for s in ('val', 'test'):
                n = len(y[s])
                idx = np.random.default_rng(12345).integers(0, n, size=(args.n_boot, n))
                rb = r2_rows(y[s], P[args.b][s], idx)
                r2b = r2_rows(y[s], P[args.b][s], np.arange(n)[None])[0]
                for run in args.a:
                    ra = r2_rows(y[s], P[run][s], idx)
                    r2a = r2_rows(y[s], P[run][s], np.arange(n)[None])[0]
                    d = ra - rb
                    rows.append({'a': run, 'b': args.b, 'feature': feature, 'target': name,
                                 'split': s, 'n': n, 'r2_a': r2a, 'r2_b': r2b, 'delta': r2a - r2b,
                                 'lo': np.percentile(d, 2.5), 'hi': np.percentile(d, 97.5)})
            print(f'  {feature:5s} {name:16s} done', flush=True)
    df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'📄 {args.out}')
    df['ci'] = df.apply(lambda r: f"{r.delta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1)
    pd.set_option('display.width', 250)
    print(df.pivot_table(index=['feature', 'a', 'target'], columns='split', values='ci',
                         aggfunc='first').to_string())


if __name__ == '__main__':
    main()
