#!/usr/bin/env python3
"""E9 latent analysis for MAIZE — clustering and t-SNE over the frozen latent.

The maize twin of `eval/latent_analysis.py`. Separate file: different target
table, different plant-id parse, different model width.

THE GENOTYPE PROBLEM IS THE SAME, AND SO IS THE FRAMING.
Every maize plant comes from one generator; `<plant phenotypeId>` is the literal
constant "BTx99" for all 15,000, and `<Tassel>@seed` is just the plant index. So
there is no genotype variable here either, and the plan's "cluster by genotype"
again has nothing to condition on. The answerable question stays:

    does the latent recover the generative factors, and is it continuous?

A near-zero silhouette is therefore a POSITIVE result — there are no genuine
clusters in a continuous generator — and that has to be said before the number
is read, not after.

WHAT IS DIFFERENT, AND BETTER, THAN SORGHUM.
Sorghum's factor table has rank ~2: height and leaf count are one variable
(r = 0.994) and the biomass proxy is that variable again (r = 0.996), so
colouring six panels by six columns showed one gradient six times. Maize is
genuinely multi-factor — height vs leaf count r = 0.199, leaf angle |r| < 0.034
with everything, effective rank 7.7 of 12 candidates. The panels below therefore
span real, distinct axes instead of restating one.

Usage
-----
    python eval/latent_analysis_maize.py --runs maize_4m --split val --device cpu

Reads the same cache `eval/linear_probe_maize.py` writes, so once the probe has
run this costs no GPU.
"""

import sys as _sys
import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))
_sys.path.insert(0, str(_HERE.parent))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import linear_probe_maize as LPM


# ── palette ──────────────────────────────────────────────────────────────────
# One hue per panel, strictly light->dark within a panel. A multi-hue ramp on a
# continuous factor reads as categorical, which is the very thing being measured.
def _ramp(anchor, name):
    import colorsys
    from matplotlib.colors import to_rgb
    h, _l, sat = colorsys.rgb_to_hls(*to_rgb(anchor))
    stops = []
    for t in np.linspace(0.0, 1.0, 9):
        stops.append(colorsys.hls_to_rgb(h, 0.93 - 0.74 * t,
                                         min(1.0, sat * (0.50 + 0.62 * t))))
    return LinearSegmentedColormap.from_list(name, stops)


SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', [
    '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
SEQ_ORANGE = _ramp('#eb6834', 'seq_orange')
SEQ_AQUA = _ramp('#1baf7a', 'seq_aqua')
SEQ_VIOLET = _ramp('#4a3aa7', 'seq_violet')
SEQ_MAGENTA = _ramp('#e87ba4', 'seq_magenta')
INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8984'

# Six genuinely distinct axes — possible here, impossible in sorghum.
PANELS = [
    ('leaf_count',        'leaf count',                  SEQ_BLUE),
    ('stem_internodeSum', 'height — internode sum',      SEQ_ORANGE),
    ('leaf_angleMean',    'leaf angle (deg)',            SEQ_AQUA),
    ('leaf_areaProxy',    'biomass — leaf-area proxy',   SEQ_VIOLET),
    ('leaf_twistAbsMean', 'leaf twist |mean| (deg)',     SEQ_MAGENTA),
]
PANEL_CLUSTER = ('_cluster', 'k-means partition (k=6)', None)


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#dcdbd6')
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_3, labelsize=7, length=3, width=0.8)
    ax.set_facecolor('#fcfcfb')


# ── analysis ─────────────────────────────────────────────────────────────────

def structure(X):
    from sklearn.decomposition import PCA
    p = PCA().fit(X - X.mean(0, keepdims=True))
    ev, lam = p.explained_variance_ratio_, p.explained_variance_
    return {
        'dim': int(X.shape[1]),
        'pc1_var': float(ev[0]),
        'pc1_3_var': float(ev[:3].sum()),
        'n_pcs_90': int(np.searchsorted(np.cumsum(ev), 0.90) + 1),
        'participation_ratio': float(lam.sum() ** 2 / (lam ** 2).sum()),
    }


def factor_alignment(X, tgt, cols, n_pc=5):
    from sklearn.decomposition import PCA
    Z = PCA(n_components=n_pc).fit_transform(X - X.mean(0, keepdims=True))
    rows = []
    for c in cols:
        y = tgt[c].to_numpy(dtype=np.float64)
        rs = [abs(float(np.corrcoef(Z[:, k], y)[0, 1])) for k in range(n_pc)]
        rows.append({'factor': c, 'best_pc': int(np.argmax(rs)) + 1,
                     'best_abs_r': max(rs)})
    return pd.DataFrame(rows)


def cluster_tendency(X, tgt, ks=(2, 3, 4, 6, 8, 10, 12), seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    size_bin = pd.qcut(tgt['leaf_count'].rank(method='first'), 10, labels=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({'k': k,
                     'silhouette': float(silhouette_score(
                         X, km.labels_, sample_size=min(2000, len(X)),
                         random_state=seed)),
                     'ari_vs_leafcount_decile': float(
                         adjusted_rand_score(size_bin, km.labels_))})
    return pd.DataFrame(rows)


def neighbourhood(X, tgt, cols, emb2d=None, seed=0, n=1500):
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X[idx]
    S = StandardScaler().fit_transform(tgt.iloc[idx][cols].to_numpy(float))
    rho, _ = spearmanr(pdist(Xs), pdist(S))
    out = {'n_sampled': int(len(idx)), 'spearman_latent_vs_shape_dist': float(rho)}
    if emb2d is not None:
        from sklearn.manifold import trustworthiness
        out['tsne_trustworthiness_k10'] = float(
            trustworthiness(Xs, emb2d[idx], n_neighbors=10))
    return out


def local_smoothness(emb, tgt, cols, seed=0, k=15):
    """How much each factor varies between t-SNE neighbours vs random pairs.

    1.0 = no local structure, 0 = perfectly smooth. Separates "the model did not
    learn it" from "t-SNE preserves neighbourhoods, not linear axes" — a factor
    can be far more linearly decodable than it is locally clustered.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    rng = np.random.default_rng(seed)
    out = {}
    for c in cols:
        v = tgt[c].to_numpy(float)
        v = (v - v.mean()) / (v.std() or 1.0)
        local = np.mean((v[:, None] - v[idx[:, 1:]]) ** 2)
        rand = np.mean((v[:, None] - v[rng.integers(0, len(v), (len(v), k))]) ** 2)
        out[c] = float(local / rand)
    return out


def embed(X, seed=0, perplexity=30):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X50 = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    out = {'tsne': TSNE(n_components=2, perplexity=perplexity, init='pca',
                        random_state=seed).fit_transform(X50)}
    try:
        import umap
        out['umap'] = umap.UMAP(n_components=2, random_state=seed).fit_transform(X50)
    except ImportError:
        print('  (umap-learn not installed — t-SNE only; not installing it, '
              'these conda envs are shared with running jobs)')
    return out


def figure(emb, tgt, clusters, run, split, method, path):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), facecolor='#fcfcfb')
    fig.suptitle(f'E9 · maize latent structure — {run} · {split} · {method.upper()}',
                 fontsize=13, color=INK, y=0.985)
    for ax, (col, title, cmap) in zip(axes.ravel(), PANELS + [PANEL_CLUSTER]):
        style(ax)
        ax.set_title(title, fontsize=9.5, color=INK, pad=7, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
        if col == '_cluster':
            k = int(clusters.max()) + 1
            for c in range(k):
                m = clusters == c
                ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=.85, linewidths=0,
                           c=[plt.cm.Greys(0.30 + 0.55 * c / max(k - 1, 1))])
            ax.text(.02, .02, f'k={k}', transform=ax.transAxes,
                    fontsize=7.5, color=INK_3)
        else:
            v = tgt[col].to_numpy(dtype=np.float64)
            lo, hi = np.percentile(v, [2, 98])
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=5, c=np.clip(v, lo, hi),
                            cmap=cmap, alpha=.9, linewidths=0)
            cb = fig.colorbar(sc, ax=ax, fraction=.045, pad=.02)
            cb.ax.tick_params(labelsize=7, colors=INK_3, length=2)
            cb.outline.set_visible(False)
    fig.text(0.5, 0.012,
             'No genotype label exists in this dataset either — one generator, '
             'phenotypeId is the constant "BTx99". A continuous latent is the '
             'expected result, not a failed clustering.',
             ha='center', fontsize=8, color=INK_3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(path, dpi=170, facecolor='#fcfcfb')
    plt.close(fig)
    print(f'  🖼  {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    ap.add_argument('--ckpt', default='best_model.pth')
    ap.add_argument('--data-root', default=LPM.DATA_ROOT)
    ap.add_argument('--feature', default='cls', choices=('cls', 'mean', 'cls+mean'))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--cache-dir',
                    default=str(LPM.REPO / 'outputs' / '_probe_cache_maize'))
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--out-dir', default=str(LPM.REPO / 'reports'))
    ap.add_argument('--fig-dir', default=str(LPM.REPO / 'figures'))
    args = ap.parse_args()
    args.split_set = [args.split]

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('⚠️  no CUDA — fine if the cache is warm, fatal if it is cold')
        args.device = 'cpu'

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tgt_all = LPM.load_targets(args.data_root)
    FACTORS = ['leaf_count', 'stem_internodeSum', 'leaf_angleMean',
               'leaf_areaProxy', 'leaf_twistAbsMean']
    print(__doc__.split('Usage')[0])

    summary = []
    for run in args.runs:
        run_dir = LPM.REPO / 'outputs' / run if not Path(run).exists() else Path(run)
        slug = Path(run_dir).name
        print(f'\n=== {slug} · {args.split} ===')

        cache = Path(args.cache_dir) / (
            f'maize_{slug}__{Path(args.ckpt).stem}__{args.split}__{args.feature}'
            f'__seed{args.seed}__rep{args.repeats}.npz')
        if cache.exists() and not args.refresh:
            z = np.load(cache, allow_pickle=True)
            plants, X = z['plants'], z['feats']
            print(f'  ⚡ cache hit {cache.name} {X.shape}')
        else:
            model, cfg, _ = LPM.build_model(run_dir, args.ckpt, args.device)
            plants, X = LPM.cached_features(model, cfg, args, slug, args.split)
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()

        tgt = tgt_all.reindex(plants)
        X = X.astype(np.float64)

        st = structure(X)
        print(f'  structure: dim {st["dim"]} · PC1 {st["pc1_var"]:.1%} · '
              f'90% at {st["n_pcs_90"]} PCs · '
              f'participation ratio {st["participation_ratio"]:.1f}')

        fa = factor_alignment(X, tgt, FACTORS)
        print('  factor alignment:')
        for r in fa.itertuples(index=False):
            print(f'    {r.factor:20s} PC{r.best_pc}  |r| {r.best_abs_r:.3f}')

        ct = cluster_tendency(X, tgt)
        best = ct.loc[ct['silhouette'].idxmax()]
        print(f'  cluster tendency: best silhouette {best["silhouette"]:.3f} at '
              f'k={int(best["k"])} (ARI {best["ari_vs_leafcount_decile"]:.3f})')
        print('    -> low silhouette means no real clusters, which is the '
              'expected result for a continuous generator.')

        embs = embed(X, seed=args.seed)
        nb = neighbourhood(X, tgt, FACTORS, emb2d=embs.get('tsne'), seed=args.seed)
        print(f'  neighbourhood: spearman(latent, factor dist) = '
              f'{nb["spearman_latent_vs_shape_dist"]:+.3f}')

        sm = local_smoothness(embs['tsne'], tgt, FACTORS, seed=args.seed)
        print('  local smoothness in the embedding (1.0 = none, 0 = perfect):')
        for c, v in sm.items():
            print(f'    {c:20s} {v:.3f}')

        from sklearn.cluster import KMeans
        clusters = KMeans(n_clusters=6, n_init=10,
                          random_state=args.seed).fit_predict(X)
        for method, E in embs.items():
            figure(E, tgt, clusters, slug, args.split, method,
                   fig_dir / f'e9_maize_{slug}_{args.split}_{method}.png')

        ct.to_csv(out_dir / f'e9_maize_cluster_tendency_{slug}_{args.split}.csv',
                  index=False)
        fa.to_csv(out_dir / f'e9_maize_factor_alignment_{slug}_{args.split}.csv',
                  index=False)
        summary.append({'run': slug, 'split': args.split, 'n_plants': len(plants),
                        **st, **nb,
                        'best_silhouette': float(best['silhouette']),
                        **{f'align_{r.factor}': float(r.best_abs_r)
                           for r in fa.itertuples(index=False)},
                        **{f'smooth_{k}': v for k, v in sm.items()}})

    df = pd.DataFrame(summary)
    p = out_dir / f'e9_maize_latent_summary_{args.split}.csv'
    df.to_csv(p, index=False)
    print(f'\n📄 {p}')
    cols = ['run', 'participation_ratio', 'pc1_var', 'best_silhouette',
            'spearman_latent_vs_shape_dist']
    print('\n' + df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == '__main__':
    main()
