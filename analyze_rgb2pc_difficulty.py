"""What actually predicts per-plant RGB->PC Chamfer error?

Tests the "hard cases are foreshortened / near-axial views" hypothesis that came
out of eyeballing gen_rgb2pc_gallery_pct.py, whose worst rows all looked top-down.

The test set is a gift for this question.  Every plant folder Sorghum_<id>_<vv>
holds the SAME complete 3-D plant cloud (Sorghum_<id>_nc.ply is byte-identical
across all ten views); _nc_cam.ply is only that cloud rotated into the view's
camera frame, and Chamfer after the dataset's centre + unit-scale is rotation
invariant.  So the view index changes the INPUT IMAGE and the frame the answer
must be expressed in, and nothing else about the target's difficulty.  And
camera_pose.json shows the view index is not an arbitrary label: it is camera
elevation, exactly, sin(elev) = -0.9 + 0.2 * view_index with zero spread.

Candidate predictors, all correlated against reports/quant_smr50_seed1_test.json:
  * |camera elevation|                       -- the foreshortening index
  * camera-frame cloud orientation / extent   -- re-measures the same thing from
    the data (the cloud is in the camera frame, so its orientation encodes the
    viewing direction; this is NOT independent evidence, it is a check that the
    elevation number really is plant-axis vs view-axis alignment)
  * rotation-invariant cloud shape            -- genuinely independent: how
    slender or bushy the plant is, regardless of where the camera sits
  * plant complexity from features.csv        -- n_leaves, stem length, etc.

Usage:
    python analyze_rgb2pc_difficulty.py            # uses/creates the feature cache
    python analyze_rgb2pc_difficulty.py --rebuild  # re-reads the 2250 .ply files
"""
import argparse, csv, json, pathlib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')

# dataviz reference palette (light surface #fcfcfb)
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']  # ordinal, steps 250..650
BLUE, ORANGE, AQUA, DIM = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2'


def build_cache(names, split, out):
    """Per-plant point-cloud shape statistics from the raw .ply (deterministic --
    the full cloud, not the dataset's random 8196-point subsample)."""
    import open3d as o3d
    n = len(names)
    cam = np.zeros((n, 6))
    shp = np.zeros((n, 9))
    for i, nm in enumerate(names):
        d = DATA / split / nm
        c = json.loads((d / 'camera_pose.json').read_text())
        M = np.array(c['cameraToWorld']).reshape(4, 4)
        cam[i] = [*np.array(c['position']), *(-M[:3, 2])]      # position, forward (OpenGL -z)
        pts = np.asarray(o3d.io.read_point_cloud(str(next(d.glob('*_nc_cam.ply')))).points)
        p = pts - pts.mean(0)
        p = p / np.linalg.norm(p, axis=1).max()                # the dataset's normalisation
        ev, evec = np.linalg.eigh(np.cov(p.T))
        o = np.argsort(ev)[::-1]
        ev, evec = ev[o], evec[:, o]
        ext = p.max(0) - p.min(0)                              # camera frame: x right, y up, z depth
        shp[i] = [len(pts), *np.sqrt(np.maximum(ev, 0)), abs(evec[2, 0]),
                  *ext, ext[2] / max(ext[0], ext[1])]
        if i % 250 == 0:
            print(f'  ply {i}/{n}', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), cam=cam, shp=shp)
    return cam, shp


def load_predictors(rep, cache, rebuild):
    names, split = rep['plants'], rep['split']
    if rebuild or not cache.exists():
        cam, shp = build_cache(names, split, cache)
    else:
        z = np.load(cache, allow_pickle=True)
        assert list(z['names']) == names, 'cache does not match this report'
        cam, shp = z['cam'], z['shp']

    vi = np.array([int(x.rsplit('_', 1)[1]) for x in names])
    sin_e = -0.9 + 0.2 * vi                       # exact, verified against camera_pose.json
    elev = np.degrees(np.arcsin(sin_e))           # + = camera below, looking up

    tbl = {}
    for r in csv.DictReader(open(DATA / 'features.csv')):
        tbl[r['plant']] = r
    pid = [x.split('_')[1] for x in names]
    fc = lambda k: np.array([float(tbl[p][k]) for p in pid])

    P = {
        # --- camera geometry ---
        '|camera elevation|':        (np.abs(elev), 'cam'),
        'cloud axis vs view axis':   (shp[:, 4], 'cam'),        # |cos(PC1, camera z)|
        'cloud depth/image extent':  (shp[:, 8], 'cam'),
        # --- rotation-invariant cloud shape (independent of where the camera is) ---
        'cloud flatness l3/l1':      (shp[:, 3] / shp[:, 1], 'shape'),
        'cloud elongation l2/l1':    (shp[:, 2] / shp[:, 1], 'shape'),
        'raw cloud point count':     (shp[:, 0], 'shape'),
        # --- plant complexity ---
        'leaf count':                (fc('n_leaves'), 'plant'),
        'stem length':               (fc('stem_length'), 'plant'),
        'mean leaf length':          (fc('leaf_len_mean'), 'plant'),
        'mean branching angle':      (fc('branch_mean'), 'plant'),
        'leaf roll spread':          (fc('roll_std'), 'plant'),
        # --- data-side control: how far apart two GT resamples of this plant are ---
        'GT resample floor':         (np.array(rep['per_sample']['gt_vs_gt']), 'ctrl'),
    }
    return vi, elev, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default=str(REPO / 'reports/quant_smr50_seed1_test.json'))
    ap.add_argument('--cache', default=str(REPO / 'vis_gallery/.difficulty_cache.npz'))
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--out', default=str(REPO / 'vis_gallery/difficulty_analysis.png'))
    a = ap.parse_args()

    rep = json.loads(pathlib.Path(a.report).read_text())
    err = np.array(rep['per_sample']['model'])
    vi, elev, P = load_predictors(rep, pathlib.Path(a.cache), a.rebuild)
    n = len(err)
    e4 = err * 1e4

    # ---------------- statistics ----------------
    groups = [err[vi == k] for k in range(10)]
    gm = err.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    sst = ((err - gm) ** 2).sum()
    eta2 = ssb / sst
    F, pF = stats.f_oneway(*groups)

    def r2(cols):
        X = np.column_stack([np.ones(n)] + cols)
        b, *_ = np.linalg.lstsq(X, err, rcond=None)
        return 1 - ((err - X @ b) ** 2).sum() / sst

    ae = np.abs(elev)
    r2_lin, r2_quad = r2([ae]), r2([ae, ae ** 2])

    print(f'n={n}  split={rep["split"]}  epoch={rep["epoch"]}  source={rep["source"]}')
    print(f'view-index ANOVA F={F:.1f} p={pF:.2e}  eta^2={eta2:.4f}')
    print(f'R2 from |elevation|: linear {r2_lin:.4f}  quadratic {r2_quad:.4f} '
          f'({r2_quad/eta2:.0%} of the whole 10-level view effect)')
    print(f'signed elevation: spearman rho={stats.spearmanr(elev, err)[0]:+.3f} '
          f'p={stats.spearmanr(elev, err)[1]:.3f}   (no up/down asymmetry)')

    rows = []
    for k, (x, fam) in P.items():
        rows.append((k, fam, stats.spearmanr(x, err)[0], stats.pearsonr(x, err)[0]))
    rows.sort(key=lambda r: -abs(r[2]))
    print(f'\n{"predictor":26s} {"family":6s} {"spearman":>9s} {"pearson":>8s} {"R2":>7s}')
    for k, fam, sp, pr in rows:
        print(f'{k:26s} {fam:6s} {sp:+9.3f} {pr:+8.3f} {pr**2:7.4f}')

    # matched |elevation| pairs: camera above vs camera below
    print(f'\n{"|elev|":>7s} {"above":>9s} {"below":>9s} {"diff":>8s} {"Welch p":>8s}')
    pair = []
    for k in range(5):
        dn, up = err[vi == k] * 1e4, err[vi == 9 - k] * 1e4
        p = stats.ttest_ind(dn, up, equal_var=False)[1]
        pair.append((abs(elev[vi == k][0]), dn, up, p))
        print(f'{abs(elev[vi==k][0]):7.1f} {dn.mean():9.2f} {up.mean():9.2f} '
              f'{dn.mean()-up.mean():+8.2f} {p:8.3f}')

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.6, 5.35), dpi=190, facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.60, 1.0, 1.22],
                          left=0.046, right=0.988, top=0.745, bottom=0.195, wspace=0.345)

    def frame(ax, title, sub):
        ax.set_facecolor(SURF)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8.2, length=3, width=0.8)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(INK2)
        ax.set_title(title, color=INK, fontsize=10.6, fontweight='bold', loc='left', pad=20)
        ax.text(0, 1.022, sub, transform=ax.transAxes, color=INK2, fontsize=8.3, va='bottom')

    # --- A: error distribution vs camera elevation -----------------------------
    ax = fig.add_subplot(gs[0, 0])
    frame(ax, 'Error is set by how steeply the camera looks',
          'box = IQR, whisker = p10-p90, dot = mean')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    xs = np.array([np.degrees(np.arcsin(-0.9 + 0.2 * k)) for k in range(10)])
    lev = {5.7: 0, 17.5: 1, 30.0: 2, 44.4: 3, 64.2: 4}
    means = []
    for k in range(10):
        g = e4[vi == k]
        x, c = xs[k], BLUE_RAMP[lev[round(abs(xs[k]), 1)]]
        q1, q3 = np.percentile(g, [25, 75])
        lo, hi = np.percentile(g, [10, 90])
        ax.plot([x, x], [lo, hi], color=c, lw=1.6, solid_capstyle='butt', zorder=2)
        ax.add_patch(Rectangle((x - 3.3, q1), 6.6, q3 - q1, facecolor=c, edgecolor=SURF,
                               lw=1.2, zorder=3))
        ax.plot([x - 3.3, x + 3.3], [np.median(g)] * 2, color=SURF, lw=1.6, zorder=4)
        ax.plot(x, g.mean(), 'o', ms=4.4, mfc=SURF, mec=c, mew=1.5, zorder=5)
        means.append(g.mean())
    ax.plot(xs, means, color=BLUE_RAMP[3], lw=1.3, alpha=0.45, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{abs(v):.0f}' for v in xs], fontsize=8)
    ax.set_xlim(-76, 76)
    ax.set_ylim(2.0, 20.6)
    ax.set_ylabel(r'symmetric Chamfer  ($\times10^{-4}$)', color=INK2, fontsize=8.8)
    ax.set_xlabel('camera elevation below / above the horizon  (degrees)',
                  color=INK2, fontsize=8.8, labelpad=1)
    for x0, lab in ((-40, 'camera above, looking down'), (40, 'camera below, looking up')):
        ax.text(x0, 19.4, lab, ha='center', color=MUTED, fontsize=8, style='italic')
    ax.annotate('', xy=(-74, 18.35), xytext=(-6, 18.35),
                arrowprops=dict(arrowstyle='-|>', color=AXIS, lw=0.9))
    ax.annotate('', xy=(74, 18.35), xytext=(6, 18.35),
                arrowprops=dict(arrowstyle='-|>', color=AXIS, lw=0.9))
    ax.text(0, -0.20, f'View index explains $\\eta^2$ = {eta2:.2f} of per-plant error, and a smooth '
                      f'curve in |elevation| alone recovers {r2_quad:.2f} of that {eta2:.2f}.',
            transform=ax.transAxes, ha='left', va='top', color=INK2, fontsize=8.4)
    for k in (0, 9):
        ax.text(xs[k], np.percentile(e4[vi == k], 90) + 0.5, f'{means[k]:.1f}', ha='center',
                color=BLUE_RAMP[4], fontsize=8.8, fontweight='bold')
    imin = int(np.argmin(means))
    ax.text(xs[imin], np.percentile(e4[vi == imin], 10) - 1.35, f'{means[imin]:.1f}',
            ha='center', color=BLUE_RAMP[2], fontsize=8.8, fontweight='bold')

    # --- B: up/down symmetry ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    frame(ax, 'Not top-down - just off-axis',
          'mean $\\pm$ 95% CI at matched |elevation|')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    xa = np.array([p[0] for p in pair])
    for j, (col, lab, idx) in enumerate(((ORANGE, 'camera above', 1), (AQUA, 'camera below', 2))):
        m = np.array([p[idx].mean() for p in pair])
        ci = np.array([1.96 * p[idx].std(ddof=1) / np.sqrt(len(p[idx])) for p in pair])
        ax.errorbar(xa + (j - .5) * 1.9, m, yerr=ci, color=col, lw=2.0, marker='o', ms=6.5,
                    mfc=SURF, mew=1.8, mec=col, capsize=3, elinewidth=1.4, label=lab, zorder=3)
    for j, idx in ((0, 1), (1, 2)):
        col = (ORANGE, AQUA)[j]
        v = pair[-1][idx].mean()
        ax.text(xa[-1] + (j - .5) * 1.9, v + (1.05 if j == 0 else -1.35), f'{v:.1f}',
                ha='center', color=col, fontsize=8.4, fontweight='bold')
    ax.set_xticks(xa)
    ax.set_xticklabels([f'{v:.0f}' for v in xa], fontsize=8)
    ax.set_xlabel('|camera elevation|  (degrees)', color=INK2, fontsize=8.8, labelpad=1)
    ax.set_ylabel(r'mean Chamfer  ($\times10^{-4}$)', color=INK2, fontsize=8.8)
    ax.set_ylim(4.8, 14.6)
    leg = ax.legend(frameon=False, fontsize=8.4, loc='upper left', handlelength=1.4,
                    borderpad=0.1, labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.30, 0.045, 'signed elevation: $\\rho$ = %+.2f' % stats.spearmanr(elev, err)[0],
            transform=ax.transAxes, ha='left', color=MUTED, fontsize=8.1)

    # --- C: everything else fails ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    frame(ax, 'Nothing about the plant predicts error',
          '|Spearman $\\rho$| vs Chamfer; blue = camera geometry')
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    lab = [r[0] for r in rows][::-1]
    val = [abs(r[2]) for r in rows][::-1]
    fam = [r[1] for r in rows][::-1]
    cols = [BLUE if f == 'cam' else DIM for f in fam]
    y = np.arange(len(lab))
    ax.barh(y, val, height=0.62, color=cols, zorder=3)
    for i, v in enumerate(val):
        ax.text(v + 0.012, i, f'{v:.2f}', va='center', color=INK2, fontsize=8.0)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8.1)
    for t, f in zip(ax.get_yticklabels(), fam):
        t.set_color(BLUE if f == 'cam' else INK2)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel(r'|Spearman $\rho$|', color=INK2, fontsize=8.8, labelpad=1)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.text(0.335, 4.15, 'cloud shape and plant complexity —\nevery one of them $|\\rho| \\leq$ 0.06',
            color=MUTED, fontsize=8.4, ha='center', va='center', linespacing=1.5)

    fig.text(0.046, 0.968, 'What makes an RGB→point-cloud reconstruction hard?',
             color=INK, fontsize=13.6, fontweight='bold', va='top')
    fig.text(0.046, 0.906,
             'smr50 seed 1, epoch %d, %s split (n = %d). Every view of a plant shares one identical target cloud, '
             'so the view index changes only the input image and the frame of the answer.'
             % (rep['epoch'], rep['split'], n),
             color=INK2, fontsize=8.6, va='top')

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
