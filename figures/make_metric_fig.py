"""Fig. 6: Chamfer distance against the frozen-probe R^2 across every sorghum arm.

The twelve points are the E2 arms, the three token-budget controls, the E3 data
arms and the E4 model sizes (val, first sorghum release), read from the tables
in paper/results.tex.  The metrics agree on the gross ordering (PC-only is worst
on both) but disagree on the decisions that matter: depth, the parameter
stream, and more data all move Chamfer and leave the probe alone.

    python figures/make_metric_fig.py        # writes paper/fig_metrics.{pdf,png}
    python figures/make_metric_fig.py --ugg  # UGG-style preview, ONLY paper/figs/ugg_preview/fig_metrics_ugg.{pdf,png}

The --ugg preview (figures/ugg_style.py): E2 arms coloured by the stream they add (ARM), the theta controls in the
recipe stream's pastel, E3 teal, E4 dark grey; hollow markers; one-row legend above the panels; no top/right spines,
light horizontal grid; arm labels placed so they do not overlap (thin leader lines where moved). Same numbers,
fonts, font sizes and figure size as the default.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'paper'
# name: (chamfer x1e-3, height R2, roll R2, leaf-length R2), group
ARMS = {
    'PC':            ((4.96, 0.899, 0.452, 0.102), 'E2'),
    'PC+RGB':        ((2.50, 0.978, 0.796, 0.273), 'E2'),
    'PC+RGB+D':      ((1.71, 0.981, 0.806, 0.362), 'E2'),
    r'+$\theta$':    ((2.50, 0.966, 0.793, 0.280), 'E2'),
    r'+$\theta$ 20%':((2.41, 0.958, 0.764, 0.288), 'ctrl'),
    r'+$\theta$ 60%':((2.72, 0.922, 0.676, 0.202), 'ctrl'),
    r'+$\theta$ target':((2.54, 0.970, 0.737, 0.259), 'ctrl'),
    '1k':            ((3.02, 0.963, 0.770, 0.269), 'E3'),
    '3k':            ((2.61, 0.963, 0.764, 0.255), 'E3'),
    '10k':           ((2.42, 0.962, 0.789, 0.301), 'E3'),
    'small':         ((3.02, 0.946, 0.761, 0.252), 'E4'),
    'large':         ((2.53, 0.967, 0.799, 0.298), 'E4'),
}
COL = {'E2': '#1f4e79', 'ctrl': '#c0504d', 'E3': '#4f9d69', 'E4': '#8064a2'}
MK = {'E2': 'o', 'ctrl': 's', 'E3': '^', 'E4': 'D'}
LAB = {'E2': 'streams (E2)', 'ctrl': r'$\theta$ controls', 'E3': 'training plants (E3)', 'E4': 'model size (E4)'}

def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
    cols = [(1, 'height $R^2$'), (2, 'leaf roll $R^2$'), (3, 'leaf length $R^2$')]
    for ax, (j, yl) in zip(axes, cols):
        xs = np.array([v[0][0] for v in ARMS.values()]); ys = np.array([v[0][j] for v in ARMS.values()])
        rho = stats.spearmanr(xs, ys)[0]
        for name, (v, g) in ARMS.items():
            ax.scatter(v[0], v[j], s=22, c=COL[g], marker=MK[g], zorder=3, label=LAB[g] if name in ('PC', r'+$\theta$ 20%', '1k', 'small') else None)
            dx = 0.08
            ha = 'left'
            if name in ('PC+RGB', '10k', '3k'):
                dx, ha = -0.08, 'right'
            ax.annotate(name, (v[0], v[j]), xytext=(dx * 10, 0), textcoords='offset points', fontsize=5.2, ha=ha, va='center', color='#333')
        ax.set_xlabel(r'Chamfer $\times10^{-3}$ (lower is better)', fontsize=6.5)
        ax.set_ylabel(yl, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlim(1.4, 5.3)
        ax.invert_xaxis()
        ax.text(0.02, 0.93, r'Spearman $\rho$ = %+.2f' % rho, transform=ax.transAxes, fontsize=6, color='#555')
        ax.grid(True, lw=0.3, alpha=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=5.5, loc='lower right', frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_metrics.{ext}', dpi=300)
    print('wrote', OUT / 'fig_metrics.pdf')

UGG_E2 = {'PC': 'PC', 'PC+RGB': 'PC+RGB', 'PC+RGB+D': 'PC+RGB+D', r'+$\theta$': 'PC+RGB+D+θ'}


def _place_labels(ax, pts, names, renderer, fs=5.2, obstacles=(), tries=400, seed=0):
    """Label placement in display space: every label gets an offset around its point (right first, as the default
    figure) that overlaps no marker, no other label and no obstacle, stays inside the axes and whose leader line
    crosses no other marker. Greedy over `tries` random label orders; the order with the lowest total offset wins.
    Geometry only (text sizes measured once), then the chosen labels are drawn, with thin leaders when moved."""
    pix = ax.figure.dpi / 72.0
    ab = ax.get_window_extent(renderer)
    disp = ax.transData.transform(pts)
    sizes = []
    for n in names:
        t = ax.text(0, 0, n, fontsize=fs); bb = t.get_window_extent(renderer); t.remove()
        sizes.append((bb.width + 1.2 * pix, bb.height + 0.4 * pix))
    mr = 3.4 * pix
    obs = [(o.x0, o.y0, o.x1, o.y1) for o in obstacles]
    angles = [0, 180, 25, -25, 155, -155, 50, -50, 130, -130, 75, -75, 105, -105, 90, -90]
    radii = (4.5, 7, 10, 13, 17, 21, 26, 32, 38, 45, 53, 62)
    cands = []                                   # per label: list of (cost, r, a, ha, va, box)
    for i, (x, y) in enumerate(disp):
        w, h = sizes[i]; lst = []
        for r in radii:
            for k, a in enumerate(angles):
                c, sn = np.cos(np.radians(a)), np.sin(np.radians(a))
                ha = 'left' if c > 0.3 else ('right' if c < -0.3 else 'center')
                va = 'bottom' if sn > 0.3 else ('top' if sn < -0.3 else 'center')
                ax_, ay_ = x + r * c * pix, y + r * sn * pix
                x0 = ax_ if ha == 'left' else (ax_ - w if ha == 'right' else ax_ - w / 2)
                y0 = ay_ if va == 'bottom' else (ay_ - h if va == 'top' else ay_ - h / 2)
                box = (x0, y0, x0 + w, y0 + h)
                if box[0] < ab.x0 + 1 or box[2] > ab.x1 - 1 or box[1] < ab.y0 + 1 or box[3] > ab.y1 - 1:
                    continue
                if any(np.hypot(min(max(px, box[0]), box[2]) - px, min(max(py, box[1]), box[3]) - py) < mr for px, py in disp):
                    continue
                if any(box[0] < o[2] and box[2] > o[0] and box[1] < o[3] and box[3] > o[1] for o in obs):
                    continue
                bad = 0
                if r > 6:                        # leader from the point to the nearest point of the box
                    tx_, ty_ = min(box[2], max(box[0], x)), min(box[3], max(box[1], y))
                    seg = np.array([tx_ - x, ty_ - y]); L2 = seg @ seg
                    for kk, q in enumerate(disp):
                        if kk == i or L2 < 1e-9: continue
                        u = np.clip((q - disp[i]) @ seg / L2, 0, 1)
                        bad += np.hypot(*(disp[i] + u * seg - q)) < mr
                lst.append((r + 0.15 * k + (4 if r > 6 else 0) + 25 * bad, r, a, ha, va, box))
        cands.append(sorted(lst, key=lambda z: z[0]))
    rng = np.random.default_rng(seed); best = None
    ov = lambda b1, b2: b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]
    for t in range(tries):
        order = list(range(len(names))) if t == 0 else list(rng.permutation(len(names)))
        placed, cost, miss = {}, 0.0, 0
        for i in order:
            for cand in cands[i]:
                if not any(ov(cand[5], placed[j][5]) for j in placed):
                    placed[i] = cand; cost += cand[0]; break
            else:
                miss += 1; cost += 100
        if best is None or (miss, cost) < best[0]:
            best = ((miss, cost), placed)
    for i, n in enumerate(names):
        if i not in best[1]:
            print('WARN: no free spot for', repr(n), 'in', ax.get_ylabel())
            ax.annotate(n, pts[i], xytext=(4.5, 0), textcoords='offset points', fontsize=fs, ha='left', va='center', color='#333')
            continue
        _, r, a, ha, va, _ = best[1][i]
        c, sn = np.cos(np.radians(a)), np.sin(np.radians(a))
        kw = dict(arrowprops=dict(arrowstyle='-', color='#a6a6a6', lw=0.4, shrinkA=0, shrinkB=1.0)) if r > 6 else {}
        ax.annotate(n, pts[i], xytext=(r * c, r * sn), textcoords='offset points', fontsize=fs, ha=ha, va=va,
                    color='#333', **kw)


def main_ugg():
    import ugg_style as U
    from matplotlib.legend_handler import HandlerTuple
    col = {'ctrl': U.STREAM['text'], 'E3': U.SERIES[4], 'E4': U.GREYS[0]}
    colour = lambda name, g: U.ARM[UGG_E2[name]] if g == 'E2' else col[g]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
    cols = [(1, 'height $R^2$'), (2, 'leaf roll $R^2$'), (3, 'leaf length $R^2$')]
    rhos = []
    for ax, (j, yl) in zip(axes, cols):
        xs = np.array([v[0][0] for v in ARMS.values()]); ys = np.array([v[0][j] for v in ARMS.values()])
        rhos.append(stats.spearmanr(xs, ys)[0])
        for name, (v, g) in ARMS.items():
            ax.scatter(v[0], v[j], s=22, marker=MK[g], facecolors='white', edgecolors=colour(name, g), linewidths=1.0, zorder=3)
        ax.set_xlabel(r'Chamfer $\times10^{-3}$ (lower is better)', fontsize=6.5)
        ax.set_ylabel(yl, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlim(1.4, 5.3)
        ax.invert_xaxis()
        U.style_axes(ax, grid='y')
    mk = lambda c, m: plt.Line2D([], [], ls='', marker=m, ms=4.2, mfc='white', mec=c, mew=1.0)
    e2 = tuple(mk(U.ARM[UGG_E2[n]], 'o') for n in UGG_E2)
    hs = [e2] + [mk(col[g], MK[g]) for g in ('ctrl', 'E3', 'E4')]
    labs = ['streams (E2, colour = stream added)'] + [LAB[g] for g in ('ctrl', 'E3', 'E4')]
    fig.legend(hs, labs, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4, frameon=False, fontsize=5.5,
               handletextpad=0.4, handlelength=3.0, columnspacing=1.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.91))
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for ax, (j, _), rho in zip(axes, cols, rhos):
        tx = ax.text(0.02, 0.93, r'Spearman $\rho$ = %+.2f' % rho, transform=ax.transAxes, fontsize=6, color='#555')
        pts = [(v[0], v[j]) for v, _ in ARMS.values()]
        _place_labels(ax, pts, list(ARMS), rend, obstacles=[tx.get_window_extent(rend)])
    out = OUT / 'figs' / 'ugg_preview'; out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_metrics_ugg.{ext}', dpi=300)
    print('wrote', out / 'fig_metrics_ugg.pdf')


if __name__ == '__main__':
    if '--ugg' in _sys.argv:
        _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
        main_ugg()
    else:
        main()
