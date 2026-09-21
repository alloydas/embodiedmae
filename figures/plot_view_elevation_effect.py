"""Two results the per-plant eval reports carry, plotted honestly.

A. RGB->PC difficulty is set by CAMERA ELEVATION, not by the plant.
   The ten view folders of a plant are not arbitrary labels: parsing every
   camera_pose.json gives sin(elevation) = -0.9 + 0.2 * view_index exactly
   (sd 0.0 within each group, |dev| < 1e-15). All ten views share ONE target --
   <plant>_nc.ply is byte-identical across the ten folders and _nc_cam.ply is
   that same cloud rotated into the view frame -- and the loader centres and
   unit-scales before Chamfer, which is rotation invariant. So the view index
   changes the INPUT IMAGE and nothing about how hard the answer is to hit.
   That makes this a clean natural experiment, and the curve is a U that is
   symmetric about the horizon: looking steeply up is as hard as steeply down.
   The original hunch was 'top-down views are hard'. That half is wrong.

B. Source masking pays off where the task is hard, graded by difficulty.
   Bucketing all 2250 plants by the CONTROL's own error and measuring the
   paired change gives a monotone gradient, not a uniform 5.9% shift.

Both panels are computed from reports/quant_*_test.json only. Nothing is
re-run and nothing is sampled -- every number is over all 2250 plants.

  python figures/plot_view_elevation_effect.py --out vis_gallery/view_elevation_effect.png
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, RULE = '#0f1619', '#71838b', '#dbe4e6'
SERIES = [
    ('smr50_seed1',         'Source masked (smr 0.5)',   '#4a2a63', 2.4, '-'),
    ('smr00_full',          'Source whole (smr 0.0)',    '#6c828b', 2.0, '-'),
    ('teacher_undistilled', 'Undistilled teacher',       '#b9c8cc', 1.7, '--'),
    ('generalist_depth',    'Generalist, depth source',  '#eb6834', 2.0, '-'),
]


def load(label, split='test'):
    return json.loads(open(f'reports/quant_{label}_{split}.json').read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='vis_gallery/view_elevation_effect.png')
    args = ap.parse_args()

    base = load('smr50_seed1')
    view = np.array([int(re.match(r'.+_(\d+)_(\d+)$', p).group(2)) for p in base['plants']])
    elev = np.degrees(np.arcsin(-0.9 + 0.2 * np.arange(10)))

    def eta2(x):
        x = np.asarray(x); gm = x.mean()
        ssb = sum((view == k).sum() * (x[view == k].mean() - gm) ** 2 for k in range(10))
        return ssb / ((x - gm) ** 2).sum()

    fig = plt.figure(figsize=(12.4, 4.5), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1], wspace=0.26,
                          left=0.062, right=0.988, top=0.86, bottom=0.155)

    # ── A. the U ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for label, nice, col, lw, ls in SERIES:
        v = np.array(load(label)['per_sample']['model'])
        m = np.array([v[view == k].mean() for k in range(10)]) * 1e4
        se = np.array([v[view == k].std(ddof=1) / np.sqrt((view == k).sum()) for k in range(10)]) * 1e4
        ax.errorbar(elev, m, yerr=se, color=col, lw=lw, ls=ls, marker='o', ms=4.2,
                    capsize=2.5, elinewidth=1, label=f'{nice}   $\\eta^2$={eta2(v):.2f}', zorder=3)
    ax.axvline(0, color=RULE, lw=1, zorder=1)
    ax.set_xticks(elev.round(1))
    ax.set_xticklabels([f'{e:.0f}' for e in elev], fontsize=8.5)
    ax.set_xlabel('camera elevation (degrees; negative = looking down, 0 = horizon)',
                  fontsize=9.5, color=MUTED)
    ax.set_ylabel(r'mean Chamfer  $\times 10^{-4}$', fontsize=9.5, color=MUTED)
    ax.set_title('A.  Difficulty is set by how far the camera is off the horizon',
                 fontsize=10.5, color=INK, loc='left', pad=9)
    ax.text(0, 1.008, '', transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=8.4, loc='upper center', ncol=2,
              handlelength=2.0, columnspacing=1.4, labelcolor=INK)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(RULE)
    ax.grid(axis='y', color=RULE, lw=.8, alpha=.7); ax.set_axisbelow(True)
    ax.set_ylim(0, 18.2)
    ax.text(.5, -.235, 'n = 208–244 plants per view · same target cloud in all ten views · '
            'cross-plant GT baseline is flat here ($\\eta^2$=0.04)',
            transform=ax.transAxes, ha='center', fontsize=7.8, color=MUTED)

    # ── B. who the gain goes to ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ctl = np.array(load('smr00_full')['per_sample']['model'])
    smr = np.array(load('smr50_seed1')['per_sample']['model'])
    q = np.quantile(ctl, [.25, .5, .75])
    masks = [ctl <= q[0], (ctl > q[0]) & (ctl <= q[1]),
             (ctl > q[1]) & (ctl <= q[2]), ctl > q[2]]
    labs = ['Q1\neasiest', 'Q2', 'Q3', 'Q4\nhardest']
    pct = [(smr[m].mean() - ctl[m].mean()) / ctl[m].mean() * 100 for m in masks]
    win = [(smr[m] < ctl[m]).mean() * 100 for m in masks]
    cols = ['#c26b4d' if p > 0 else '#4a2a63' for p in pct]
    b = ax.bar(range(4), pct, color=cols, width=.66, zorder=3)
    for i, (p, w) in enumerate(zip(pct, win)):
        ax.text(i, p + (0.42 if p > 0 else -0.42), f'{p:+.1f}%',
                ha='center', va='bottom' if p > 0 else 'top',
                fontsize=9.5, color=INK, fontweight='semibold')
        ax.text(i, -15.9, f'wins\n{w:.0f}%', ha='center', va='bottom',
                fontsize=8, color=MUTED)
    ax.axhline(0, color=INK, lw=1, zorder=4)
    ax.axhline(-5.9, color=MUTED, lw=1, ls=':', zorder=2)
    ax.text(3.46, -5.9, 'overall\n−5.9%', fontsize=7.8, color=MUTED, va='center', ha='left')
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.8, color=MUTED)
    ax.set_ylabel('change in Chamfer from source masking', fontsize=9.5, color=MUTED)
    ax.set_title('B.  The gain goes to the hard plants', fontsize=10.5, color=INK, loc='left', pad=9)
    ax.set_ylim(-17.0, 5.2); ax.set_xlim(-.62, 3.42)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(RULE)
    ax.grid(axis='y', color=RULE, lw=.8, alpha=.7); ax.set_axisbelow(True)
    ax.text(.5, -.235, 'all 2250 test plants, bucketed by the control\'s own error · '
            'paired per plant', transform=ax.transAxes, ha='center', fontsize=7.8, color=MUTED)

    fig.savefig(args.out, bbox_inches='tight', facecolor='white')
    print(f'wrote {args.out}')
    print('per-view means x1e-4:', np.round([np.array(base['per_sample']['model'])[view == k].mean() * 1e4 for k in range(10)], 2))
    print('quartile deltas:', [f'{p:+.1f}%' for p in pct], 'wins:', [f'{w:.0f}%' for w in win])


if __name__ == '__main__':
    main()
