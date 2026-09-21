"""
sweeps/mask_sweep_all_modalities.py — post-process vis_mask_sweep/metrics.json to report,
for EVERY masking condition, the reconstruction metric of ALL FOUR modalities (not
just the swept target). Reads the already-computed metrics; no model re-run.

Produces (in the same --dir):
    heatmaps_all_modalities.png   4x4 grid: row = swept target, col = metric modality
    summary_all_modalities.txt    per-target tables, one per reported modality

For a swept target T, the y-axis is T's own mask %, the x-axis is the shared mask %
of the OTHER three modalities. Each column then reads out one modality's metric under
that same config — so off-diagonal panels show how the non-target modalities hold up
while T is being masked. The diagonal panels (bold titles) are the target's own metric.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS   = ['rgb', 'depth', 'pc', 'text']
TGT_LABEL = {'rgb': 'RGB', 'depth': 'Depth', 'pc': 'Point cloud', 'text': 'Spline'}
METRICS   = [('rgb_mse', 'RGB MSE'), ('depth_mse', 'Depth MSE'),
             ('pc_chamfer', 'PC Chamfer'), ('spline_mae', 'Spline MAE')]
# which metric column "belongs to" which swept target (the diagonal)
DIAG = {'rgb': 'rgb_mse', 'depth': 'depth_mse', 'pc': 'pc_chamfer', 'text': 'spline_mae'}


def build_grids(cells, own_r, other_r):
    """grids[target][metric_key] -> (len(own) x len(other)) array."""
    idx = {(c['target'], c['own_mask'], c['other_mask']): c for c in cells}
    grids = {}
    for t in TARGETS:
        grids[t] = {}
        for mkey, _ in METRICS:
            M = np.full((len(own_r), len(other_r)), np.nan)
            for i, o in enumerate(own_r):
                for j, c in enumerate(other_r):
                    cell = idx.get((t, o, c))
                    if cell is not None:
                        M[i, j] = cell[mkey]
            grids[t][mkey] = M
    return grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='vis_mask_sweep')
    args = ap.parse_args()
    d = Path(args.dir)

    data = json.load(open(d / 'metrics.json'))
    own_r, other_r = data['own_ratios'], data['other_ratios']
    epoch, vloss = data.get('epoch'), data.get('val_loss')
    grids = build_grids(data['cells'], own_r, other_r)

    # per-metric colour scale (shared down a column so rows are comparable),
    # robust max via 95th pct so the 0%-row outliers don't wash out the scale
    vlim = {}
    for mkey, _ in METRICS:
        allv = np.concatenate([grids[t][mkey].ravel() for t in TARGETS])
        allv = allv[~np.isnan(allv)]
        vlim[mkey] = (float(np.nanmin(allv)), float(np.nanpercentile(allv, 95)))

    nrow, ncol = len(TARGETS), len(METRICS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow))
    for r, t in enumerate(TARGETS):
        for cI, (mkey, mlabel) in enumerate(METRICS):
            ax = axes[r, cI]
            M = grids[t][mkey]
            lo, hi = vlim[mkey]
            im = ax.imshow(M, cmap='viridis', aspect='auto', origin='lower',
                           vmin=lo, vmax=hi)
            ax.set_xticks(range(len(other_r)))
            ax.set_xticklabels([f"{int(c*100)}" for c in other_r], fontsize=8)
            ax.set_yticks(range(len(own_r)))
            ax.set_yticklabels([f"{int(o*100)}" for o in own_r], fontsize=8)
            if r == nrow - 1:
                ax.set_xlabel('other modalities mask %', fontsize=9)
            if cI == 0:
                ax.set_ylabel(f"swept: {TGT_LABEL[t]}\nown mask %", fontsize=9)
            is_diag = (mkey == DIAG[t])
            ax.set_title((f"◆ {mlabel} (target)" if is_diag else mlabel),
                         fontweight='bold' if is_diag else 'normal',
                         color='darkred' if is_diag else 'black', fontsize=10)
            if is_diag:
                for s in ax.spines.values():
                    s.set_edgecolor('darkred'); s.set_linewidth(2.2)
            mid = (lo + hi) / 2
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    txt = 'nan' if np.isnan(v) else f"{v:.3f}"
                    ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                            color='white' if (np.isnan(v) or v < mid) else 'black')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("All-modality reconstruction across the mask-ratio grid\n"
                 "rows = swept target modality, columns = reported modality metric "
                 f"(◆ = target's own; epoch {epoch})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(d / 'heatmaps_all_modalities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"wrote {d/'heatmaps_all_modalities.png'}")

    # ── expanded text tables ────────────────────────────────────────────────
    with open(d / 'summary_all_modalities.txt', 'w') as f:
        f.write(f"All-modality mask-ratio sweep — EmbodiedMAE-4M (epoch {epoch}, "
                f"val_loss {vloss})\n")
        f.write("For each swept target: rows = target own mask %, cols = other "
                "modalities mask %.\nEach block reports one modality's reconstruction "
                "metric under that config.\n")
        f.write("(◆ marks the target's own metric.)\n\n")
        for t in TARGETS:
            f.write(f"################  SWEPT TARGET: {TGT_LABEL[t]}  ################\n\n")
            for mkey, mlabel in METRICS:
                tag = "  ◆ (target's own)" if mkey == DIAG[t] else ""
                f.write(f"  [{mlabel}]{tag}\n")
                f.write("   own\\other " + "".join(f"{int(c*100):>9d}%" for c in other_r) + "\n")
                M = grids[t][mkey]
                for i, o in enumerate(own_r):
                    row = "".join(f"{M[i, j]:>10.4f}" for j in range(len(other_r)))
                    f.write(f"   {int(o*100):>6d}%   {row}\n")
                f.write("\n")
            f.write("\n")
    print(f"wrote {d/'summary_all_modalities.txt'}")


if __name__ == '__main__':
    main()
