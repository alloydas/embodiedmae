#!/usr/bin/env python3
"""Compare the E2 modality value-add arms.

The one rule this script exists to enforce: arms are compared on pc_chamfer,
never on total loss. `total` sums a different number of terms per arm (PC only
vs PC+RGB+D+params), so it runs ~1.6 to ~9.8 at init purely from term count --
reading it as quality would say the PC-only arm is 6x better than the 4-modality
one. PC is active in every arm, so pc_chamfer is the arm-invariant yardstick.

    python eval/analyze_e2.py                       # table over all arms found
    python eval/analyze_e2.py --epoch 500           # compare at a fixed epoch
    python eval/analyze_e2.py --csv e2.csv
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse, json, csv
from pathlib import Path

ARMS = [('pc',      'PC only'),
        ('pcrgb',   'PC + RGB'),
        ('pcrgbd',  'PC + RGB + D'),
        ('pcrgbdt', 'PC + RGB + D + params')]


def load(slug, out_root):
    h = Path(out_root) / f'e2_{slug}' / 'training_history.json'
    if not h.exists():
        return None
    d = json.loads(h.read_text())
    if not d.get('val_pc_chamfer'):
        return None
    return d


def at_epoch(d, want, val_freq):
    """Index into the val series. Validation runs at epoch 1 then every val_freq,
    so the series index is not the epoch number."""
    series = d['val_pc_chamfer']
    if want is None:
        return len(series) - 1, series[-1]
    # reconstruct the epochs at which validation actually ran
    epochs = [1] + [e for e in range(val_freq, 100000, val_freq) if e != 1]
    epochs = sorted(set(epochs))[:len(series)]
    best = min(range(len(series)), key=lambda i: abs(epochs[i] - want))
    return best, series[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default='./outputs')
    ap.add_argument('--epoch', type=int, default=None,
                    help='compare at this epoch (default: latest available)')
    ap.add_argument('--val_freq', type=int, default=50)
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()

    rows, baseline = [], None
    for slug, label in ARMS:
        d = load(slug, a.out_root)
        if d is None:
            rows.append({'arm': slug, 'label': label, 'status': 'no results yet'})
            continue
        i, ch = at_epoch(d, a.epoch, a.val_freq)
        n = len(d['val_pc_chamfer'])
        best = min(d['val_pc_chamfer'])
        r = {'arm': slug, 'label': label, 'status': f'{n} val points',
             'pc_chamfer': ch, 'best_pc_chamfer': best,
             'train_epochs': len(d.get('train_loss', []))}
        # report the other modalities' metrics where the arm has them, but never
        # fold them into the cross-arm comparison
        for k in ('val_rgb_mse', 'val_depth_mse', 'val_param_mae_masked'):
            if d.get(k):
                r[k] = d[k][min(i, len(d[k]) - 1)]
        if slug == 'pc':
            baseline = ch
        rows.append(r)

    w = max(len(l) for _, l in ARMS) + 2
    tag = 'latest' if a.epoch is None else f'epoch ~{a.epoch}'
    print(f"\nE2 modality value-add — compared on val pc_chamfer ({tag})")
    print(f"{'arm':<{w}} {'pc_chamfer':>12} {'best':>10} {'vs PC-only':>12}  extra metrics")
    print('-' * (w + 52))
    for r in rows:
        if 'pc_chamfer' not in r:
            print(f"{r['label']:<{w}} {r['status']:>12}")
            continue
        if baseline:
            delta = (r['pc_chamfer'] - baseline) / baseline * 100
            d_s = f"{delta:+.1f}%"
        else:
            d_s = '—'
        extra = '  '.join(f"{k.replace('val_',''):s}={r[k]:.4f}"
                          for k in ('val_rgb_mse', 'val_depth_mse',
                                    'val_param_mae_masked') if k in r)
        print(f"{r['label']:<{w}} {r['pc_chamfer']:>12.6f} {r['best_pc_chamfer']:>10.6f} "
              f"{d_s:>12}  {extra}")
    print("\nNegative 'vs PC-only' = lower Chamfer = the added modality helped PC "
          "reconstruction.\nTotal loss is deliberately not shown: it is not comparable "
          "across arms.\n")

    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader(); wr.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == '__main__':
    main()
