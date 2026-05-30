"""Rebuild training_history.json + loss_curves.png for 4m_run_v3 from SLURM logs.

The duplicate concurrent job clobbered the on-disk training_history.json. Parse
both .out files instead:

  logs/embmae4m_10632614.out  — original run, epochs 1 .. 1125 (died at walltime)
  logs/embmae4m_10727040.out  — resumed run,  epochs 1101 .. 2400 (completed)

We prefer the completed-run rows for epochs 1101+, since that's the trajectory
that actually reached the final 2400-epoch checkpoint.

Outputs:
  outputs/4m_run_v3/training_history_recovered.json
  outputs/4m_run_v3/loss_curves_2400.png
  outputs/4m_run_v3/metrics_summary_2400.json
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path('/work/mech-ai-scratch/alloy/embodiedmae')
LOG_ORIG = REPO / 'logs/embmae4m_10632614.out'
LOG_DONE = REPO / 'logs/embmae4m_10727040.out'
OUT_DIR  = REPO / 'outputs/4m_run_v3'

EPOCH_RE   = re.compile(r'^Epoch (\d+)/\d+\s+lr=([0-9.eE+-]+)')
TRAIN_RE   = re.compile(r'^Train — Loss:\s*([0-9.]+)\s+RGB:\s*([0-9.]+)\s+Depth:\s*([0-9.]+)\s+PC:\s*([0-9.]+)\s+Text:\s*([0-9.]+)')
VAL_RE     = re.compile(r'^Val   — Loss:\s*([0-9.]+)\s+RGB:\s*([0-9.]+)\s+Depth:\s*([0-9.]+)\s+PC:\s*([0-9.]+)\s+Text:\s*([0-9.]+)')
METRICS_RE = re.compile(r'^Metrics — RGB MSE:\s*([0-9.]+)\s+Depth MSE:\s*([0-9.]+)\s+PC Chamfer:\s*([0-9.]+)\s+PC EMD:\s*([0-9.]+)')
PARAM_RE   = re.compile(r'^Param   — MSE:\s*([0-9.]+)\s+MAE:\s*([0-9.]+)\s+MAE\(masked\):\s*([0-9.]+)\s+acc@0\.05:\s*([0-9.]+)')


def parse_log(path: Path):
    """Return dict: epoch -> {train: {...}, val: {...} (maybe), lr: float}."""
    rows = {}
    cur_epoch = None
    cur_lr = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = EPOCH_RE.match(line)
            if m:
                cur_epoch = int(m.group(1))
                cur_lr = float(m.group(2))
                rows.setdefault(cur_epoch, {'lr': cur_lr})
                continue
            if cur_epoch is None:
                continue
            m = TRAIN_RE.match(line)
            if m:
                rows[cur_epoch]['train'] = dict(
                    loss=float(m.group(1)),  rgb=float(m.group(2)),
                    depth=float(m.group(3)), pc=float(m.group(4)),
                    text=float(m.group(5)),
                )
                continue
            m = VAL_RE.match(line)
            if m:
                rows[cur_epoch].setdefault('val', {}).update(dict(
                    loss=float(m.group(1)),  rgb=float(m.group(2)),
                    depth=float(m.group(3)), pc=float(m.group(4)),
                    text=float(m.group(5)),
                ))
                continue
            m = METRICS_RE.match(line)
            if m:
                rows[cur_epoch].setdefault('val', {}).update(dict(
                    rgb_mse=float(m.group(1)),  depth_mse=float(m.group(2)),
                    pc_chamfer=float(m.group(3)), pc_emd=float(m.group(4)),
                ))
                continue
            m = PARAM_RE.match(line)
            if m:
                rows[cur_epoch].setdefault('val', {}).update(dict(
                    param_mse=float(m.group(1)),       param_mae=float(m.group(2)),
                    param_mae_masked=float(m.group(3)), param_acc05=float(m.group(4)),
                ))
    return rows


def main():
    orig = parse_log(LOG_ORIG)
    done = parse_log(LOG_DONE)

    # Merge: epochs 1..1100 from orig, 1101..end from done
    merged = {}
    for ep, v in orig.items():
        if ep <= 1100:
            merged[ep] = v
    for ep, v in done.items():
        merged[ep] = v

    epochs = sorted(merged.keys())
    print(f'orig={len(orig)} epochs, done={len(done)} epochs, merged={len(epochs)} (max ep {epochs[-1]})')

    # Build flat lists indexed by *position* (per-epoch, with NaNs for missing val)
    def lst(key_path):
        out = []
        for ep in epochs:
            cur = merged[ep]
            for k in key_path:
                cur = cur.get(k, None) if isinstance(cur, dict) else None
                if cur is None:
                    break
            out.append(cur if cur is not None else float('nan'))
        return out

    history = {
        'epoch': epochs,
        'lr':    [merged[e].get('lr', float('nan')) for e in epochs],
        # train rolling
        'train_loss':  lst(['train', 'loss']),
        'train_rgb':   lst(['train', 'rgb']),
        'train_depth': lst(['train', 'depth']),
        'train_pc':    lst(['train', 'pc']),
        'train_text':  lst(['train', 'text']),
        # val (only present every val_freq epochs)
        'val_loss':  lst(['val', 'loss']),
        'val_rgb':   lst(['val', 'rgb']),
        'val_depth': lst(['val', 'depth']),
        'val_pc':    lst(['val', 'pc']),
        'val_text':  lst(['val', 'text']),
        'val_rgb_mse':         lst(['val', 'rgb_mse']),
        'val_depth_mse':       lst(['val', 'depth_mse']),
        'val_pc_chamfer':      lst(['val', 'pc_chamfer']),
        'val_pc_emd':          lst(['val', 'pc_emd']),
        'val_param_mse':       lst(['val', 'param_mse']),
        'val_param_mae':       lst(['val', 'param_mae']),
        'val_param_mae_masked':lst(['val', 'param_mae_masked']),
        'val_param_acc05':     lst(['val', 'param_acc05']),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'training_history_recovered.json', 'w') as f:
        json.dump(history, f)
    print(f'Wrote {OUT_DIR / "training_history_recovered.json"}')

    # ── Summary metrics ──────────────────────────────────────────────────────
    val_loss = np.array(history['val_loss'], dtype=float)
    val_ep   = np.array(epochs)
    finite   = np.isfinite(val_loss)
    best_idx = int(np.nanargmin(val_loss))
    best_ep  = int(val_ep[best_idx])
    final_idx = int(np.where(finite)[0][-1])  # last validated epoch
    final_ep  = int(val_ep[final_idx])

    def at(idx, key):
        v = history[key][idx]
        return float(v) if v is not None and np.isfinite(v) else None

    summary = {
        'best_epoch':  best_ep,
        'final_epoch': final_ep,
        'total_epochs_completed': epochs[-1],
        'best': {k: at(best_idx, k) for k in (
            'val_loss','val_rgb','val_depth','val_pc','val_text',
            'val_rgb_mse','val_depth_mse','val_pc_chamfer','val_pc_emd',
            'val_param_mse','val_param_mae','val_param_mae_masked','val_param_acc05')},
        'final': {k: at(final_idx, k) for k in (
            'val_loss','val_rgb','val_depth','val_pc','val_text',
            'val_rgb_mse','val_depth_mse','val_pc_chamfer','val_pc_emd',
            'val_param_mse','val_param_mae','val_param_mae_masked','val_param_acc05')},
    }
    with open(OUT_DIR / 'metrics_summary_2400.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Best  ep {best_ep:4d}: val_loss={summary["best"]["val_loss"]:.6f}')
    print(f'Final ep {final_ep:4d}: val_loss={summary["final"]["val_loss"]:.6f}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()

    def smooth(y, win=15):
        y = np.asarray(y, dtype=float)
        if len(y) < win:
            return y
        k = np.ones(win) / win
        return np.convolve(y, k, mode='same')

    train_pairs = [
        ('train_loss',  'val_loss',  'Total loss'),
        ('train_rgb',   'val_rgb',   'RGB (weighted, per-patch MSE)'),
        ('train_depth', 'val_depth', 'Depth (weighted)'),
        ('train_pc',    'val_pc',    'PC (× pc_loss_weight=10)'),
        ('train_text',  'val_text',  'Text/params (× spline_w=5)'),
    ]
    for ax, (tk, vk, title) in zip(axes, train_pairs):
        tr = np.array(history[tk], dtype=float)
        va = np.array(history[vk], dtype=float)
        ax.plot(epochs, tr, color='steelblue', alpha=0.25, linewidth=0.7,
                label='train (raw)')
        ax.plot(epochs, smooth(tr), color='steelblue', linewidth=1.6,
                label='train (smooth)')
        v_ep = [e for e, x in zip(epochs, va) if np.isfinite(x)]
        v_y  = [x for x in va if np.isfinite(x)]
        ax.plot(v_ep, v_y, color='crimson', marker='.', linewidth=1.4,
                markersize=4, label='val')
        ax.axvline(best_ep, color='gray', linestyle='--', linewidth=0.7)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('epoch')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # Last panel: validation accuracy@0.05 + LR
    ax = axes[5]
    acc = np.array(history['val_param_acc05'], dtype=float)
    v_ep = [e for e, x in zip(epochs, acc) if np.isfinite(x)]
    v_y  = [x for x in acc if np.isfinite(x)]
    ax.plot(v_ep, v_y, color='seagreen', marker='.', linewidth=1.6,
            label='val param acc@0.05')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc@0.05', color='seagreen')
    ax.tick_params(axis='y', labelcolor='seagreen')
    ax.grid(alpha=0.3)
    ax.axvline(best_ep, color='gray', linestyle='--', linewidth=0.7)

    ax2 = ax.twinx()
    ax2.plot(epochs, history['lr'], color='goldenrod', linewidth=1.2,
             label='lr', alpha=0.8)
    ax2.set_ylabel('learning rate', color='goldenrod')
    ax2.tick_params(axis='y', labelcolor='goldenrod')
    ax.set_title('Param acc@0.05 + LR schedule', fontsize=12, weight='bold')

    fig.suptitle(
        f'EmbodiedMAE-4M run v3 — full training history (epochs 1–{epochs[-1]})\n'
        f'Best val_loss = {summary["best"]["val_loss"]:.4f} @ epoch {best_ep}     '
        f'·  Final val_loss = {summary["final"]["val_loss"]:.4f} @ epoch {final_ep}',
        fontsize=14, weight='bold', y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / 'loss_curves_2400.png'
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
