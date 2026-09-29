"""Read the recipe-sweep runs and print one comparison table.

Every candidate is trained on the same plants for the same number of optimiser
steps, so the numbers line up column by column.  The table is cut at the last
validation epoch all the runs reached, so a run that is still going (or died
early) is compared at a budget the others also reached rather than at its own
final epoch.

    python compare_recipes.py                      # all outputs/recipe_*
    python compare_recipes.py --runs outputs/a outputs/b
    python compare_recipes.py --output recipe_comparison.md
    python compare_recipes.py --loss-study         # QAL / Chamfer / Sinkhorn arms

Validation is clean: `val_loss` is what `best_model.pth` is chosen on. Runs
trained before 2026-09-29 also logged an occluded pass (`val_occ_*`); those
columns appear only when a run has them.
"""

import argparse
import json
from pathlib import Path

# (history key, column header, "lower is better", format)
COLUMNS = [
    ('val_loss',             'val loss',      True,  '{:.4f}'),
    ('val_occ_loss',         'val(occ) loss', True,  '{:.4f}'),
    ('val_rgb_mse',          'RGB MSE',       True,  '{:.4f}'),
    ('val_depth_mse',        'Depth MSE',     True,  '{:.5f}'),
    ('val_pc_chamfer',       'PC Chamfer',    True,  '{:.6f}'),
    ('val_pc_f1@0.03',       'F1@0.03',       False, '{:.4f}'),
    ('val_pc_recall@0.03',   'Recall@0.03',   False, '{:.4f}'),
    ('val_pc_f1@0.01',       'PC F1@0.01',    False, '{:.4f}'),
    ('val_param_mae_masked', 'Param MAE',     True,  '{:.4f}'),
]

# Loss study: every arm optimises a different PC loss, so val loss (and the
# val_pc term inside it) is a different quantity per arm. Only metrics computed
# the same way for all arms go in this table. EMD is evaluated at the final
# epoch only (0 elsewhere in the history).
LOSS_COLUMNS = [
    ('val_pc_chamfer',       'PC Chamfer',    True,  '{:.6f}'),
    ('val_pc_f1@0.01',       'F1@0.01',       False, '{:.4f}'),
    ('val_pc_f1@0.02',       'F1@0.02',       False, '{:.4f}'),
    ('val_pc_f1@0.03',       'F1@0.03',       False, '{:.4f}'),
    ('val_pc_recall@0.03',   'Recall@0.03',   False, '{:.4f}'),
    ('val_pc_emd',           'EMD (final)',   True,  '{:.4f}'),
    ('val_occ_pc_chamfer',   'occ Chamfer',   True,  '{:.6f}'),
    ('val_rgb_mse',          'RGB MSE',       True,  '{:.4f}'),
    ('val_depth_mse',        'Depth MSE',     True,  '{:.5f}'),
]


def load_run(run_dir):
    run_dir = Path(run_dir)
    history_path = run_dir / 'training_history.json'
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text())
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {'name': run_dir.name, 'dir': run_dir,
            'history': history, 'config': config}


def describe_loss(run):
    """(loss, setting) for the loss-study table, from the run's own config."""
    cfg = run['config']
    name = cfg.get('loss_name', 'chamfer')
    if name == 'qal_loss':
        return 'QAL', f"t={cfg.get('qal_threshold')} a={cfg.get('qal_alpha'):g}"
    if name == 'sinkhorn':
        return 'Sinkhorn', (f"blur={cfg.get('sinkhorn_blur')} "
                            f"w={cfg.get('sinkhorn_loss_weight'):g}")
    return 'Chamfer', f"w={cfg.get('pc_loss_weight'):g}"


def describe(run):
    """The two knobs the sweep varies, straight from the run's own config."""
    cfg = run['config']
    sm = cfg.get('structured_mask') or {}
    occ = cfg.get('occlusion') or {}
    blob = sm.get('length_scale', '-')
    noise = occ.get('rgb_noise_std', '-')
    return f"ls={blob}", f"rgb_noise={noise}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', default=None,
                    help='Run directories (default: outputs/recipe_*)')
    ap.add_argument('--output', default=None, help='Also write the table here (Markdown)')
    ap.add_argument('--loss-study', action='store_true',
                    help='QAL/Chamfer/Sinkhorn arms: loss-independent metrics only')
    ap.add_argument('--at-epoch', type=int, default=None,
                    help='Compare at this epoch instead of the last common one')
    args = ap.parse_args()

    if args.runs:
        dirs = args.runs
    elif args.loss_study:
        dirs = ['outputs/params_v1_seed1'] + sorted(
            str(p) for p in Path('outputs').glob('loss_*'))
    else:
        dirs = sorted(str(p) for p in Path('outputs').glob('recipe_*'))
    columns = LOSS_COLUMNS if args.loss_study else COLUMNS
    runs = [r for r in (load_run(d) for d in dirs) if r]
    missing = [d for d in dirs if not (Path(d) / 'training_history.json').exists()]
    if not runs:
        raise SystemExit(f"No training_history.json under: {', '.join(dirs) or 'outputs/recipe_*'}")

    # Validation runs every val_freq epochs, so entry i of a val_* list is a
    # different epoch from entry i of train_loss. Compare at a shared count of
    # validation points, which is the same epoch for all runs of this sweep.
    n_val = min(len(r['history'].get('val_loss', [])) for r in runs)
    if args.at_epoch is not None:
        freq = max(int(r['config'].get('val_freq', 10) or 10) for r in runs)
        n_val = min(n_val, max(1, args.at_epoch // freq))
    if n_val == 0:
        raise SystemExit('No run has completed a validation pass yet.')

    for r in runs:
        h = r['history']
        r['epochs_done'] = len(h.get('train_loss', []))
        r['values'] = {}
        for key, _, _, _ in columns:
            series = h.get(key) or []
            r['values'][key] = series[n_val - 1] if len(series) >= n_val else None
            if key == 'val_pc_emd' and not r['values'][key]:
                r['values'][key] = None      # EMD exists only at the final epoch
        # Best-so-far clean val, the quantity best_model.pth tracks.
        clean = (h.get('val_loss') or [])[:n_val]
        r['best_val'] = min(clean) if clean else None

    present = [c for c in columns if any(r['values'][c[0]] is not None for r in runs)]
    for key, _, lower_better, _ in present:
        vals = [(r['values'][key], r) for r in runs if r['values'][key] is not None]
        if vals:
            winner = (min if lower_better else max)(vals, key=lambda t: t[0])[1]
            winner.setdefault('wins', set()).add(key)

    if args.loss_study:
        header = ['run', 'loss', 'setting', 'epochs'] + [c[1] for c in present]
    else:
        header = (['run', 'blobs', 'noise', 'epochs'] + [c[1] for c in present]
                  + ['best clean val'])
    rows = []
    for r in runs:
        if args.loss_study:
            cells = [r['name'], *describe_loss(r), str(r['epochs_done'])]
        else:
            blob, noise = describe(r)
            cells = [r['name'], blob.split('=')[1], noise.split('=')[1],
                     str(r['epochs_done'])]
        for key, _, _, fmt in present:
            v = r['values'][key]
            cell = '—' if v is None else fmt.format(v)
            if key in r.get('wins', set()):
                cell = f"**{cell}**"
            cells.append(cell)
        if not args.loss_study:
            cells.append('—' if r['best_val'] is None else f"{r['best_val']:.4f}")
        rows.append(cells)

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    def fmt_row(cells):
        return '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'

    freq = runs[0]['config'].get('val_freq', 10) or 10
    lines = [
        f"### {'Loss study' if args.loss_study else 'Recipe sweep'} — compared at "
        f"validation point {n_val} (epoch ~{n_val * freq})",
        '',
        fmt_row(header),
        '| ' + ' | '.join('-' * w for w in widths) + ' |',
        *(fmt_row(row) for row in rows),
        '',
        '**bold** = best in column. Validation inputs are clean; the `val(occ)` / '
        '`occ Chamfer` columns, where present, are the occluded pass older runs logged.',
    ]
    if missing:
        lines += ['', 'Not started / no history yet: ' + ', '.join(missing)]

    text = '\n'.join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text + '\n')
        print(f"\nWritten to {args.output}")


if __name__ == '__main__':
    main()
