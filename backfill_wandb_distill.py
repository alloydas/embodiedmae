"""Backfill a finished/in-progress distillation run into W&B from its
training_history.json (no re-training). Re-runnable: each call makes a fresh run.

  python backfill_wandb_distill.py --run_dir outputs/4m_distill_v1 \
      --project embodied-mae-sorghum --name 4m_crossmodal_distill_v1 \
      --baseline crossmodal_baseline.json
"""
import argparse, json
from pathlib import Path
import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='outputs/4m_distill_v1')
    ap.add_argument('--project', default='embodied-mae-sorghum')
    ap.add_argument('--entity', default=None)
    ap.add_argument('--name', default='4m_crossmodal_distill_v1')
    ap.add_argument('--baseline', default='crossmodal_baseline.json')
    ap.add_argument('--viz_glob', default='visualizations/epoch_*_src-*.png')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    hist = json.load(open(run_dir / 'training_history.json'))
    cfg = {}
    if (run_dir / 'config.json').exists():
        cfg = json.load(open(run_dir / 'config.json'))

    run = wandb.init(project=args.project, entity=args.entity,
                     name=args.name, config=cfg,
                     tags=['crossmodal', 'distillation', 'backfill'])

    # Baseline (zero-shot, pre-distillation) as reference lines at epoch 0.
    if Path(args.baseline).exists():
        base = json.load(open(args.baseline))
        b = {'val/mean_gen': base['mean_gen']}
        for s, d in base['per_source'].items():
            for k, v in d.items():
                b[f'val/{s}/{k}'] = v
        wandb.log({**b, 'epoch': 0}, step=0)
        wandb.run.summary['baseline_mean_gen'] = base['mean_gen']

    train_by_ep = {e['epoch']: e for e in hist.get('train', [])}
    val_by_ep   = {e['epoch']: e for e in hist.get('val', [])}
    for ep in sorted(set(train_by_ep) | set(val_by_ep)):
        log = {'epoch': ep}
        if ep in train_by_ep:
            for k, v in train_by_ep[ep].items():
                if k != 'epoch':
                    log[f'train/{k}'] = v
        if ep in val_by_ep:
            ve = val_by_ep[ep]
            log['val/mean_gen'] = ve['mean_gen']
            for s, d in ve.get('per_source', {}).items():
                for k, v in d.items():
                    log[f'val/{s}/{k}'] = v
        wandb.log(log, step=ep)

    if val_by_ep:
        best = min(val_by_ep.values(), key=lambda e: e['mean_gen'])
        wandb.run.summary['best_mean_gen'] = best['mean_gen']
        wandb.run.summary['best_epoch'] = best['epoch']

    imgs = sorted(run_dir.glob(args.viz_glob))
    if imgs:
        wandb.log({'visualizations': [wandb.Image(str(p), caption=p.name)
                                       for p in imgs[-12:]]})

    print(f"\n✅ W&B run: {run.url}")
    wandb.finish()


if __name__ == '__main__':
    main()
