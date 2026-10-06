#!/usr/bin/env python3
"""Score every saved checkpoint of the occluded-scene arms on clean AND occluded val.

The trainer's in-run validation only gained point-cloud F1 / precision / recall
(at 0.01 / 0.02 / 0.03) on 2026-10-01, after sm_scene_s1, sm_scene_off_s1 and
sm_scene_noparam_s1 had already logged most of their val points. This backfills
them -- and scores every arm with ONE code path, so the arms are comparable
however their own runs were interrupted or resumed.

It reproduces the trainer's validation exactly: the same val dataset (view 0
per plant), the same 2-rank DistributedSampler shards, the same scene seeds
(val_seed * 1000 + rank, then per batch), and train_sorghum_4m.evaluate()
itself, so `pc_chamfer` here should land on the run's logged val_pc_chamfer /
val_occ_pc_chamfer to within masking noise (which this script prints).

Two things are held fixed ACROSS checkpoints and arms, which the trainer does
not do: the val batches are read once and kept in memory (so every checkpoint
sees the same cloud subsample -- load_pointcloud's is unseeded), and torch is
re-seeded before every pass (so the Dirichlet split and token shuffles are the
same). Differences between checkpoints or arms are therefore paired.

Incremental: rows already in --out are skipped, so re-running after an arm has
saved more checkpoints scores only the new ones.

--wandb-runs R ...: after scoring, resume each named run's own W&B run (the id
its checkpoints carry) and log the rows under backfill/val/* and
backfill/val_occ/* against `epoch` (plot them on an epoch x-axis: they land
after the run's last step), plus clean and occluded-scene figures from its last
checkpoint under backfill/visualizations and backfill/scene_visualizations.
Only for a run that has FINISHED -- W&B cannot resume a run that is still
logging. Every row of the run in --out is logged, so do it once per run.

    python eval/eval_scene_ckpts.py --runs sm_scene_s1 sm_scene_off_s1 \\
        sm_scene_noparam_s1 --out reports/scene_ckpts.json
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler

import train_sorghum_4m as T
from occlusion_scene import SceneConfig
from sorghum_dataset_4m import SorghumDataset4M

SHOW = ['pc_chamfer', *T.PC_FSCORE_KEYS]


def load_val_batches(cfg, world, batch_size, workers, max_batches=None):
    """The trainer's val shards, read once: {rank: [batch, ...]}."""
    ds = SorghumDataset4M(cfg['data_root'], img_size=cfg['img_size'],
                          num_points=cfg['num_points'], split='val',
                          max_leaves=cfg['max_leaves'],
                          view_sampling=cfg['view_sampling'], deterministic_view=True,
                          return_pose=True, spline_root=cfg.get('spline_root'))
    shards = {}
    for r in range(world):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        sampler=DistributedSampler(ds, world, r, shuffle=False))
        # clone: a worker's tensors arrive in shared memory, one file descriptor
        # each, and ~1,100 of them held at once can exhaust the fd limit.
        shards[r] = []
        for b in dl:
            shards[r].append([t.clone() if torch.is_tensor(t) else t for t in b])
            if max_batches and len(shards[r]) >= max_batches:
                break
        print(f'  rank {r}: {len(shards[r])} batches', flush=True)
    return shards


def data_key(cfg):
    """Arms whose val batches are interchangeable share one in-memory copy."""
    return (cfg['data_root'], cfg['img_size'], cfg['num_points'], cfg['max_leaves'],
            cfg['view_sampling'], cfg.get('spline_root'), cfg['batch_size'], cfg['world_size'])


def score(model, shards, device, mask_ratio, scene, seed):
    """Mean over ranks of evaluate()'s per-rank means = the trainer's all-reduce
    (every DistributedSampler shard has the same number of batches)."""
    per = []
    for r, batches in shards.items():
        torch.manual_seed(seed)
        _, m = T.evaluate(model, batches, device, mask_ratio=mask_ratio,
                          scene=scene, scene_seed=(scene.val_seed * 1000 + r) if scene else 0)
        per.append(m)
    return {k: sum(m[k] for m in per) / len(per) for k in per[0]}


def logged(hist, key, epoch, epochs_key=None, val_freq=None, total=None):
    """The run's own logged value at `epoch`, or None."""
    s = hist.get(key) or []
    if epochs_key:
        eps = hist.get(epochs_key) or []
    else:   # clean val: epoch 1, every val_freq, and the last epoch
        eps = sorted({1, total, *range(val_freq, total + 1, val_freq)})[:len(s)]
    return s[eps.index(epoch)] if epoch in eps and eps.index(epoch) < len(s) else None


def log_wandb(run, rd, cfg, scene, rows, model, shards, device, args):
    """Backfilled rows + last-checkpoint figures into the run's own W&B run."""
    import wandb
    last = max(r['epoch'] for r in rows)
    ck = torch.load(rd / 'checkpoints' / f'checkpoint_epoch_{last}.pth',
                    map_location='cpu', weights_only=False)
    run_id = ck.get('wandb_run_id')
    if not run_id:
        print(f'{run}: no wandb_run_id in its checkpoint; not logged')
        return
    model.load_state_dict({k.replace('module.', '', 1): v
                           for k, v in ck['model_state_dict'].items()})
    del ck
    wandb.init(project=cfg['wandb_project'], entity=cfg.get('wandb_entity'),
               id=run_id, resume='must')
    wandb.define_metric('epoch')
    wandb.define_metric('backfill/*', step_metric='epoch')
    for r in rows:
        wandb.log({'epoch': r['epoch'],
                   **{f'backfill/val/{k}': v for k, v in r['clean'].items()},
                   **{f'backfill/val_occ/{k}': v for k, v in r['occ'].items()}})
    viz = Path(rd) / 'visualizations'
    viz.mkdir(exist_ok=True)
    first = shards[0]
    figs = {'backfill/visualizations': T.visualize_scene_4m(
                model, first, device, last, viz, args.num_viz,
                mask_ratio=cfg['mask_ratio'], tag='clean'),
            'backfill/scene_visualizations': T.visualize_scene_4m(
                model, first, device, last, viz, args.num_viz,
                mask_ratio=cfg['mask_ratio'], scene=scene, seed=scene.val_seed,
                tag='scene')}
    wandb.log({'epoch': last, **{k: [wandb.Image(p, caption=Path(p).name) for p in v]
                                 for k, v in figs.items() if v}})
    wandb.finish()
    print(f'{run}: {len(rows)} rows + {sum(map(len, figs.values()))} figures -> W&B {run_id}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='names under --outputs-root')
    ap.add_argument('--outputs-root', default=str(_REPO / 'outputs'))
    ap.add_argument('--out', required=True, help='JSON, updated in place')
    ap.add_argument('--epochs', default=None, help='comma list; default every checkpoint')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wandb-runs', nargs='*', default=[],
                    help='finished runs to log the backfill into (see above)')
    ap.add_argument('--num-viz', type=int, default=6)
    ap.add_argument('--max-batches', type=int, default=None,
                    help='smoke test: first N batches per rank (numbers then mean nothing)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['run'], r['epoch']) for r in rows}
    want = {int(e) for e in args.epochs.split(',')} if args.epochs else None
    data = {}
    t0 = time.time()

    for run in args.wandb_runs:
        assert run in args.runs, f'--wandb-runs {run} is not in --runs'

    for run in args.runs:
        rd = Path(args.outputs_root) / run
        cfg = json.loads((rd / 'config.json').read_text())
        scene = SceneConfig.from_dict(cfg['occlusion_scene'])
        ckpts = sorted((int(re.search(r'_(\d+)\.pth$', p.name).group(1)), p)
                       for p in (rd / 'checkpoints').glob('checkpoint_epoch_*.pth'))
        todo = [(e, p) for e, p in ckpts
                if (run, e) not in done and (want is None or e in want)]
        print(f'{run}: {len(ckpts)} checkpoints, {len(todo)} to score', flush=True)
        if not todo and run not in args.wandb_runs:
            continue
        key = data_key(cfg)
        if key not in data:
            print(f'[{time.time() - t0:.0f}s] reading val shards', flush=True)
            data[key] = load_val_batches(cfg, cfg['world_size'], cfg['batch_size'],
                                         args.workers, args.max_batches)
        shards = data[key]
        try:
            hist = json.loads((rd / 'training_history.json').read_text())
        except Exception:
            hist = {}
        model = T.build_model_from_args(argparse.Namespace(**cfg)).to(device)
        for epoch, path in todo:
            try:    # a live run saves non-atomically; its newest file may be mid-write
                ck = torch.load(path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f'{run} ep {epoch}: unreadable ({type(e).__name__}), skipped; '
                      f'a later re-run picks it up', flush=True)
                continue
            model.load_state_dict({k.replace('module.', '', 1): v
                                   for k, v in ck['model_state_dict'].items()})
            del ck
            model.eval()
            clean = score(model, shards, device, cfg['mask_ratio'], None, args.seed)
            occ = score(model, shards, device, cfg['mask_ratio'], scene, args.seed)
            row = {'run': run, 'epoch': epoch,
                   'clean': {k: clean[k] for k in SHOW},
                   'occ': {k: occ[k] for k in SHOW + ['occ_hidden_px', 'occ_nb_points']},
                   'logged_val_pc_chamfer': logged(
                       hist, 'val_pc_chamfer', epoch, val_freq=cfg['val_freq'],
                       total=cfg['epochs']),
                   'logged_val_occ_pc_chamfer': logged(
                       hist, 'val_occ_pc_chamfer', epoch, epochs_key='val_occ_epoch')}
            rows.append(row)
            rows.sort(key=lambda r: (r['run'], r['epoch']))
            tmp = out.with_name(out.name + f'.tmp{os.getpid()}')
            tmp.write_text(json.dumps(rows, indent=1))
            os.replace(tmp, out)
            print(f'[{time.time() - t0:.0f}s] {run} ep {epoch}: clean chamfer '
                  f'{clean["pc_chamfer"]:.6f} (logged {row["logged_val_pc_chamfer"]})  '
                  f'occ chamfer {occ["pc_chamfer"]:.6f} '
                  f'(logged {row["logged_val_occ_pc_chamfer"]})', flush=True)
        if run in args.wandb_runs:
            log_wandb(run, rd, cfg, scene, [r for r in rows if r['run'] == run],
                      model, shards, device, args)
        del model
        torch.cuda.empty_cache()

    # The table: every arm, every checkpoint, clean then occluded.
    hdr = f"{'run':<22}{'ep':>5} {'input':<6}{'chamfer':>10}" + ''.join(
        f"{'F1/P/R@' + str(t):>24}" for t in T.PC_THRESHOLDS)
    print('\n' + hdr + '\n' + '-' * len(hdr))
    for r in rows:
        if r['run'] not in args.runs:
            continue
        for lv in ('clean', 'occ'):
            m = r[lv]
            print(f"{r['run']:<22}{r['epoch']:>5} {lv:<6}{m['pc_chamfer']:>10.6f}" + ''.join(
                f"{m[f'f1@{t}']:.4f}/{m[f'precision@{t}']:.4f}/{m[f'recall@{t}']:.4f}".rjust(24)
                for t in T.PC_THRESHOLDS))
    print(f'\nwrote {out}  [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
