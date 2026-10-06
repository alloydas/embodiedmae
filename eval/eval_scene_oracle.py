#!/usr/bin/env python3
"""Score occluded-scene arms on occluded val under TWO test-time maskings.

  uniform : the regime every arm is validated under (SceneConfig.for_validation)
            -- nothing at test time says which tokens show a neighbour, so they
            are masked at random. The deployable number; it should land on the
            run's logged val_occ_* to within masking noise.
  oracle  : the tokens that show a neighbour are masked FIRST, exactly as
            neighbour_first training masks them, the rest of the budget at
            random. This assumes a segmentation of the target plant is
            available at inference (e.g. an instance mask). It is the setting
            structured masking was designed for: there training and test match.

Every arm gets both, so "does structured training help once the test masking
matches it?" is a paired comparison against the uniform-trained arm under the
same oracle masking, not against its uniform number.

Same val dataset (view 0 per plant), same 2-rank DistributedSampler shards,
same scene seeds (val_seed * 1000 + rank, then per batch) and the trainer's own
evaluate() as the in-run validation. Val batches are read once per species and
torch is re-seeded before every pass, so arms and maskings see identical
scenes, cloud subsamples and Dirichlet draws. Sorghum and maize runs may be
mixed in --runs (a run is maize when its name starts with maize_).

    python eval/eval_scene_oracle.py --runs sm_scene_s1 sm_scene_struct_s1 \\
        maize_scene_s1 maize_scene_struct_s1 --out reports/scene_oracle.json
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler

import train_sorghum_4m as TS
import train_maize_4m as TM
from occlusion_scene import SceneConfig
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

SHOW = ['pc_chamfer', *TS.PC_FSCORE_KEYS, 'rgb_mse', 'depth_mse']


def is_maize(run):
    return run.startswith('maize_')


def build_model(run, cfg):
    ns = argparse.Namespace(**cfg)
    if not is_maize(run):
        return TS.build_model_from_args(ns)
    fn = {'small': TM.embodied_mae_4m_maize_small, 'base': TM.embodied_mae_4m_maize_base,
          'large': TM.embodied_mae_4m_maize_large}[ns.model_size]
    # mirrors train_maize_4m.train_worker's constructor call
    return fn(active_modalities=ns.active_modalities, img_size=ns.img_size, num_pc_tokens=196,
              target_points=ns.num_points, pc_loss_weight=ns.pc_loss_weight,
              max_leaves=ns.max_leaves, spline_loss_weight=ns.spline_loss_weight,
              depth_norm_type=ns.depth_norm_type, pc_loss_name=ns.pc_loss_name,
              qal_threshold=ns.qal_threshold, qal_alpha=ns.qal_alpha,
              qal_use_squared=ns.qal_use_squared, text_mask_ratio=ns.text_mask_ratio)


def load_val_shards(run, cfg, workers, max_batches=None):
    kw = dict(img_size=cfg['img_size'], num_points=cfg['num_points'], split='val',
              max_leaves=cfg['max_leaves'], view_sampling=cfg['view_sampling'],
              deterministic_view=True, return_pose=True)
    if is_maize(run):
        ds = MaizeDataset4M(cfg['data_root'], **kw)
    else:
        ds = SorghumDataset4M(cfg['data_root'], spline_root=cfg.get('spline_root'), **kw)
    world = cfg['world_size']
    shards = {}
    for r in range(world):
        dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=workers,
                        sampler=DistributedSampler(ds, world, r, shuffle=False))
        shards[r] = []
        for b in dl:
            shards[r].append([t.clone() if torch.is_tensor(t) else t for t in b])
            if max_batches and len(shards[r]) >= max_batches:
                break
        print(f'  rank {r}: {len(shards[r])} batches', flush=True)
    return shards


def score(T, model, shards, device, mask_ratio, scene, seed, oracle):
    """Mean over ranks of evaluate()'s per-rank means = the trainer's all-reduce."""
    per = []
    for r, batches in shards.items():
        torch.manual_seed(seed)
        _, m = T.evaluate(model, batches, device, mask_ratio=mask_ratio, scene=scene,
                          scene_seed=scene.val_seed * 1000 + r, oracle=oracle)
        per.append(m)
    return {k: sum(m[k] for m in per) / len(per) for k in per[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--epoch', type=int, default=200, help='checkpoint_epoch_<N>.pth to score')
    ap.add_argument('--outputs-root', default=str(_REPO / 'outputs'))
    ap.add_argument('--out', required=True, help='JSON, updated in place; scored rows are skipped')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-batches', type=int, default=None,
                    help='smoke test: first N batches per rank (numbers then mean nothing)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['run'], r['epoch']) for r in rows}
    data = {}
    t0 = time.time()
    for run in args.runs:
        if (run, args.epoch) in done:
            print(f'{run}: already scored, skipped', flush=True)
            continue
        rd = Path(args.outputs_root) / run
        ck_path = rd / 'checkpoints' / f'checkpoint_epoch_{args.epoch}.pth'
        if not ck_path.exists():
            print(f'{run}: no {ck_path.name}, skipped', flush=True)
            continue
        cfg = json.loads((rd / 'config.json').read_text())
        scene = SceneConfig.from_dict(cfg['occlusion_scene'])
        T = TM if is_maize(run) else TS
        key = ('maize' if is_maize(run) else 'sorghum', cfg['batch_size'], cfg['world_size'],
               cfg['num_points'])
        if key not in data:
            print(f'[{time.time() - t0:.0f}s] reading {key[0]} val shards', flush=True)
            data[key] = load_val_shards(run, cfg, args.workers, args.max_batches)
        model = build_model(run, cfg).to(device)
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
        del ck
        model.eval()
        row = {'run': run, 'epoch': args.epoch,
               'mask_policy_trained': scene.mask_policy, 'scene_prob_trained': scene.prob}
        for name, oracle in (('uniform', False), ('oracle', True)):
            m = score(T, model, data[key], device, cfg['mask_ratio'], scene, args.seed, oracle)
            row[name] = {k: m[k] for k in SHOW if k in m}
            print(f'[{time.time() - t0:.0f}s] {run} {name}: chamfer {m["pc_chamfer"]:.5f} '
                  f'F1@.01 {m["f1@0.01"]:.3f} P@.03 {m["precision@0.03"]:.3f} '
                  f'R@.03 {m["recall@0.03"]:.3f}', flush=True)
        rows.append(row)
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(out)
        del model
        torch.cuda.empty_cache()
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
