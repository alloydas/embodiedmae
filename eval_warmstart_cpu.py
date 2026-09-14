"""Epoch-0 cross-modal baseline for the ->params arms, on CPU.

Same measurement as eval_warmstart.py -- the undistilled warm start scored on
the subset the runs validated on -- but sharded so it reports progress, writes
partial results as it goes, and survives being cut short. CPU because both GPU
queues are days deep and the model has no CUDA-only ops.

Faithfulness: this calls the trainer's own `evaluate_crossmodal` unmodified, on
consecutive shards of the SAME first `max_batches` batches. Each shard holds an
identical number of full batches, so the plain mean of the shard means is the
mean over all batches -- the exact quantity the runs logged. No reimplementation
of the metric, which is the part that would be easy to get subtly wrong.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import train_sorghum_4m_distill as DIS
from sorghum_dataset_4m import SorghumDataset4M

FIELDS = ('total', 'rgb', 'depth', 'pc_chamfer', 'param_mae_masked',
          'pc_loss', 'text_loss')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint',
                    default='./outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth')
    ap.add_argument('--sources', default='rgb,pc')
    ap.add_argument('--shards', type=int, default=10)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)

    cfg = yaml.safe_load(open(a.config))
    args = DIS.config_to_namespace(cfg)
    bs, nb = args.batch_size, args.val_max_batches
    assert nb % a.shards == 0, f'{nb} batches does not divide into {a.shards} shards'
    per_shard = nb // a.shards                      # batches per shard
    n_samples = nb * bs

    ds = SorghumDataset4M(f"{cfg['data']['data_root']}/val",
                          img_size=args.img_size, num_points=args.num_points)
    print(f"val: {len(ds)} samples; scoring the first {n_samples} "
          f"({nb} batches of {bs}) in {a.shards} shards of {per_shard}", flush=True)

    model = DIS.build_model(args, 'cpu')
    DIS.load_weights_into(model, a.checkpoint, 'cpu', 'warmstart')
    model.eval()

    sources = a.sources.split(',')
    shard_vals = {s: [] for s in sources}
    out = Path(a.out)
    t0 = time.time()

    for sh in range(a.shards):
        lo = sh * per_shard * bs
        loader = DataLoader(Subset(ds, range(lo, lo + per_shard * bs)),
                            batch_size=bs, shuffle=False, num_workers=4)
        for s in sources:
            per, _ = DIS.evaluate_crossmodal(model, loader, 'cpu', [s],
                                             target=args.targets,
                                             max_batches=per_shard)
            shard_vals[s].append({k: per[s][k] for k in FIELDS})
            run = {k: float(np.mean([d[k] for d in shard_vals[s]])) for k in FIELDS}
            el = time.time() - t0
            done = sh * len(sources) + sources.index(s) + 1
            tot = a.shards * len(sources)
            print(f"[shard {sh+1}/{a.shards} src={s}] "
                  f"param_mae {per[s]['param_mae_masked']:.5f} "
                  f"(running {run['param_mae_masked']:.5f})  "
                  f"chamfer {per[s]['pc_chamfer']:.5f}  "
                  f"{el/60:.1f} min elapsed, ~{el/done*(tot-done)/60:.0f} min left",
                  flush=True)

        # Partial result after every shard: a run cut short is still usable, and
        # the shard spread shows how stable the number is.
        res = {'checkpoint': a.checkpoint, 'config': a.config,
               'batch_size': bs, 'batches_total': nb, 'shards_done': sh + 1,
               'samples_scored': (sh + 1) * per_shard * bs,
               'per_source': {s: {k: float(np.mean([d[k] for d in shard_vals[s]]))
                                  for k in FIELDS} for s in sources},
               'per_source_shard_std': {
                   s: {k: float(np.std([d[k] for d in shard_vals[s]]))
                       for k in FIELDS} for s in sources},
               'shards': shard_vals}
        res['mean_gen'] = float(np.mean(
            [res['per_source'][s]['param_mae_masked'] for s in sources]))
        json.dump(res, open(out, 'w'), indent=1)

    print("\n=== EPOCH 0 (warm start, no distillation) ===")
    for s in sources:
        p = res['per_source'][s]
        print(f"  src={s:<4} total {p['total']:.4f}  rgb {p['rgb']:.4f}  "
              f"depth {p['depth']:.4f}  pc_chamfer {p['pc_chamfer']:.5f}  "
              f"param_mae {p['param_mae_masked']:.5f} "
              f"(shard sd {res['per_source_shard_std'][s]['param_mae_masked']:.5f})")
    print(f"  mean over sources: {res['mean_gen']:.5f}")
    print(f"wrote {out}")
    print("EVAL_DONE")


if __name__ == '__main__':
    main()
