#!/usr/bin/env python3
"""Does mixed masking / Sinkhorn help where occlusion is HEAVY, and at the leaf tips?

On Alloy's rendered maize rows the three matched arms (random + QAL, random + QAL + Sinkhorn,
mixed + QAL + Sinkhorn = row_best) tie on overall F1@10mm. Mixed masking exists to teach filling
in behind neighbours and those scenes are lightly occluded (~9 % of the target's pixels hidden),
so the hypothesis is that its benefit sits in the most-occluded plants; on the synthetic scenes
the Sinkhorn gain sat at the leaf tips. This scores every arm on the SAME occluded test scenes,
masks and seeds as eval/card_ensemble.py (no card), PER PLANT, in one invocation (so the rows are
paired on one GPU), and reports:

  overall means (cd_mm, F1 / R / P at 5, 10, 20 mm), per quartile of the share of the target's
  pixels the neighbours hide, recall@10mm in the inner / middle / outer third of the plant's
  radius from the stem (outer = leaf tips), and paired comparisons: share of plants where A beats
  B and a bootstrap 95 % interval of the mean difference, overall and in the top quartile.

Exploratory: report it next to the overall result, never instead of it.

    python eval/occlusion_strata.py --out reports/occlusion_strata_maize_row.json
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'eval'))

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import eval_test_scene as E
import embodied_mae_4m_maize as M4
from maize_row_dataset import build_maize_dataset
from occlusion_scene import SceneConfig, compose, to_world

# name: (run, card passes K -- None = no card; K = its own predicted card averaged over K masks, fed back)
ARMS = {'random': ('maize_row_random_s1', None), 'random_sink': ('maize_row_random_sink_s1', None),
        'mixed_sink': ('maize_row_best_s1', None), 'mixed_sink_fb': ('maize_row_best_fb_s1', None),
        'mixed_sink_fb_card10': ('maize_row_best_fb_s1', 10)}
PAIRS = [('random_sink', 'random'), ('mixed_sink', 'random_sink'), ('mixed_sink', 'random'),
         ('mixed_sink_fb_card10', 'random'), ('mixed_sink_fb_card10', 'mixed_sink_fb')]


def band_recall(pred, tgt, band, mm, t_mm=10.0, chunk=1024):
    """recall@t_mm per radial band (B, 3): share of target points in the band within t_mm of pred."""
    d = torch.full(tgt.shape[:2], float('inf'), device=tgt.device)
    for s in range(0, pred.shape[1], chunk):
        d = torch.minimum(d, torch.cdist(tgt, pred[:, s:s + chunk]).min(2).values)
    hit = (d * mm[:, None] < t_mm).float()
    return torch.stack([(hit * (band == k)).sum(1) / (band == k).sum(1).clamp(min=1) for k in range(3)], 1)


@torch.no_grad()
def score_arm(run, epoch, dev, max_batches, card_k=None):
    rc = json.loads((_REPO / 'outputs' / run / 'config.json').read_text())
    m = E.build_model('maize', rc).to(dev)
    ck = torch.load(_REPO / 'outputs' / run / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth',
                    map_location='cpu', weights_only=False, mmap=True)
    m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
    m.pc_sinkhorn_weight = 0.0
    m.eval()
    scene = SceneConfig.from_dict(rc['occlusion_scene']).for_validation()
    ds = build_maize_dataset(rc, 'test', deterministic_view=True)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, generator=torch.Generator().manual_seed(0))
    rows = {}
    for bi, b in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break
        rgb, depth, pc, pf, tv = (t.to(dev) for t in b[:5])
        pn, c2w, nf = b[6].to(dev), b[7].to(dev), b[8].to(dev)
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16, near_far=nf,
                     generator=torch.Generator().manual_seed(bi), given=b[10])
        _, world = to_world(pc, pn, c2w)
        r = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * r / r.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        cn = sc.get('crop_norm')
        kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': sc.get('target_pc', pc)}, 'loss_tokens': sc['loss_tokens']}
        x = (sc['rgb'], sc['depth'], sc['pc'])
        p_in, tv_in = pf, tv
        if card_k:
            # card_ensemble's two-pass readout: K card predictions (card hidden), averaged, fed back visible
            m.text_mask_ratio = 1.0
            pps = []
            for k in range(card_k):
                torch.manual_seed(40000 + bi + 1_000_003 * k)
                pps.append(m(*x, pf, tv, mask_ratio=0.8, **kw)[2][3])
            pp = torch.stack(pps).mean(0)
            L = tv.shape[1] - 1
            n_hat = (pp[:, 0, 0].clamp(0, 1) * float(M4._PLANT_SCALE[0])).round().clamp(1, L)
            tv_in = torch.cat([torch.ones_like(tv[:, :1]), (torch.arange(L, device=dev)[None] < n_hat[:, None]).to(tv.dtype)], 1)
            p_in = pp * tv_in[..., None]
        m.text_mask_ratio = 0.0 if card_k else 1.0
        torch.manual_seed(30000 + bi)                       # card_ensemble's reconstruction seed
        pr = m(*x, p_in, tv_in, mask_ratio=0.8, **kw)[2][2]
        if cn is not None:
            pr = pr * cn[:, None, 3:4] + cn[:, None, :3]
        s = E.scores(pr, pc, band, pn[:, 3])
        br = band_recall(pr, pc, band, pn[:, 3] * 1000.0)
        for j, name in enumerate(b[5]):
            rows[name] = {k: float(s[k][j]) for k in ('cd_mm', 'f15mm', 'r5mm', 'p5mm', 'f110mm', 'r10mm', 'p10mm', 'f120mm')}
            rows[name].update(hidden=float(b[10]['hidden'][j]), r10_inner=float(br[j, 0]),
                              r10_middle=float(br[j, 1]), r10_outer=float(br[j, 2]))
    return rows


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/occlusion_strata_maize_row.json')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--max-batches', type=int, default=None)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    per = {}
    for arm, (run, k) in ARMS.items():
        per[arm] = score_arm(run, a.epoch, dev, a.max_batches, card_k=k)
        print(f'[{time.time() - t0:.0f}s] {arm}: {len(per[arm])} plants', flush=True)
    names = sorted(set.intersection(*(set(v) for v in per.values())))
    hid = np.array([per['random'][n]['hidden'] for n in names])
    q = np.quantile(hid, [0.25, 0.5, 0.75])
    quart = np.digitize(hid, q)                              # 0 = least occluded ... 3 = most
    keys = ['cd_mm', 'f15mm', 'r5mm', 'p5mm', 'f110mm', 'r10mm', 'p10mm', 'f120mm', 'r10_inner', 'r10_middle', 'r10_outer']
    arr = {arm: {k: np.array([per[arm][n][k] for n in names]) for k in keys} for arm in ARMS}
    out = {'n_plants': len(names), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           'hidden_quartile_edges': q.tolist(), 'hidden_mean_by_quartile': [float(hid[quart == i].mean()) for i in range(4)],
           'overall': {arm: {k: float(v.mean()) for k, v in d.items()} for arm, d in arr.items()},
           'by_quartile': {arm: {k: [float(v[quart == i].mean()) for i in range(4)] for k, v in d.items()} for arm, d in arr.items()},
           'paired': {}}
    for A, B in PAIRS:
        res = {}
        for k in ('f15mm', 'f110mm', 'cd_mm', 'r10_outer'):
            for tag, sel in (('all', np.ones(len(names), bool)), ('most_occluded_q4', quart == 3)):
                d = arr[A][k][sel] - arr[B][k][sel]
                better = (d < 0) if k == 'cd_mm' else (d > 0)
                res[f'{k}_{tag}'] = {'mean_diff': float(d.mean()), 'ci95': boot_ci(d), 'share_A_better': float(better.mean())}
        out['paired'][f'{A}_vs_{B}'] = res
    dst = _REPO / a.out
    tmp = dst.with_suffix('.tmp'); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(dst)
    print(f"\n{out['n_plants']} occluded test plants, {out['gpu']}; hidden share by quartile:",
          [round(x, 3) for x in out['hidden_mean_by_quartile']])
    print(f"{'arm':12s} {'err mm':>6s} {'F1@5':>6s} {'F1@10':>6s} {'R@10':>6s} {'P@10':>6s} {'R10 tips':>8s}")
    for arm, v in out['overall'].items():
        print(f"{arm:12s} {v['cd_mm']:6.2f} {v['f15mm']:6.3f} {v['f110mm']:6.3f} {v['r10mm']:6.3f} {v['p10mm']:6.3f} {v['r10_outer']:8.3f}")
    print('F1@5mm by occlusion quartile (least -> most):')
    for arm, v in out['by_quartile'].items():
        print(f"  {arm:12s}", ' '.join(f'{x:.3f}' for x in v['f15mm']), '| F1@10mm', ' '.join(f'{x:.3f}' for x in v['f110mm']))
    for pair, res in out['paired'].items():
        print(pair)
        for k, v in res.items():
            print(f"  {k:24s} diff {v['mean_diff']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  A better on {v['share_A_better']:.0%}")
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
