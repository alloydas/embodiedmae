#!/usr/bin/env python3
"""Does a BETTER predicted card help the maize model rebuild an occluded plant?

eval/pc_generation.py feeds the model its own predicted card (one pass, mask seed 40000+bi)
and finds it barely helps (deploy: no card 0.446, + predicted card 0.444, + TRUE card 0.521
F1@10mm). Each pass sees a different 20 % of the scene, so the card can be predicted K times
under K masks and averaged. This scores, on the same occluded test scenes, loader order and
seeds as pc_generation (so 'no card', 'predicted card K=1' and 'true card' reproduce it):

  no card                 card hidden (text_mask_ratio 1.0)
  predicted card, K=k     the card averaged over k passes (normalised params; leaf count
                          read off the averaged plant token), fed back fully visible
  true card               the true card inside the Dirichlet budget, as trained

and how good each averaged card is: leaf-count exact share and |error| in leaves, and the mean
absolute error of the normalised params on tokens real in both cards.

    python eval/card_ensemble.py --run maize_scene_anyparam_deploy_s1 --epoch 200 \
        --out reports/card_ensemble_maize_deploy.json
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'eval'))

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

import eval_test_scene as E
import embodied_mae_4m_maize as M4
from maize_dataset_4m import MaizeDataset4M
from maize_row_dataset import build_maize_dataset
from occlusion_scene import SceneConfig, compose, to_world

KEYS = ('cd_mm', 'f15mm', 'r5mm', 'p5mm', 'f110mm', 'r10mm', 'p10mm', 'f120mm')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5, 10])
    ap.add_argument('--max-batches', type=int, default=None)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    rc = json.loads((_REPO / 'outputs' / a.run / 'config.json').read_text())
    m = E.build_model('maize', rc).to(dev)
    ck = torch.load(_REPO / 'outputs' / a.run / 'checkpoints' / f'checkpoint_epoch_{a.epoch}.pth',
                    map_location='cpu', weights_only=False, mmap=True)
    m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
    m.pc_sinkhorn_weight = 0.0
    m.eval()
    scene = SceneConfig.from_dict(rc['occlusion_scene']).for_validation()
    L = int(rc['max_leaves'])
    ds = build_maize_dataset(rc, 'test', deterministic_view=True) if rc.get('dataset') == 'maize_row' else MaizeDataset4M(rc['data_root'], img_size=rc['img_size'], num_points=rc['num_points'], split='test',
                        max_leaves=L, view_sampling=rc['view_sampling'], deterministic_view=True, return_pose=True,
                        return_visibility=scene.visible_only, depth_planes=rc.get('depth_planes'))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, generator=torch.Generator().manual_seed(0))
    K = max(a.ks)
    acc, card = {}, {}
    for bi, b in enumerate(dl):
        if a.max_batches is not None and bi >= a.max_batches:
            break
        rgb, depth, pc, pf, tv = (t.to(dev) for t in b[:5])
        pn, c2w, nf = b[6].to(dev), b[7].to(dev), b[8].to(dev)
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16, near_far=nf,
                     generator=torch.Generator().manual_seed(bi), vis=b[9].to(dev) if len(b) > 9 else None,
                     given=b[10] if len(b) > 10 else None)
        _, world = to_world(pc, pn, c2w)
        r = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * r / r.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        x = (sc['rgb'], sc['depth'], sc['pc'])
        kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': sc.get('target_pc', pc)}, 'loss_tokens': sc['loss_tokens']}
        cn = sc.get('crop_norm')
        back = (lambda q: q) if cn is None else (lambda q: q * cn[:, None, 3:4] + cn[:, None, :3])

        # K card predictions, card hidden; pass 0 uses pc_generation's seed, so K=1 reproduces it
        m.text_mask_ratio = 1.0
        pps = []
        for k in range(K):
            torch.manual_seed(40000 + bi + 1_000_003 * k)
            pps.append(m(*x, pf, tv, mask_ratio=0.8, **kw)[2][3])

        def run(p_in, tv_in, tmr):
            m.text_mask_ratio = tmr
            torch.manual_seed(30000 + bi)
            return back(m(*x, p_in, tv_in, mask_ratio=0.8, **kw)[2][2])

        preds = {'no card': run(pf, tv, 1.0), 'true card': run(pf, tv, None)}
        n_true = tv[:, 1:].sum(1).round().long()
        for k in a.ks:
            pp = torch.stack(pps[:k]).mean(0)
            n_hat = (pp[:, 0, 0].clamp(0, 1) * float(M4._PLANT_SCALE[0])).round().clamp(1, L).long()
            tv_h = torch.zeros_like(tv); tv_h[:, 0] = 1
            tv_h[:, 1:] = (torch.arange(L, device=dev)[None] < n_hat[:, None]).float()
            ph = pp * tv_h[..., None]
            preds[f'predicted card, K={k}'] = run(ph, tv_h, 0.0)
            both = (tv * tv_h)[..., None]
            c = card.setdefault(k, {'exact': [], 'leaf_err': [], 'param_mae': []})
            c['exact'].append((n_hat == n_true).float().cpu())
            c['leaf_err'].append((n_hat - n_true).abs().float().cpu())
            c['param_mae'].append((((ph - pf).abs() * both).sum((1, 2)) / (both.sum((1, 2)) * pf.shape[-1]).clamp(min=1)).cpu())
        for cond, pr in preds.items():
            s = E.scores(pr, pc, band, pn[:, 3])
            for key in KEYS:
                acc.setdefault(cond, {}).setdefault(key, []).append(s[key].float().cpu())
        if bi % 20 == 0:
            print(f'[{time.time() - t0:.0f}s] batch {bi}', flush=True)

    per = {c: {k: torch.cat(v) for k, v in d.items()} for c, d in acc.items()}
    means = {c: {k: float(v.mean()) for k, v in d.items()} for c, d in per.items()}
    cards = {k: {m_: float(torch.cat(v).mean()) for m_, v in d.items()} for k, d in card.items()}
    ref = per['no card']['f110mm']
    better = {c: float((per[c]['f110mm'] > ref).float().mean()) for c in per if c != 'no card'}
    out = {'run': a.run, 'epoch': a.epoch, 'n_plants': int(ref.numel()),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           'means': means, 'share_better_than_no_card': better, 'card_quality': cards}
    dst = _REPO / a.out
    tmp = dst.with_suffix('.tmp'); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(dst)
    print(f'\n{a.run} ep{a.epoch}, occluded maize test ({out["n_plants"]} plants):')
    print(f'{"condition":24s} err mm  F1@5   F1@10  R@10   P@10   F1@20  >no-card')
    for c, v in means.items():
        bt = better.get(c)
        print(f"{c:24s} {v['cd_mm']:6.1f}  {v['f15mm']:.3f}  {v['f110mm']:.3f}  {v['r10mm']:.3f}  {v['p10mm']:.3f}  {v['f120mm']:.3f}"
              + ('' if bt is None else f'  {bt:.0%}'))
    print('card quality (K: leaf count exact / |err| leaves / param MAE):')
    for k, v in cards.items():
        print(f"  K={k:<3d} {v['exact']:.3f} / {v['leaf_err']:.2f} / {v['param_mae']:.4f}")
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
