#!/usr/bin/env python3
"""Occluded TEST scenes (maize or sorghum): what point cloud does an any-to-any model (E) build
from (a) no recipe card, (b) the scene + its OWN predicted card, (c) the scene + the
true card (inside the Dirichlet budget, as trained), (d) its predicted card ONLY (no
photo / depth / cloud: the plant generated from the guessed recipe), (e) the true card
only. All 2,250 test plants are scored (mm); three example plants -- the same ones as
reports/test_scene/maize_mm.json (picked by occlusion) -- are saved with their clouds,
per-point errors and the true / predicted card in physical units. Same scenes, masks and
loader order as eval/eval_test_scene.py, so its reports/test_scene/<species>_mm.json means
are directly comparable.

The predicted card is fed back with a predicted leaf count: maize reads it off the
plant token (leafCount); sorghum has no such field, but its generator sets
stem_length = 0.05 x n_leaves - 0.001, so it is read off the predicted stem length.

    python eval/pc_generation.py      # maize E -> reports/pc_generation_maize.json
    python eval/pc_generation.py --run maize_scene_anyparam_card_s1 --out reports/pc_generation_maize_card.json
    python eval/pc_generation.py --species sorghum --run sm_scene_anyparam_pre_s1 --out reports/pc_generation_sorghum.json

A run trained with occlusion_scene.norm: crop is fed crop-frame clouds and its
predictions are mapped back to the target's frame before scoring.
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
import embodied_mae_4m as S4
import embodied_mae_4m_maize as M4
from maize_dataset_4m import MaizeDataset4M
from maize_row_dataset import build_maize_dataset
from sorghum_dataset_4m import SorghumDataset4M
from occlusion_scene import SceneConfig, compose, to_world, no_scene_frame

RUN, EP = 'maize_scene_anyparam_pre_s1', 200    # defaults; see --run / --epoch
CONDS = ['no card', 'scene + predicted card', 'scene + true card', 'predicted card only', 'true card only',
         'no card, true centre + size', 'no card, true centre']
N_SHOW = 2400


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--species', choices=('maize', 'sorghum'), default='maize')
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--epoch', type=int, default=EP)
    ap.add_argument('--out', default='reports/pc_generation_maize.json')
    a = ap.parse_args()
    run_name, ep, sp = a.run, a.epoch, a.species
    examples_of = [e['name'] for e in json.loads((_REPO / f'reports/test_scene/{sp}_mm.json').read_text())['examples']]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    rc = json.loads((_REPO / 'outputs' / run_name / 'config.json').read_text())
    m = E.build_model(sp, rc).to(dev)
    ck = torch.load(_REPO / 'outputs' / run_name / 'checkpoints' / f'checkpoint_epoch_{ep}.pth', map_location='cpu', weights_only=False, mmap=True)
    m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
    m.pc_sinkhorn_weight = 0.0
    m.eval()
    scene = SceneConfig.from_dict(rc['occlusion_scene']).for_validation()
    L = int(rc['max_leaves'])
    if sp == 'maize':
        ds = build_maize_dataset(rc, 'test', deterministic_view=True) if rc.get('dataset') == 'maize_row' else MaizeDataset4M(rc['data_root'], img_size=rc['img_size'], num_points=rc['num_points'], split='test', max_leaves=L,
                            view_sampling=rc['view_sampling'], deterministic_view=True, return_pose=True,
                            return_visibility=scene.visible_only, depth_planes=rc.get('depth_planes'))
    else:
        ds = SorghumDataset4M(rc['data_root'], img_size=rc['img_size'], num_points=rc['num_points'], split='test', max_leaves=L,
                              view_sampling=rc['view_sampling'], deterministic_view=True, return_pose=True,
                              spline_root=rc.get('spline_root'), return_visibility=scene.visible_only)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, generator=torch.Generator().manual_seed(0))
    acc = {c: {} for c in CONDS}
    examples = {}
    rng = np.random.default_rng(0)
    for bi, b in enumerate(dl):
        rgb, depth, pc, pf, tv = (t.to(dev) for t in b[:5])
        names = list(b[5])
        pn, c2w = b[6].to(dev), b[7].to(dev)
        nf = b[8].to(dev) if sp == 'maize' else None      # sorghum: fixed render planes
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16, near_far=nf, generator=torch.Generator().manual_seed(bi),
                     vis=b[9].to(dev) if len(b) > 9 else None,
                     given=b[10] if len(b) > 10 else None)
        _, world = to_world(pc, pn, c2w)
        r = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * r / r.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        x = (sc['rgb'], sc['depth'], sc['pc'])
        kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': sc.get('target_pc', pc)}, 'loss_tokens': sc['loss_tokens']}
        cn = sc.get('crop_norm')       # norm 'crop' / 'metric' / 'stem': back to the target's frame for scoring
        back = (lambda q: q) if cn is None else (lambda q: q * cn[:, None, 3:4] + cn[:, None, :3])
        # card only sees no scene, so a 'metric' / 'stem' run predicts in the frame its clean
        # training samples used (own centroid / stem base, metres): undo that frame
        if scene.norm in ('metric', 'stem'):
            cn_c = no_scene_frame(pc, pn, c2w, scene)[1]
            back_card = lambda q: q * cn_c[:, None, 3:4] + cn_c[:, None, :3]
        else:
            back_card = lambda q: q

        # pass 1: predict the card from the scene, no card given
        m.text_mask_ratio = 1.0
        torch.manual_seed(40000 + bi)
        pp = m(*x, pf, tv, mask_ratio=0.8, **kw)[2][3]
        if sp == 'maize':
            n_hat = pp[:, 0, 0].clamp(0, 1) * float(M4._PLANT_SCALE[0])
        else:
            n_hat = (pp[:, 0, 0].clamp(0, 1) * float(S4._PLANT_SCALE[0]) + 0.001) / 0.05
        n_hat = n_hat.round().clamp(1, L).long()
        tv_h = torch.zeros_like(tv); tv_h[:, 0] = 1
        tv_h[:, 1:] = (torch.arange(L, device=dev)[None] < n_hat[:, None]).float()
        ph = pp * tv_h[..., None]

        def run(p_in, tv_in, tmr=None, visible=None):
            m.text_mask_ratio = tmr
            torch.manual_seed(30000 + bi)
            if visible is None:
                return back(m(*x, p_in, tv_in, mask_ratio=0.8, **kw)[2][2])
            # card only: no scene token is seen, so there is no crop frame to undo
            return back_card(m(*x, p_in, tv_in, visible=visible, source_mask_ratio=0.0, targets=kw['targets'])[2][2])

        preds = {'no card': run(pf, tv, 1.0), 'scene + predicted card': run(ph, tv_h, 0.0),
                 'scene + true card': run(pf, tv, None),
                 'predicted card only': run(ph, tv_h, None, visible={'text'}),
                 'true card only': run(pf, tv, None, visible={'text'})}
        # DIAGNOSTIC, uses the answer: the no-card prediction moved and rescaled so its
        # centroid and RMS radius match the target's. The gap to 'no card' is error in
        # WHERE / HOW BIG the target is, as opposed to its shape (matters for norm: crop).
        p0 = preds['no card']
        pcen, tcen = p0.mean(1, keepdim=True), pc.mean(1, keepdim=True)
        rms = lambda q, c: (q - c).pow(2).sum(-1).mean(1).sqrt()
        preds['no card, true centre + size'] = (p0 - pcen) * (rms(pc, tcen) / rms(p0, pcen).clamp(min=1e-6))[:, None, None] + tcen
        # translation only: the no-card prediction shifted so its centroid sits on the
        # target's (no rescaling, which the row above showed is harmful)
        preds['no card, true centre'] = p0 - pcen + tcen
        for c, pr in preds.items():
            s = E.scores(pr, pc, band, pn[:, 3])
            for k in ('cd_mm', 'f110mm', 'r10mm', 'p10mm', 'f120mm'):
                acc[c].setdefault(k, []).append(s[k])
        for j, nm in enumerate(names):
            if nm not in examples_of:
                continue
            sel_t = rng.choice(pc.shape[1], N_SHOW, replace=False)
            tgt = pc[j][sel_t]
            ex = {'name': nm, 'radius_m': round(float(pn[j, 3]), 4),
                  'hidden_px': round(float(((sc['shown'][j, 0]) & (depth[j, 0] > 0)).float().sum() / (depth[j, 0] > 0).float().sum().clamp(min=1)), 4),
                  'rgb_scene': E.rgb_png(sc['rgb'][j]),
                  'pc_in': E.r3(back(sc['pc'])[j][rng.choice(sc['pc'].shape[1], N_SHOW, replace=False)].cpu()),
                  'target': E.r3(tgt.cpu()),
                  'card_true': E.phys_params(sp, pf[j], tv[j]), 'card_pred': E.phys_params(sp, ph[j], tv_h[j]),
                  'pred': {}, 'nn_dist': {}, 'scores': {}}
            for c, pr in preds.items():
                p = pr[j]
                ex['pred'][c] = E.r3(p[rng.choice(p.shape[0], N_SHOW, replace=False)].cpu())
                ex['nn_dist'][c] = np.round(torch.cdist(tgt[None], p[None])[0].min(dim=1).values.cpu().numpy(), 3).tolist()
                s1 = E.scores(pr[j:j + 1], pc[j:j + 1], band[j:j + 1], pn[j:j + 1, 3])
                ex['scores'][c] = {k: round(float(s1[k][0]), 4) for k in ('cd_mm', 'f110mm', 'r10mm', 'p10mm')}
            examples[nm] = ex
        if bi % 20 == 0:
            print(f'[{time.time() - t0:.0f}s] batch {bi}', flush=True)
    out = {'species': sp, 'run': run_name, 'epoch': ep,
           'means': {c: {k: float(torch.cat([v.reshape(-1) for v in d[k]]).mean()) for k in d} for c, d in acc.items()},
           'examples': [examples[n] for n in examples_of if n in examples]}
    dst = _REPO / a.out
    dst.write_text(json.dumps(out, separators=(',', ':')))
    print(f'\n{run_name} on occluded {sp} test: error mm / F1@10mm / R@10mm / P@10mm / F1@20mm')
    for c, v in out['means'].items():
        print(f"{c:26s} {v['cd_mm']:6.1f}  {v['f110mm']:.3f}  {v['r10mm']:.3f}  {v['p10mm']:.3f}  {v['f120mm']:.3f}")
    for ex in out['examples']:
        print(ex['name'], 'true', ex['card_true'], '\n   pred', ex['card_pred'])
    print(f'wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
