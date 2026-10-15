#!/usr/bin/env python3
"""3D reconstructions of representative occluded TEST plants, for the update page.

Three plants per species are picked from the per-plant scores in
reports/test_scene/<species>_mm.json (random 400 vs mixed + Sinkhorn 400, occluded
F1@10mm): a TYPICAL plant (both F1 and the gain near their medians), a CLEAR WIN
(gain near the 85th percentile) and a typical plant among those where RANDOM WINS
(gain near the median of the negative gains). Each is rebuilt exactly as eval/eval_test_scene.py scored it (same
seeded loader, scene seed = batch index, mask seed 30000 + batch index), and three
models reconstruct it: random 400, mixed + Sinkhorn 400 (both given the true
parameters), and the any-to-any 400 model given NO parameters.

Everything is exported in WORLD coordinates (metres, stem base at the origin, y up), so
plants stand upright whatever the camera. Per model: a display subsample of the
predicted points with each point's distance to the FULL true plant (the precision view)
and whether it is stacked (another predicted point within 0.001 of the unit-sphere
cloud, the dup0.001 rule); per true-plant display point, the distance to the FULL
prediction (the recall view); and the plant's scores.

    python export/export_recon_3d.py --species maize sorghum   # -> reports/recon_3d.json
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
from occlusion_scene import SceneConfig, compose, to_world
from structured_mask import foreground
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

MODELS = {
    'sorghum': [('random400', 'sm_scene400_s1', 400, None), ('ours400', 'sm_scene_mix_sink400_s1', 400, None),
                ('noparam400', 'sm_scene_anyparam_pre400_s1', 400, 1.0)],
    'maize': [('random400', 'maize_scene400_s1', 400, None), ('ours400', 'maize_scene_mix_sink400_s1', 400, None),
              ('noparam400', 'maize_scene_anyparam_pre400_s1', 400, 1.0)],
}
N_SHOW = 2500


def pick(sp):
    d = json.loads((_REPO / 'reports' / 'test_scene' / f'{sp}_mm.json').read_text())
    o = np.array(d['per_plant']['mixed_sink400']['occluded']['f110mm'])
    r = np.array(d['per_plant']['random400']['occluded']['f110mm'])
    hid = np.array(d['hidden_px'])
    g = o - r
    ok = (hid >= np.percentile(hid, 25)) & (hid <= np.percentile(hid, 75))    # ordinary occlusion
    sd_o, sd_g = o.std(), g.std()
    out = []
    for label, target_g, target_o in (('typical', np.median(g), np.median(o)),
                                      ('clear win', np.percentile(g, 85), np.median(o)),
                                      ('random wins', np.median(g[g < 0]), np.median(r))):
        cost = np.abs(g - target_g) / sd_g + np.abs(o - target_o) / sd_o + 1e3 * ~ok
        if label == 'random wins':
            cost += 1e3 * (g >= 0)
        i = int(np.argmin(cost))
        out.append((label, i, float(o[i]), float(r[i])))
    meta = {'median_gain': float(np.median(g)), 'median_loss': float(np.median(g[g < 0])),
            'p85_gain': float(np.percentile(g, 85)), 'share_ours_better': float((g > 0).mean())}
    return out, meta


@torch.no_grad()
def run_species(sp, dev, workers):
    t0 = time.time()
    picks, meta = pick(sp)
    print(sp, 'picks', picks, meta, flush=True)
    base = json.loads((_REPO / 'outputs' / E.ARMS[sp][0][1] / 'config.json').read_text())
    scene = SceneConfig.from_dict(base['occlusion_scene']).for_validation()
    kw = dict(img_size=base['img_size'], num_points=base['num_points'], split='test', max_leaves=base['max_leaves'],
              view_sampling=base['view_sampling'], deterministic_view=True, return_pose=True)
    ds = (SorghumDataset4M(base['data_root'], spline_root=base.get('spline_root'), **kw)
          if sp == 'sorghum' else MaizeDataset4M(base['data_root'], **kw))
    # identical to eval_test_scene.py's loader, so sorghum's worker-side subsample matches
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=workers, generator=torch.Generator().manual_seed(0))
    want = {i // 16 for _, i, _, _ in picks}
    batches = {}
    for bi, b in enumerate(dl):
        if bi in want:
            batches[bi] = [t.clone() if torch.is_tensor(t) else t for t in b]
        if len(batches) == len(want):
            break
    print(f'[{time.time() - t0:.0f}s] {sp}: loaded batches {sorted(batches)}', flush=True)

    models = []
    for key, run, ep, tmr in MODELS[sp]:
        rc = json.loads((_REPO / 'outputs' / run / 'config.json').read_text())
        m = E.build_model(sp, rc).to(dev)
        ck = torch.load(_REPO / 'outputs' / run / 'checkpoints' / f'checkpoint_epoch_{ep}.pth',
                        map_location='cpu', weights_only=False, mmap=True)
        m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
        m.pc_sinkhorn_weight = 0.0
        if tmr is not None:
            m.text_mask_ratio = tmr        # 1.0: the parameters are fully hidden (any-to-any)
        # otherwise keep the run's own setting (0.0 = parameters fully visible), as
        # eval_test_scene.py does -- overriding it to None put them in the masking budget
        m.eval()
        models.append((key, m, rc))

    rng = np.random.default_rng(0)
    plants = []
    for label, idx, o_ref, r_ref in picks:
        bi, j = idx // 16, idx % 16
        b = batches[bi]
        rgb, depth, pc, pf, tv = (t.to(dev) for t in b[:5])
        names = list(b[5])
        pn, c2w = b[6].to(dev), b[7].to(dev)
        extra = {'near_far': b[8].to(dev)} if sp == 'maize' else {}
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16,
                     generator=torch.Generator().manual_seed(bi), **extra)
        fg = foreground(depth)
        hid = float(((fg & sc['shown'])[j].sum() / fg[j].sum().clamp(min=1)).item())
        _, world = to_world(pc, pn, c2w)
        r = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * r / r.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        to_w = lambda q: to_world(q, pn, c2w)[1][j]          # unit-sphere camera frame -> world metres
        tgt_n = pc[j]
        sel_t = rng.choice(tgt_n.shape[0], N_SHOW, replace=False)
        in_n = sc['pc'][j]
        sel_i = rng.choice(in_n.shape[0], N_SHOW, replace=False)
        mm = 1000.0 * float(pn[j, 3])
        P = {'label': label, 'name': names[j], 'index': idx, 'radius_m': round(float(pn[j, 3]), 4),
             'hidden_px': round(hid, 4), 'rgb_scene': E.rgb_png(sc['rgb'][j]), 'rgb_target': E.rgb_png(rgb[j]),
             'input': E.r3(to_w(sc['pc'])[sel_i]), 'input_nb': sc['nb_point'][j][sel_i].int().tolist(),
             'target': E.r3(to_w(pc)[sel_t]), 'models': {}}
        for key, m, rc in models:
            torch.manual_seed(30_000 + bi)
            pred = m(sc['rgb'], sc['depth'], sc['pc'], pf, tv, mask_ratio=rc['mask_ratio'],
                     targets={'rgb': rgb, 'depth': depth, 'pc': pc}, loss_tokens=sc['loss_tokens'])[2][2]
            s = E.scores(pred[j:j + 1], pc[j:j + 1], band[j:j + 1], pn[j:j + 1, 3])
            p = pred[j]
            d_pt = torch.cdist(p[None], tgt_n[None])[0].min(dim=1).values          # each predicted point -> plant
            d_tp = torch.cdist(tgt_n[sel_t][None], p[None])[0].min(dim=1).values  # each shown plant point -> prediction
            dd = torch.cdist(p[None], p[None])[0]
            dd.fill_diagonal_(float('inf'))
            stacked = dd.min(dim=1).values < 0.001
            sel_p = rng.choice(p.shape[0], N_SHOW, replace=False)
            P['models'][key] = {
                'pred': E.r3(to_w(pred)[sel_p]),
                'pred_mm': np.round(d_pt[sel_p].cpu().numpy() * mm, 1).tolist(),
                'stacked': stacked[sel_p].int().tolist(),
                'reach_mm': np.round(d_tp.cpu().numpy() * mm, 1).tolist(),
                'scores': {k: round(float(s[k][0]), 4) for k in ('f110mm', 'p10mm', 'r10mm', 'cd_mm', 'dup0.001')},
            }
        print(f"{sp} {label:12s} {names[j]} hidden {hid:.3f} | F1@10mm random {P['models']['random400']['scores']['f110mm']:.3f} "
              f"(report {r_ref:.3f}) ours {P['models']['ours400']['scores']['f110mm']:.3f} (report {o_ref:.3f}) "
              f"no-param {P['models']['noparam400']['scores']['f110mm']:.3f}", flush=True)
        plants.append(P)
    del models
    torch.cuda.empty_cache()
    return {'meta': meta, 'plants': plants}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--species', nargs='+', default=['maize', 'sorghum'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=str(_REPO / 'reports' / 'recon_3d.json'))
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = {sp: run_species(sp, dev, a.workers) for sp in a.species}
    Path(a.out).write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
