#!/usr/bin/env python3
"""How much does a scene model lean on depth, and on the hints depth carries?

Maize depth.png is (z - near) / (far - near) with near / far set PER PLANT, centred on
the clean target's centre (it decodes to exactly 0.5) and spanning ~3.3x its radius, so
the input depth tells the model where the target is and how big it is. Sorghum uses
fixed planes (0.025 / 50 m) and carries no such hint. This scores a finished run on the
occluded TEST scenes of eval/pc_generation.py (same loader order, scenes and seeds; no
card) under:

  trained masking        Dirichlet 0.8, as trained -- reproduces pc_generation's 'no card'
  depth fixed planes     (maize) depth re-encoded with ONE near / far pair for every plant:
                         same geometry, the per-plant hint removed. Out of distribution for
                         the model, so a drop is an upper bound on what the hint is worth.
  rgb+depth+pc 117 tok   each modality 80 % masked (the select path's control, ~117 tokens)
  rgb+pc 117 tok         depth removed, the same token count (each 70 % masked)
  rgb+pc 78 tok          depth removed, the same per-modality rate (80 %)
  depth+pc 117 tok       RGB removed instead, for scale
  origin off N cm        (norm 'stem' runs) the stem base the frame is built on moved N cm
                         in a random direction: how exact must a real stem estimate be?

    python eval/depth_reliance.py --species maize --run maize_scene_anyparam_stem_s1 --epoch 200 \
        --out reports/depth_reliance_maize_stem.json
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
from maize_dataset_4m import MaizeDataset4M
from sorghum_dataset_4m import SorghumDataset4M
from occlusion_scene import SceneConfig, compose, to_world

KEYS = ('cd_mm', 'f110mm', 'r10mm', 'p10mm', 'f120mm')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--species', choices=('maize', 'sorghum'), required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--near', type=float, default=1.0, help='fixed depth plane (m), maize re-encode')
    ap.add_argument('--far', type=float, default=3.0, help='fixed depth plane (m), maize re-encode')
    ap.add_argument('--max-batches', type=int, default=None, help='smoke test')
    ap.add_argument('--fp32', action='store_true',
                    help='disable TF32 in cuDNN convolutions and matmuls (GPU-dependence check)')
    a = ap.parse_args()
    if a.fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    sp = a.species
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    rc = json.loads((_REPO / 'outputs' / a.run / 'config.json').read_text())
    m = E.build_model(sp, rc).to(dev)
    ck = torch.load(_REPO / 'outputs' / a.run / 'checkpoints' / f'checkpoint_epoch_{a.epoch}.pth',
                    map_location='cpu', weights_only=False, mmap=True)
    m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
    m.pc_sinkhorn_weight = 0.0
    m.eval()
    scene = SceneConfig.from_dict(rc['occlusion_scene']).for_validation()
    L = int(rc['max_leaves'])
    if sp == 'maize':
        ds = MaizeDataset4M(rc['data_root'], img_size=rc['img_size'], num_points=rc['num_points'], split='test',
                            max_leaves=L, view_sampling=rc['view_sampling'], deterministic_view=True, return_pose=True,
                            return_visibility=scene.visible_only, depth_planes=rc.get('depth_planes'))
    else:
        ds = SorghumDataset4M(rc['data_root'], img_size=rc['img_size'], num_points=rc['num_points'], split='test',
                              max_leaves=L, view_sampling=rc['view_sampling'], deterministic_view=True,
                              return_pose=True, spline_root=rc.get('spline_root'),
                              return_visibility=scene.visible_only)
    # identical to eval/pc_generation.py, so 'trained masking' reproduces its 'no card'
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, generator=torch.Generator().manual_seed(0))
    stem = scene.norm == 'stem'
    acc = {}
    for bi, b in enumerate(dl):
        if a.max_batches is not None and bi >= a.max_batches:
            break
        rgb, depth, pc, pf, tv = (t.to(dev) for t in b[:5])
        pn, c2w = b[6].to(dev), b[7].to(dev)
        nf = b[8].to(dev) if sp == 'maize' else None
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16, near_far=nf,
                     generator=torch.Generator().manual_seed(bi), vis=b[9].to(dev) if len(b) > 9 else None)
        _, world = to_world(pc, pn, c2w)
        rad = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * rad / rad.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        cn = sc.get('crop_norm')
        tgt = {'rgb': rgb, 'depth': depth, 'pc': sc.get('target_pc', pc)}

        def fwd(x_depth=None, x_pc=None, back_cn=None, visible=None, src=0.0):
            torch.manual_seed(30000 + bi)
            xd = sc['depth'] if x_depth is None else x_depth
            xp = sc['pc'] if x_pc is None else x_pc
            if visible is None:
                m.text_mask_ratio = 1.0                   # no card, as pc_generation's 'no card'
                pr = m(sc['rgb'], xd, xp, pf, tv, mask_ratio=0.8, targets=tgt,
                       loss_tokens=sc['loss_tokens'])[2][2]
            else:
                m.text_mask_ratio = None                  # text not in `visible` -> fully hidden
                pr = m(sc['rgb'], xd, xp, pf, tv, visible=visible, source_mask_ratio=src, targets=tgt)[2][2]
            c = cn if back_cn is None else back_cn
            return pr if c is None else pr * c[:, None, 3:4] + c[:, None, :3]

        preds = {'trained masking': fwd()}
        if nf is not None:
            near, span = nf[:, 0].view(-1, 1, 1, 1), (nf[:, 1] - nf[:, 0]).view(-1, 1, 1, 1)
            fg = sc['depth'] > 0
            z = near + sc['depth'] * span
            d_fixed = torch.where(fg, ((z - a.near) / (a.far - a.near)).clamp(1e-6, 1), torch.zeros_like(z))
            preds['depth fixed planes'] = fwd(x_depth=d_fixed)
        preds['rgb+depth+pc 117 tok'] = fwd(visible={'rgb', 'depth', 'pc'}, src=0.8)
        preds['rgb+pc 117 tok'] = fwd(visible={'rgb', 'pc'}, src=0.7)
        preds['rgb+pc 78 tok'] = fwd(visible={'rgb', 'pc'}, src=0.8)
        preds['depth+pc 117 tok'] = fwd(visible={'depth', 'pc'}, src=0.7)
        if stem:
            # crop_norm = (origin c in the target's normalised camera frame, r = unit_m / radius);
            # an origin error e (metres) moves the input by -e / unit_m and c by e / radius.
            g = torch.Generator(device='cpu').manual_seed(50000 + bi)
            u = torch.randn(pc.shape[0], 3, generator=g).to(dev)
            u = u / u.norm(dim=1, keepdim=True)
            for cm in (1, 2, 5):
                e = u * (cm / 100.0)
                x_pc = sc['pc'] - (e / float(scene.unit_m))[:, None]
                c2 = torch.cat([cn[:, :3] + e / pn[:, 3:4], cn[:, 3:4]], dim=1)
                preds[f'origin off {cm} cm'] = fwd(x_pc=x_pc, back_cn=c2)
        for c, pr in preds.items():
            s = E.scores(pr, pc, band, pn[:, 3])
            for k in KEYS:
                acc.setdefault(c, {}).setdefault(k, []).append(s[k].float().cpu())
        if bi % 20 == 0:
            print(f'[{time.time() - t0:.0f}s] batch {bi}', flush=True)

    per = {c: {k: torch.cat(v) for k, v in d.items()} for c, d in acc.items()}
    means = {c: {k: float(v.mean()) for k, v in d.items()} for c, d in per.items()}
    # paired: share of plants whose F1@10mm is LOWER than under the reference condition
    ref_of = lambda c: 'rgb+depth+pc 117 tok' if 'tok' in c else 'trained masking'
    paired = {c: {'vs': ref_of(c), 'share_lower_f110mm': float((per[c]['f110mm'] < per[ref_of(c)]['f110mm']).float().mean())}
              for c in per if c != ref_of(c)}
    out = {'species': sp, 'run': a.run, 'epoch': a.epoch, 'n_plants': int(per['trained masking']['f110mm'].numel()),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'fp32': a.fp32,
           'fixed_planes_m': [a.near, a.far] if sp == 'maize' else None, 'means': means, 'paired': paired}
    dst = _REPO / a.out
    tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(dst)
    print(f'\n{a.run} ep{a.epoch}, occluded {sp} test, no card ({out["n_plants"]} plants):'
          f'\n{"condition":24s} err mm  F1@10  R@10   P@10   F1@20  lower-than-ref')
    for c, v in means.items():
        pl = paired.get(c, {}).get('share_lower_f110mm')
        print(f"{c:24s} {v['cd_mm']:6.1f}  {v['f110mm']:.3f}  {v['r10mm']:.3f}  {v['p10mm']:.3f}  {v['f120mm']:.3f}"
              + ('' if pl is None else f'  {pl:.0%}'))
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
