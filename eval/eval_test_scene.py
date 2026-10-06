#!/usr/bin/env python3
"""Score the occluded-scene arms on the TEST split (reports/test_scene/<species>.json).

The test split was never used to choose anything: every arm is scored at its
last checkpoint (not best_model.pth, which was selected on val loss), with
settings chosen on val. For each species and arm, every test plant is scored

  clean     the plant alone, uniform masking at mask_ratio
  occluded  an occluded scene from occlusion_scene.compose (prob 1, uniform
            masking -- the deployable regime), scored against the clean plant

with train_sorghum_4m.pc_scores' metrics kept per plant (chamfer; precision /
recall / F1 at 0.01 and 0.03), plus recall@0.03 split by distance from the
stem: target points in the inner, middle and outer third of the plant's
radius (world hypot(x, z); the stem is on the world y axis). Scenes, cloud
subsamples and masks are identical across arms with the same modalities
(batches read once, torch re-seeded per batch); the no-parameter long run has
a different token set, so its masks differ from the others'.

Example plants for the page are chosen by OCCLUSION, not by any model's score:
the plants at the 50th / 75th / 90th percentile of the share of their pixels a
neighbour hides.

    python eval/eval_test_scene.py --species sorghum maize
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import argparse
import base64
import io
import json
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import train_sorghum_4m as TS
import train_maize_4m as TM
from occlusion_scene import SceneConfig, compose, to_world
from structured_mask import foreground
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

ARMS = {
    'sorghum': [('random', 'sm_scene_s1', 200), ('structured', 'sm_scene_struct_s1', 200),
                ('mixed', 'sm_scene_mix_s1', 200), ('random_sink', 'sm_scene_sink_s1', 200),
                ('mixed_sink', 'sm_scene_mix_sink_s1', 200), ('long', 'long_sorghum_mixsink_s1', 600)],
    'maize': [('random', 'maize_scene_s1', 200), ('structured', 'maize_scene_struct_s1', 200),
              ('mixed', 'maize_scene_mix_s1', 200), ('random_sink', 'maize_scene_sink_s1', 200),
              ('mixed_sink', 'maize_scene_mix_sink_s1', 200)],
}
# More arms, scored only when named with --arms (they write <species>_<tag>.json).
EXTRA = {
    'sorghum': {'q02sq': ('sm_scene_q02sq_s1', 200), 'q02sq_sink': ('sm_scene_q02sq_sink_s1', 200),
                'cham': ('sm_scene_cham_s1', 200), 'sinkonly': ('sm_scene_sinkonly_s1', 200),
                'mixed_sink400': ('sm_scene_mix_sink400_s1', 400)},
    'maize': {'cham': ('maize_scene_cham_s1', 200), 'sinkonly': ('maize_scene_sinkonly_s1', 200),
              'mixed_sink400': ('maize_scene_mix_sink400_s1', 400)},
}
SHOW_ARMS = ('random', 'mixed_sink', 'long', 'mixed_sink400')   # reconstructions kept for the examples
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
PCTL = (50, 75, 90)
N_SHOW = 2400


def build_model(species, cfg):
    ns = argparse.Namespace(**cfg)
    if species == 'sorghum':
        return TS.build_model_from_args(ns)
    fn = {'small': TM.embodied_mae_4m_maize_small, 'base': TM.embodied_mae_4m_maize_base,
          'large': TM.embodied_mae_4m_maize_large}[ns.model_size]
    return fn(active_modalities=ns.active_modalities, img_size=ns.img_size, num_pc_tokens=196,
              target_points=ns.num_points, pc_loss_weight=ns.pc_loss_weight,
              max_leaves=ns.max_leaves, spline_loss_weight=ns.spline_loss_weight,
              depth_norm_type=ns.depth_norm_type, pc_loss_name=ns.pc_loss_name,
              qal_threshold=ns.qal_threshold, qal_alpha=ns.qal_alpha,
              qal_use_squared=ns.qal_use_squared, text_mask_ratio=ns.text_mask_ratio,
              pc_sinkhorn_weight=getattr(ns, 'pc_sinkhorn_weight', 0.0),
              pc_sinkhorn_points=getattr(ns, 'pc_sinkhorn_points', 2048),
              pc_sinkhorn_blur=getattr(ns, 'pc_sinkhorn_blur', 0.01))


@torch.no_grad()
def scores(pred, target, r_band, chunk=512):
    """pc_scores per plant, plus recall@0.03 per radius band (B, 3)."""
    d_pp = []
    d_pt = torch.full(target.shape[:2], float('inf'), device=pred.device, dtype=pred.dtype)
    for s in range(0, pred.shape[1], chunk):
        d = ((pred[:, s:s + chunk, None] - target[:, None]) ** 2).sum(-1)
        d_pp.append(d.min(dim=2).values)
        d_pt = torch.minimum(d_pt, d.min(dim=1).values)
        del d
    d_pp = torch.cat(d_pp, dim=1)
    out = {'chamfer': d_pp.mean(1) + d_pt.mean(1)}
    for t in (0.01, 0.03):
        p = (d_pp < t * t).float().mean(1)
        r = (d_pt < t * t).float().mean(1)
        out[f'p{t}'], out[f'r{t}'], out[f'f1{t}'] = p, r, 2 * p * r / (p + r).clamp(min=1e-9)
    hit = (d_pt < 0.03 ** 2).float()
    for k, name in enumerate(('r0.03_inner', 'r0.03_middle', 'r0.03_outer')):
        m = (r_band == k).float()
        out[name] = (hit * m).sum(1) / m.sum(1).clamp(min=1)
    return {k: v.cpu() for k, v in out.items()}


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def rgb_png(x):
    img = (x.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return png_b64((img * 255).round().astype(np.uint8))


def r3(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
    return np.round(np.asarray(x, dtype=np.float64), 3).tolist()


def run_species(sp, args, device):
    t0 = time.time()
    root = Path(args.outputs_root)
    if args.arms:
        known = {a: (r, e) for a, r, e in ARMS[sp]} | EXTRA.get(sp, {})
        arms = [(a, *known[a]) for a in args.arms if a in known]
    else:
        arms = ARMS[sp]
    base_cfg = json.loads((root / ARMS[sp][0][1] / 'config.json').read_text())
    scene = SceneConfig.from_dict(base_cfg['occlusion_scene']).for_validation()
    kw = dict(img_size=base_cfg['img_size'], num_points=base_cfg['num_points'], split=args.split,
              max_leaves=base_cfg['max_leaves'], view_sampling=base_cfg['view_sampling'],
              deterministic_view=True, return_pose=True)
    ds = (SorghumDataset4M(base_cfg['data_root'], spline_root=base_cfg.get('spline_root'), **kw)
          if sp == 'sorghum' else MaizeDataset4M(base_cfg['data_root'], **kw))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=args.workers)
    batches = []
    for b in dl:
        batches.append([t.clone() if torch.is_tensor(t) else t for t in b])
        if args.max_batches and len(batches) >= args.max_batches:
            break
    print(f'[{time.time() - t0:.0f}s] {sp}: {len(batches)} {args.split} batches', flush=True)

    def inputs(bi):
        b = batches[bi]
        rgb, depth, pc, pf, tv = (t.to(device) for t in b[:5])
        pn, c2w = b[6].to(device), b[7].to(device)
        extra = {'near_far': b[8].to(device)} if sp == 'maize' else {}
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16,
                     generator=torch.Generator().manual_seed(args.seed * 1_000_003 + bi), **extra)
        _, world = to_world(pc, pn, c2w)
        r = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * r / r.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        fg = foreground(depth)
        hid = ((fg & sc['shown']).flatten(1).sum(1) / fg.flatten(1).sum(1).clamp(min=1))
        return rgb, depth, pc, pf, tv, sc, band, hid

    res, keep_pred = {}, {}
    hidden = None
    for arm, run, ep in arms:
        rc = json.loads((root / run / 'config.json').read_text())
        m = build_model(sp, rc).to(device)
        ck = torch.load(root / run / 'checkpoints' / f'checkpoint_epoch_{ep}.pth', map_location='cpu', weights_only=False)
        m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
        del ck
        m.eval()
        # The Sinkhorn term is a training loss only; computing it here is wasted
        # work, and geomloss re-enables autograd inside its call, which leaks out
        # of torch.no_grad() (job 16734136 died on it). Predictions do not depend on it.
        m.pc_sinkhorn_weight = 0.0
        acc = {'clean': {}, 'occluded': {}}
        hid_all = []
        with torch.no_grad():
            for bi in range(len(batches)):
                rgb, depth, pc, pf, tv, sc, band, hid = inputs(bi)
                hid_all.append(hid.cpu())
                torch.manual_seed(20_000 + bi)
                _, _, (_, _, pc_clean, _), _ = m(rgb, depth, pc, pf, tv, mask_ratio=(args.mask_ratio if args.mask_ratio is not None else rc['mask_ratio']))
                torch.manual_seed(30_000 + bi)
                _, _, (_, _, pc_occ, _), _ = m(sc['rgb'], sc['depth'], sc['pc'], pf, tv, mask_ratio=(args.mask_ratio if args.mask_ratio is not None else rc['mask_ratio']),
                                               targets={'rgb': rgb, 'depth': depth, 'pc': pc},
                                               loss_tokens=sc['loss_tokens'])
                for cond, pred in (('clean', pc_clean), ('occluded', pc_occ)):
                    for k, v in scores(pred, pc, band).items():
                        acc[cond].setdefault(k, []).append(v)
        res[arm] = {c: {k: torch.cat(v).numpy() for k, v in d.items()} for c, d in acc.items()}
        hidden = torch.cat(hid_all).numpy()
        print(f'[{time.time() - t0:.0f}s] {sp} {arm} ({run} ep {ep}): '
              + ' | '.join(f"{c} cham {res[arm][c]['chamfer'].mean():.5f} F1@.01 {res[arm][c]['f10.01'].mean():.3f} "
                           f"R@.03 {res[arm][c]['r0.03'].mean():.3f} P@.03 {res[arm][c]['p0.03'].mean():.3f}"
                           for c in ('clean', 'occluded')), flush=True)
        if arm in SHOW_ARMS:
            keep_pred[arm] = (m, rc)
        else:
            del m
            torch.cuda.empty_cache()

    if args.no_examples:
        picks = []
    pool = np.where(hidden > 0)[0]
    order = pool[np.argsort(hidden[pool])]
    if not args.no_examples:
        picks = [int(order[min(len(order) - 1, int(round(q / 100 * (len(order) - 1))))]) for q in PCTL]
    rng = np.random.default_rng(0)
    examples = []
    with torch.no_grad():
        for q, idx in zip(PCTL, picks):
            bi, j = divmod(idx, 16)
            rgb, depth, pc, pf, tv, sc, band, hid = inputs(bi)
            preds = {}
            for arm, (m, rc) in keep_pred.items():
                torch.manual_seed(30_000 + bi)
                _, _, (_, _, pc_occ, _), _ = m(sc['rgb'], sc['depth'], sc['pc'], pf, tv, mask_ratio=(args.mask_ratio if args.mask_ratio is not None else rc['mask_ratio']),
                                               targets={'rgb': rgb, 'depth': depth, 'pc': pc},
                                               loss_tokens=sc['loss_tokens'])
                p = pc_occ[j]
                preds[arm] = r3(p[rng.choice(p.shape[0], N_SHOW, replace=False)].cpu())
            sel_in = rng.choice(sc['pc'].shape[1], N_SHOW, replace=False)
            sel_t = rng.choice(pc.shape[1], N_SHOW, replace=False)
            name = batches[bi][5][j] if isinstance(batches[bi][5], (list, tuple)) else str(idx)
            examples.append({
                'name': name, 'percentile': q, 'hidden_px': round(float(hidden[idx]), 4),
                'rgb_scene': rgb_png(sc['rgb'][j]), 'rgb_target': rgb_png(rgb[j]),
                'pc_in': r3(sc['pc'][j][sel_in].cpu()), 'pc_nb': sc['nb_point'][j][sel_in].int().cpu().tolist(),
                'target': r3(pc[j][sel_t].cpu()), 'pred': preds,
                'scores': {arm: {k: round(float(res[arm]['occluded'][k][idx]), 6)
                                 for k in ('chamfer', 'f10.01', 'p0.03', 'r0.03')} for arm in preds},
            })

    out = {
        'species': sp, 'split': args.split, 'n_plants': int(len(hidden)), 'seed': args.seed,
        'mask_ratio': args.mask_ratio,
        'arms': [{'arm': a, 'run': r, 'epoch': e} for a, r, e in arms],
        'hidden_px': np.round(hidden, 4).tolist(),
        'means': {arm: {c: {k: float(v.mean()) for k, v in d.items()} for c, d in res[arm].items()} for arm in res},
        'per_plant': {arm: {c: {k: np.round(d[k], 5).tolist() for k in ('chamfer', 'f10.01', 'r0.03')}
                            for c, d in res[arm].items()} for arm in res},
        'examples': examples,
    }
    dst = Path(args.out_dir) / (f'{sp}_{args.tag}.json' if args.tag else f'{sp}.json')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':')))
    tmp.replace(dst)
    print(f'[{time.time() - t0:.0f}s] {sp}: wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)', flush=True)
    keep_pred.clear()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species', nargs='+', default=['sorghum', 'maize'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--outputs-root', default=str(_REPO / 'outputs'))
    ap.add_argument('--out-dir', default=str(_REPO / 'reports' / 'test_scene'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-batches', type=int, default=None, help='smoke test only')
    ap.add_argument('--arms', nargs='+', default=None, help='score only these arms (ARMS or EXTRA names)')
    ap.add_argument('--tag', default=None, help='output <species>_<tag>.json instead of <species>.json')
    ap.add_argument('--no-examples', action='store_true')
    ap.add_argument('--mask-ratio', type=float, default=None,
                    help='test-time masking ratio (default: what each arm trained with, 0.8)')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for sp in args.species:
        run_species(sp, args, device)


if __name__ == '__main__':
    main()
