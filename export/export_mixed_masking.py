#!/usr/bin/env python3
"""Data for the mixed-masking explainer page (reports/mixed_masking/<species>.json).

For each species, over every occluded val scene (view 0 per plant, scenes built
by occlusion_scene.compose with a fixed generator):

  1. Per-plant scores of the random / structured / mixed arms' epoch-200
     checkpoints on the occluded input under uniform test masking. All three
     arms get the SAME scene, the SAME cloud subsample and the SAME masks
     (torch re-seeded per batch), so the per-plant differences are paired.
     Scores follow train_sorghum_4m.pc_scores exactly, kept per sample.
  2. How much neighbour content stays visible to the encoder under each
     TRAINING masking policy on these scenes: uniform, neighbour_first, and
     mixed (a per-sample coin at nf_prob 0.5 between the two). Both policies
     use the same Dirichlet split per batch (re-seeded), so only WHICH tokens
     are masked differs.
  3. A few example scenes for the page's viewer: the scene and target RGB, the
     neighbour pixels, each policy's visible patches and visible point-cloud
     points (a point is visible when it lies in the 32-point group of a visible
     token), the clean target cloud and each arm's reconstruction. Examples sit
     at fixed percentiles of how much worse pure structured masking did on that
     plant than random masking, among plants with at least 8 % of their pixels
     hidden -- chosen by rule, not by eye.

    python export/export_mixed_masking.py --species sorghum maize
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
from dataclasses import replace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import train_sorghum_4m as TS
import train_maize_4m as TM
from occlusion_scene import SceneConfig, compose
from structured_mask import foreground
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M

ARMS = {
    'sorghum': {'random': 'sm_scene_s1', 'structured': 'sm_scene_struct_s1', 'mixed': 'sm_scene_mix_s1'},
    'maize': {'random': 'maize_scene_s1', 'structured': 'maize_scene_struct_s1', 'mixed': 'maize_scene_mix_s1'},
}
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
              qal_use_squared=ns.qal_use_squared, text_mask_ratio=ns.text_mask_ratio)


@torch.no_grad()
def per_sample_scores(pred, target, chunk=512):
    """train_sorghum_4m.pc_scores, per sample instead of batch means."""
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
        out[f'p{t}'] = p
        out[f'r{t}'] = r
        out[f'f1{t}'] = 2 * p * r / (p + r).clamp(min=1e-9)
    return {k: v.cpu() for k, v in out.items()}


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def rgb_png(x):
    img = (x.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return png_b64((img * 255).round().astype(np.uint8))


def overlay_png(shown):
    """RGBA: neighbour pixels in a warm tint, everything else transparent."""
    s = shown.cpu().numpy().astype(bool)
    a = np.zeros(s.shape + (4,), dtype=np.uint8)
    a[s] = (255, 170, 0, 150)
    return png_b64(a)


def group_visible(pc, fps_idx, vis_tok, k):
    """(N,) bool: points inside the k-NN group of a visible token."""
    cen = pc[fps_idx]                                            # (S, 3)
    gidx = torch.cdist(cen, pc).topk(k, dim=1, largest=False).indices   # (S, k)
    out = torch.zeros(pc.shape[0], dtype=torch.bool, device=pc.device)
    out[gidx[vis_tok].flatten()] = True
    return out


def r3(x):
    return np.round(np.asarray(x, dtype=np.float64), 3).tolist()


def run_species(sp, args, device):
    t0 = time.time()
    runs = ARMS[sp]
    root = Path(args.outputs_root)
    cfg = json.loads((root / runs['random'] / 'config.json').read_text())
    scene_val = SceneConfig.from_dict(cfg['occlusion_scene']).for_validation()
    scene_nf = replace(scene_val, mask_policy='neighbour_first')     # same scene, plus the flags
    kw = dict(img_size=cfg['img_size'], num_points=cfg['num_points'], split='val',
              max_leaves=cfg['max_leaves'], view_sampling=cfg['view_sampling'],
              deterministic_view=True, return_pose=True)
    ds = (SorghumDataset4M(cfg['data_root'], spline_root=cfg.get('spline_root'), **kw)
          if sp == 'sorghum' else MaizeDataset4M(cfg['data_root'], **kw))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=args.workers)
    batches = []
    for b in dl:
        batches.append([t.clone() if torch.is_tensor(t) else t for t in b])
        if args.max_batches and len(batches) >= args.max_batches:
            break
    print(f'[{time.time() - t0:.0f}s] {sp}: {len(batches)} batches read', flush=True)

    models = {}
    for arm, run in runs.items():
        rc = json.loads((root / run / 'config.json').read_text())
        m = build_model(sp, rc).to(device)
        ck = torch.load(root / run / 'checkpoints' / 'checkpoint_epoch_200.pth',
                        map_location='cpu', weights_only=False)
        m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
        del ck
        models[arm] = m.eval()
    print(f'[{time.time() - t0:.0f}s] {sp}: models loaded', flush=True)
    P = models['random'].patch_size
    K = models['random'].pc_embed.group_size

    def scene_of(bi, b):
        rgb, depth, pc, pf, tv = (t.to(device) for t in b[:5])
        extra = {'near_far': b[8].to(device)} if sp == 'maize' else {}
        sc = compose(rgb, depth, pc, b[6].to(device), b[7].to(device), scene_nf, patch_size=P,
                     generator=torch.Generator().manual_seed(args.seed * 1_000_003 + bi), **extra)
        return rgb, depth, pc, pf, tv, sc

    def policy_masks(bi, sc, pf):
        """Training-policy masks on this scene: uniform and neighbour_first, same Dirichlet split."""
        m = models['random']
        emb = m._embed_active(sc['rgb'], sc['depth'], sc['pc'], pf)
        fps = m.pc_embed.last_fps_idx.clone()
        scores = m._flag_scores(sc['mask_flags'])
        torch.manual_seed(50_000 + bi)
        mu = m.random_masking_dirichlet(emb, cfg['mask_ratio'])
        torch.manual_seed(50_000 + bi)
        mn = m.random_masking_dirichlet(emb, cfg['mask_ratio'], scores=scores)
        return fps, {'uniform': {n: mu[n][1] for n in ('rgb', 'pc')},
                     'neighbour_first': {n: mn[n][1] for n in ('rgb', 'pc')}}

    scores = {arm: {} for arm in runs}
    hidden, vis = [], {p: {'pc_vis': 0, 'pc_vis_nb': 0, 'img_vis': 0, 'img_vis_nb30': 0}
                       for p in ('uniform', 'neighbour_first', 'mixed')}
    coin_gen = torch.Generator().manual_seed(args.seed + 7)
    with torch.no_grad():
        for bi, b in enumerate(batches):
            rgb, depth, pc, pf, tv, sc = scene_of(bi, b)
            fg = foreground(depth)
            hidden.append(((fg & sc['shown']).flatten(1).sum(1) / fg.flatten(1).sum(1).clamp(min=1)).cpu())
            kwm = {'targets': {'rgb': rgb, 'depth': depth, 'pc': pc}, 'loss_tokens': sc['loss_tokens']}
            for arm, m in models.items():
                torch.manual_seed(10_000 + bi)
                _, _, (_, _, pred_pc, _), _ = m(sc['rgb'], sc['depth'], sc['pc'], pf, tv,
                                                mask_ratio=cfg['mask_ratio'], **kwm)
                for k, v in per_sample_scores(pred_pc, pc).items():
                    scores[arm].setdefault(k, []).append(v)
            fps, pm = policy_masks(bi, sc, pf)
            cnb = torch.gather(sc['nb_point'], 1, fps)                       # (B, S) neighbour-centred
            img_nb30 = torch.nn.functional.avg_pool2d(sc['shown'].float(), P).flatten(1) >= 0.3
            coin = (torch.rand(rgb.shape[0], generator=coin_gen) < 0.5).to(device)
            for name in ('uniform', 'neighbour_first', 'mixed'):
                if name == 'mixed':
                    mpc = torch.where(coin[:, None], pm['neighbour_first']['pc'], pm['uniform']['pc'])
                    mrgb = torch.where(coin[:, None], pm['neighbour_first']['rgb'], pm['uniform']['rgb'])
                else:
                    mpc, mrgb = pm[name]['pc'], pm[name]['rgb']
                pv, iv = mpc == 0, mrgb == 0
                vis[name]['pc_vis'] += pv.sum().item()
                vis[name]['pc_vis_nb'] += (pv & cnb).sum().item()
                vis[name]['img_vis'] += iv.sum().item()
                vis[name]['img_vis_nb30'] += (iv & img_nb30).sum().item()
            if bi % 20 == 0:
                print(f'[{time.time() - t0:.0f}s] {sp}: batch {bi}/{len(batches)}', flush=True)

    hidden = torch.cat(hidden).numpy()
    S = {arm: {k: torch.cat(v).numpy() for k, v in d.items()} for arm, d in scores.items()}
    gap = S['structured']['chamfer'] - S['random']['chamfer']
    pool = np.where(hidden >= 0.08)[0]
    order = pool[np.argsort(gap[pool])]
    picks = [int(order[min(len(order) - 1, int(round(q / 100 * (len(order) - 1))))]) for q in PCTL]
    print(f'{sp}: examples {picks} (gap percentiles {PCTL})', flush=True)

    examples = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for q, idx in zip(PCTL, picks):
            bi, j = divmod(idx, 16)
            rgb, depth, pc, pf, tv, sc = scene_of(bi, batches[bi])
            kwm = {'targets': {'rgb': rgb, 'depth': depth, 'pc': pc}, 'loss_tokens': sc['loss_tokens']}
            preds = {}
            for arm, m in models.items():
                torch.manual_seed(10_000 + bi)
                _, _, (_, _, pred_pc, _), _ = m(sc['rgb'], sc['depth'], sc['pc'], pf, tv,
                                                mask_ratio=cfg['mask_ratio'], **kwm)
                preds[arm] = pred_pc[j]
            fps, pm = policy_masks(bi, sc, pf)
            x_in = sc['pc'][j]
            sel_in = rng.choice(x_in.shape[0], N_SHOW, replace=False)
            sel_t = rng.choice(pc.shape[1], N_SHOW, replace=False)
            name = batches[bi][5][j] if isinstance(batches[bi][5], (list, tuple)) else str(idx)
            ex = {
                'name': name, 'percentile': q, 'hidden_px': round(float(hidden[idx]), 4),
                'nb_points': round(float(sc['nb_point'][j].float().mean()), 4),
                'rgb_scene': rgb_png(sc['rgb'][j]), 'rgb_target': rgb_png(rgb[j]),
                'nb_overlay': overlay_png(sc['shown'][j, 0]),
                'nb_frac': r3(torch.nn.functional.avg_pool2d(sc['shown'][j:j + 1].float(), P).flatten().cpu()),
                'patch_vis': {p: pm[p]['rgb'][j].eq(0).int().cpu().tolist() for p in pm},
                'pc_in': r3(x_in[sel_in].cpu()),
                'pc_nb': sc['nb_point'][j][sel_in].int().cpu().tolist(),
                'pc_vis': {p: group_visible(x_in, fps[j], pm[p]['pc'][j].eq(0), K)[sel_in].int().cpu().tolist()
                           for p in pm},
                'tok_vis_nb': {p: int((pm[p]['pc'][j].eq(0) & sc['nb_point'][j][fps[j]]).sum()) for p in pm},
                'tok_vis': {p: int(pm[p]['pc'][j].eq(0).sum()) for p in pm},
                'target': r3(pc[j][sel_t].cpu()),
                'pred': {arm: r3(p[rng.choice(p.shape[0], N_SHOW, replace=False)].cpu()) for arm, p in preds.items()},
                'scores': {arm: {k: round(float(S[arm][k][idx]), 6) for k in S[arm]} for arm in S},
            }
            examples.append(ex)

    out = {
        'species': sp, 'runs': runs, 'n_plants': int(len(hidden)), 'seed': args.seed,
        'hidden_px': np.round(hidden, 4).tolist(),
        'per_plant': {arm: {k: np.round(v, 6).tolist() for k, v in S[arm].items() if k in ('chamfer', 'f10.01', 'p0.03', 'r0.03')}
                      for arm in S},
        'means': {arm: {k: float(v.mean()) for k, v in S[arm].items()} for arm in S},
        'visibility': {p: {'pc_vis_nb_share': v['pc_vis_nb'] / max(v['pc_vis'], 1),
                           'img_vis_nb30_share': v['img_vis_nb30'] / max(v['img_vis'], 1)} for p, v in vis.items()},
        'examples': examples,
    }
    dst = Path(args.out_dir) / f'{sp}.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':')))
    tmp.replace(dst)
    print(f'[{time.time() - t0:.0f}s] {sp}: wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB); '
          f'means {json.dumps(out["means"])}; visibility {json.dumps(out["visibility"])}', flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species', nargs='+', default=['sorghum', 'maize'])
    ap.add_argument('--outputs-root', default=str(_REPO / 'outputs'))
    ap.add_argument('--out-dir', default=str(_REPO / 'reports' / 'mixed_masking'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-batches', type=int, default=None, help='smoke test only')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for sp in args.species:
        run_species(sp, args, device)


if __name__ == '__main__':
    main()
