#!/usr/bin/env python3
"""Score reconstruction under neighbour occlusion -- does structured masking help?

The 2026-09-29 meeting's objection to the earlier structured-masking result was
that it was judged on CLEAN input, which is not what the method is for. This
script scores every run on the same OCCLUDED test inputs: each test plant gets
two neighbours (other test plants, same camera view index) placed beside it by
structured_mask.occluders, with a per-pixel depth test deciding what they hide.
The target is always the clean plant.

Every run sees identical inputs: plants, views, neighbours, placements, the
cloud subsample, the FPS start point, the per-batch Dirichlet token budget and
the uniform shuffle noise are all drawn from seeds that do not depend on the
model. So two runs differ ONLY in their weights, and per-sample differences
between them are paired.

Levels (input / which tokens are masked):
  clean          clean input   / uniform at the run's mask_ratio (= validation)
  clean_occmask  clean input   / the tokens a neighbour WOULD hide go first,
                                 uniform fill to the same budget -- the
                                 structured arm's own training regime
  occ            occluded input/ uniform -- the model is not told where the
                                 neighbours are and sees their pixels
  occ_occmask    occluded input/ every patch showing a neighbour goes first,
                                 uniform fill to the same budget
  occ_deploy     occluded input/ ONLY patches showing a neighbour are masked,
                                 everything else visible, params visible as
                                 conditioning (the meeting's setup). B=1, since
                                 the visible count differs per sample.

Occluded input: RGB and depth are composited (the nearer surface wins per
pixel); the cloud loses every point behind a neighbour and is refilled to its
size by resampling the survivors. It stays in the CLEAN cloud's normalisation
frame -- re-normalising the partial cloud would move the prediction into a
different frame from the target and score the frame shift, not the
reconstruction. Neighbour POINTS are not added: that is the target-vs-neighbour
problem Alloy's scene data is for, which no masking scheme addresses.

Metrics, all against the clean target:
  pc_chamfer           as embodied_mae.chamfer_distance (squared, both ways)
  f1/precision/recall  at 0.01 / 0.02 / 0.03 on the unit-normalised cloud
  hidden_recall@0.03   recall over ONLY the target points a neighbour hides --
                       did the model complete what it could not see?
  visible_recall@0.03  the same over the points it could see
  rgb_mse / depth_mse  in the training loss's space (norm-pix RGB, min-max
                       depth), over masked patches
  rgb_occ_mse / depth_occ_mse  the same over masked patches the neighbour hides
Results are also split by how much of the plant the neighbours hide.

    python eval/eval_occlusion.py --runs outputs/sm_off_s1 outputs/sm_nbr_s1 \\
        --ckpt last --num-plants 1000 --out reports/occlusion_eval.json
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from embodied_mae_4m import (
    embodied_mae_4m_base, embodied_mae_4m_large, embodied_mae_4m_small,
    _visible_token_counts,
)
from sorghum_dataset_4m import SorghumDataset4M
from structured_mask import (
    RENDER_FAR, RENDER_NEAR, NeighbourMaskConfig, draw_params, hidden_points,
    image_token_flags, occluders,
)

DATA_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
LEVELS = ('clean', 'clean_occmask', 'occ', 'occ_occmask', 'occ_deploy')
THRESHOLDS = (0.01, 0.02, 0.03)
# Hidden-fraction bins (share of the target's plant pixels a neighbour hides).
BINS = ((0.0, 0.10), (0.10, 0.25), (0.25, 1.01))
# A patch "shows a neighbour" when this share of its pixels does.
SHOWN_PATCH_FRAC = 0.3

_BUILD = {'small': embodied_mae_4m_small, 'base': embodied_mae_4m_base,
          'large': embodied_mae_4m_large}


# ── test set ──────────────────────────────────────────────────────────────────

def test_occlusion_config():
    """Placement for the TEST occluders: the training ranges, widened.

    Wider than training on both ends, so the test spans lighter and heavier
    occlusion than the structured arm ever saw and the result can be read as a
    function of how much is hidden (BINS), not at one strength.
    """
    return NeighbourMaskConfig(prob=1.0, neighbours=2, offset=(0.0, 0.8),
                               depth_offset=(-0.6, 0.2))


def build_pairs(ds, num_plants, seed):
    """One view per test plant and two neighbour plants seen from the same view.

    The fibonacci view directions are identical for every plant (capture.py), so
    "same view index" means the neighbour was rendered from the same camera
    direction -- the same elevation as the target.
    """
    by_plant = {}
    for i, folder in enumerate(ds.samples):
        plant, view = folder.name.rsplit('_', 1)
        by_plant.setdefault(plant, {})[int(view)] = i
    plants = sorted(p for p, v in by_plant.items() if len(v) == 10)
    rng = random.Random(seed)
    chosen = rng.sample(plants, min(num_plants, len(plants)))
    pairs = []
    for plant in chosen:
        view = rng.randrange(10)
        nbs = rng.sample([p for p in plants if p != plant], 2)
        pairs.append((by_plant[plant][view],
                      [by_plant[n][view] for n in nbs]))
    return pairs


class OcclusionPairs(Dataset):
    """Target sample (with pc_norm) + its two neighbours' RGB and depth."""

    def __init__(self, base, pairs, params, seed):
        self.base, self.pairs, self.params, self.seed = base, pairs, params, seed

    def __len__(self):
        return len(self.pairs)

    def _image(self, idx):
        folder = self.base.samples[idx]
        with Image.open(folder / 'rgb.png') as im:
            rgb = self.base.rgb_transform(im.convert('RGB'))
        return rgb, self.base.load_depth(folder / 'depth.png')

    def __getitem__(self, i):
        tgt, nbs = self.pairs[i]
        # load_pointcloud subsamples with the unseeded global numpy RNG.
        np.random.seed((self.seed * 1_000_003 + i) % 2 ** 32)
        rgb, depth, pc, pf, tv, name, pc_norm = self.base[tgt]
        (r1, d1), (r2, d2) = self._image(nbs[0]), self._image(nbs[1])
        params = {k: v[i] for k, v in self.params.items()}
        return (rgb, depth, pc, pf, tv, pc_norm, r1, d1, r2, d2, params, i)


# ── model ─────────────────────────────────────────────────────────────────────

def resolve_ckpt(run_dir, ckpt):
    if ckpt != 'last':
        return run_dir / ckpt
    found = sorted((int(p.stem.rsplit('_', 1)[1]), p)
                   for p in (run_dir / 'checkpoints').glob('checkpoint_epoch_*.pth'))
    if not found:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    return found[-1][1]


def build_model(run_dir, ckpt_path, device):
    """The exact model a run trained, from its own config.json, loaded strictly.

    structured_mask is NOT passed: it only acts in train mode, and this script
    sets every mask itself.
    """
    cfg = json.loads((run_dir / 'config.json').read_text())
    model = _BUILD[cfg['model_size']](
        active_modalities=cfg.get('active_modalities'),
        img_size=cfg.get('img_size', 224),
        num_pc_tokens=196,
        target_points=cfg.get('num_points', 8196),
        pc_loss_weight=cfg.get('pc_loss_weight', 1.0),
        max_leaves=cfg.get('max_leaves', 24),
        spline_loss_weight=cfg.get('spline_loss_weight', 5.0),
        depth_norm_type=cfg.get('depth_norm_type', 'minmax'),
        pc_loss_name=cfg.get('pc_loss_name', 'qal_loss'),
        qal_threshold=cfg.get('qal_threshold', 0.01),
        qal_alpha=cfg.get('qal_alpha', 100.0),
        qal_use_squared=cfg.get('qal_use_squared', False),
        text_mask_ratio=cfg.get('text_mask_ratio'),
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k[7:] if k.startswith('module.') else k: v
             for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), cfg, int(ckpt.get('epoch', -1))


# ── masking ───────────────────────────────────────────────────────────────────

def budget(model, lengths, mask_ratio, rng):
    """Per-modality visible counts for one batch: the model's own allocation.

    Same Dirichlet + bounds as random_masking_dirichlet, but drawn from `rng`
    so every run gets the same budget for the same batch.
    """
    names = list(lengths)
    split_text = model.text_mask_ratio is not None and 'text' in names
    budget_names = [n for n in names if not (split_text and n == 'text')]
    props = rng.dirichlet([model.dirichlet_alpha] * len(budget_names)).tolist()
    counts = _visible_token_counts([lengths[n] for n in budget_names], props,
                                   mask_ratio, 0.25)
    nv = dict(zip(budget_names, counts))
    if split_text:
        r = model.text_mask_ratio
        nv['text'] = 0 if r >= 1.0 else max(1, int(math.floor(
            lengths['text'] * (1.0 - r) + 1e-12)))
    return nv


def mask_tokens(x, nv, scores):
    """random_masking_dirichlet's _mask with given scores: lowest stay visible."""
    ids_shuf = torch.argsort(scores, dim=1)
    ids_rest = torch.argsort(ids_shuf, dim=1)
    ids_keep = ids_shuf[:, :nv]
    x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
    mask = torch.ones(x.shape[:2], device=x.device)
    mask[:, :nv] = 0
    return x_vis, torch.gather(mask, 1, ids_rest), ids_rest


@torch.no_grad()
def run_masked(model, rgb, depth, pc, pf, nv, scores, fps_seed):
    """Encoder + decoder with explicit per-modality counts and scores."""
    torch.manual_seed(fps_seed)            # FPS start point
    embeds = model._embed_active(rgb, depth, pc, pf)
    centers = model.pc_embed.last_centers
    masked = {}
    for n in model.active_modalities:
        s = scores[n](centers) if callable(scores[n]) else scores[n]
        masked[n] = mask_tokens(embeds[n], nv[n], s)
    latent = model._run_encoder([masked[n][0] for n in model.active_modalities])
    (latent, mr, md, mp, mt, rr, rd, rp, rt,
     lr_, ld_, lp_, lt_) = model._pack(masked, latent)
    preds = model.forward_decoder(latent, rr, rd, rp, rt, lr_, ld_, lp_, lt_)
    return preds, {'rgb': mr, 'depth': md, 'pc': mp, 'text': mt}


# ── metrics ───────────────────────────────────────────────────────────────────

def nn_dists(a, b, chunk=2048):
    """For every point in a (B, N, 3): distance to its nearest point in b."""
    out = []
    for s in range(0, a.shape[1], chunk):
        out.append(torch.cdist(a[:, s:s + chunk], b).min(dim=2).values)
    return torch.cat(out, dim=1)


def pc_metrics(pred, target, hidden):
    d_pt = nn_dists(target, pred)          # target -> prediction (recall side)
    d_pp = nn_dists(pred, target)          # prediction -> target (precision side)
    m = {'pc_chamfer': (d_pp.pow(2).mean(1) + d_pt.pow(2).mean(1))}
    for t in THRESHOLDS:
        p = (d_pp < t).float().mean(1)
        r = (d_pt < t).float().mean(1)
        m[f'precision@{t}'] = p
        m[f'recall@{t}'] = r
        m[f'f1@{t}'] = 2 * p * r / (p + r).clamp(min=1e-9)
    hit = (d_pt < 0.03).float()
    nh = hidden.sum(1)
    m['hidden_recall@0.03'] = torch.where(
        nh >= 50, (hit * hidden).sum(1) / nh.clamp(min=1), torch.nan)
    nv = (~hidden).sum(1)
    m['visible_recall@0.03'] = (hit * ~hidden).sum(1) / nv.clamp(min=1)
    return m


def patch_errors(model, rgb, depth, pred_rgb, pred_depth):
    """Per-patch MSE (B, L) in the loss's own target space."""
    t = model.patchify(rgb, model.patch_size, rgb.shape[1])
    if model.norm_pix_loss:
        t = (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True) + 1e-6) ** .5
    e_rgb = ((pred_rgb - t) ** 2).mean(-1)
    td = model.patchify(depth, model.patch_size, 1)
    B, N = td.shape[:2]
    tf = td.reshape(B, -1)
    lo, hi = tf.min(1, keepdim=True).values, tf.max(1, keepdim=True).values
    td = ((tf - lo) / (hi - lo).clamp(min=1e-6)).reshape(B, N, -1)
    e_depth = ((pred_depth - td) ** 2).mean(-1)
    return e_rgb, e_depth


def masked_mean(err, sel):
    n = sel.sum(1)
    return torch.where(n > 0, (err * sel).sum(1) / n.clamp(min=1), torch.nan)


# ── evaluation ────────────────────────────────────────────────────────────────

def occluded_inputs(rgb, depth, pc, pc_norm, occ, gen):
    """Composite the neighbours into the input; drop the points they hide."""
    shown = occ['shown']
    rgb_o = torch.where(shown, occ['rgb'], rgb)
    nb_depth = ((occ['occ_z'] - RENDER_NEAR) / (RENDER_FAR - RENDER_NEAR)).clamp(0, 1)
    depth_o = torch.where(shown, nb_depth, depth)
    hid = hidden_points(pc, pc_norm, occ['occ_z'])
    pc_o = pc.clone()
    for b in range(pc.shape[0]):
        keep = (~hid[b]).nonzero().squeeze(1)
        if 0 < keep.numel() < pc.shape[1]:
            pick = keep[torch.randint(keep.numel(), (pc.shape[1],),
                                      generator=gen, device=gen.device)]
            pc_o[b] = pc[b, pick]
    return rgb_o, depth_o, pc_o, hid


def evaluate_run(model, cfg, loader, levels, device, seed, mask_cfg):
    mask_ratio = float(cfg.get('mask_ratio', 0.8))
    act = model.active_modalities
    lengths = {n: model._token_len[n] for n in act}
    per = {lv: {} for lv in levels}
    meta = {'hidden_px_frac': [], 'hidden_pts_frac': [], 'occ_patches': [], 'index': []}

    def add(level, metrics):
        for k, v in metrics.items():
            per[level].setdefault(k, []).append(v.detach().float().cpu())

    for bi, batch in enumerate(loader):
        (rgb, depth, pc, pf, tv, pn, r1, d1, r2, d2, params, idx) = batch
        rgb, depth, pc, pf, pn = (t.to(device) for t in (rgb, depth, pc, pf, pn))
        r1, d1, r2, d2 = (t.to(device) for t in (r1, d1, r2, d2))
        params = {k: v.to(device) for k, v in params.items()}
        B = rgb.shape[0]

        occ = occluders(depth, [d1, d2], params, nb_rgbs=[r1, r2])
        gen = torch.Generator(device=device).manual_seed(seed * 7919 + bi)
        rgb_o, depth_o, pc_o, hid = occluded_inputs(rgb, depth, pc, pn, occ, gen)
        occ_img = image_token_flags(occ, model.patch_size, mask_cfg)   # hidden plant patches
        shown_patch = (F.avg_pool2d(occ['shown'].float(), model.patch_size)
                       .flatten(1) >= SHOWN_PATCH_FRAC)
        known_img = occ_img | shown_patch

        fg_px = occ['tgt_fg'].flatten(1).sum(1).clamp(min=1)
        meta['hidden_px_frac'].append((occ['hidden'].flatten(1).sum(1) / fg_px).cpu())
        meta['hidden_pts_frac'].append(hid.float().mean(1).cpu())
        meta['occ_patches'].append(occ_img.sum(1).cpu())
        meta['index'].append(idx)

        nv = budget(model, lengths, mask_ratio, np.random.RandomState(seed * 104729 + bi))
        ngen = torch.Generator(device=device).manual_seed(seed * 15485863 + bi)
        noise = {n: torch.rand(B, lengths[n], generator=ngen, device=device) for n in act}
        fps_seed = seed * 32452843 + bi

        def pc_flags(occ_z):
            return lambda centers: hidden_points(centers, pn, occ_z).float() + noise['pc']

        for level in levels:
            if level == 'occ_deploy':
                continue
            clean_in = level.startswith('clean')
            x = (rgb, depth, pc) if clean_in else (rgb_o, depth_o, pc_o)
            if level.endswith('occmask'):
                img_flags = occ_img if clean_in else known_img
                scores = dict(noise)
                for n in ('rgb', 'depth'):
                    if n in act:
                        scores[n] = img_flags.float() + noise[n]
                scores['pc'] = pc_flags(occ['occ_z'])
            else:
                img_flags = occ_img
                scores = noise
            (p_rgb, p_depth, p_pc, _), masks = run_masked(
                model, *x, pf, nv, scores, fps_seed)
            add(level, collect(model, rgb, depth, pc, hid, p_rgb, p_depth, p_pc,
                               masks, occ_img))

        if 'occ_deploy' in levels:
            add('occ_deploy', deploy(model, rgb, depth, pc, pf, rgb_o, depth_o, pc_o,
                                     pn, occ, hid, known_img, occ_img, fps_seed))
        if bi % 10 == 0:
            print(f'    batch {bi + 1}/{len(loader)}', flush=True)

    out = {lv: {k: torch.cat(v).numpy() for k, v in per[lv].items()} for lv in levels}
    meta = {k: torch.cat(v).numpy() for k, v in meta.items()}
    return out, meta


def collect(model, rgb, depth, pc, hid, p_rgb, p_depth, p_pc, masks, occ_img):
    m = pc_metrics(p_pc, pc, hid)
    if p_rgb is not None and p_depth is not None:
        e_rgb, e_depth = patch_errors(model, rgb, depth, p_rgb, p_depth)
        mr, md = masks['rgb'].bool(), masks['depth'].bool()
        m['rgb_mse'] = masked_mean(e_rgb, mr)
        m['depth_mse'] = masked_mean(e_depth, md)
        m['rgb_occ_mse'] = masked_mean(e_rgb, mr & occ_img)
        m['depth_occ_mse'] = masked_mean(e_depth, md & occ_img)
    return m


@torch.no_grad()
def deploy(model, rgb, depth, pc, pf, rgb_o, depth_o, pc_o, pn, occ, hid,
           known_img, occ_img, fps_seed):
    """Only neighbour patches masked, all else (params included) visible. B=1."""
    rows = []
    act = model.active_modalities
    for b in range(rgb.shape[0]):
        sl = slice(b, b + 1)
        flags = known_img[sl]
        n_known = int(flags.sum())
        nv, scores = {}, {}
        for n in act:
            L = model._token_len[n]
            if n in ('rgb', 'depth'):
                # Every non-neighbour patch visible; keep >= 1 masked so the
                # decoder restore path is the one the model trained with.
                nv[n] = max(1, L - max(n_known, 1))
                scores[n] = flags.float() + torch.linspace(0, 0.5, L, device=rgb.device)[None]
            else:
                nv[n] = L
                scores[n] = torch.zeros(1, L, device=rgb.device)
        (p_rgb, p_depth, p_pc, _), masks = run_masked(
            model, rgb_o[sl], depth_o[sl], pc_o[sl], pf[sl], nv, scores, fps_seed + b)
        rows.append(collect(model, rgb[sl], depth[sl], pc[sl], hid[sl],
                            p_rgb, p_depth, p_pc, masks, occ_img[sl]))
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]}


def summarise(per, meta):
    hp = meta['hidden_px_frac']
    out = {}
    for level, metrics in per.items():
        row = {k: float(np.nanmean(v)) for k, v in metrics.items()}
        row['bins'] = {}
        for lo, hi in BINS:
            sel = (hp >= lo) & (hp < hi)
            row['bins'][f'{lo:.2f}-{min(hi, 1):.2f}'] = {
                'n': int(sel.sum()),
                **{k: float(np.nanmean(v[sel])) if sel.any() else None
                   for k, v in metrics.items()}}
        out[level] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', nargs='+', required=True, help='run output dirs')
    ap.add_argument('--ckpt', default='last',
                    help="'last' (highest checkpoint_epoch_N) or a path inside the run")
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split', default='test')
    ap.add_argument('--num-plants', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--levels', nargs='+', default=list(LEVELS), choices=LEVELS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', required=True, help='json written atomically')
    ap.add_argument('--per-sample', action='store_true',
                    help='also store per-sample values (for paired comparisons)')
    args = ap.parse_args()

    device = torch.device(args.device)
    base = SorghumDataset4M(args.data_root, split=args.split, view_sampling=False,
                            return_pc_norm=True)
    pairs = build_pairs(base, args.num_plants, args.seed)
    test_cfg = test_occlusion_config()
    params = draw_params(len(pairs), test_cfg, 'cpu',
                         generator=torch.Generator().manual_seed(args.seed))
    ds = OcclusionPairs(base, pairs, params, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    # Which tokens count as "occluded" uses the training defaults, so the
    # occluded-patch metrics mean the same thing the structured arm trained on.
    mask_cfg = NeighbourMaskConfig()
    print(f'{len(pairs)} test plants, levels {args.levels}', flush=True)

    result = {'args': vars(args), 'test_occluders': test_cfg.to_dict(),
              'mask_flags': mask_cfg.to_dict(), 'runs': {}}
    for run in args.runs:
        run_dir = Path(run)
        ckpt_path = resolve_ckpt(run_dir, args.ckpt)
        model, cfg, epoch = build_model(run_dir, ckpt_path, device)
        print(f'== {run_dir.name}  {ckpt_path.name} (epoch {epoch})', flush=True)
        t0 = time.time()
        per, meta = evaluate_run(model, cfg, loader, args.levels, device,
                                 args.seed, mask_cfg)
        entry = {'checkpoint': str(ckpt_path), 'epoch': epoch,
                 'structured_mask': cfg.get('structured_mask'),
                 'seed': cfg.get('seed'),
                 'summary': summarise(per, meta),
                 'occlusion': {k: float(np.mean(v)) for k, v in meta.items()
                               if k != 'index'}}
        if args.per_sample:
            entry['per_sample'] = {lv: {k: v.tolist() for k, v in m.items()}
                                   for lv, m in per.items()}
            entry['per_sample_meta'] = {k: v.tolist() for k, v in meta.items()}
        result['runs'][run_dir.name] = entry
        for lv in args.levels:
            s = entry['summary'][lv]
            print(f'  {lv:14s} F1@.03 {s["f1@0.03"]:.4f}  recall@.03 {s["recall@0.03"]:.4f}  '
                  f'hidden_recall {s["hidden_recall@0.03"]:.4f}  chamfer {s["pc_chamfer"]:.5f}  '
                  f'rgb_occ {s.get("rgb_occ_mse", float("nan")):.4f}  '
                  f'depth_occ {s.get("depth_occ_mse", float("nan")):.4f}', flush=True)
        print(f'  ({time.time() - t0:.0f}s)', flush=True)
        del model
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
