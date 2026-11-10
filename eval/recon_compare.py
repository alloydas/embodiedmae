"""Masked reconstruction quality of the frozen encoders that have a decoder: the reconstruction companion of the
encoder comparison (paper Table 9). One deterministic view (view 00) of every validation plant.

Image block: the photograph and the depth frame in, 75 % of each image's 16-px patches hidden (49 of 196 visible
per image; MultiMAE's pretraining budget of 98 visible tokens out of 392), the point cloud and the recipe hidden.
Scored on the plant pixels of the hidden patches (the MultiMAE adapter's eroded plant mask, depth_valid):
  PSNR      of the photograph, colours in [0, 1]. Both models predict per-patch standardized colour (CropCode's RGB
            head, MultiMAE's norm_rgb head), so both are de-standardized with the true patch mean and variance,
            the usual MAE convention.
  depth RMSE in mm. Each model's normalized depth is mapped back with the two per-image numbers its own
            normalization used (CropCode: the image's min and max; MultiMAE: the truncated mean and standard
            deviation of the plant pixels), then scaled to metres with the adapter's depth scale.
Point-cloud block: the point cloud alone in, 60 % of its point groups hidden (Point-MAE's pretraining ratio).
Point-MAE is also given the centres of the hidden groups, as in its pretraining; CropCode is not. Clouds are
compared at matched density (prediction and truth each subsampled to 2,048 points) in the unit-sphere frame:
  Chamfer  mean squared nearest-neighbour distance, both directions summed
  P / R / F1 at tau = 0.01 and 0.02: share of predicted points within tau of the truth, share of true points
           within tau of the prediction, and their harmonic mean (per plant, then averaged).
DINOv2, the supervised network and EmbodiedMAE-B (its decoder is not ported here) have no decoder in this repo.

usage: python eval/recon_compare.py --out reports/recon_compare_<job> [--species sorghum maize] [--n-plants N] [--device cuda]
"""
import sys, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'eval')]
import argparse, json, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

import baselines as B
import baselines.multimae as MMA
import baselines.pointmae as PMA
from eval_row import _unpack, bootstrap_ci
from train_sorghum_4m import unpatchify, _unnorm_pix

MULTIMAE_REPO = pathlib.Path('/work/mech-ai-scratch/alloy/MultiMAE')     # upstream @ 66910f5, multimae/ unmodified
WEIGHTS = '/work/mech-ai-scratch/alloy/e8_weights'
DATA = {'sorghum': '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K', 'maize': '/work/mech-ai-scratch/alloy/Maize_1'}
RUNS = {'sorghum': [('CropCode PC only', 'e2_pc_d2', 'checkpoints/checkpoint_epoch_600.pth'),
                    ('CropCode PC+RGB+D', 'e2_pcrgbd_d2', 'checkpoints/checkpoint_epoch_600.pth'),
                    ('CropCode four streams', 'e2_pcrgbdt_d2', 'checkpoints/checkpoint_epoch_600.pth'),
                    ('CropCode headline (1,000 ep)', '4m_pretrain_15k_d2_sink', 'checkpoints/checkpoint_epoch_1000.pth')],
        'maize': [('CropCode PC only', 'maize_e2_pc_d2', 'checkpoints/checkpoint_epoch_600.pth'),
                  ('CropCode PC+RGB+D', 'maize_e2_pcrgbd_d2', 'checkpoints/checkpoint_epoch_600.pth'),
                  ('CropCode four streams', 'maize_4m_d2', 'checkpoints/checkpoint_epoch_600.pth'),
                  ('CropCode headline (1,000 ep)', 'maize_4m_pretrain_15k_d2', 'checkpoints/checkpoint_epoch_1000.pth')]}
IMG_HIDE, PC_HIDE, N_MATCH, TAUS = 0.75, 0.60, 2048, (0.01, 0.02)
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ap = argparse.ArgumentParser()
ap.add_argument('--species', nargs='+', default=['sorghum', 'maize'])
ap.add_argument('--split', default='val'); ap.add_argument('--n-plants', type=int, default=None)
ap.add_argument('--batch-size', type=int, default=16); ap.add_argument('--num-workers', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = torch.device(a.device)
hub = B.configure_weight_cache(WEIGHTS)


# ------------------------------------------------------------------ models
def build_multimae():
    sys.path.insert(0, str(MULTIMAE_REPO))
    from multimae.multimae import pretrain_multimae_base
    from multimae.input_adapters import PatchedInputAdapter
    from multimae.output_adapters import SpatialOutputAdapter
    ins = {'rgb': PatchedInputAdapter(num_channels=3, stride_level=1, patch_size_full=16),
           'depth': PatchedInputAdapter(num_channels=1, stride_level=1, patch_size_full=16)}
    outs = {d: SpatialOutputAdapter(num_channels=c, stride_level=1, patch_size_full=16, dim_tokens=256, depth=2, num_heads=8,
                                    use_task_queries=True, task=d, context_tasks=['rgb', 'depth', 'semseg'], use_xattn=True)
            for d, c in (('rgb', 3), ('depth', 1))}
    outs['norm_rgb'] = SpatialOutputAdapter(num_channels=3, stride_level=1, patch_size_full=16, dim_tokens=256, depth=2, num_heads=8,
                                            use_task_queries=True, task='rgb', context_tasks=['rgb', 'depth', 'semseg'], use_xattn=True)
    m = pretrain_multimae_base(input_adapters=ins, output_adapters=outs, num_global_tokens=1)
    sd = torch.load(B.verify_file(MMA.checkpoint_path(hub), size=MMA.CKPT_SIZE, sha256=MMA.CKPT_SHA256), map_location='cpu', weights_only=True)['model']
    keep = {k: v for k, v in sd.items() if not k.startswith(('input_adapters.semseg', 'output_adapters.semseg'))}
    missing, unexpected = m.load_state_dict(keep, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'MultiMAE full model: {len(missing)} missing {missing[:3]}, {len(unexpected)} unexpected {unexpected[:3]}')
    print(f'MultiMAE-B with its rgb, norm_rgb and depth decoders: {len(keep)} keys loaded strictly', flush=True)
    return m.eval().to(dev)


def build_pointmae():
    sd = {k[len('module.'):]: v for k, v in PMA.load_checkpoint(hub).items()}
    m = PMA.V.PointMAE(); B.strict_load(m, sd, 'Point-MAE full pretraining model')
    return m.eval().to(dev)


def build_ours(species, run, ckpt):
    import linear_probe as LP, linear_probe_maize as LPM
    return (LP if species == 'sorghum' else LPM).build_model(REPO / 'outputs' / run, ckpt, dev)


# ------------------------------------------------------------------ data
def dataset(species, rgb_file, gain, param_schema='v2'):
    if species == 'sorghum':
        import linear_probe as LP
        return LP.SorghumDataset4M(DATA['sorghum'], img_size=224, num_points=8196, split=a.split, max_leaves=24,
                                   view_sampling=True, deterministic_view=True, rgb_file=rgb_file, rgb_gain=gain)
    import linear_probe_maize as LPM
    return LPM.MaizeDataset4M(DATA['maize'], img_size=224, num_points=8192, split=a.split, max_leaves=LPM.MAX_LEAVES,
                              view_sampling=True, deterministic_view=True, rgb_file=rgb_file, param_schema=param_schema)


def loader(ds):
    import linear_probe as LP
    if a.n_plants: ds.plant_views = ds.plant_views[:a.n_plants]
    np.random.seed(a.seed)
    return DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, worker_init_fn=LP._worker_init,
                      persistent_workers=False)


# ------------------------------------------------------------------ metrics
def destd_patches(pred_img, gt_img, p=16):
    """per-patch standardized image prediction -> image space, with the true patch mean and variance (MAE convention)."""
    def to_p(x):
        b, c, h, w = x.shape
        return x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, (h // p) * (w // p), p * p * c)
    def from_p(x, shape):
        b, c, h, w = shape
        return x.reshape(b, h // p, w // p, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(shape)
    t = to_p(gt_img); m, v = t.mean(-1, keepdim=True), t.var(-1, keepdim=True)
    return from_p(to_p(pred_img) * (v + 1e-6) ** .5 + m, gt_img.shape)


def multimae_depth_stats(depth, species, names):
    """the truncated mean and variance depth_to_multimae standardizes with (metres), per image."""
    valid = MMA.depth_valid(depth, names); d = depth * MMA.DEPTH_M_PER_UNIT[species]
    nd = d.clone(); nd[~valid] = float('nan'); srt = torch.sort(nd.flatten(1), dim=1)[0]; n = valid.flatten(1).sum(1)
    lo, hi = (n * 0.1).long().tolist(), (n * 0.9).long().tolist()
    mean = torch.stack([srt[i, a_:b_].mean() for i, (a_, b_) in enumerate(zip(lo, hi))])
    var = torch.stack([srt[i, a_:b_].var() for i, (a_, b_) in enumerate(zip(lo, hi))])
    return mean, (var + MMA.DEPTH_STD_EPS) ** .5


def rmse_mm(pred_m, true_m, sel):
    return [1000 * float(((pred_m[i][sel[i]] - true_m[i][sel[i]]) ** 2).mean() ** .5) for i in range(pred_m.shape[0])]


def mae_mm(pred_m, true_m, sel):
    return [1000 * float((pred_m[i][sel[i]] - true_m[i][sel[i]]).abs().mean()) for i in range(pred_m.shape[0])]


def nmse(pred, tgt, sel):
    """per image: SSE / SST over the selected pixels (all channels), SST about each channel's mean."""
    out = []
    for i in range(pred.shape[0]):
        s = sel[i, 0]
        p, t = pred[i][:, s], tgt[i][:, s]
        sst = ((t - t.mean(1, keepdim=True)) ** 2).sum()
        out.append(float(((p - t) ** 2).sum() / sst.clamp(min=1e-12)))
    return out


def psnr(pred_n, tgt_n, sel):
    p = (pred_n.cpu() * IMNET_STD + IMNET_MEAN).clamp(0, 1); t = (tgt_n.cpu() * IMNET_STD + IMNET_MEAN).clamp(0, 1)
    out = []
    for i in range(p.shape[0]):
        s = sel[i, 0].cpu(); mse = float(((p[i][:, s] - t[i][:, s]) ** 2).mean())
        out.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return out


def patch_mask_to_pixels(mask):            # (B, 196) 1 = hidden -> (B, 1, 224, 224) bool
    return mask.view(-1, 1, 14, 14).repeat_interleave(16, 2).repeat_interleave(16, 3) > 0.5


def cloud_scores(pred, gt, g):
    """matched density: each side subsampled to N_MATCH points; Chamfer and P/R/F1 per plant."""
    rows = []
    for i in range(pred.shape[0]):
        p = pred[i][torch.randperm(pred.shape[1], generator=g)[:N_MATCH].to(pred.device)] if pred.shape[1] > N_MATCH else pred[i]
        t = gt[i][torch.randperm(gt.shape[1], generator=g)[:N_MATCH].to(gt.device)]
        d = torch.cdist(p[None].float(), t[None].float())[0]
        dp, dt = d.min(1).values, d.min(0).values
        r = {'chamfer': float((dp ** 2).mean() + (dt ** 2).mean())}
        for tau in TAUS:
            P, R = float((dp < tau).float().mean()), float((dt < tau).float().mean())
            r[f'P@{tau}'], r[f'R@{tau}'], r[f'F1@{tau}'] = P, R, (2 * P * R / (P + R) if P + R > 0 else 0.0)
        rows.append(r)
    return rows


@torch.no_grad()
def ours_forward(model, visible, ratio, rgb, depth, pc, seed):
    torch.manual_seed(seed)
    pe, emb = getattr(model, 'pos_embed_text', None), getattr(model, 'param_embed', None)
    n_tok = pe.shape[1] if pe is not None else 1 + int(getattr(model, 'max_leaves', 24))   # text inactive: never read
    n_par = emb.proj.in_features if emb is not None else 14
    zeros = torch.zeros(rgb.shape[0], n_tok, n_par, device=rgb.device)
    out = model.forward_encoder_select(rgb, depth, pc, zeros, visible=set(visible), source_mask_ratio=ratio)
    latent, masks, rests, nvis = _unpack(out)
    pr, pd_, ppc, _ = model.forward_decoder(latent, *rests, *nvis.values())
    return masks, pr, pd_, ppc


# ------------------------------------------------------------------ run
rows = []
for species in a.species:
    import linear_probe as LP
    ours = [(label, run, ckpt) + build_ours(species, run, ckpt) for label, run, ckpt in RUNS[species]]
    groups = {}                                           # (rgb_file, gain) -> models fed those images
    base_file = 'rgb_nobg.png' if species == 'sorghum' else 'rgba.png'
    groups.setdefault((base_file, 1.0 if species == 'sorghum' else None), []).extend(['multimae', 'pointmae'])
    for label, run, ckpt, model, cfg, ep in ours:
        key = (cfg.get('rgb_file', 'rgb.png'), LP.run_rgb_gain(cfg) if species == 'sorghum' else None)
        groups.setdefault(key, []).append((label, run, model, ep))
    mm, pm = build_multimae(), build_pointmae()
    for (rgb_file, gain), members in groups.items():
        ds = dataset(species, rgb_file, gain)
        print(f'{species}: {len(ds.plant_views)} plants, {rgb_file} at gain {gain}: {[m if isinstance(m, str) else m[0] for m in members]}', flush=True)
        g = torch.Generator().manual_seed(a.seed); t0 = time.time()
        for bi, batch in enumerate(loader(ds)):
            rgb, depth, pc = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)
            names = list(batch[5]); n = rgb.shape[0]
            valid = MMA.depth_valid(depth.cpu(), names).to(dev)
            pc_gt = pc
            for mem in members:
                if mem == 'multimae':
                    z = MMA.depth_to_multimae(depth.cpu(), species, names).to(dev)
                    for i in range(n):                        # one sample per forward (upstream sums kept tokens over the batch)
                        gm = torch.Generator().manual_seed(a.seed * 7919 + bi * 1000 + i)
                        tm = {d: torch.ones(1, 196) for d in ('rgb', 'depth')}
                        for d in tm:
                            tm[d][0, torch.randperm(196, generator=gm)[:int(196 * (1 - IMG_HIDE))]] = 0
                        tm = {d: v.to(dev).long() for d, v in tm.items()}
                        preds, _ = mm({'rgb': rgb[i:i + 1], 'depth': z[i:i + 1]}, mask_inputs=True, task_masks=tm)
                        sel_r = patch_mask_to_pixels(tm['rgb'].float()) & valid[i:i + 1]
                        sel_d = patch_mask_to_pixels(tm['depth'].float()) & valid[i:i + 1]
                        img = destd_patches(preds['norm_rgb'], rgb[i:i + 1])
                        mu, sd = multimae_depth_stats(depth[i:i + 1].cpu(), species, names[i:i + 1])
                        dm = (preds['depth'].cpu() * sd.view(-1, 1, 1, 1) + mu.view(-1, 1, 1, 1))
                        tm_ = depth[i:i + 1].cpu() * MMA.DEPTH_M_PER_UNIT[species]
                        rows.append(dict(species=species, model='MultiMAE-B (RGB+D)', block='image', plant=names[i],
                                         rgb_psnr=psnr(img, rgb[i:i + 1], sel_r)[0], depth_rmse_mm=rmse_mm(dm, tm_, sel_d.cpu())[0],
                                         depth_mae_mm=mae_mm(dm, tm_, sel_d.cpu())[0]))
                elif mem == 'pointmae':
                    x = PMA.to_upstream(PMA.orient(pc, species, names))
                    nb, center = pm.group_divider(x)
                    mask = PMA.V.rand_mask(n, PMA.NUM_GROUP, PC_HIDE, generator=g, device=dev)
                    rebuild, _ = pm.reconstruct(nb, center, mask)
                    rebuild = rebuild.reshape(n, -1, PMA.GROUP_SIZE, 3) + center[mask].reshape(n, -1, 1, 3)
                    vis = nb[~mask].reshape(n, -1, PMA.GROUP_SIZE, 3) + center[~mask].reshape(n, -1, 1, 3)
                    full = torch.cat([vis.reshape(n, -1, 3), rebuild.reshape(n, -1, 3)], 1)
                    gt = PMA.orient(pc, species, names)
                    for i, r in enumerate(cloud_scores(full, gt, g)):
                        rows.append(dict(species=species, model='Point-MAE (PC)', block='cloud', plant=names[i], **r))
                else:
                    label, run, model, ep = mem
                    act = set(model.active_modalities)
                    if {'rgb', 'depth'} <= act:
                        masks, pr, pd_, _ = ours_forward(model, ('rgb', 'depth'), IMG_HIDE, rgb, depth, pc, a.seed + bi)
                        img = unpatchify(_unnorm_pix(model, pr, rgb), model.patch_size, 3, model.img_size)
                        dimg = unpatchify(pd_, model.patch_size, 1, model.img_size)
                        flat = depth.flatten(1); dmin, dmax = flat.min(1).values.view(-1, 1, 1, 1), flat.max(1).values.view(-1, 1, 1, 1)
                        k = MMA.DEPTH_M_PER_UNIT[species]
                        dm = (dimg * (dmax - dmin) + dmin).cpu() * k; tm_ = depth.cpu() * k
                        sel_r = patch_mask_to_pixels(masks['rgb'].float()) & valid
                        sel_d = patch_mask_to_pixels(masks['depth'].float()) & valid
                        for i, (rp, dr, da) in enumerate(zip(psnr(img, rgb, sel_r), rmse_mm(dm, tm_, sel_d.cpu()), mae_mm(dm, tm_, sel_d.cpu()))):
                            rows.append(dict(species=species, model=label, run=run, epoch=ep, block='image', plant=names[i], rgb_psnr=rp,
                                             depth_rmse_mm=dr, depth_mae_mm=da))
                    _, _, _, ppc = ours_forward(model, ('pc',), PC_HIDE, rgb, depth, pc, a.seed + bi)
                    for i, r in enumerate(cloud_scores(ppc, pc_gt, g)):
                        rows.append(dict(species=species, model=label, run=run, epoch=ep, block='cloud', plant=names[i], **r))
            if bi % 20 == 0:
                print(f'  {min((bi + 1) * a.batch_size, len(ds.plant_views))}/{len(ds.plant_views)} plants ({time.time() - t0:.0f} s)', flush=True)

df = pd.DataFrame(rows); df.to_csv(f'{a.out}.csv', index=False)
summary = []
for (species, model, block), gdf in df.groupby(['species', 'model', 'block'], sort=False):
    e = dict(species=species, model=model, block=block, n=len(gdf))
    cols = ['rgb_psnr', 'depth_rmse_mm', 'depth_mae_mm'] if block == 'image' else ['chamfer'] + [f'{m}@{t}' for t in TAUS for m in ('P', 'R', 'F1')]
    for c in cols:
        mean, lo, hi = bootstrap_ci(gdf[c].values); e[c] = dict(mean=mean, lo=lo, hi=hi)
    summary.append(e)
json.dump(summary, open(f'{a.out}.json', 'w'), indent=1)
print(f'wrote {a.out}.csv ({len(df)} rows) and {a.out}.json')
for e in summary:
    if e['block'] == 'image':
        print(f"{e['species']:8s} {e['model']:30s} image  RGB PSNR {e['rgb_psnr']['mean']:.1f} dB  depth RMSE {e['depth_rmse_mm']['mean']:.1f} mm, MAE {e['depth_mae_mm']['mean']:.1f} mm  (n={e['n']})")
    else:
        print(f"{e['species']:8s} {e['model']:30s} cloud  Chamfer {e['chamfer']['mean'] * 1e3:.2f}e-3  P/R/F1@0.01 {e['P@0.01']['mean']:.3f}/{e['R@0.01']['mean']:.3f}/{e['F1@0.01']['mean']:.3f}  F1@0.02 {e['F1@0.02']['mean']:.3f}  (n={e['n']})")
