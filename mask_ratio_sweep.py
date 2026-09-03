"""
mask_ratio_sweep.py — test-time mask-ratio sweep for EmbodiedMAE-4M.

For each *target* modality (RGB, Depth, PC, Spline) we vary its OWN mask ratio
from 0%..100% in 10% steps, while holding the OTHER three modalities together at
a chosen context mask ratio drawn from {0, 20, 50, 70, 100}%. For every cell we
re-run the (frozen) encoder+decoder and measure how well the target modality is
reconstructed. This is a generalisation of generate_crossmodal.py from the binary
visible/absent regime to a continuous partial-masking grid.

No weights are changed. We only re-implement the encoder masking step with a
chosen per-modality keep fraction, then reuse model.forward_decoder.

Outputs (to --output_dir):
    metrics.json                    full grid (every cell, every modality metric)
    metrics.csv                     flat table for spreadsheets
    heatmaps_target_recon.png       4 heatmaps: target recon error vs (own × context)
    gallery_<modality>.png          reconstruction galleries across own mask ratio
    summary.txt                     human-readable headline numbers

Usage:
    python mask_ratio_sweep.py \
        --checkpoint outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth \
        --config config_4m.yaml \
        --output_dir vis_mask_sweep \
        --num_samples 8
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from embodied_mae_4m import (
    embodied_mae_4m_base, embodied_mae_4m_small,
    N_PARAMS, chamfer_distance,
)
from sorghum_dataset_4m import SorghumDataset4M


MODALITIES   = ['rgb', 'depth', 'pc', 'text']
MOD_LABEL    = {'rgb': 'RGB', 'depth': 'Depth', 'pc': 'Point cloud', 'text': 'Spline'}
OWN_RATIOS   = [round(0.1 * i, 1) for i in range(11)]        # 0.0 .. 1.0
OTHER_RATIOS = [0.0, 0.2, 0.5, 0.7, 1.0]
DISPLAY_OWN  = [0.0, 0.2, 0.5, 0.7, 1.0]                      # columns in galleries

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Masking + forward (frozen model) ──────────────────────────────────────────

def _mask_embed(emb, ratio, gen):
    """Partial random masking of one modality's token embeddings.
    Returns (x_vis, mask, ids_restore, L). Mirrors random_masking_dirichlet._mask
    but with an explicit keep-fraction = (1 - ratio)."""
    B, L, D = emb.shape
    n_keep   = int(round((1.0 - ratio) * L))
    n_keep   = max(0, min(L, n_keep))
    noise    = torch.rand(B, L, device=emb.device, generator=gen)
    ids_shuf = torch.argsort(noise, dim=1)
    ids_rest = torch.argsort(ids_shuf, dim=1)
    ids_keep = ids_shuf[:, :n_keep]
    x_vis    = torch.gather(emb, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask     = torch.ones(B, L, device=emb.device)
    mask[:, :n_keep] = 0
    mask     = torch.gather(mask, 1, ids_rest)               # 1 = masked, at orig pos
    return x_vis, mask, ids_rest, L


@torch.no_grad()
def forward_sweep(model, rgb, depth, pc, params, ratios, gen):
    """ratios: dict modality -> mask ratio in [0,1]. Returns preds + masks."""
    e_rgb   = model.rgb_embed(rgb)     + model.pos_embed_2d   + model.modality_embed_rgb
    e_depth = model.depth_embed(depth) + model.pos_embed_2d   + model.modality_embed_depth
    e_pc    = model.pc_embed(pc)       + model.pos_embed_pc   + model.modality_embed_pc
    e_text  = model.param_embed(params)+ model.pos_embed_text + model.modality_embed_text
    embeds  = {'rgb': e_rgb, 'depth': e_depth, 'pc': e_pc, 'text': e_text}

    vis, mask, rest, lvis = {}, {}, {}, {}
    for m in MODALITIES:
        xv, mk, rs, _ = _mask_embed(embeds[m], ratios[m], gen)
        vis[m], mask[m], rest[m], lvis[m] = xv, mk, rs, xv.shape[1]

    x   = torch.cat([vis['rgb'], vis['depth'], vis['pc'], vis['text']], dim=1)
    cls = model.cls_token.expand(x.shape[0], -1, -1)
    x   = torch.cat([cls, x], dim=1)
    for blk in model.encoder_blocks:
        x = blk(x)
    latent = model.encoder_norm(x)

    pred_rgb, pred_depth, pred_pc, pred_params = model.forward_decoder(
        latent, rest['rgb'], rest['depth'], rest['pc'], rest['text'],
        lvis['rgb'], lvis['depth'], lvis['pc'], lvis['text'])

    preds = {'rgb': pred_rgb, 'depth': pred_depth, 'pc': pred_pc, 'text': pred_params}
    return preds, mask


# ── Metrics ───────────────────────────────────────────────────────────────────

def _masked_mean(per_tok, mask):
    s = mask.sum()
    return ((per_tok * mask).sum() / s).item() if s > 0 else per_tok.mean().item()


def rgb_mse(model, pred, rgb, mask):
    tgt = model.patchify(rgb, model.patch_size, 3)
    if model.norm_pix_loss:
        mu = tgt.mean(-1, keepdim=True)
        var = tgt.var(-1, keepdim=True)
        tgt = (tgt - mu) / (var + 1e-6) ** .5
    return _masked_mean(((pred - tgt) ** 2).mean(-1), mask)


def depth_mse(model, pred, depth, mask):
    tgt = model.patchify(depth, model.patch_size, 1)
    B, N = tgt.shape[0], tgt.shape[1]
    tf = tgt.reshape(B, -1)
    if model.depth_norm_type == 'minmax':
        lo = tf.min(1, keepdim=True).values
        hi = tf.max(1, keepdim=True).values
        tf = (tf - lo) / (hi - lo).clamp(min=1e-6)
    elif model.depth_norm_type == 'standard':
        tf = (tf - tf.mean(1, keepdim=True)) / tf.std(1, keepdim=True).clamp(min=1e-6)
    tgt = tf.reshape(B, N, -1)
    return _masked_mean(((pred - tgt) ** 2).mean(-1), mask)


def pc_chamfer(pred_pc, gt_pc):
    """mean per-sample chamfer + random-plant baseline (chamfer to a different plant)."""
    B = pred_pc.shape[0]
    cd  = [chamfer_distance(pred_pc[i:i+1], gt_pc[i:i+1]).item() for i in range(B)]
    rnd = [chamfer_distance(gt_pc[(i+1) % B:(i+1) % B+1], gt_pc[i:i+1]).item()
           for i in range(B)]
    return float(np.mean(cd)), float(np.mean(rnd))


def param_mae(pred, gt, text_valid, mask_text):
    """Returns (plant_mae, leaf_mae, n_plant, n_leaf) over (valid & masked) tokens.
    Falls back to all valid tokens at 0% masking (nothing masked)."""
    valid = text_valid.bool()
    msel  = mask_text > 0.5
    eff   = valid & msel if msel.any() else valid          # fall back when nothing masked
    err   = (pred - gt).abs()
    plant = eff.clone(); plant[:, 1:] = False
    leaf  = eff.clone(); leaf[:, 0]   = False

    def mm(sel, p):
        s3 = sel.unsqueeze(-1).expand_as(err)
        v  = err[..., :p][s3[..., :p]]
        n  = int(sel.sum())
        return (v.mean().item() if v.numel() else float('nan')), n

    pm, npl = mm(plant, 9)
    lm, nlf = mm(leaf, 7)
    return pm, lm, npl, nlf


def all_metrics(model, preds, masks, rgb, depth, pc, params, text_valid):
    cd, cd_rand = pc_chamfer(preds['pc'], pc)
    plant, leaf, npl, nlf = param_mae(preds['text'], params, text_valid, masks['text'])
    # combined spline MAE over all masked valid tokens — token-count weighted, NaN-safe
    parts, wts = [], []
    if npl > 0 and not np.isnan(plant): parts.append(plant); wts.append(npl)
    if nlf > 0 and not np.isnan(leaf):  parts.append(leaf);  wts.append(nlf)
    spline_mae = float(np.average(parts, weights=wts)) if parts else float('nan')
    return {
        'rgb_mse':    rgb_mse(model, preds['rgb'], rgb, masks['rgb']),
        'depth_mse':  depth_mse(model, preds['depth'], depth, masks['depth']),
        'pc_chamfer': cd, 'pc_chamfer_rand': cd_rand,
        'spline_mae': spline_mae, 'plant_mae': plant, 'leaf_mae': leaf,
    }


# target modality -> the headline metric key used for its heatmap / gallery
TARGET_METRIC = {'rgb': 'rgb_mse', 'depth': 'depth_mse',
                 'pc': 'pc_chamfer', 'text': 'spline_mae'}
METRIC_NICE   = {'rgb_mse': 'RGB recon MSE (norm-pix)',
                 'depth_mse': 'Depth recon MSE (norm)',
                 'pc_chamfer': 'PC Chamfer',
                 'spline_mae': 'Spline param MAE (masked)'}


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _unpatchify(x, p, c, img):
    h = w = img // p
    x = x.reshape(x.shape[0], h, w, p, p, c)
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(x.shape[0], c, h * p, w * p)


def _rgb_recon_disp(model, pred_rgb, rgb):
    """Invert per-patch norm-pix using GT patch stats, then un-ImageNet-normalise."""
    tgt = model.patchify(rgb, model.patch_size, 3)
    mu, var = tgt.mean(-1, keepdim=True), tgt.var(-1, keepdim=True)
    pred = pred_rgb * (var + 1e-6) ** .5 + mu if model.norm_pix_loss else pred_rgb
    img = _unpatchify(pred, model.patch_size, 3, rgb.shape[2])[0].cpu()
    return torch.clamp(img * _STD + _MEAN, 0, 1).permute(1, 2, 0).numpy()


def _depth_recon_disp(model, pred_depth, img_size):
    img = _unpatchify(pred_depth, model.patch_size, 1, img_size)[0, 0].cpu().numpy()
    return np.clip(img, 0, 1)


def _gray_patches(disp_img, mask_1d, p=16, gray=0.55):
    """disp_img: (H,W) or (H,W,3) in [0,1]; mask_1d: (L,) with 1=masked."""
    out = disp_img.copy()
    h = w = out.shape[0] // p
    m = mask_1d.reshape(h, w).cpu().numpy()
    for i in range(h):
        for j in range(w):
            if m[i, j] > 0.5:
                out[i*p:(i+1)*p, j*p:(j+1)*p] = gray
    return out


def _sub(pc_np, n=2000):
    if pc_np.shape[0] > n:
        return pc_np[np.random.choice(pc_np.shape[0], n, replace=False)]
    return pc_np


def gallery_image_modality(model, mod, gt_chw_or_img, gallery, save, val_loss_note):
    """RGB / Depth gallery: rows = [masked input, reconstruction], cols = own ratios."""
    is_rgb = (mod == 'rgb')
    ncol = len(DISPLAY_OWN)
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.2))
    for c, r in enumerate(DISPLAY_OWN):
        disp, mask, mse = gallery[r]
        masked_in = _gray_patches(disp, mask)
        ax = axes[0, c]
        ax.imshow(masked_in if is_rgb else masked_in, cmap=None if is_rgb else 'viridis',
                  vmin=None if is_rgb else 0, vmax=None if is_rgb else 1)
        ax.set_title(f"{int(r*100)}% masked", fontweight='bold', fontsize=11)
        ax.axis('off')
        if c == 0:
            ax.set_ylabel('input', fontsize=11)
        ax = axes[1, c]
        ax.imshow(disp, cmap=None if is_rgb else 'viridis',
                  vmin=None if is_rgb else 0, vmax=None if is_rgb else 1)
        ax.set_title(f"MSE {mse:.4f}", fontsize=10)
        ax.axis('off')
    axes[0, 0].text(-0.12, 0.5, 'MASKED INPUT', rotation=90, va='center', ha='right',
                    transform=axes[0, 0].transAxes, fontweight='bold', fontsize=11)
    axes[1, 0].text(-0.12, 0.5, 'RECONSTRUCTION', rotation=90, va='center', ha='right',
                    transform=axes[1, 0].transAxes, fontweight='bold', fontsize=11)
    fig.suptitle(f"{MOD_LABEL[mod]} reconstruction vs own mask ratio "
                 f"(other modalities fully visible){val_loss_note}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()


def gallery_pc(gt_pc, gallery, save, val_loss_note):
    ncol = len(DISPLAY_OWN) + 1
    fig = plt.figure(figsize=(3.2 * ncol, 3.6))
    gtv = _sub(gt_pc)
    ax = fig.add_subplot(1, ncol, 1, projection='3d')
    ax.scatter(gtv[:, 0], gtv[:, 1], gtv[:, 2], c=gtv[:, 2], cmap='viridis', s=2)
    ax.set_title('GROUND TRUTH', fontweight='bold', fontsize=11)
    ax.view_init(elev=20, azim=45); ax.set_axis_off()
    for k, r in enumerate(DISPLAY_OWN):
        genv, cd = gallery[r]
        genv = _sub(genv)
        ax = fig.add_subplot(1, ncol, k + 2, projection='3d')
        ax.scatter(genv[:, 0], genv[:, 1], genv[:, 2], c=genv[:, 2], cmap='plasma', s=2)
        ax.set_title(f"{int(r*100)}% masked\nChamfer {cd:.4f}", fontsize=10)
        ax.view_init(elev=20, azim=45); ax.set_axis_off()
    fig.suptitle(f"Point-cloud reconstruction vs own mask ratio "
                 f"(other modalities fully visible){val_loss_note}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()


def heatmaps(grid, save, epoch_note):
    """grid[target][own][other] = headline metric. 2x2 heatmap panel."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, mod in zip(axes.ravel(), MODALITIES):
        key = TARGET_METRIC[mod]
        M = np.array([[grid[mod][o][c] for c in OTHER_RATIOS] for o in OWN_RATIOS])
        im = ax.imshow(M, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xticks(range(len(OTHER_RATIOS)))
        ax.set_xticklabels([f"{int(c*100)}" for c in OTHER_RATIOS])
        ax.set_yticks(range(len(OWN_RATIOS)))
        ax.set_yticklabels([f"{int(o*100)}" for o in OWN_RATIOS])
        ax.set_xlabel('other modalities mask %')
        ax.set_ylabel(f'{MOD_LABEL[mod]} (own) mask %')
        ax.set_title(f"{MOD_LABEL[mod]} target — {METRIC_NICE[key]}",
                     fontweight='bold')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha='center', va='center',
                        color='white' if M[i, j] < (np.nanmin(M)+np.nanmax(M))/2 else 'black',
                        fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Target-modality reconstruction error across the mask-ratio grid{epoch_note}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth')
    ap.add_argument('--config', default='config_4m.yaml')
    ap.add_argument('--output_dir', default='vis_mask_sweep')
    ap.add_argument('--num_samples', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = SorghumDataset4M(
        cfg['data']['data_root'], img_size=cfg['data']['img_size'],
        num_points=cfg['data']['num_points'], split='val',
        max_leaves=cfg['model']['max_leaves'])
    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=False, num_workers=4)

    build = embodied_mae_4m_small if cfg['model']['model_size'] == 'small' else embodied_mae_4m_base
    model = build(
        img_size=cfg['data']['img_size'], num_pc_tokens=196,
        target_points=cfg['data']['num_points'],
        pc_loss_weight=cfg['model']['pc_loss_weight'],
        max_leaves=cfg['model']['max_leaves'],
        spline_loss_weight=cfg['model']['spline_loss_weight'],
        depth_norm_type=cfg['model']['depth_norm_type'])

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ckpt['model_state_dict']
    if list(sd)[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dev).eval()
    epoch = ckpt.get('epoch')
    vloss = ckpt.get('val_loss', ckpt.get('best_val_loss'))
    epoch_note = f"  (epoch {epoch})" if epoch is not None else ""
    vnote = f"  |  val_loss {vloss:.4f}" if vloss is not None else ""
    print(f"Loaded {args.checkpoint} (epoch {epoch}, val_loss {vloss})")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    rgb, depth, pc, params, text_valid, names = next(iter(loader))
    rgb, depth, pc = rgb.to(dev), depth.to(dev), pc.to(dev)
    params, text_valid = params.to(dev), text_valid.to(dev)
    B = rgb.shape[0]
    print(f"Gallery plants ({B}): {list(names)}")

    gen = torch.Generator(device=dev)

    # grid[target][own][other] = headline metric ; full[...] = all metrics
    grid = {m: {o: {} for o in OWN_RATIOS} for m in MODALITIES}
    full_rows = []
    # stash sample-0 reconstructions for the galleries (other_ratio = 0)
    stash = {m: {} for m in MODALITIES}

    n_cells = len(MODALITIES) * len(OWN_RATIOS) * len(OTHER_RATIOS)
    done = 0
    for target in MODALITIES:
        for own in OWN_RATIOS:
            for other in OTHER_RATIOS:
                gen.manual_seed(args.seed + done)            # reproducible per cell
                ratios = {m: (own if m == target else other) for m in MODALITIES}
                preds, masks = forward_sweep(model, rgb, depth, pc, params, ratios, gen)
                met = all_metrics(model, preds, masks, rgb, depth, pc, params, text_valid)
                grid[target][own][other] = met[TARGET_METRIC[target]]
                row = {'target': target, 'own_mask': own, 'other_mask': other, **met}
                full_rows.append(row)

                # collect gallery data for sample 0 at full-context (other=0)
                if other == 0.0 and own in DISPLAY_OWN:
                    if target == 'rgb':
                        disp = _rgb_recon_disp(model, preds['rgb'][0:1], rgb[0:1])
                        stash['rgb'][own] = (disp, masks['rgb'][0], met['rgb_mse'])
                    elif target == 'depth':
                        disp = _depth_recon_disp(model, preds['depth'][0:1], rgb.shape[2])
                        stash['depth'][own] = (disp, masks['depth'][0], met['depth_mse'])
                    elif target == 'pc':
                        stash['pc'][own] = (preds['pc'][0].cpu().numpy(), met['pc_chamfer'])
                done += 1
                print(f"  [{done:3d}/{n_cells}] target={target:5s} own={int(own*100):3d}% "
                      f"other={int(other*100):3d}%  {TARGET_METRIC[target]}="
                      f"{grid[target][own][other]:.4f}")

    # ── write metrics ────────────────────────────────────────────────────────
    json.dump({'epoch': epoch, 'val_loss': vloss, 'plants': list(names),
               'own_ratios': OWN_RATIOS, 'other_ratios': OTHER_RATIOS,
               'grid': grid, 'cells': full_rows},
              open(out / 'metrics.json', 'w'), indent=2)

    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader(); w.writerows(full_rows)

    # ── figures ──────────────────────────────────────────────────────────────
    heatmaps(grid, out / 'heatmaps_target_recon.png', epoch_note)

    gt_rgb_disp = torch.clamp(rgb[0].cpu() * _STD + _MEAN, 0, 1).permute(1, 2, 0).numpy()
    gallery_image_modality(model, 'rgb', gt_rgb_disp, stash['rgb'],
                           out / 'gallery_rgb.png', vnote)
    gallery_image_modality(model, 'depth', None, stash['depth'],
                           out / 'gallery_depth.png', vnote)
    gallery_pc(pc[0].cpu().numpy(), stash['pc'], out / 'gallery_pc.png', vnote)

    # ── summary ──────────────────────────────────────────────────────────────
    with open(out / 'summary.txt', 'w') as f:
        f.write(f"Mask-ratio sweep — EmbodiedMAE-4M (epoch {epoch}, val_loss {vloss})\n")
        f.write(f"Plants: {list(names)}\n")
        f.write(f"Own ratios: {[int(o*100) for o in OWN_RATIOS]}%   "
                f"Other ratios: {[int(o*100) for o in OTHER_RATIOS]}%\n\n")
        for target in MODALITIES:
            key = TARGET_METRIC[target]
            f.write(f"[{MOD_LABEL[target]} target]  metric = {METRIC_NICE[key]}\n")
            f.write("  own\\other " + "".join(f"{int(c*100):>9d}%" for c in OTHER_RATIOS) + "\n")
            for o in OWN_RATIOS:
                f.write(f"  {int(o*100):>6d}%   " +
                        "".join(f"{grid[target][o][c]:>10.4f}" for c in OTHER_RATIOS) + "\n")
            f.write("\n")
    print(f"\nDone. metrics + {3 + 1} figures written to {out}/")
    print(open(out / 'summary.txt').read())


if __name__ == '__main__':
    main()
