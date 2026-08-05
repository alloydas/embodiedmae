"""
generate_crossmodal.py — empirical cross-modal generation probe for EmbodiedMAE-4M.

Question: given ONLY some modalities visible (e.g. RGB+Depth), can the *current*
checkpoint generate the others (point cloud + spline params)?

The model was trained at mask_ratio≈0.15 total with a Dirichlet split that always
kept >=1 visible token per modality (min_mask_ratio=0.25). It NEVER saw "two
modalities fully present, two fully absent", so this is an out-of-distribution
masking regime. This script measures what the weights already do there.

It does NOT modify the model. It re-implements the encoder masking step with a
deterministic choice of which modalities are fully visible vs fully masked
(0 visible tokens), then reuses model.forward_decoder / model.forward_loss.

Usage:
    python generate_crossmodal.py \
        --checkpoint outputs/4m_run_v3/best_model.pth \
        --config config_4m.yaml \
        --output_dir vis_crossmodal \
        --num_samples 6 \
        --visible rgb_depth          # or: rgb, depth, rgb_depth_pc (control)
"""

import argparse
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


MODALITIES = ['rgb', 'depth', 'pc', 'text']

# which param indices are "real" per token type, for readable error reporting
_PLANT_KEYS = ['sl', 'sdx', 'sdy', 'sdz', 'psx', 'psy', 'psz', 'pa', 'pr']
_LEAF_KEYS  = ['sp', 'ln', 'ra', 'ba', 'wf', 'wp0', 'wp1']   # last 2 of 9 are pad


# ── Cross-modal forward (no model changes) ────────────────────────────────────

@torch.no_grad()
def forward_crossmodal(model, rgb, depth, pc, params, visible):
    """Run encoder+decoder with `visible` modalities fully present and the rest
    fully masked (0 visible tokens). `visible` is a set/list drawn from MODALITIES.

    Returns pred_rgb, pred_depth, pred_pc, pred_params plus the per-modality
    mask tensors (1 = masked / to-be-generated).
    """
    B = rgb.shape[0]
    dev = rgb.device

    # 1. Embed every modality exactly as forward_encoder does. We still feed real
    #    tensors for the "absent" modalities — their embeddings are simply dropped
    #    (0 visible tokens), so no degenerate FPS-on-zeros etc.
    e_rgb   = model.rgb_embed(rgb)     + model.pos_embed_2d    + model.modality_embed_rgb
    e_depth = model.depth_embed(depth) + model.pos_embed_2d    + model.modality_embed_depth
    e_pc    = model.pc_embed(pc)       + model.pos_embed_pc    + model.modality_embed_pc
    e_text  = model.param_embed(params)+ model.pos_embed_text  + model.modality_embed_text
    embeds  = {'rgb': e_rgb, 'depth': e_depth, 'pc': e_pc, 'text': e_text}

    def build(name, e):
        L = e.shape[1]
        ids_rest = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        if name in visible:                       # keep all tokens, in order
            x_vis = e
            mask  = torch.zeros(B, L, device=dev)
        else:                                     # 0 visible tokens -> all masked
            x_vis = e[:, :0]
            mask  = torch.ones(B, L, device=dev)
        return x_vis, mask, ids_rest, L

    xr_v, mr, rr, lr_ = build('rgb',   e_rgb)
    xd_v, md, rd, ld_ = build('depth', e_depth)
    xp_v, mp, rp, lp_ = build('pc',    e_pc)
    xt_v, mt, rt, lt_ = build('text',  e_text)

    # 2. Encoder over visible tokens + CLS
    x   = torch.cat([xr_v, xd_v, xp_v, xt_v], dim=1)
    cls = model.cls_token.expand(B, -1, -1)
    x   = torch.cat([cls, x], dim=1)
    for blk in model.encoder_blocks:
        x = blk(x)
    latent = model.encoder_norm(x)

    # 3. Decoder restores masked positions and predicts every modality
    lr_v, ld_v, lp_v, lt_v = xr_v.shape[1], xd_v.shape[1], xp_v.shape[1], xt_v.shape[1]
    pred_rgb, pred_depth, pred_pc, pred_params = model.forward_decoder(
        latent, rr, rd, rp, rt, lr_v, ld_v, lp_v, lt_v)

    return pred_rgb, pred_depth, pred_pc, pred_params, (mr, md, mp, mt)


# ── Param error reporting ─────────────────────────────────────────────────────

def param_errors(model, pred_params, gt_params, text_valid):
    """Mean-abs-error on *normalised* [0,1] params over real tokens, split into
    plant token vs leaf tokens. Returns dict."""
    valid = text_valid.bool()                       # (B, T)
    err   = (pred_params - gt_params).abs()         # (B, T, P)

    plant_mask = valid.clone(); plant_mask[:, 1:] = False
    leaf_mask  = valid.clone(); leaf_mask[:, 0]   = False

    def masked_mae(m, p_slice):
        m3 = m.unsqueeze(-1).expand_as(err)
        sel = err[..., :p_slice][m3[..., :p_slice]]
        return sel.mean().item() if sel.numel() else float('nan')

    return {
        'plant_mae_norm': masked_mae(plant_mask, 9),   # all 9 plant params real
        'leaf_mae_norm':  masked_mae(leaf_mask, 7),    # leaves use first 7
        'n_plant_tokens': int(plant_mask.sum()),
        'n_leaf_tokens':  int(leaf_mask.sum()),
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def _subsample(pc_np, n=1500):
    if pc_np.shape[0] > n:
        idx = np.random.choice(pc_np.shape[0], n, replace=False)
        return pc_np[idx]
    return pc_np


def visualize(sample, save_path, visible):
    rgb, depth = sample['rgb'], sample['depth']
    gt_pc, gen_pc = sample['pc'].numpy(), sample['pred_pc'].numpy()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_vis = torch.clamp(rgb * std + mean, 0, 1).permute(1, 2, 0).numpy()

    fig = plt.figure(figsize=(16, 9))
    cond = '+'.join(visible).upper()

    ax = plt.subplot(2, 3, 1)
    ax.imshow(rgb_vis); ax.axis('off')
    ax.set_title(f"RGB ({'INPUT' if 'rgb' in visible else 'unused'})",
                 fontweight='bold')

    ax = plt.subplot(2, 3, 2)
    im = ax.imshow(depth[0].numpy(), cmap='viridis'); ax.axis('off')
    ax.set_title(f"Depth ({'INPUT' if 'depth' in visible else 'unused'})",
                 fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    gtv, genv = _subsample(gt_pc), _subsample(gen_pc)
    ax = plt.subplot(2, 3, 4, projection='3d')
    ax.scatter(gtv[:, 0], gtv[:, 1], gtv[:, 2], c=gtv[:, 2], cmap='viridis', s=2)
    ax.set_title(f"GROUND-TRUTH point cloud\n{gt_pc.shape[0]} pts", fontweight='bold')
    ax.view_init(elev=20, azim=45)

    ax = plt.subplot(2, 3, 5, projection='3d')
    ax.scatter(genv[:, 0], genv[:, 1], genv[:, 2], c=genv[:, 2], cmap='plasma', s=2)
    ax.set_title(f"GENERATED from {cond}\nChamfer {sample['chamfer']:.5f} "
                 f"(rand baseline {sample['chamfer_rand']:.5f})", fontweight='bold')
    ax.view_init(elev=20, azim=45)

    # text panel: GT vs generated params
    ax = plt.subplot(1, 3, 3); ax.axis('off')
    lines = [f"SPLINE PARAMS  (generated from {cond})", "=" * 38, ""]
    lines.append(f"plant MAE(norm): {sample['param_err']['plant_mae_norm']:.3f}")
    lines.append(f"leaf  MAE(norm): {sample['param_err']['leaf_mae_norm']:.3f}")
    lines += ["", "PLANT token:", f"  GT : {sample['gt_text'][0]}",
              f"  GEN: {sample['gen_text'][0]}", "", "First leaves:"]
    n_real = int(sample['text_valid'].sum())
    for ti in range(1, min(n_real, 4)):
        lines.append(f"  leaf{ti} GT : {sample['gt_text'][ti]}")
        lines.append(f"  leaf{ti} GEN: {sample['gen_text'][ti]}")
        lines.append("")
    ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left',
            family='monospace', fontsize=7.5, transform=ax.transAxes)

    plt.suptitle(f"Cross-modal generation — {sample['name']}  |  visible: {cond}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


def save_ply(pc_np, path):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pc_np.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pc_np:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_run_v3/best_model.pth')
    ap.add_argument('--config', default='config_4m.yaml')
    ap.add_argument('--output_dir', default='vis_crossmodal')
    ap.add_argument('--num_samples', type=int, default=6)
    ap.add_argument('--visible', default='rgb_depth',
                    help="underscore-joined visible modalities, e.g. rgb_depth, "
                         "rgb, depth, rgb_depth_pc")
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    visible = [m for m in args.visible.split('_') if m in MODALITIES]
    assert visible, f"no valid modalities in --visible={args.visible}"
    assert set(visible) != set(MODALITIES), "leave at least one modality to generate"
    print(f"Visible (conditioning) modalities : {visible}")
    print(f"Generating                        : {[m for m in MODALITIES if m not in visible]}")

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
        pc_loss_name=cfg['model'].get('loss_name', 'chamfer'),
        qal_threshold=cfg['model'].get('qal_threshold', 0.01),
        qal_alpha=cfg['model'].get('qal_alpha', 100.0),
        qal_use_squared=cfg['model'].get('qal_use_squared', False),
        # Generation discards the training objective, so avoid the extra
        # full-cloud Sinkhorn computation here.
        sinkhorn_loss_weight=0.0,
        max_leaves=cfg['model']['max_leaves'],
        spline_loss_weight=cfg['model']['spline_loss_weight'],
        depth_norm_type=cfg['model']['depth_norm_type'])

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ckpt['model_state_dict']
    if list(sd)[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dev).eval()
    _vl = ckpt.get('val_loss', ckpt.get('best_val_loss'))
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"val_loss {_vl:.4f})" if _vl is not None
          else f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    rgb, depth, pc, params, text_valid, names = next(iter(loader))
    rgb, depth, pc = rgb.to(dev), depth.to(dev), pc.to(dev)
    params, text_valid = params.to(dev), text_valid.to(dev)

    pred_rgb, pred_depth, pred_pc, pred_params, masks = forward_crossmodal(
        model, rgb, depth, pc, params, visible)

    gt_text  = model.decode_params_to_text(params)
    gen_text = model.decode_params_to_text(pred_params)

    summary = []
    B = rgb.shape[0]
    for i in range(B):
        cd = chamfer_distance(pred_pc[i:i+1], pc[i:i+1]).item()
        # random-plant baseline: chamfer vs a DIFFERENT sample's GT cloud
        j = (i + 1) % B
        cd_rand = chamfer_distance(pc[j:j+1], pc[i:i+1]).item()
        perr = param_errors(model, pred_params[i:i+1], params[i:i+1], text_valid[i:i+1])

        sample = {
            'name': names[i], 'rgb': rgb[i].cpu(), 'depth': depth[i].cpu(),
            'pc': pc[i].cpu(), 'pred_pc': pred_pc[i].cpu(),
            'text_valid': text_valid[i].cpu(),
            'gt_text': gt_text[i], 'gen_text': gen_text[i],
            'chamfer': cd, 'chamfer_rand': cd_rand, 'param_err': perr,
        }
        sdir = out / f"sample_{i+1:02d}_{names[i]}"
        sdir.mkdir(exist_ok=True)
        visualize(sample, sdir / "crossmodal.png", visible)
        save_ply(pc[i].cpu().numpy(),      sdir / "gt_pointcloud.ply")
        save_ply(pred_pc[i].cpu().numpy(), sdir / "generated_pointcloud.ply")
        with open(sdir / "params.json", 'w') as f:
            json.dump({'name': names[i], 'visible': visible,
                       'chamfer': cd, 'chamfer_random_baseline': cd_rand,
                       'param_err_norm': perr,
                       'gt_text': gt_text[i], 'gen_text': gen_text[i]}, f, indent=2)

        summary.append({'name': names[i], 'chamfer': cd, 'chamfer_rand': cd_rand,
                        'plant_mae': perr['plant_mae_norm'],
                        'leaf_mae': perr['leaf_mae_norm']})
        print(f"  [{i+1}/{B}] {names[i]:<22} chamfer={cd:.5f}  "
              f"(rand {cd_rand:.5f})  plantMAE={perr['plant_mae_norm']:.3f}  "
              f"leafMAE={perr['leaf_mae_norm']:.3f}")

    # aggregate
    arr = lambda k: np.array([s[k] for s in summary])
    agg = {
        'visible': visible, 'n': B, 'epoch': ckpt.get('epoch'),
        'chamfer_mean': float(arr('chamfer').mean()),
        'chamfer_random_baseline_mean': float(arr('chamfer_rand').mean()),
        'plant_mae_norm_mean': float(np.nanmean(arr('plant_mae'))),
        'leaf_mae_norm_mean': float(np.nanmean(arr('leaf_mae'))),
        'per_sample': summary,
    }
    json.dump(agg, open(out / "summary.json", 'w'), indent=2)

    print("\n" + "=" * 70)
    print(f"CROSS-MODAL GENERATION  ({'+'.join(visible).upper()}  ->  "
          f"{'+'.join(m for m in MODALITIES if m not in visible).upper()})")
    print("=" * 70)
    print(f"PC Chamfer (generated vs GT) : {agg['chamfer_mean']:.5f}")
    print(f"PC Chamfer (random plant)    : {agg['chamfer_random_baseline_mean']:.5f}  <- baseline")
    print(f"Plant param MAE (norm [0,1]) : {agg['plant_mae_norm_mean']:.3f}")
    print(f"Leaf  param MAE (norm [0,1]) : {agg['leaf_mae_norm_mean']:.3f}")
    print("=" * 70)
    print("Interpretation: chamfer << random-baseline  => generation is")
    print("plant-specific, not just an average shape. MAE near 0 => params")
    print("recovered; MAE near random (~0.33 for U[0,1]) => not recovered.")
    print(f"\nSaved per-sample PLYs + figures to: {out}/")


if __name__ == '__main__':
    main()
