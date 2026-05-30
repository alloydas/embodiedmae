"""
Evaluation script for EmbodiedMAE-4M.

Saves all four modalities (RGB / Depth / PointCloud / Spline params) for every
evaluated sample — both the original inputs and the model's reconstructions —
plus the masking pattern used and per-sample metrics.

Usage:
    python validate_4m.py \
        --checkpoint outputs/<run>/best_model.pth \
        --output_dir vis_val_4m \
        --config config_4m.yaml \
        --mask_ratio 0.75 \
        --num_samples 12

Output layout:
    output_dir/
        config_used.json            — every knob this run was launched with
        metrics_aggregate.json      — averaged metrics across saved samples
        sample_<name>/
            inputs/
                rgb.png, rgb.pt
                depth.png, depth.npy
                pointcloud.ply, pointcloud.npy
                params.json
            outputs/
                rgb.png, rgb.pt
                depth.png, depth.npy
                pointcloud.ply, pointcloud.npy
                params.json
            masks/
                rgb_mask.npy, depth_mask.npy
                pc_mask.npy, text_mask.npy
            metrics.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from embodied_mae import chamfer_distance, earth_movers_distance
from embodied_mae_4m import (
    N_PARAMS,
    embodied_mae_4m_base,
    embodied_mae_4m_small,
    _PLANT_SCALE, _PLANT_SHIFT,
    _LEAF_SCALE,  _LEAF_SHIFT,
)
from sorghum_dataset_4m import SorghumDataset4M


# ── I/O helpers ──────────────────────────────────────────────────────────────

_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_RGB_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def save_rgb(t: torch.Tensor, path: Path, denormalize=True):
    if denormalize:
        t = t * _RGB_STD + _RGB_MEAN
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_depth(t: torch.Tensor, path_png: Path, path_npy: Path):
    d = t.squeeze().numpy()
    np.save(path_npy, d)
    lo, hi = float(d.min()), float(d.max())
    if hi > lo:
        d_vis = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        d_vis = (d * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(d_vis, mode='L').save(path_png)


def save_pointcloud(t: torch.Tensor, path_ply: Path, path_npy: Path):
    pts = t.numpy()
    np.save(path_npy, pts)
    with open(path_ply, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")


def unpatchify(x: torch.Tensor, patch: int, channels: int, img_size: int):
    h = w = img_size // patch
    x = x.reshape(x.shape[0], h, w, patch, patch, channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(x.shape[0], channels, h * patch, w * patch)


# ── Param decoding ───────────────────────────────────────────────────────────

_PLANT_KEYS = ['sl', 'sd_x', 'sd_y', 'sd_z', 'ps_x', 'ps_y', 'ps_z', 'pa', 'pr']
_LEAF_KEYS  = ['sp', 'ln', 'ra', 'ba', 'wf', 'wp0', 'wp1']  # last 2 unused


def _denorm(norm: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return np.clip(norm, 0.0, 1.0) * scale - shift


def _params_dict(norm: np.ndarray, scale, shift, keys, keep=None):
    """Return {key: raw_value} for the first `keep` entries (or all)."""
    raw = _denorm(norm, scale, shift)
    if keep is None:
        keep = len(keys)
    return {k: float(raw[i]) for i, k in enumerate(keys[:keep])}


def build_params_record(target_norm: np.ndarray,
                        pred_norm:   np.ndarray,
                        text_valid:  np.ndarray,
                        text_mask:   np.ndarray) -> dict:
    """Build a per-sample dict of plant + leaves with norm / raw / diff per token."""
    rec = {'plant': {}, 'leaves': []}
    n_tokens = target_norm.shape[0]

    for ti in range(n_tokens):
        if text_valid[ti] < 0.5:
            continue
        is_masked = bool(text_mask[ti] > 0.5)
        if ti == 0:
            tgt_raw  = _params_dict(target_norm[ti], _PLANT_SCALE, _PLANT_SHIFT, _PLANT_KEYS)
            pred_raw = _params_dict(pred_norm[ti],   _PLANT_SCALE, _PLANT_SHIFT, _PLANT_KEYS)
            rec['plant'] = {
                'masked':       is_masked,
                'target_norm':  target_norm[ti].tolist(),
                'pred_norm':    pred_norm[ti].tolist(),
                'target_raw':   tgt_raw,
                'pred_raw':     pred_raw,
                'abs_diff_raw': {k: abs(tgt_raw[k] - pred_raw[k]) for k in tgt_raw},
            }
        else:
            tgt_raw  = _params_dict(target_norm[ti], _LEAF_SCALE, _LEAF_SHIFT, _LEAF_KEYS, keep=7)
            pred_raw = _params_dict(pred_norm[ti],   _LEAF_SCALE, _LEAF_SHIFT, _LEAF_KEYS, keep=7)
            rec['leaves'].append({
                'leaf_index':   ti,
                'masked':       is_masked,
                'target_norm':  target_norm[ti, :7].tolist(),
                'pred_norm':    pred_norm[ti, :7].tolist(),
                'target_raw':   tgt_raw,
                'pred_raw':     pred_raw,
                'abs_diff_raw': {k: abs(tgt_raw[k] - pred_raw[k]) for k in tgt_raw},
            })
    return rec


# ── Metrics helpers ──────────────────────────────────────────────────────────

def per_sample_metrics(rgb, pred_rgb_img, depth, pred_depth_img, pc, pred_pc,
                       params, pred_params, text_valid, text_mask, compute_emd):
    rgb_mse    = float(((pred_rgb_img - rgb) ** 2).mean())
    depth_mse  = float(((pred_depth_img - depth) ** 2).mean())
    pc_chamfer = float(chamfer_distance(pred_pc.unsqueeze(0), pc.unsqueeze(0)))
    pc_emd     = (float(earth_movers_distance(pred_pc.unsqueeze(0), pc.unsqueeze(0), num_samples=500))
                  if compute_emd else 0.0)

    valid_mask = (text_valid > 0.5)
    eff_mask   = valid_mask & (text_mask > 0.5)
    abs_diff   = (pred_params - params).abs()
    param_mae_all       = float(abs_diff[valid_mask].mean()) if valid_mask.any() else 0.0
    param_mae_masked    = float(abs_diff[eff_mask].mean())   if eff_mask.any()   else 0.0
    return {
        'rgb_mse':           rgb_mse,
        'depth_mse':         depth_mse,
        'pc_chamfer':        pc_chamfer,
        'pc_emd':            pc_emd,
        'param_mae_all':     param_mae_all,
        'param_mae_masked':  param_mae_masked,
    }


# ── Per-sample writer ────────────────────────────────────────────────────────

def save_sample(out_dir: Path, name: str,
                rgb, depth, pc, params_norm, text_valid,
                pred_rgb_img, pred_depth_img, pred_pc, pred_params_norm,
                m_rgb, m_depth, m_pc, m_text,
                metrics: dict):
    s = out_dir / f'sample_{name}'
    (s / 'inputs').mkdir(parents=True, exist_ok=True)
    (s / 'outputs').mkdir(parents=True, exist_ok=True)
    (s / 'masks').mkdir(parents=True, exist_ok=True)

    # Inputs
    torch.save(rgb, s / 'inputs' / 'rgb.pt')
    save_rgb(rgb, s / 'inputs' / 'rgb.png')
    save_depth(depth, s / 'inputs' / 'depth.png', s / 'inputs' / 'depth.npy')
    save_pointcloud(pc, s / 'inputs' / 'pointcloud.ply', s / 'inputs' / 'pointcloud.npy')

    # Outputs
    torch.save(pred_rgb_img, s / 'outputs' / 'rgb.pt')
    save_rgb(pred_rgb_img, s / 'outputs' / 'rgb.png')
    save_depth(pred_depth_img, s / 'outputs' / 'depth.png', s / 'outputs' / 'depth.npy')
    save_pointcloud(pred_pc, s / 'outputs' / 'pointcloud.ply', s / 'outputs' / 'pointcloud.npy')

    # Params (input + output combined into one record)
    rec = build_params_record(
        target_norm=params_norm.numpy(),
        pred_norm=pred_params_norm.numpy(),
        text_valid=text_valid.numpy(),
        text_mask=m_text.numpy(),
    )
    with open(s / 'inputs' / 'params.json', 'w') as f:
        json.dump({'plant': {'target_raw': rec['plant'].get('target_raw'),
                              'target_norm': rec['plant'].get('target_norm'),
                              'masked': rec['plant'].get('masked')},
                   'leaves': [{k: v for k, v in lf.items()
                               if k in ('leaf_index', 'masked', 'target_raw', 'target_norm')}
                              for lf in rec['leaves']]},
                  f, indent=2)
    with open(s / 'outputs' / 'params.json', 'w') as f:
        json.dump(rec, f, indent=2)

    # Masks (1 = masked, 0 = visible)
    np.save(s / 'masks' / 'rgb_mask.npy',   m_rgb.numpy().astype(np.uint8))
    np.save(s / 'masks' / 'depth_mask.npy', m_depth.numpy().astype(np.uint8))
    np.save(s / 'masks' / 'pc_mask.npy',    m_pc.numpy().astype(np.uint8))
    np.save(s / 'masks' / 'text_mask.npy',  m_text.numpy().astype(np.uint8))

    with open(s / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate EmbodiedMAE-4M and dump all modalities')
    parser.add_argument('--checkpoint',     type=str, required=True)
    parser.add_argument('--output_dir',     type=str, required=True)
    parser.add_argument('--config',         type=str, default='config_4m.yaml')
    parser.add_argument('--data_root',      type=str, default=None,
                        help='Override data.data_root from the YAML')
    parser.add_argument('--split',          type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--mask_ratio',     type=float, default=None,
                        help='Override model.mask_ratio (the only knob you '
                             'usually want to sweep at eval time)')
    parser.add_argument('--num_samples',    type=int, default=12,
                        help='Stop after saving this many samples')
    parser.add_argument('--batch_size',     type=int, default=8)
    parser.add_argument('--num_workers',    type=int, default=4)
    parser.add_argument('--device',         type=str, default='cuda')
    parser.add_argument('--seed',           type=int, default=42,
                        help='Seeds the Dirichlet draw + per-modality token '
                             'shuffles so masks are reproducible run-to-run')
    parser.add_argument('--compute_emd',    action='store_true',
                        help='Also compute EMD per sample (slow)')
    args = parser.parse_args()

    # Config — YAML defaults, CLI overrides
    cfg = yaml.safe_load(open(args.config))
    data_root      = args.data_root or cfg['data']['data_root']
    img_size       = cfg['data'].get('img_size',       224)
    num_points     = cfg['data'].get('num_points',     8196)
    model_size     = cfg['model'].get('model_size',    'base')
    mask_ratio     = args.mask_ratio if args.mask_ratio is not None \
                     else cfg['model'].get('mask_ratio', 0.15)
    pc_loss_weight = cfg['model'].get('pc_loss_weight',     10.0)
    spline_loss_w  = cfg['model'].get('spline_loss_weight',  5.0)
    depth_norm     = cfg['model'].get('depth_norm_type', 'minmax')
    max_leaves     = cfg['model'].get('max_leaves',         24)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used so eval runs are auditable
    with open(out_dir / 'config_used.json', 'w') as f:
        json.dump({
            'timestamp':         datetime.now().isoformat(timespec='seconds'),
            'checkpoint':        args.checkpoint,
            'data_root':         data_root,
            'split':             args.split,
            'img_size':          img_size,
            'num_points':        num_points,
            'model_size':        model_size,
            'mask_ratio':        mask_ratio,
            'pc_loss_weight':    pc_loss_weight,
            'spline_loss_weight': spline_loss_w,
            'depth_norm_type':   depth_norm,
            'max_leaves':        max_leaves,
            'seed':              args.seed,
            'num_samples':       args.num_samples,
            'batch_size':        args.batch_size,
            'compute_emd':       args.compute_emd,
        }, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Reproducible masking
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print(f"Loading dataset from: {data_root} (split={args.split})")
    ds = SorghumDataset4M(data_root, img_size=img_size, num_points=num_points,
                           split=args.split, max_leaves=max_leaves)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # Model
    build_fn = embodied_mae_4m_small if model_size == 'small' else embodied_mae_4m_base
    model = build_fn(
        img_size=img_size,
        num_pc_tokens=196,
        target_points=num_points,
        pc_loss_weight=pc_loss_weight,
        max_leaves=max_leaves,
        spline_loss_weight=spline_loss_w,
        depth_norm_type=depth_norm,
    ).to(device)

    # Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck['model_state_dict']
    if next(iter(sd.keys())).startswith('module.'):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"  epoch={ck.get('epoch', '?')}  best_val={ck.get('best_val_loss', '?')}")

    # Eval loop
    n_saved = 0
    agg_metrics = {k: 0.0 for k in
                   ('rgb_mse', 'depth_mse', 'pc_chamfer', 'pc_emd',
                    'param_mae_all', 'param_mae_masked')}

    with torch.no_grad():
        for rgb, depth, pc, params, text_valid, names in tqdm(dl, desc='Eval'):
            if n_saved >= args.num_samples:
                break

            rgb_d        = rgb.to(device)
            depth_d      = depth.to(device)
            pc_d         = pc.to(device)
            params_d     = params.to(device)
            text_valid_d = text_valid.to(device)

            _, _, (pred_rgb_p, pred_depth_p, pred_pc, pred_params), \
                (m_rgb, m_depth, m_pc, m_text) = model(
                    rgb_d, depth_d, pc_d, params_d, text_valid_d,
                    mask_ratio=mask_ratio,
            )

            B = rgb.shape[0]
            pred_rgb_img   = unpatchify(pred_rgb_p,   model.patch_size, 3, model.img_size).cpu()
            pred_depth_img = unpatchify(pred_depth_p, model.patch_size, 1, model.img_size).cpu()
            pred_pc_cpu    = pred_pc.cpu()
            pred_params_cp = pred_params.clamp(0, 1).cpu()
            m_rgb_cp       = m_rgb.cpu()
            m_depth_cp     = m_depth.cpu()
            m_pc_cp        = m_pc.cpu()
            m_text_cp      = m_text.cpu()

            for i in range(B):
                if n_saved >= args.num_samples:
                    break
                metrics = per_sample_metrics(
                    rgb=rgb[i], pred_rgb_img=pred_rgb_img[i],
                    depth=depth[i], pred_depth_img=pred_depth_img[i],
                    pc=pc[i], pred_pc=pred_pc_cpu[i],
                    params=params[i], pred_params=pred_params_cp[i],
                    text_valid=text_valid[i], text_mask=m_text_cp[i],
                    compute_emd=args.compute_emd,
                )
                save_sample(
                    out_dir, names[i],
                    rgb=rgb[i], depth=depth[i], pc=pc[i],
                    params_norm=params[i], text_valid=text_valid[i],
                    pred_rgb_img=pred_rgb_img[i], pred_depth_img=pred_depth_img[i],
                    pred_pc=pred_pc_cpu[i], pred_params_norm=pred_params_cp[i],
                    m_rgb=m_rgb_cp[i], m_depth=m_depth_cp[i],
                    m_pc=m_pc_cp[i], m_text=m_text_cp[i],
                    metrics=metrics,
                )
                for k, v in metrics.items():
                    agg_metrics[k] += v
                n_saved += 1

    if n_saved == 0:
        print("⚠️  No samples evaluated — check dataset paths.")
        return

    agg = {k: v / n_saved for k, v in agg_metrics.items()}
    agg['n_samples']     = n_saved
    agg['mask_ratio']    = mask_ratio
    with open(out_dir / 'metrics_aggregate.json', 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"\nSaved {n_saved} samples to {out_dir}")
    print("Aggregate:")
    for k, v in agg.items():
        print(f"  {k:20s} {v}")


if __name__ == '__main__':
    main()
