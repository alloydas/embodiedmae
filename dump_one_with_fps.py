"""Run the model on val samples, capturing the FPS center indices so we can
visualise *exactly* which points the model saw / masked.

Outputs (per sample) added to the same layout as validate_4m.py, plus:
    masks/fps_idx.npy          — (num_pc_tokens,) int64, indices into the input PC
    masks/pc_vis_point_idx.npy — indices of points belonging to visible tokens
"""
import argparse, json, sys
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
    N_PARAMS, embodied_mae_4m_base, embodied_mae_4m_small,
    _PLANT_SCALE, _PLANT_SHIFT, _LEAF_SCALE, _LEAF_SHIFT,
)
from sorghum_dataset_4m import SorghumDataset4M
from validate_4m import (
    save_rgb, save_depth, save_pointcloud, unpatchify,
    build_params_record, per_sample_metrics, save_sample,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config',     default='config_4m.yaml')
    ap.add_argument('--mask_ratio', type=float, default=None,
                    help='Override model.mask_ratio from the YAML config')
    ap.add_argument('--num_samples', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--seed',       type=int, default=7)
    ap.add_argument('--device',     default='cpu')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root  = cfg['data']['data_root']
    img_size   = cfg['data'].get('img_size', 224)
    num_points = cfg['data'].get('num_points', 8196)
    model_size = cfg['model'].get('model_size', 'base')
    pc_loss_w  = cfg['model'].get('pc_loss_weight', 10.0)
    spline_w   = cfg['model'].get('spline_loss_weight', 5.0)
    depth_norm = cfg['model'].get('depth_norm_type', 'minmax')
    max_leaves = cfg['model'].get('max_leaves', 24)
    mask_ratio = (args.mask_ratio if args.mask_ratio is not None
                  else cfg['model'].get('mask_ratio', 0.15))
    pc_deterministic_fps = cfg['model'].get('pc_deterministic_fps', False)
    pc_add_center_coordinates = cfg['model'].get(
        'pc_add_center_coordinates', False)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = SorghumDataset4M(data_root, img_size=img_size, num_points=num_points,
                          split='val', max_leaves=max_leaves)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)

    build_fn = embodied_mae_4m_small if model_size == 'small' else embodied_mae_4m_base
    model = build_fn(img_size=img_size, num_pc_tokens=196, target_points=num_points,
                    pc_loss_weight=pc_loss_w, max_leaves=max_leaves,
                    spline_loss_weight=spline_w, depth_norm_type=depth_norm,
                    pc_deterministic_fps=pc_deterministic_fps,
                    pc_add_center_coordinates=pc_add_center_coordinates).to(device)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck['model_state_dict']
    if next(iter(sd.keys())).startswith('module.'):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # Monkey-patch the PC embed to capture FPS indices each call
    captured = {}
    orig_fps = model.pc_embed.fps
    def fps_capture(xyz, npoint):
        idx = orig_fps(xyz, npoint)
        captured['fps_idx'] = idx.detach().cpu().numpy()   # (B, num_tokens)
        return idx
    model.pc_embed.fps = fps_capture

    # Also capture kNN grouping idx by patching knn_grouping
    orig_knn = model.pc_embed.knn_grouping
    def knn_capture(xyz, points, fps_idx):
        # mirror the internal computation to save knn indices
        # xyz: (B, N, 3); fps_idx: (B, S)
        B, N, _ = xyz.shape
        S = fps_idx.shape[1]
        centroids = xyz[torch.arange(B, device=xyz.device)[:, None], fps_idx]  # (B, S, 3)
        dist = torch.cdist(centroids, xyz)                                     # (B, S, N)
        _, knn_idx = torch.topk(dist, model.pc_embed.group_size, dim=2, largest=False)
        captured['knn_idx'] = knn_idx.detach().cpu().numpy()                  # (B, S, k)
        return orig_knn(xyz, points, fps_idx)
    model.pc_embed.knn_grouping = knn_capture

    n_saved = 0
    with torch.no_grad():
        for rgb, depth, pc, params, text_valid, names in tqdm(dl):
            if n_saved >= args.num_samples: break
            rgb_d, depth_d, pc_d = rgb.to(device), depth.to(device), pc.to(device)
            params_d, tv_d = params.to(device), text_valid.to(device)

            _, _, (pred_rgb_p, pred_depth_p, pred_pc, pred_params), \
                (m_rgb, m_depth, m_pc, m_text) = model(
                    rgb_d, depth_d, pc_d, params_d, tv_d,
                    mask_ratio=mask_ratio)

            B = rgb.shape[0]
            pred_rgb_img   = unpatchify(pred_rgb_p,   model.patch_size, 3, model.img_size).cpu()
            pred_depth_img = unpatchify(pred_depth_p, model.patch_size, 1, model.img_size).cpu()
            pred_pc_cpu    = pred_pc.cpu()
            pred_params_cp = pred_params.clamp(0, 1).cpu()
            m_rgb_cp, m_depth_cp = m_rgb.cpu(), m_depth.cpu()
            m_pc_cp,  m_text_cp  = m_pc.cpu(),  m_text.cpu()
            fps_idx_b = captured['fps_idx']   # (B, S)
            knn_idx_b = captured['knn_idx']   # (B, S, k)

            for i in range(B):
                if n_saved >= args.num_samples: break
                metrics = per_sample_metrics(
                    rgb=rgb[i], pred_rgb_img=pred_rgb_img[i],
                    depth=depth[i], pred_depth_img=pred_depth_img[i],
                    pc=pc[i], pred_pc=pred_pc_cpu[i],
                    params=params[i], pred_params=pred_params_cp[i],
                    text_valid=text_valid[i], text_mask=m_text_cp[i],
                    compute_emd=False,
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
                # Save the FPS + visible point indices for this sample
                s_dir = Path(out_dir) / f'sample_{names[i]}'
                np.save(s_dir / 'masks' / 'fps_idx.npy', fps_idx_b[i])
                # visible point indices = union of kNN groups of visible (m_pc==0) tokens
                vis_tok = np.where(m_pc_cp[i].numpy() < 0.5)[0]
                vis_pts = np.unique(knn_idx_b[i, vis_tok].reshape(-1))
                np.save(s_dir / 'masks' / 'pc_vis_point_idx.npy', vis_pts)
                n_saved += 1

    print(f'Saved {n_saved} samples to {out_dir}')


if __name__ == '__main__':
    main()
