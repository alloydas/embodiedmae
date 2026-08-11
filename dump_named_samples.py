"""Dump a specific list of sample names through the EmbodiedMAE-4M eval pipeline.

Wraps `validate_4m.py`'s logic but filters the dataset to only the samples
whose folder name appears in `--names`.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from embodied_mae_4m import embodied_mae_4m_base, embodied_mae_4m_small
from sorghum_dataset_4m import SorghumDataset4M
from validate_4m import (
    per_sample_metrics, save_sample, unpatchify,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default='config_4m.yaml')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--mask_ratio', type=float, default=None)
    ap.add_argument('--names', nargs='+', required=True,
                    help='Sample folder names (e.g. Sorghum_3_05)')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--compute_emd', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root  = cfg['data']['data_root']
    img_size   = cfg['data'].get('img_size', 224)
    num_points = cfg['data'].get('num_points', 8196)
    model_size = cfg['model'].get('model_size', 'base')
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg['model'].get('mask_ratio', 0.15)
    pc_w       = cfg['model'].get('pc_loss_weight', 10.0)
    spline_w   = cfg['model'].get('spline_loss_weight', 5.0)
    depth_norm = cfg['model'].get('depth_norm_type', 'minmax')
    max_leaves = cfg['model'].get('max_leaves', 24)
    pc_deterministic_fps = cfg['model'].get('pc_deterministic_fps', False)
    pc_add_center_coordinates = cfg['model'].get(
        'pc_add_center_coordinates', False)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

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
            'pc_loss_weight':    pc_w,
            'spline_loss_weight': spline_w,
            'depth_norm_type':   depth_norm,
            'max_leaves':        max_leaves,
            'pc_deterministic_fps': pc_deterministic_fps,
            'pc_add_center_coordinates': pc_add_center_coordinates,
            'seed':              args.seed,
            'names':             args.names,
            'compute_emd':       args.compute_emd,
        }, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Loading dataset from: {data_root}  (split={args.split})")
    ds = SorghumDataset4M(data_root, img_size=img_size, num_points=num_points,
                          split=args.split, max_leaves=max_leaves)
    name_set = set(args.names)
    ds.samples = [p for p in ds.samples if p.name in name_set]
    # Preserve the user-supplied order
    order = {n: i for i, n in enumerate(args.names)}
    ds.samples.sort(key=lambda p: order.get(p.name, 9999))
    print(f"Filtered to {len(ds.samples)} requested samples")
    if len(ds.samples) != len(args.names):
        missing = name_set - {p.name for p in ds.samples}
        print(f"⚠️  Missing samples: {sorted(missing)}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    build = embodied_mae_4m_small if model_size == 'small' else embodied_mae_4m_base
    model = build(
        img_size=img_size, num_pc_tokens=196, target_points=num_points,
        pc_loss_weight=pc_w, max_leaves=max_leaves,
        spline_loss_weight=spline_w, depth_norm_type=depth_norm,
        pc_deterministic_fps=pc_deterministic_fps,
        pc_add_center_coordinates=pc_add_center_coordinates,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck['model_state_dict']
    if next(iter(sd.keys())).startswith('module.'):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd); model.eval()
    print(f"  epoch={ck.get('epoch', '?')}  best_val={ck.get('best_val_loss', '?')}")

    n_saved = 0
    agg = {k: 0.0 for k in
           ('rgb_mse', 'depth_mse', 'pc_chamfer', 'pc_emd',
            'param_mae_all', 'param_mae_masked')}

    with torch.no_grad():
        for rgb, depth, pc, params, text_valid, names in tqdm(dl, desc='Eval'):
            rgb_d = rgb.to(device); depth_d = depth.to(device)
            pc_d  = pc.to(device);  params_d = params.to(device)
            tv_d  = text_valid.to(device)

            _, _, (pred_rgb_p, pred_depth_p, pred_pc, pred_params), \
                (m_rgb, m_depth, m_pc, m_text) = model(
                    rgb_d, depth_d, pc_d, params_d, tv_d,
                    mask_ratio=mask_ratio,
                )

            B = rgb.shape[0]
            pred_rgb_img   = unpatchify(pred_rgb_p,   model.patch_size, 3, model.img_size).cpu()
            pred_depth_img = unpatchify(pred_depth_p, model.patch_size, 1, model.img_size).cpu()
            pred_pc_cpu    = pred_pc.cpu()
            pred_params_cp = pred_params.clamp(0, 1).cpu()
            m_rgb_cp       = m_rgb.cpu(); m_depth_cp = m_depth.cpu()
            m_pc_cp        = m_pc.cpu();  m_text_cp  = m_text.cpu()

            for i in range(B):
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
                    agg[k] += v
                n_saved += 1

    if n_saved == 0:
        print('⚠️  No samples saved. Check --names.')
        return

    agg = {k: v / n_saved for k, v in agg.items()}
    agg['n_samples']  = n_saved
    agg['mask_ratio'] = mask_ratio
    with open(out_dir / 'metrics_aggregate.json', 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved {n_saved} samples to {out_dir}")
    for k, v in agg.items():
        print(f"  {k:20s} {v}")


if __name__ == '__main__':
    main()
