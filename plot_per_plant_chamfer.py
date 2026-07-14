"""
plot_per_plant_chamfer.py — per-plant RGB→PointCloud chamfer for the 4M model.

Runs the encoder with ONLY RGB visible (depth/pc/text fully masked), generates
the point cloud for every validation plant, and records the per-plant bidirectional
Chamfer distance vs ground truth. Saves a sorted bar chart + a JSON of the numbers.

Usage:
    python plot_per_plant_chamfer.py \
        --checkpoint outputs/4m_distill_rgb2pc/best_model.pth \
        --config configs/config_4m_distill_rgb2pc.yaml \
        --output per_plant_chamfer_rgb2pc.png
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from embodied_mae_4m import embodied_mae_4m_base, embodied_mae_4m_small, chamfer_distance
from sorghum_dataset_4m import SorghumDataset4M
from generate_crossmodal import forward_crossmodal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_distill_rgb2pc/best_model.pth')
    ap.add_argument('--config',     default='configs/config_4m_distill_rgb2pc.yaml')
    ap.add_argument('--visible',    default='rgb')
    ap.add_argument('--output',     default='per_plant_chamfer_rgb2pc.png')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--data_root',  default=None,
                    help='override cfg data.data_root (e.g. an external test set)')
    ap.add_argument('--split',      default='val',
                    help="subfolder under data_root, or 'none' if data_root IS the split dir")
    ap.add_argument('--limit',      type=int, default=0,
                    help='evaluate only the first N plants (0 = all)')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    visible = [m for m in args.visible.split('_') if m in ('rgb', 'depth', 'pc', 'text')]

    data_root = args.data_root or cfg['data']['data_root']
    split = None if args.split.lower() == 'none' else args.split
    ds = SorghumDataset4M(data_root, img_size=cfg['data']['img_size'],
                          num_points=cfg['data']['num_points'], split=split,
                          max_leaves=cfg['model']['max_leaves'])
    if args.limit and args.limit < len(ds):
        ds.samples = ds.samples[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    build = embodied_mae_4m_small if cfg['model']['model_size'] == 'small' else embodied_mae_4m_base
    model = build(img_size=cfg['data']['img_size'], num_pc_tokens=196,
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
    print(f"Loaded {args.checkpoint} (epoch {epoch}); conditioning on {visible}")

    names, chamfers = [], []
    with torch.no_grad():
        for rgb, depth, pc, params, tv, nm in loader:
            rgb, depth, pc = rgb.to(dev), depth.to(dev), pc.to(dev)
            params, tv = params.to(dev), tv.to(dev)
            _, _, pred_pc, _, _ = forward_crossmodal(model, rgb, depth, pc, params, visible)
            for i in range(rgb.shape[0]):
                cd = chamfer_distance(pred_pc[i:i+1], pc[i:i+1]).item()
                names.append(nm[i]); chamfers.append(cd)

    chamfers = np.array(chamfers)
    order = np.argsort(chamfers)
    names_s = [names[i] for i in order]
    ch_s = chamfers[order]
    mean, med = chamfers.mean(), np.median(chamfers)
    print(f"N={len(chamfers)}  mean={mean:.5f}  median={med:.5f}  "
          f"min={ch_s[0]:.5f} ({names_s[0]})  max={ch_s[-1]:.5f} ({names_s[-1]})")

    # ── plot ──────────────────────────────────────────────────────────────────
    n = len(ch_s)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.16), 5.5))
    x = np.arange(n)
    ax.bar(x, ch_s, color='#2a9d8f', width=0.85, zorder=3)
    ax.axhline(mean, color='#e76f51', lw=1.6, ls='--', zorder=4,
               label=f'mean {mean:.5f}')
    ax.axhline(med, color='#264653', lw=1.4, ls=':', zorder=4,
               label=f'median {med:.5f}')
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=90, fontsize=6)
    ax.set_ylabel('Chamfer distance (RGB → point cloud)')
    ax.set_xlabel('validation plant (sorted best → worst)')
    ax.set_title(f"Per-plant RGB→PC chamfer  |  {Path(args.checkpoint).parent.name} "
                 f"epoch {epoch}  |  N={n}", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(frameon=False)
    ax.set_xlim(-1, n)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved {args.output}")

    js = Path(args.output).with_suffix('.json')
    json.dump({'checkpoint': args.checkpoint, 'epoch': epoch, 'visible': visible,
               'mean': float(mean), 'median': float(med),
               'per_plant': [{'name': n_, 'chamfer': float(c)}
                             for n_, c in zip(names_s, ch_s)]},
              open(js, 'w'), indent=2)
    print(f"Saved {js}")


if __name__ == '__main__':
    main()
