#!/usr/bin/env python3
"""Regenerate all-four-source cross-modal visualizations from the best distill
checkpoint. Reuses the exact `visualize_crossmodal` from the training script.

Runs on CPU by default so it never contends for GPU memory with a live training
job sharing the node. Override with --device cuda if the GPU is free.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

import train_sorghum_4m_distill as T
from sorghum_dataset_4m import SorghumDataset4M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_distill_v1/best_model.pth')
    ap.add_argument('--data_root', default='./Dataset/new_data')
    ap.add_argument('--out_dir', default='outputs/4m_distill_v1/visualizations_best')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--num_samples', type=int, default=6)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--max_leaves', type=int, default=24)
    a = ap.parse_args()

    device = torch.device(a.device)

    args = SimpleNamespace(
        model_size='base', img_size=a.img_size, num_points=a.num_points,
        pc_loss_weight=10.0, max_leaves=a.max_leaves,
        spline_loss_weight=5.0, depth_norm_type='minmax',
    )
    model = T.build_model(args, device)
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    epoch = ck.get('epoch', 0)
    T.load_weights_into(model, a.checkpoint, device, 'best')
    model.eval()

    val_ds = SorghumDataset4M(a.data_root, img_size=a.img_size,
                              num_points=a.num_points, split='val',
                              max_leaves=a.max_leaves)
    # deterministic order (no shuffle) so the same plants appear as before
    loader = DataLoader(val_ds, batch_size=a.num_samples, shuffle=False,
                        num_workers=2, collate_fn=getattr(val_ds, 'collate_fn', None))

    out = Path(a.out_dir)
    for src in ['rgb', 'depth', 'pc', 'text']:
        print(f"→ generating source={src} …")
        with torch.no_grad():
            T.visualize_crossmodal(model, loader, device, epoch, out, src,
                                   num_samples=a.num_samples)
    print(f"\n✅ saved to {out}  (epoch {epoch})")


if __name__ == '__main__':
    main()
