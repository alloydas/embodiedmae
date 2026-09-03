#!/usr/bin/env python3
"""Generate RGB-source -> all-modality cross-modal figures for N plants from the
best distilled checkpoint, and assemble them into a single multi-page PDF
(one plant per page).

Reuses the exact `visualize_crossmodal` panels from the training script.
Runs on CPU by default to avoid contending with a live GPU training job.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image

import train_sorghum_4m_distill as T
from sorghum_dataset_4m import SorghumDataset4M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_distill_v1/best_model.pth')
    ap.add_argument('--data_root', default='./Dataset/new_data')
    ap.add_argument('--pdf', default='outputs/4m_distill_v1/rgb_source_generation.pdf')
    ap.add_argument('--png_dir', default='outputs/4m_distill_v1/viz_rgb_source_best')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--num_plants', type=int, default=10)
    ap.add_argument('--views_per_plant', type=int, default=4)
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

    # Folder names are Sorghum_<plant>_<view>; the _NN suffix indexes different
    # camera VIEWS of the same plant. Group indices by plant, then keep the
    # first `views_per_plant` views of the first `num_plants` distinct plants.
    from collections import OrderedDict
    by_plant = OrderedDict()
    for idx, folder in enumerate(val_ds.samples):
        plant = folder.name.split('_')[1]          # "Sorghum_3_07" -> "3"
        by_plant.setdefault(plant, []).append(idx)

    selected = OrderedDict()
    for plant, idxs in by_plant.items():
        if len(selected) >= a.num_plants:
            break
        selected[plant] = idxs[:a.views_per_plant]
    total = sum(len(v) for v in selected.values())
    print(f"  {len(selected)} plants x up to {a.views_per_plant} views = {total} figures")

    out_png = Path(a.png_dir)
    with torch.no_grad():
        for plant, idxs in selected.items():
            names = [val_ds.samples[i].name for i in idxs]
            print(f"→ plant {plant}: {names}")
            loader = DataLoader(Subset(val_ds, idxs), batch_size=len(idxs),
                                shuffle=False, num_workers=2)
            T.visualize_crossmodal(model, loader, device, epoch, out_png, 'rgb',
                                   num_samples=len(idxs))

    # Assemble every view into one PDF, ordered by (plant, view) so each plant's
    # views appear together and in sequence.
    def sort_key(p):
        name = p.stem.split('_sample_')[1].split('_', 1)[1]  # -> "Sorghum_3_01"
        _, plant, view = name.split('_')
        return (int(plant), int(view))
    pngs = sorted(out_png.glob(f'epoch_{epoch:03d}_src-rgb_sample_*.png'), key=sort_key)
    imgs = [Image.open(p).convert('RGB') for p in pngs]
    pdf_path = Path(a.pdf)
    imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
    print(f"\n✅ {len(imgs)} figures ({len(selected)} plants) → {pdf_path}  (epoch {epoch})")


if __name__ == '__main__':
    main()
