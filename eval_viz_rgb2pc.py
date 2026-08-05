"""
eval_viz_rgb2pc.py — visualise what the RGB→PC distilled model actually produces,
per dataset. Conditions on RGB only (depth/PC/text fully masked) and renders the
generated point cloud against ground truth.

  <out>_recon.png  — per dataset: best / median / worst sample.
                     RGB input | GT cloud | predicted cloud | overlay, with Chamfer.
  <out>_dist.png   — Chamfer distribution per dataset (same N each, like-for-like).

Usage:
    python eval_viz_rgb2pc.py --checkpoint outputs/4m_distill_rgb2pc/best_model.pth \
                              --config configs/config_4m_distill_rgb2pc.yaml --n 150
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from embodied_mae_4m import embodied_mae_4m_base, embodied_mae_4m_small, chamfer_distance
from sorghum_dataset_4m import SorghumDataset4M
from generate_crossmodal import forward_crossmodal

GT_C, PR_C = '#264653', '#e76f51'
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

GROUPS = [
    ('old10k / train\n(model trained on these)', '/work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/train'),
    ('old10k / val\n(same plants, new views)',   '/work/mech-ai-scratch/alloy/embodiedmae/Dataset/new_data/val'),
    ('15K / test  (OOD)\n(unseen plants)',       '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/test'),
]


class FolderDS(SorghumDataset4M):
    """SorghumDataset4M over an explicit folder list — skips the (slow) 22.5k-folder scan."""
    def __init__(self, folders, img_size, num_points, max_leaves):
        Dataset.__init__(self)
        self.samples, self.img_size = list(folders), img_size
        self.num_points, self.max_leaves = num_points, max_leaves
        self.rgb_transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                                        T.Normalize(*IMNET)])
        self.depth_transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


def pick(root, n, seed=3):
    fs = sorted(p for p in Path(root).iterdir() if p.is_dir()
                and (p / 'rgb.png').exists() and list(p.glob('*_spline.yml')))
    return random.Random(seed).sample(fs, min(n, len(fs)))


def denorm_rgb(t):
    m, s = np.array(IMNET[0]), np.array(IMNET[1])
    return np.clip(t.permute(1, 2, 0).cpu().numpy() * s + m, 0, 1)


def scat(ax, pts, c, s=0.5, a=0.5):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=c, alpha=a, linewidths=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_distill_rgb2pc/best_model.pth')
    ap.add_argument('--config',     default='configs/config_4m_distill_rgb2pc.yaml')
    ap.add_argument('--n',          type=int, default=150, help='samples scored per dataset')
    ap.add_argument('--out',        default='eval_rgb2pc')
    ap.add_argument('--device',     default='cuda')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    build = embodied_mae_4m_small if cfg['model']['model_size'] == 'small' else embodied_mae_4m_base
    model = build(img_size=cfg['data']['img_size'], num_pc_tokens=196,
                  target_points=cfg['data']['num_points'],
                  pc_loss_weight=cfg['model']['pc_loss_weight'],
                  pc_loss_name=cfg['model'].get('loss_name', 'chamfer'),
                  qal_threshold=cfg['model'].get('qal_threshold', 0.01),
                  qal_alpha=cfg['model'].get('qal_alpha', 100.0),
                  qal_use_squared=cfg['model'].get('qal_use_squared', False),
                  # This script scores reconstructions directly and does not
                  # consume the training objective.
                  sinkhorn_loss_weight=0.0,
                  max_leaves=cfg['model']['max_leaves'],
                  spline_loss_weight=cfg['model']['spline_loss_weight'],
                  depth_norm_type=cfg['model']['depth_norm_type'])
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ck['model_state_dict']
    if list(sd)[0].startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd); model.to(dev).eval()
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch')}); conditioning on RGB only")

    results = {}
    for name, root in GROUPS:
        ds = FolderDS(pick(root, args.n), cfg['data']['img_size'],
                      cfg['data']['num_points'], cfg['model']['max_leaves'])
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8)
        recs = []
        with torch.no_grad():
            for rgb, depth, pc, params, tv, nm in dl:
                rgb, depth, pc = rgb.to(dev), depth.to(dev), pc.to(dev)
                params = params.to(dev)
                _, _, pred, _, _ = forward_crossmodal(model, rgb, depth, pc, params, ['rgb'])
                for i in range(rgb.shape[0]):
                    cd = chamfer_distance(pred[i:i+1], pc[i:i+1]).item()
                    recs.append(dict(name=nm[i], cd=cd, rgb=rgb[i].cpu(),
                                     gt=pc[i].cpu().numpy(), pr=pred[i].cpu().numpy()))
        recs.sort(key=lambda r: r['cd'])
        cds = np.array([r['cd'] for r in recs])
        results[name] = recs
        print(f"{name.splitlines()[0]:22s} n={len(recs):3d}  mean={cds.mean():.5f}  "
              f"median={np.median(cds):.5f}  best={cds[0]:.5f}  worst={cds[-1]:.5f}")

    def recon_fig(groups, path, title):
        """best / median / worst reconstruction rows for each group in `groups`."""
        ncols = 4
        nrows = len(groups) * 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
        for g, (name, _) in enumerate(groups):
            recs = results[name]
            picks = [('best', recs[0]), ('median', recs[len(recs) // 2]), ('worst', recs[-1])]
            for k, (tag, r) in enumerate(picks):
                row = g * 3 + k
                ax = axes[row, 0]
                ax.imshow(denorm_rgb(r['rgb'])); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"RGB input — {r['name']}", fontsize=9)
                ax.set_ylabel(f"{name}\n{tag}  (CD {r['cd']:.4f})", fontsize=9, fontweight='bold')

                for j, (pts, c, ttl) in enumerate([(r['gt'], GT_C, 'ground truth'),
                                                   (r['pr'], PR_C, 'predicted from RGB')]):
                    ax = axes[row, 1 + j]
                    scat(ax, pts, c)
                    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=9)

                ax = axes[row, 3]
                scat(ax, r['gt'], GT_C, a=0.35)
                scat(ax, r['pr'], PR_C, a=0.35)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title('overlay  (GT dark / pred orange)', fontsize=9)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")

    ep = ck.get('epoch')
    recon_fig(GROUPS, f'{args.out}_recon.png',
              f"RGB→point-cloud reconstruction, epoch {ep} — best / median / worst per dataset")
    for i, grp in enumerate(GROUPS):        # one page per dataset, for the PDF
        recon_fig([grp], f'{args.out}_recon_p{i}.png',
                  f"{grp[0].replace(chr(10), ' ')} — best / median / worst  (epoch {ep})")


    # ── chamfer distribution per dataset ──────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    labels = [n.splitlines()[0] for n, _ in GROUPS]
    data = [[r['cd'] for r in results[n]] for n, _ in GROUPS]
    parts = a1.violinplot(data, showmedians=True)
    for pc_, c in zip(parts['bodies'], ['#2a9d8f', '#e9c46a', '#e76f51']):
        pc_.set_facecolor(c); pc_.set_alpha(0.7)
    a1.set_xticks(range(1, len(labels) + 1)); a1.set_xticklabels(labels, fontsize=9)
    a1.set_ylabel('Chamfer distance (RGB → PC)')
    a1.set_title('linear scale — OOD dwarfs the rest', fontsize=10, fontweight='bold')
    a1.grid(axis='y', alpha=0.3)

    for d, l, c in zip(data, labels, ['#2a9d8f', '#e9c46a', '#e76f51']):
        a2.hist(np.log10(d), bins=30, alpha=0.6, label=f"{l}  (mean {np.mean(d):.4f})", color=c)
    a2.set_xlabel('log10 Chamfer distance'); a2.set_ylabel('samples')
    a2.set_title('log scale — the two regimes are ~2 orders apart', fontsize=10, fontweight='bold')
    a2.legend(fontsize=8, frameon=False); a2.grid(alpha=0.3)
    fig.suptitle('RGB→PC generalisation gap', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'{args.out}_dist.png', dpi=140, bbox_inches='tight')
    print(f"Saved {args.out}_dist.png")


if __name__ == '__main__':
    main()
