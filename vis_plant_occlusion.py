"""
Preview structured masking + procedural occlusion on real samples.

Standalone (no model / checkpoint): reads a few sample folders, applies the
`occlusion:` and `model.structured_mask:` settings from a training config, and
saves one figure:

  per sample : clean RGB | encoder-input RGB (occluded + noise) | input depth |
               structured token mask | point cloud with the hidden points
  bottom row : how often each radius is covered -- occluder pixels, and masked
               tokens for structured vs uniform masking -- which is the
               "centre plant less occluded than the edges" property in numbers.

    python vis_plant_occlusion.py --config config_4m_occlusion.yaml
    python vis_plant_occlusion.py --config config_4m_occlusion.yaml \
        --split val --num_samples 4 --seed 0 --output vis_plant_occlusion.png

Only reads the dataset.
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

from embodied_mae import PointCloudEmbed
from plant_occlusion import (
    ProceduralOcclusion,
    StructuredMaskConfig,
    image_grid,
    pc_to_image,
    sample_field,
    structured_mask_scores,
)
from sorghum_dataset import SorghumDataset

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _Folders(SorghumDataset):
    """SorghumDataset over an explicit folder list.

    The base constructor globs every folder of the split (22k+ on Sorghum_15K);
    a preview needs a handful, loaded with the exact same preprocessing.
    """

    def __init__(self, folders, img_size, num_points):
        self.samples = list(folders)
        self.img_size = img_size
        self.num_points = num_points
        self._deterministic_point_sampling = True
        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN.flatten().tolist(), std=_STD.flatten().tolist()),
        ])


def _visible_mask(scores, n_visible):
    """1 = masked, lowest scores stay visible (as random_masking_dirichlet)."""
    order = scores.argsort(dim=1)
    mask = torch.ones_like(scores)
    mask.scatter_(1, order[:, :n_visible], 0.0)
    return mask


def radial_profile(values, radius, bins):
    idx = np.clip(np.digitize(radius, bins) - 1, 0, len(bins) - 2)
    return np.array([values[..., idx == b].mean() if (idx == b).any() else np.nan
                     for b in range(len(bins) - 1)])


def radial_statistics(occluder, sm_cfg, mask_ratio, img_size, patch, draws, seed):
    """Occluder-pixel and masked-token rates vs radius, over synthetic draws."""
    torch.manual_seed(seed)
    bins = np.linspace(0, 1, 11)
    centres = 0.5 * (bins[1:] + bins[:-1])
    out = {'r': centres}

    grid = image_grid(img_size).numpy()
    r_pix = np.linalg.norm(grid, axis=1) / math.sqrt(2)
    if occluder is not None:
        B = 64
        # Empty frame: measures where the leaf silhouettes land. (With a
        # plant-free depth map, depth_quantile's z-test never rejects a pixel.)
        rgb = torch.zeros(B, 3, img_size, img_size)
        depth = torch.zeros(B, 1, img_size, img_size)
        pc = torch.zeros(B, 16, 3)
        acc = []
        for _ in range(max(1, draws // B)):
            info = occluder(rgb, depth, pc)[3]
            acc.append((info['alpha'][:, 0].reshape(B, -1) > 0.5).float().numpy())
        out['occluder'] = radial_profile(np.concatenate(acc), r_pix, bins)

    uv = image_grid(img_size // patch)
    r_tok = (uv.norm(dim=1) / math.sqrt(2)).numpy()
    L = uv.shape[0]
    n_vis = int(round(L * (1 - mask_ratio)))
    B = draws
    uniform = _visible_mask(torch.rand(B, L), n_vis).numpy()
    out['uniform'] = radial_profile(uniform, r_tok, bins)
    if sm_cfg is not None:
        s = structured_mask_scores(uv.expand(B, -1, -1), sm_cfg,
                                   sample_field(B, sm_cfg.length_scale, 'cpu'))
        out['structured'] = radial_profile(_visible_mask(s, n_vis).numpy(), r_tok, bins)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--config', default='config_4m_occlusion.yaml')
    p.add_argument('--data_root', default=None, help='override data.data_root')
    p.add_argument('--split', default='val')
    p.add_argument('--num_samples', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mask_ratio', type=float, default=None,
                   help='per-modality mask ratio to illustrate (default: model.mask_ratio)')
    p.add_argument('--draws', type=int, default=512,
                   help='synthetic draws for the radial statistics')
    p.add_argument('--output', default='vis_plant_occlusion.png')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_root = Path(args.data_root or cfg['data']['data_root'])
    img_size = cfg['data'].get('img_size', 224)
    num_points = cfg['data'].get('num_points', 8196)
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg['model'].get('mask_ratio', 0.75)
    patch = 16

    occluder = ProceduralOcclusion.from_config(cfg.get('occlusion'))
    sm_cfg = StructuredMaskConfig.from_dict(cfg['model'].get('structured_mask'))
    if occluder is None and sm_cfg is None:
        raise SystemExit(f"{args.config} enables neither `occlusion` nor "
                         "`model.structured_mask`; nothing to preview")
    # Show the structured regime on every preview sample, not a prob-weighted mix.
    if occluder is not None:
        occluder.prob = 1.0

    split_dir = data_root / args.split
    folders = sorted(d for d in split_dir.iterdir() if d.is_dir())
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(folders), size=min(args.num_samples, len(folders)), replace=False)
    ds = _Folders([folders[i] for i in sorted(pick)], img_size, num_points)
    batch = [ds[i] for i in range(len(ds))]
    rgb = torch.stack([b[0] for b in batch])
    depth = torch.stack([b[1] for b in batch])
    pc = torch.stack([b[2] for b in batch])
    names = [b[3] for b in batch]
    B = len(batch)

    torch.manual_seed(args.seed)
    if occluder is not None:
        rgb_in, depth_in, pc_in, info = occluder(rgb, depth, pc)
    else:
        rgb_in, depth_in, pc_in = rgb, depth, pc
        info = {k: torch.zeros(B) for k in ('plant_coverage', 'pc_dropped')}

    uv = image_grid(img_size // patch)
    L = uv.shape[0]
    n_vis = int(round(L * (1 - mask_ratio)))
    if sm_cfg is not None:
        field = sample_field(B, sm_cfg.length_scale, 'cpu')
        tok_scores = structured_mask_scores(uv.expand(B, -1, -1), sm_cfg, field)
        fps = PointCloudEmbed(num_tokens=L, deterministic_fps=True).fps(pc_in, L)
        centres = pc_in[torch.arange(B)[:, None], fps]
        pc_scores = structured_mask_scores(pc_to_image(centres), sm_cfg, field)
        mask_label = f'structured, bias {sm_cfg.center_bias:g}'
    else:
        tok_scores = torch.rand(B, L)
        centres = pc_scores = None
        mask_label = 'uniform'
    tok_mask = _visible_mask(tok_scores, n_vis)

    def show_rgb(ax, t):
        ax.imshow((t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy())

    def show_depth(ax, d):
        d = d[0].numpy().copy()
        d[d <= 0] = np.nan
        ax.imshow(d, cmap='viridis')

    stats = radial_statistics(occluder, sm_cfg, mask_ratio, img_size, patch,
                              args.draws, args.seed)

    cols = 5
    fig = plt.figure(figsize=(3.3 * cols, 3.4 * (B + 1)))
    gs = fig.add_gridspec(B + 1, 2 * cols, hspace=0.35, wspace=0.3)
    for i in range(B):
        panel = lambda c: fig.add_subplot(gs[i, 2 * c:2 * c + 2])
        ax = panel(0)
        show_rgb(ax, rgb[i]); ax.set_title(f'{names[i]}\nclean RGB', fontsize=9)

        ax = panel(1)
        show_rgb(ax, rgb_in[i])
        ax.set_title(f'encoder input RGB\nplant covered {info["plant_coverage"][i]:.0%}',
                     fontsize=9)

        ax = panel(2)
        show_depth(ax, depth_in[i]); ax.set_title('encoder input depth', fontsize=9)

        ax = panel(3)
        g = img_size // patch
        m = np.kron(tok_mask[i].reshape(g, g).numpy(), np.ones((patch, patch)))
        img = (rgb_in[i] * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img * (1 - 0.85 * m[..., None]))
        ax.set_title(f'token mask ({mask_label})\n{n_vis}/{L} visible', fontsize=9)

        ax = panel(4)
        uv_pts = pc_to_image(pc[i]).numpy()
        hidden = (info['pc_hidden'][i].numpy() if 'pc_hidden' in info
                  else np.zeros(len(uv_pts), dtype=bool))
        ax.scatter(uv_pts[~hidden, 0], uv_pts[~hidden, 1], s=0.3, c='#4a7a3a')
        ax.scatter(uv_pts[hidden, 0], uv_pts[hidden, 1], s=0.3, c='#d62728')
        if centres is not None:
            c_uv = pc_to_image(centres[i]).numpy()
            vis = _visible_mask(pc_scores[i:i + 1], n_vis)[0].numpy() == 0
            ax.scatter(c_uv[vis, 0], c_uv[vis, 1], s=8, facecolors='none',
                       edgecolors='k', linewidths=0.6)
        ax.set_xlim(-1, 1); ax.set_ylim(1, -1); ax.set_aspect('equal')
        ax.set_title(f'cloud, image frame\nred hidden {info["pc_dropped"][i]:.0%}'
                     f'  o visible tokens', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for k, ax in enumerate(fig.axes):
        if k % cols != cols - 1:
            ax.axis('off')

    ax = fig.add_subplot(gs[B, :cols])
    if 'occluder' in stats:
        ax.plot(stats['r'], stats['occluder'], 'o-', c='#4a7a3a')
    ax.set_xlabel('radius from image centre (1 = corner)')
    ax.set_ylabel('P(pixel under an occluding leaf)')
    ax.set_title('Procedural occluders (occluded samples)', fontsize=10)
    ax.set_ylim(0, None); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[B, cols:])
    ax.plot(stats['r'], stats['uniform'], 's--', c='grey', label='uniform')
    if 'structured' in stats:
        ax.plot(stats['r'], stats['structured'], 'o-', c='#d62728', label=mask_label)
    ax.axhline(mask_ratio, c='k', lw=0.6, ls=':')
    ax.set_xlabel('radius from image centre (1 = corner)')
    ax.set_ylabel('P(token masked)')
    ax.set_title(f'Token masking by radius (mask ratio {mask_ratio:g})', fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'{args.config}  |  split={args.split}  seed={args.seed}',
                 fontsize=11, fontweight='bold', y=0.995)
    fig.savefig(args.output, dpi=110, bbox_inches='tight')
    print(f"Saved {args.output}")

    print("\nradius  " + "  ".join(f"{r:4.2f}" for r in stats['r']))
    for key in ('occluder', 'uniform', 'structured'):
        if key in stats:
            print(f"{key:10s}" + "  ".join(f"{v:4.2f}" for v in stats[key]))
    print(f"\nplant coverage per sample: "
          + ", ".join(f"{v:.0%}" for v in info['plant_coverage'].tolist()))
    print("PC points hidden per sample: "
          + ", ".join(f"{v:.0%}" for v in info['pc_dropped'].tolist()))


if __name__ == '__main__':
    main()
