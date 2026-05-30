"""
Point-cloud masking visualizer.

Loads a few real samples, runs through PointCloudEmbed (FPS + KNN grouping)
and random_masking_dirichlet, then plots:
  - Full point cloud
  - FPS centres coloured by visible (green) / masked (red)
  - Visible points only (the raw points that belong to visible tokens)
  - Masked points only
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sorghum_dataset import SorghumDataset
from embodied_mae import EmbodiedMAE

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "./Dataset/new_data"
NUM_POINTS  = 8196
NUM_SAMPLES = 4          # samples to visualise
MASK_RATIO  = 0.15
NUM_TOKENS  = 196
GROUP_SIZE  = 32
SAVE_PATH   = "vis_pc_masking.png"
DEVICE      = torch.device("cpu")

# ── helpers ───────────────────────────────────────────────────────────────────

def fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest-point sampling; returns (B, npoint) indices."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    dist      = torch.ones(B, N) * 1e10
    farthest  = torch.randint(0, N, (B,), dtype=torch.long)
    batch_idx = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)          # (B,1,3)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.where(d < dist, d, dist)
        farthest = dist.max(-1)[1]
    return centroids


def points_for_token(xyz: torch.Tensor, fps_idx: torch.Tensor,
                     group_size: int) -> torch.Tensor:
    """
    Returns (B, num_tokens, group_size) point indices — which original points
    belong to each FPS token.
    """
    B, N, _ = xyz.shape
    S = fps_idx.shape[1]
    centers = xyz[torch.arange(B)[:, None], fps_idx]          # (B, S, 3)
    dist    = torch.cdist(centers, xyz)                        # (B, S, N)
    _, idx  = torch.topk(dist, group_size, dim=2, largest=False)  # (B, S, k)
    return idx


def scatter3d(ax, pts, c, s=2, alpha=0.6, **kw):
    # Map: data-X → plot-X, data-Z → plot-Y, data-Y → plot-Z
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               c=c, s=s, alpha=alpha, **kw)
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Z", fontsize=7)
    ax.set_zlabel("Y", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=20, azim=45)


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading dataset …")
dataset = SorghumDataset(DATA_ROOT, num_points=NUM_POINTS, split='train')
NUM_SAMPLES = min(NUM_SAMPLES, len(dataset))

# ── build a minimal model just for masking ────────────────────────────────────
model = EmbodiedMAE(
    num_pc_tokens=NUM_TOKENS,
    pc_group_size=GROUP_SIZE,
    dirichlet_alpha=1.0,
).to(DEVICE).eval()

# ── figure layout: NUM_SAMPLES rows × 4 cols ─────────────────────────────────
fig = plt.figure(figsize=(20, 5 * NUM_SAMPLES))
fig.suptitle(
    f"Point-cloud masking check  |  mask_ratio={MASK_RATIO}  "
    f"num_tokens={NUM_TOKENS}  num_points={NUM_POINTS}",
    fontsize=13, fontweight='bold'
)

for row, sample_idx in enumerate(range(NUM_SAMPLES)):
    rgb, depth, pc, name = dataset[sample_idx]

    # pc: (N, 3) → (1, N, 3)
    pc_batch = pc.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 1. Embed all modalities (need rgb/depth tensors too for masking call)
        rgb_batch   = rgb.unsqueeze(0).to(DEVICE)
        depth_batch = depth.unsqueeze(0).to(DEVICE)

        x_rgb   = model.rgb_embed(rgb_batch)   + model.pos_embed_2d + model.modality_embed_rgb
        x_depth = model.depth_embed(depth_batch) + model.pos_embed_2d + model.modality_embed_depth
        x_pc    = model.pc_embed(pc_batch)      + model.pos_embed_pc  + model.modality_embed_pc

        # 2. Dirichlet masking
        (_, _, _,
         mask_rgb, mask_depth, mask_pc,
         _, _, _) = model.random_masking_dirichlet(
            x_rgb, x_depth, x_pc, mask_ratio_total=MASK_RATIO
        )

        # 3. FPS to get token centres in 3D space
        fps_idx = model.pc_embed.fps(pc_batch, NUM_TOKENS)          # (1, S)
        centers = pc_batch[0, fps_idx[0]].cpu().numpy()             # (S, 3)

        # 4. Which original points belong to each token
        member_idx = points_for_token(pc_batch, fps_idx, GROUP_SIZE)  # (1, S, k)
        member_idx = member_idx[0].cpu().numpy()                      # (S, k)

    pc_np   = pc.cpu().numpy()                                       # (N, 3)
    mask    = mask_pc[0].cpu().numpy()                               # (S,) 1=masked 0=visible
    n_masked  = int(mask.sum())
    n_visible = NUM_TOKENS - n_masked

    # ── collect visible / masked point indices ──────────────────────────────
    visible_pts_idx = np.unique(member_idx[mask == 0].flatten())
    masked_pts_idx  = np.unique(member_idx[mask == 1].flatten())

    # ── subplots ────────────────────────────────────────────────────────────
    base = row * 4 + 1

    # Col 1 — full point cloud (subsample for speed)
    ax1 = fig.add_subplot(NUM_SAMPLES, 4, base, projection='3d')
    vis_n = min(4000, len(pc_np))
    idx_vis = np.random.choice(len(pc_np), vis_n, replace=False)
    scatter3d(ax1, pc_np[idx_vis], c=pc_np[idx_vis, 2], cmap='viridis')
    ax1.set_title(f"[{name}]\nFull PC  ({len(pc_np)} pts)", fontsize=8, fontweight='bold')

    # Col 2 — FPS token centres, green=visible / red=masked
    ax2 = fig.add_subplot(NUM_SAMPLES, 4, base + 1, projection='3d')
    vis_centers = centers[mask == 0]
    msk_centers = centers[mask == 1]
    if len(vis_centers):
        scatter3d(ax2, vis_centers, c='#2ecc71', s=18, alpha=0.9, label='visible')
    if len(msk_centers):
        scatter3d(ax2, msk_centers, c='#e74c3c', s=18, alpha=0.9, label='masked')
    ax2.legend(fontsize=7, loc='upper left')
    pct = 100 * n_masked / NUM_TOKENS
    ax2.set_title(
        f"FPS token centres\n"
        f"visible={n_visible}  masked={n_masked}  ({pct:.1f}% masked)",
        fontsize=8
    )

    # Col 3 — visible raw points
    ax3 = fig.add_subplot(NUM_SAMPLES, 4, base + 2, projection='3d')
    if len(visible_pts_idx):
        vis_sub = visible_pts_idx[np.random.choice(len(visible_pts_idx),
                                                    min(3000, len(visible_pts_idx)),
                                                    replace=False)]
        scatter3d(ax3, pc_np[vis_sub], c='#2ecc71', s=3)
    ax3.set_title(f"Visible points\n({len(visible_pts_idx)} pts)", fontsize=8)

    # Col 4 — masked raw points
    ax4 = fig.add_subplot(NUM_SAMPLES, 4, base + 3, projection='3d')
    if len(masked_pts_idx):
        msk_sub = masked_pts_idx[np.random.choice(len(masked_pts_idx),
                                                   min(3000, len(masked_pts_idx)),
                                                   replace=False)]
        scatter3d(ax4, pc_np[msk_sub], c='#e74c3c', s=3)
    ax4.set_title(f"Masked points\n({len(masked_pts_idx)} pts)", fontsize=8)

    print(f"  {name}: tokens visible={n_visible}/{NUM_TOKENS} "
          f"({100*n_visible/NUM_TOKENS:.1f}%)  "
          f"raw pts visible≈{len(visible_pts_idx)}  masked≈{len(masked_pts_idx)}")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=130, bbox_inches='tight')
print(f"\nSaved → {SAVE_PATH}")
