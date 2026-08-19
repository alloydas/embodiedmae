"""
Comparison figures from the best 4M pretrain checkpoint (outputs/4m_pretrain_15k).

Produces, into --out_dir:
  recon_<name>.png       per-plant GT | masked | reconstruction, all 4 modalities
  mask_sweep.png         one plant reconstructed at increasing mask ratios
  ckpt_progression.png   one plant reconstructed by successive checkpoints
  param_scatter.png      GT vs predicted spline params over N test samples
  curves.png             loss / metric curves from training_history.json
  metrics.json           aggregate numbers quoted by the report

Style follows the deck conventions: point cloud is a single colour at every
stage, panels carry real modality data rather than text, and any figure built
from an unfinished run shows an explicit epoch progress bar.

Usage:
  python make_comparison_figs.py --out_dir figures_compare --n_param_samples 400
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from embodied_mae import chamfer_distance
from embodied_mae_4m import (
    N_PARAMS, _LEAF_SCALE, _LEAF_SHIFT, _PLANT_SCALE, _PLANT_SHIFT,
    embodied_mae_4m_base,
)
from sorghum_dataset_4m import SorghumDataset4M
from train_sorghum_4m import unpatchify

# ── Style ─────────────────────────────────────────────────────────────────────

PC_COLOUR   = '#2e8b57'      # seagreen — same object colour at every stage
PC_MUTED    = '#c9d6cf'      # masked-away points
ACCENT      = '#c0563a'
GRID        = '#e3e3e3'
RGB_MEAN    = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
RGB_STD     = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

LEAF_NAMES  = ['starting_point', 'length', 'roll_angle', 'branching_angle',
               'waviness_freq', 'wav_period_x', 'wav_period_y']
PLANT_NAMES = ['stem_length', 'stem_dir_x', 'stem_dir_y', 'stem_dir_z',
               'panicle_sz_x', 'panicle_sz_y', 'panicle_sz_z',
               'panicle_seeds', 'seed_radius']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.edgecolor': '#888888',
    'axes.linewidth': 0.7,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def denorm_rgb(x):
    """(3,H,W) normalised tensor → (H,W,3) displayable array."""
    return np.clip(x * RGB_STD + RGB_MEAN, 0, 1).transpose(1, 2, 0)


def mask_to_image(mask_1d, img_size):
    """(L,) patch mask → (H,W) pixel mask via nearest-neighbour upsample."""
    g = int(round(len(mask_1d) ** 0.5))
    return np.kron(mask_1d.reshape(g, g), np.ones((img_size // g, img_size // g)))


def scatter3d(ax, pts, colour, s=1.6, alpha=0.75, elev=18, azim=48):
    """Consistent 3D point-cloud styling. Y is up in the data, so plot X-Z-Y."""
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=colour, s=s, alpha=alpha,
               linewidths=0, edgecolors='none', depthshade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1.5))
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-1.0, 1.0)
    ax.set_axis_off()      # the cloud carries the shape; axes are clutter here


def subsample(pts, n=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(pts) <= n:
        return pts
    return pts[rng.choice(len(pts), n, replace=False)]


def depth_gt(r, idx):
    """Ground-truth depth on the head's [0,1] scale, plus the background mask."""
    bg = r['depth'][idx, 0] < 0.01
    g = r['depth_norm'][idx, 0].copy(); g[bg] = np.nan
    return g, bg


def panel_title(ax, text, bold=False):
    ax.set_title(text, fontsize=9, fontweight='bold' if bold else 'normal',
                 color='#222222', pad=6)


def progress_bar(fig, epoch, total, y=0.012, height=0.008):
    """Explicit 'run is not finished' indicator across the bottom of a figure."""
    frac = epoch / total
    fig.add_artist(Rectangle((0.30, y), 0.40, height, transform=fig.transFigure,
                             facecolor='#e8e8e8', edgecolor='none', zorder=5))
    fig.add_artist(Rectangle((0.30, y), 0.40 * frac, height, transform=fig.transFigure,
                             facecolor=ACCENT, edgecolor='none', zorder=6))
    fig.text(0.72, y + height / 2, f'epoch {epoch} / {total}  ({frac:.0%})',
             fontsize=8, va='center', color='#555555', zorder=6)


# ── Model / data ──────────────────────────────────────────────────────────────

def build_model(ckpt_path, args, device):
    model = embodied_mae_4m_base(
        img_size=args.img_size, num_pc_tokens=196, target_points=args.num_points,
        pc_loss_weight=args.pc_loss_weight, max_leaves=args.max_leaves,
        spline_loss_weight=args.spline_loss_weight,
        depth_norm_type=args.depth_norm_type,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    if list(sd.keys())[0].startswith('module.'):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt.get('epoch', -1)


def forward_once(model, batch, device, mask_ratio, seed, visible=None,
                 want_chamfer=True):
    """Deterministic forward — same seed reproduces the same mask, so runs at
    different checkpoints / mask ratios stay visually comparable.

    visible: None for normal Dirichlet masking, or a subset of
    {'rgb','depth','pc','text'} to run the cross-modal regime where only those
    modalities are shown and the rest are generated."""
    rgb, depth, pc, params, valid, names = batch
    rgb, depth, pc = rgb.to(device), depth.to(device), pc.to(device)
    params, valid = params.to(device), valid.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        total, losses, preds, masks = model(rgb, depth, pc, params, valid,
                                            mask_ratio=mask_ratio, visible=visible)
    pred_rgb, pred_depth, pred_pc, pred_params = preds

    # The two image heads are trained against *normalised* targets, so their raw
    # output is not in display space. Undo each normalisation the same way the
    # loss applied it, otherwise the reconstructions look mis-coloured and the
    # depth panels sit on a different scale from their ground truth.
    with torch.no_grad():
        # RGB: norm_pix_loss divides out each patch's mean and variance, so the
        # head is never asked to predict them. Re-apply the target's per-patch
        # statistics to see the structure the model does predict.
        tgt_patches = model.patchify(rgb, model.patch_size, 3)
        pm = tgt_patches.mean(-1, keepdim=True)
        pv = tgt_patches.var(-1, keepdim=True)
        pred_rgb_disp = unpatchify(pred_rgb * (pv + 1e-6).sqrt() + pm,
                                   model.patch_size, 3, model.img_size)
        # Depth: target is min-max normalised over the whole image, so put the
        # ground truth on the same [0, 1] scale the head predicts in.
        dflat = depth.reshape(depth.shape[0], -1)
        dmin = dflat.min(1, keepdim=True).values
        dmax = dflat.max(1, keepdim=True).values
        depth_norm = ((dflat - dmin) / (dmax - dmin).clamp(min=1e-6)).reshape(depth.shape)

    out = {
        'names': list(names),
        'rgb': rgb.cpu().numpy(),
        'depth': depth.cpu().numpy(),              # raw, for the background mask
        'depth_norm': depth_norm.cpu().numpy(),    # display scale, matches the head
        'pc': pc.cpu().numpy(),
        'params': params.cpu().numpy(),
        'valid': valid.cpu().numpy(),
        'pred_rgb': pred_rgb_disp.cpu().numpy(),
        'pred_depth': unpatchify(pred_depth, model.patch_size, 1, model.img_size).cpu().numpy(),
        'pred_pc': pred_pc.cpu().numpy(),
        'pred_params': pred_params.cpu().numpy(),
        'm_rgb': masks[0].cpu().numpy(),
        'm_depth': masks[1].cpu().numpy(),
        'm_pc': masks[2].cpu().numpy(),
        'm_text': masks[3].cpu().numpy(),
        'fps': model.pc_embed.fps(pc, model.num_pc_tokens).cpu().numpy(),
        'loss': total.item(),
        'l_rgb': losses[0].item(), 'l_depth': losses[1].item(),
        'l_pc': losses[2].item(), 'l_text': losses[3].item(),
    }
    # per-sample chamfer, for honest per-figure numbers
    if want_chamfer:
        with torch.no_grad():
            out['chamfer'] = [chamfer_distance(pred_pc[i:i + 1], pc[i:i + 1]).item()
                              for i in range(pc.shape[0])]
    else:
        out['chamfer'] = [float('nan')] * pc.shape[0]
    return out


# ── Figure 1: per-plant reconstruction ────────────────────────────────────────

def fig_reconstruction(r, i, out_path, epoch, total_epochs, mask_ratio):
    rng = np.random.default_rng(0)
    name = r['names'][i]
    img_size = r['rgb'].shape[-1]

    rgb_gt = denorm_rgb(r['rgb'][i])
    rgb_pr = denorm_rgb(r['pred_rgb'][i])
    m_rgb = mask_to_image(r['m_rgb'][i], img_size)

    bg = r['depth'][i, 0] < 0.01                  # background from the raw map
    d_gt = r['depth_norm'][i, 0]                  # scale the head predicts in
    m_d = mask_to_image(r['m_depth'][i], img_size)
    d_pr = r['pred_depth'][i, 0]

    def masked_depth(d, m):
        v = d.copy(); v[bg] = np.nan; v[m > 0.5] = np.nan
        return v

    d_show = d_gt.copy(); d_show[bg] = np.nan
    d_pr_show = d_pr.copy(); d_pr_show[bg] = np.nan
    vmin, vmax = 0.0, 1.0                         # shared, so panels are comparable

    pc_gt = r['pc'][i]
    pc_pr = r['pred_pc'][i]
    centres = pc_gt[r['fps'][i]]
    m_pc = r['m_pc'][i]

    fig = plt.figure(figsize=(9.2, 9.6))
    gs = fig.add_gridspec(3, 3, hspace=0.16, wspace=0.06,
                          left=0.04, right=0.97, top=0.90, bottom=0.06)

    # Row 1 — RGB
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(rgb_gt); ax.axis('off')
    panel_title(ax, 'RGB — ground truth', bold=True)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rgb_gt * (1 - m_rgb[:, :, None])); ax.axis('off')
    panel_title(ax, f'encoder input ({r["m_rgb"][i].mean():.0%} hidden)')
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(rgb_pr); ax.axis('off')
    panel_title(ax, 'reconstruction')

    # Row 2 — Depth
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(d_show, cmap='viridis', vmin=vmin, vmax=vmax); ax.axis('off')
    panel_title(ax, 'Depth — ground truth', bold=True)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(masked_depth(d_gt, m_d), cmap='viridis', vmin=vmin, vmax=vmax); ax.axis('off')
    panel_title(ax, f'encoder input ({r["m_depth"][i].mean():.0%} hidden)')
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(d_pr_show, cmap='viridis', vmin=vmin, vmax=vmax); ax.axis('off')
    panel_title(ax, 'reconstruction')

    # Row 3 — Point cloud (one colour throughout)
    ax = fig.add_subplot(gs[2, 0], projection='3d')
    scatter3d(ax, subsample(pc_gt, 3000, rng), PC_COLOUR)
    panel_title(ax, f'Point cloud — ground truth', bold=True)

    ax = fig.add_subplot(gs[2, 1], projection='3d')
    vis_c, msk_c = centres[m_pc == 0], centres[m_pc == 1]
    if len(msk_c):
        scatter3d(ax, msk_c, PC_MUTED, s=9, alpha=0.55)
    if len(vis_c):
        scatter3d(ax, vis_c, PC_COLOUR, s=13, alpha=0.95)
    panel_title(ax, f'encoder input ({m_pc.mean():.0%} of tokens hidden)')

    ax = fig.add_subplot(gs[2, 2], projection='3d')
    scatter3d(ax, subsample(pc_pr, 3000, rng), PC_COLOUR)
    panel_title(ax, 'reconstruction')

    fig.suptitle(f'{name}    ·    checkpoint epoch {epoch}    ·    mask ratio {mask_ratio:.2f}',
                 fontsize=11.5, fontweight='bold', y=0.965)
    fig.text(0.5, 0.925,
             f'Chamfer {r["chamfer"][i]:.5f}   ·   RGB shown with the target\'s '
             f'per-patch mean/variance restored (norm_pix_loss)',
             fontsize=8.5, color='#666666', ha='center')
    progress_bar(fig, epoch, total_epochs)
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)


# ── Figure 2: mask-ratio sweep ────────────────────────────────────────────────

def fig_mask_sweep(model, batch, device, idx, ratios, out_path, epoch, total_epochs):
    rng = np.random.default_rng(0)
    n = len(ratios)
    fig = plt.figure(figsize=(2.35 * (n + 1), 7.4))
    gs = fig.add_gridspec(3, n + 1, hspace=0.10, wspace=0.05,
                          left=0.05, right=0.98, top=0.88, bottom=0.07)

    base = forward_once(model, batch, device, ratios[0], seed=1234)
    name = base['names'][idx]

    # Column 0 — ground truth reference
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(denorm_rgb(base['rgb'][idx])); ax.axis('off')
    panel_title(ax, 'ground truth', bold=True)
    d_gt, bg = depth_gt(base, idx)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(d_gt, cmap='viridis', vmin=0, vmax=1); ax.axis('off')
    ax = fig.add_subplot(gs[2, 0], projection='3d')
    scatter3d(ax, subsample(base['pc'][idx], 2500, rng), PC_COLOUR)

    chamfers = []
    for c, mr in enumerate(ratios):
        r = forward_once(model, batch, device, mr, seed=1234)
        chamfers.append(r['chamfer'][idx])
        ax = fig.add_subplot(gs[0, c + 1])
        ax.imshow(denorm_rgb(r['pred_rgb'][idx])); ax.axis('off')
        panel_title(ax, f'{mr:.0%} masked', bold=True)
        ax = fig.add_subplot(gs[1, c + 1])
        dp = r['pred_depth'][idx, 0].copy(); dp[bg] = np.nan
        ax.imshow(dp, cmap='viridis', vmin=0, vmax=1); ax.axis('off')
        ax = fig.add_subplot(gs[2, c + 1], projection='3d')
        scatter3d(ax, subsample(r['pred_pc'][idx], 2500, rng), PC_COLOUR)
        ax.text2D(0.5, -0.02, f'CD {r["chamfer"][idx]:.5f}', transform=ax.transAxes,
                  ha='center', fontsize=8, color='#666666')

    for row, lab in enumerate(['RGB', 'Depth', 'Point cloud']):
        fig.text(0.012, 0.78 - row * 0.272, lab, fontsize=10, fontweight='bold',
                 rotation=90, va='center', color='#333333')

    fig.suptitle(f'Reconstruction vs. how much of the input is hidden\n'
                 f'{name}   ·   checkpoint epoch {epoch}',
                 fontsize=12, fontweight='bold', y=0.975)
    progress_bar(fig, epoch, total_epochs)
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    return dict(zip([f'{r:.2f}' for r in ratios], chamfers))


# ── Figure 3: checkpoint progression ──────────────────────────────────────────

def fig_ckpt_progression(ckpt_epochs, ckpt_dir, args, batch, device, idx,
                         out_path, mask_ratio, total_epochs):
    rng = np.random.default_rng(0)
    n = len(ckpt_epochs)
    fig = plt.figure(figsize=(2.35 * (n + 1), 7.4))
    gs = fig.add_gridspec(3, n + 1, hspace=0.10, wspace=0.05,
                          left=0.05, right=0.98, top=0.88, bottom=0.07)

    chamfers = {}
    ref = None
    for c, ep in enumerate(ckpt_epochs):
        path = ckpt_dir / f'checkpoint_epoch_{ep}.pth'
        model, _ = build_model(path, args, device)
        r = forward_once(model, batch, device, mask_ratio, seed=1234)
        chamfers[str(ep)] = r['chamfer'][idx]
        if ref is None:
            ref = r
            ax = fig.add_subplot(gs[0, 0]); ax.imshow(denorm_rgb(r['rgb'][idx])); ax.axis('off')
            panel_title(ax, 'ground truth', bold=True)
            d_gt, bg = depth_gt(r, idx)
            ax = fig.add_subplot(gs[1, 0])
            ax.imshow(d_gt, cmap='viridis', vmin=0, vmax=1); ax.axis('off')
            ax = fig.add_subplot(gs[2, 0], projection='3d')
            scatter3d(ax, subsample(r['pc'][idx], 2500, rng), PC_COLOUR)

        ax = fig.add_subplot(gs[0, c + 1])
        ax.imshow(denorm_rgb(r['pred_rgb'][idx])); ax.axis('off')
        panel_title(ax, f'epoch {ep}', bold=True)
        ax = fig.add_subplot(gs[1, c + 1])
        dp = r['pred_depth'][idx, 0].copy(); dp[bg] = np.nan
        ax.imshow(dp, cmap='viridis', vmin=0, vmax=1); ax.axis('off')
        ax = fig.add_subplot(gs[2, c + 1], projection='3d')
        scatter3d(ax, subsample(r['pred_pc'][idx], 2500, rng), PC_COLOUR)
        ax.text2D(0.5, -0.02, f'CD {r["chamfer"][idx]:.5f}', transform=ax.transAxes,
                  ha='center', fontsize=8, color='#666666')
        del model
        torch.cuda.empty_cache()

    for row, lab in enumerate(['RGB', 'Depth', 'Point cloud']):
        fig.text(0.012, 0.78 - row * 0.272, lab, fontsize=10, fontweight='bold',
                 rotation=90, va='center', color='#333333')

    fig.suptitle(f'What the model learns over training\n'
                 f'{ref["names"][idx]}   ·   identical mask at every checkpoint '
                 f'(mask ratio {mask_ratio:.2f})',
                 fontsize=12, fontweight='bold', y=0.975)
    progress_bar(fig, ckpt_epochs[-1], total_epochs)
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    return chamfers


# ── Figure 3b: cross-modal generation ─────────────────────────────────────────

def fig_crossmodal(model, batch, device, idx, out_path, epoch, total_epochs):
    """Give the encoder exactly one modality; ask it to produce all four."""
    rng = np.random.default_rng(0)
    sources = [('rgb', 'RGB only'), ('depth', 'Depth only'),
               ('pc', 'Point cloud only'), ('text', 'Spline params only')]

    fig = plt.figure(figsize=(2.35 * (len(sources) + 1), 7.4))
    gs = fig.add_gridspec(3, len(sources) + 1, hspace=0.10, wspace=0.05,
                          left=0.05, right=0.98, top=0.86, bottom=0.07)

    ref = None
    chamfers = {}
    for c, (src, label) in enumerate(sources):
        r = forward_once(model, batch, device, 0.0, seed=1234, visible={src})
        chamfers[src] = r['chamfer'][idx]
        if ref is None:
            ref = r
            ax = fig.add_subplot(gs[0, 0]); ax.imshow(denorm_rgb(r['rgb'][idx])); ax.axis('off')
            panel_title(ax, 'ground truth', bold=True)
            d_gt, bg = depth_gt(r, idx)
            ax = fig.add_subplot(gs[1, 0])
            ax.imshow(d_gt, cmap='viridis', vmin=0, vmax=1); ax.axis('off')
            ax = fig.add_subplot(gs[2, 0], projection='3d')
            scatter3d(ax, subsample(r['pc'][idx], 2500, rng), PC_COLOUR)

        # Show the given modality in its own panel; generate the other rows.
        ax = fig.add_subplot(gs[0, c + 1])
        if src == 'rgb':
            ax.imshow(denorm_rgb(r['rgb'][idx]))
            ax.add_patch(Rectangle((0, 0), 223, 223, fill=False, ec=PC_COLOUR, lw=3))
        else:
            ax.imshow(denorm_rgb(r['pred_rgb'][idx]))
        ax.axis('off'); panel_title(ax, label, bold=True)

        ax = fig.add_subplot(gs[1, c + 1])
        if src == 'depth':
            d, _ = depth_gt(r, idx)
            ax.imshow(d, cmap='viridis', vmin=0, vmax=1)
            ax.add_patch(Rectangle((0, 0), 223, 223, fill=False, ec=PC_COLOUR, lw=3))
        else:
            dp = r['pred_depth'][idx, 0].copy(); dp[bg] = np.nan
            ax.imshow(dp, cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, c + 1], projection='3d')
        pts = r['pc'][idx] if src == 'pc' else r['pred_pc'][idx]
        scatter3d(ax, subsample(pts, 2500, rng), PC_COLOUR)
        if src == 'pc':
            ax.text2D(0.5, -0.02, 'given', transform=ax.transAxes, ha='center',
                      fontsize=8, color=PC_COLOUR, fontweight='bold')
        else:
            ax.text2D(0.5, -0.02, f'CD {r["chamfer"][idx]:.5f}', transform=ax.transAxes,
                      ha='center', fontsize=8, color='#666666')

    for row, lab in enumerate(['RGB', 'Depth', 'Point cloud']):
        fig.text(0.012, 0.76 - row * 0.265, lab, fontsize=10, fontweight='bold',
                 rotation=90, va='center', color='#333333')

    fig.suptitle('Cross-modal generation — one modality in, all four out\n'
                 f'{ref["names"][idx]}   ·   checkpoint epoch {epoch}   ·   '
                 'green outline marks the modality that was given',
                 fontsize=12, fontweight='bold', y=0.975)
    progress_bar(fig, epoch, total_epochs)
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    return chamfers


# ── Figure 4: parameter regression ────────────────────────────────────────────

def crossmodal_aggregate(model, loader, device, n_samples, mask_ratio):
    """Chamfer per single-modality regime over many plants, so the cross-modal
    ordering rests on a distribution rather than one example. `masked` is the
    normal Dirichlet regime on the same plants, as the reference point."""
    regimes = {'rgb': {'rgb'}, 'depth': {'depth'}, 'pc': {'pc'}, 'text': {'text'},
               'rgb+depth': {'rgb', 'depth'}}
    acc = {k: [] for k in list(regimes) + ['masked']}
    seen = 0
    for bi, batch in enumerate(loader):
        for name, vis in regimes.items():
            r = forward_once(model, batch, device, 0.0, seed=2000 + bi, visible=vis)
            acc[name].extend(r['chamfer'])
        r = forward_once(model, batch, device, mask_ratio, seed=2000 + bi)
        acc['masked'].extend(r['chamfer'])
        seen += batch[0].shape[0]
        if seen >= n_samples:
            break
    return {k: {'mean': float(np.mean(v)), 'median': float(np.median(v)),
                'n': int(len(v))} for k, v in acc.items()}


def collect_params(model, loader, device, n_samples, mask_ratio):
    """Gather GT/pred spline params for masked, real tokens only."""
    leaf_gt, leaf_pr, plant_gt, plant_pr = [], [], [], []
    seen = 0
    for bi, batch in enumerate(loader):
        r = forward_once(model, batch, device, mask_ratio, seed=1000 + bi,
                         want_chamfer=False)
        gt, pr = r['params'], r['pred_params']
        keep = (r['valid'] > 0.5) & (r['m_text'] > 0.5)     # real AND hidden
        plant_sel = keep[:, 0]
        if plant_sel.any():
            plant_gt.append(gt[:, 0][plant_sel]); plant_pr.append(pr[:, 0][plant_sel])
        leaf_sel = keep[:, 1:]
        if leaf_sel.any():
            leaf_gt.append(gt[:, 1:][leaf_sel]); leaf_pr.append(pr[:, 1:][leaf_sel])
        seen += gt.shape[0]
        if seen >= n_samples:
            break
    cat = lambda xs: np.concatenate(xs, 0) if xs else np.zeros((0, N_PARAMS), np.float32)
    return cat(leaf_gt), cat(leaf_pr), cat(plant_gt), cat(plant_pr), seen


def fig_param_scatter(leaf_gt, leaf_pr, plant_gt, plant_pr, out_path,
                      epoch, total_epochs, n_seen):
    entries = []
    for j, nm in enumerate(LEAF_NAMES):
        entries.append((nm, 'leaf',
                        leaf_gt[:, j] * _LEAF_SCALE[j] - _LEAF_SHIFT[j],
                        leaf_pr[:, j] * _LEAF_SCALE[j] - _LEAF_SHIFT[j]))
    for j, nm in enumerate(PLANT_NAMES):
        entries.append((nm, 'plant',
                        plant_gt[:, j] * _PLANT_SCALE[j] - _PLANT_SHIFT[j],
                        plant_pr[:, j] * _PLANT_SCALE[j] - _PLANT_SHIFT[j]))

    fig, axes = plt.subplots(4, 4, figsize=(13.2, 13.4))
    stats = {}
    for ax, (nm, kind, g, p) in zip(axes.ravel(), entries):
        if len(g) == 0:
            ax.axis('off'); continue
        lo = float(min(g.min(), p.min())); hi = float(max(g.max(), p.max()))
        pad = 0.05 * (hi - lo + 1e-9); lo -= pad; hi += pad
        ax.scatter(g, p, s=3, alpha=0.18, color=PC_COLOUR if kind == 'leaf' else ACCENT,
                   linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color='#444444', lw=0.9, ls='--', zorder=3)
        ss_res = float(((g - p) ** 2).sum())
        ss_tot = float(((g - g.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
        mae = float(np.abs(g - p).mean())
        stats[f'{kind}.{nm}'] = {'r2': r2, 'mae': mae, 'n': int(len(g))}
        ax.set_title(f'{nm}  ({kind})', fontsize=9.5, fontweight='bold')
        ax.text(0.04, 0.94, f'R² {r2:.3f}\nMAE {mae:.4g}', transform=ax.transAxes,
                fontsize=8, va='top', color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.9))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, color=GRID, lw=0.6)
        ax.set_xlabel('ground truth', fontsize=8)
        ax.set_ylabel('predicted', fontsize=8)
    for ax in axes.ravel()[len(entries):]:
        ax.axis('off')

    fig.suptitle('Spline parameter recovery from hidden tokens\n'
                 f'checkpoint epoch {epoch}   ·   {n_seen} held-out test plants   '
                 f'·   {len(leaf_gt):,} leaf tokens, {len(plant_gt):,} plant tokens',
                 fontsize=13, fontweight='bold', y=0.985)
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    progress_bar(fig, epoch, total_epochs)
    fig.savefig(out_path, dpi=115, bbox_inches='tight')
    plt.close(fig)
    return stats


# ── Figure 5: training curves ─────────────────────────────────────────────────

def fig_curves(history, out_path, epoch, total_epochs, val_freq, test_freq):
    h = history
    ep_train = np.arange(1, len(h['train_loss']) + 1)
    # trainer validates when `epoch % val_freq == 0 or epoch == 1`, so the first
    # entry sits at epoch 1 and the rest on the val_freq grid.
    ep_val = np.array([1] + [i * val_freq for i in range(1, len(h['val_loss']))])
    ep_test = np.array(h['test_epoch'])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))

    ax = axes[0, 0]
    ax.plot(ep_train, h['train_loss'], color='#9fb8ac', lw=1.0, label='train')
    ax.plot(ep_val, h['val_loss'], color=PC_COLOUR, lw=1.8, marker='o', ms=3, label='val')
    ax.plot(ep_test, h['test_loss'], color=ACCENT, lw=1.8, marker='s', ms=3, label='test')
    ax.set_title('Total loss', fontsize=11, fontweight='bold')
    ax.set_yscale('log'); ax.legend(fontsize=8, frameon=False)

    for ax, key, lab in [(axes[0, 1], 'rgb', 'RGB (MSE, masked patches)'),
                         (axes[0, 2], 'depth', 'Depth (MSE, masked patches)'),
                         (axes[1, 0], 'pc', 'Point cloud (Chamfer × weight)'),
                         (axes[1, 1], 'text', 'Spline params (Smooth-L1 × weight)')]:
        ax.plot(ep_train, h[f'train_{key}'], color='#9fb8ac', lw=1.0, label='train')
        ax.plot(ep_val, h[f'val_{key}'], color=PC_COLOUR, lw=1.8, marker='o', ms=3, label='val')
        ax.plot(ep_test, h[f'test_{key}'], color=ACCENT, lw=1.8, marker='s', ms=3, label='test')
        ax.set_title(lab, fontsize=11, fontweight='bold')
        ax.set_yscale('log'); ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    ax.plot(ep_val, np.array(h['val_param_acc05']) * 100, color=PC_COLOUR,
            lw=1.8, marker='o', ms=3, label='val')
    ax.plot(ep_test, np.array(h['test_param_acc05']) * 100, color=ACCENT,
            lw=1.8, marker='s', ms=3, label='test')
    ax.set_title('Spline param accuracy @ 0.05', fontsize=11, fontweight='bold')
    ax.set_ylabel('% within 0.05 (normalised)', fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes.ravel():
        ax.set_xlabel('epoch', fontsize=9)
        ax.grid(alpha=0.3, color=GRID, lw=0.6)
        ax.tick_params(labelsize=8)
        ax.axvline(epoch, color='#bbbbbb', lw=0.8, ls=':')

    fig.suptitle('EmbodiedMAE-4M pretraining on Sorghum_15K — held-out val and test',
                 fontsize=13, fontweight='bold', y=0.985)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    progress_bar(fig, epoch, total_epochs)
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='./outputs/4m_pretrain_15k')
    ap.add_argument('--checkpoint', default=None, help='defaults to <run_dir>/best_model.pth')
    ap.add_argument('--data_root',
                    default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--out_dir', default='./figures_compare')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--max_leaves', type=int, default=24)
    ap.add_argument('--pc_loss_weight', type=float, default=10.0)
    ap.add_argument('--spline_loss_weight', type=float, default=5.0)
    ap.add_argument('--depth_norm_type', default='minmax')
    ap.add_argument('--mask_ratio', type=float, default=0.80)
    ap.add_argument('--total_epochs', type=int, default=1000)
    ap.add_argument('--val_freq', type=int, default=20)
    ap.add_argument('--test_freq', type=int, default=50)
    ap.add_argument('--num_recon', type=int, default=6)
    ap.add_argument('--n_param_samples', type=int, default=400)
    ap.add_argument('--n_crossmodal_samples', type=int, default=96)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / 'best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  checkpoint={ckpt_path}')

    model, epoch = build_model(ckpt_path, args, device)
    print(f'Loaded checkpoint from epoch {epoch}')

    test_ds = SorghumDataset4M(args.data_root, img_size=args.img_size,
                               num_points=args.num_points, split='test',
                               max_leaves=args.max_leaves)
    # Deterministic, spread-out sample choice so the gallery is not all one plant.
    stride = max(1, len(test_ds) // args.num_recon)
    gallery_idx = [i * stride for i in range(args.num_recon)]
    gallery = torch.utils.data.Subset(test_ds, gallery_idx)
    gal_loader = DataLoader(gallery, batch_size=args.num_recon, shuffle=False,
                            num_workers=min(args.num_workers, args.num_recon))
    gal_batch = next(iter(gal_loader))

    results = {'checkpoint_epoch': int(epoch), 'checkpoint': str(ckpt_path),
               'mask_ratio': args.mask_ratio, 'n_test': len(test_ds)}

    # 1 — per-plant reconstruction gallery
    print('→ reconstruction gallery')
    r = forward_once(model, gal_batch, device, args.mask_ratio, seed=1234)
    gallery_files = []
    for i in range(len(r['names'])):
        p = out_dir / f'recon_{r["names"][i]}.png'
        fig_reconstruction(r, i, p, epoch, args.total_epochs, args.mask_ratio)
        gallery_files.append(p.name)
        print(f'   {p.name}')
    results['gallery'] = gallery_files
    results['gallery_chamfer'] = dict(zip(r['names'], r['chamfer']))
    results['gallery_losses'] = {'total': r['loss'], 'rgb': r['l_rgb'],
                                 'depth': r['l_depth'], 'pc': r['l_pc'],
                                 'text': r['l_text']}

    # 2 — mask-ratio sweep
    print('→ mask sweep')
    results['mask_sweep_chamfer'] = fig_mask_sweep(
        model, gal_batch, device, 0, [0.50, 0.70, 0.80, 0.90, 0.95],
        out_dir / 'mask_sweep.png', epoch, args.total_epochs)

    # 2b — cross-modal generation
    print('→ cross-modal generation')
    results['crossmodal_chamfer'] = fig_crossmodal(
        model, gal_batch, device, 0, out_dir / 'crossmodal.png',
        epoch, args.total_epochs)

    # 3 — parameter regression (needs many samples → its own loader)
    print('→ parameter scatter')
    n_take = min(args.n_param_samples, len(test_ds))
    pstride = max(1, len(test_ds) // n_take)
    psub = torch.utils.data.Subset(test_ds, [i * pstride for i in range(n_take)])
    ploader = DataLoader(psub, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    print('→ cross-modal aggregate')
    n_cm = min(args.n_crossmodal_samples, len(test_ds))
    cstride = max(1, len(test_ds) // n_cm)
    csub = torch.utils.data.Subset(test_ds, [i * cstride for i in range(n_cm)])
    cloader = DataLoader(csub, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    results['crossmodal_aggregate'] = crossmodal_aggregate(
        model, cloader, device, n_cm, args.mask_ratio)
    for k, v in results['crossmodal_aggregate'].items():
        print(f'   {k:10s} mean CD {v["mean"]:.5f}  median {v["median"]:.5f}  n={v["n"]}')
    lg, lp, pg, pp, seen = collect_params(model, ploader, device,
                                          n_take, args.mask_ratio)
    results['param_stats'] = fig_param_scatter(lg, lp, pg, pp,
                                               out_dir / 'param_scatter.png',
                                               epoch, args.total_epochs, seen)
    results['param_n_leaf_tokens'] = int(len(lg))
    results['param_n_plant_tokens'] = int(len(pg))
    results['param_n_samples'] = int(seen)

    del model
    torch.cuda.empty_cache()

    # 4 — checkpoint progression (loads several checkpoints, so do it last)
    print('→ checkpoint progression')
    ckpt_dir = run_dir / 'checkpoints'
    avail = sorted(int(p.stem.split('_')[-1]) for p in ckpt_dir.glob('checkpoint_epoch_*.pth'))
    picks = [e for e in (20, 100, 240, 500, 760) if e in avail] or avail[-5:]
    results['ckpt_progression_chamfer'] = fig_ckpt_progression(
        picks, ckpt_dir, args, gal_batch, device, 0,
        out_dir / 'ckpt_progression.png', args.mask_ratio, args.total_epochs)
    results['ckpt_progression_epochs'] = picks

    # 5 — curves
    print('→ curves')
    with open(run_dir / 'training_history.json') as f:
        hist = json.load(f)
    fig_curves(hist, out_dir / 'curves.png', epoch, args.total_epochs,
               args.val_freq, args.test_freq)
    results['history_tail'] = {
        'epochs_done': len(hist['train_loss']),
        'best_val_loss': float(min(hist['val_loss'])),
        'last_val_loss': float(hist['val_loss'][-1]),
        'last_test': {k: float(hist[k][-1]) for k in
                      ('test_loss', 'test_pc_chamfer', 'test_param_acc05',
                       'test_param_mae_masked', 'test_rgb_mse', 'test_depth_mse')},
        'last_test_epoch': int(hist['test_epoch'][-1]),
    }

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ wrote {out_dir}/  ({len(list(out_dir.glob("*.png")))} figures)')


if __name__ == '__main__':
    main()
