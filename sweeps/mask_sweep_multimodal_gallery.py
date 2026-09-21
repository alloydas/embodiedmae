"""
sweeps/mask_sweep_multimodal_gallery.py — GOOD VISUALS for the mask sweep.

Unlike the per-target galleries (which only show the swept modality), this dumps a
JOINT multi-modal reconstruction: every modality is masked together at the same ratio
and RGB, Depth and Point cloud are reconstructed side-by-side, so you can SEE how all
modalities degrade as visibility drops. One figure, real modality data.

Layout: 3 modality rows (RGB / Depth / Point cloud) x [GT | 0% | 20% | 50% | 70% | 100%]
masked-uniformly columns, per-cell metric annotated.

Run on a GPU node (imports the frozen-model forward from mask_ratio_sweep).
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

import mask_ratio_sweep as S
from embodied_mae_4m import embodied_mae_4m_base, embodied_mae_4m_small
from sorghum_dataset_4m import SorghumDataset4M

RATIOS = [0.0, 0.2, 0.5, 0.7, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth')
    ap.add_argument('--config', default='config_4m.yaml')
    ap.add_argument('--output_dir', default='vis_mask_sweep')
    ap.add_argument('--num_samples', type=int, default=8)
    ap.add_argument('--samples', default='0',
                    help='comma-separated plant indices to visualise, e.g. 0,1,2')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = SorghumDataset4M(cfg['data']['data_root'], img_size=cfg['data']['img_size'],
                          num_points=cfg['data']['num_points'], split='val',
                          max_leaves=cfg['model']['max_leaves'])
    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=False, num_workers=4)

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
    model.load_state_dict(sd); model.to(dev).eval()
    epoch = ckpt.get('epoch'); vloss = ckpt.get('val_loss', ckpt.get('best_val_loss'))

    rgb, depth, pc, params, text_valid, names = next(iter(loader))
    rgb, depth, pc = rgb.to(dev), depth.to(dev), pc.to(dev)
    params, text_valid = params.to(dev), text_valid.to(dev)

    # one forward per ratio reconstructs the whole batch; we slice per plant below
    gen = torch.Generator(device=dev)
    all_preds = []
    for k, r in enumerate(RATIOS):
        gen.manual_seed(args.seed + k)
        ratios = {m: r for m in S.MODALITIES}
        preds, masks = S.forward_sweep(model, rgb, depth, pc, params, ratios, gen)
        met = S.all_metrics(model, preds, masks, rgb, depth, pc, params, text_valid)
        all_preds.append((r, preds, masks, met))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    samples = [int(x) for x in args.samples.split(',') if x.strip() != '']
    for s in samples:
        render_plant(s, names, rgb, depth, pc, model, all_preds,
                     text_valid, params, epoch, vloss, out)
    print("done")


def render_plant(s, names, rgb, depth, pc, model, all_preds, text_valid, params,
                 epoch, vloss, out):
    # collect per-plant reconstructions + metrics at each ratio
    cols = []
    for (r, preds, masks, met) in all_preds:
        # per-plant metrics for annotation
        cd_s = float(S.chamfer_distance(preds['pc'][s:s+1], pc[s:s+1]).item())
        rmse = S.rgb_mse(model, preds['rgb'][s:s+1], rgb[s:s+1], masks['rgb'][s:s+1])
        dmse = S.depth_mse(model, preds['depth'][s:s+1], depth[s:s+1], masks['depth'][s:s+1])
        _, _, _, _ = (0, 0, 0, 0)
        sp_p, sp_l, npl, nlf = S.param_mae(preds['text'][s:s+1], params[s:s+1],
                                           text_valid[s:s+1], masks['text'][s:s+1])
        import numpy as _np
        parts, wts = [], []
        if npl > 0 and not _np.isnan(sp_p): parts.append(sp_p); wts.append(npl)
        if nlf > 0 and not _np.isnan(sp_l): parts.append(sp_l); wts.append(nlf)
        spl = float(_np.average(parts, weights=wts)) if parts else float('nan')
        cols.append({
            'r': r,
            'rgb': S._rgb_recon_disp(model, preds['rgb'][s:s+1], rgb[s:s+1]),
            'depth': S._depth_recon_disp(model, preds['depth'][s:s+1], rgb.shape[2]),
            'pc': preds['pc'][s].cpu().numpy(),
            'rgb_mse': rmse, 'depth_mse': dmse, 'pc_cd': cd_s, 'spline': spl,
        })

    # ── figure: 3 rows (RGB, Depth, PC) x (GT + 5 ratios) ────────────────────
    ncol = len(RATIOS) + 1
    fig = plt.figure(figsize=(3.1 * ncol, 9.6))
    gt_rgb = torch.clamp(rgb[s].cpu() * S._STD + S._MEAN, 0, 1).permute(1, 2, 0).numpy()
    gt_depth = depth[s, 0].cpu().numpy()
    gt_pc = pc[s].cpu().numpy()

    def img_ax(row, col):
        return fig.add_subplot(3, ncol, row * ncol + col + 1)

    # Row 0: RGB
    ax = img_ax(0, 0); ax.imshow(gt_rgb); ax.axis('off')
    ax.set_title('GROUND TRUTH', fontweight='bold', fontsize=11)
    ax.text(-0.18, 0.5, 'RGB', rotation=90, va='center', ha='center',
            transform=ax.transAxes, fontweight='bold', fontsize=13)
    for k, c in enumerate(cols):
        ax = img_ax(0, k + 1); ax.imshow(c['rgb']); ax.axis('off')
        ax.set_title(f"{int(c['r']*100)}% masked\nMSE {c['rgb_mse']:.4f}", fontsize=10)

    # Row 1: Depth
    ax = img_ax(1, 0); ax.imshow(gt_depth, cmap='viridis'); ax.axis('off')
    ax.set_title('GROUND TRUTH', fontweight='bold', fontsize=11)
    ax.text(-0.18, 0.5, 'DEPTH', rotation=90, va='center', ha='center',
            transform=ax.transAxes, fontweight='bold', fontsize=13)
    for k, c in enumerate(cols):
        ax = img_ax(1, k + 1); ax.imshow(c['depth'], cmap='viridis', vmin=0, vmax=1)
        ax.axis('off'); ax.set_title(f"MSE {c['depth_mse']:.4f}", fontsize=10)

    # Row 2: Point cloud (3D)
    def pc_ax(col):
        return fig.add_subplot(3, ncol, 2 * ncol + col + 1, projection='3d')
    gtv = S._sub(gt_pc)
    ax = pc_ax(0); ax.scatter(gtv[:, 0], gtv[:, 1], gtv[:, 2], c=gtv[:, 2],
                              cmap='viridis', s=2)
    ax.set_title('GROUND TRUTH', fontweight='bold', fontsize=11)
    ax.view_init(elev=20, azim=45); ax.set_axis_off()
    ax.text2D(-0.12, 0.5, 'POINT\nCLOUD', rotation=90, va='center', ha='center',
              transform=ax.transAxes, fontweight='bold', fontsize=13)
    for k, c in enumerate(cols):
        genv = S._sub(c['pc'])
        ax = pc_ax(k + 1)
        ax.scatter(genv[:, 0], genv[:, 1], genv[:, 2], c=genv[:, 2], cmap='plasma', s=2)
        ax.set_title(f"Chamfer {c['pc_cd']:.4f}", fontsize=10)
        ax.view_init(elev=20, azim=45); ax.set_axis_off()

    fig.suptitle(f"Joint multi-modal reconstruction under uniform masking — {names[s]}\n"
                 f"all modalities masked at the same ratio  |  epoch {epoch}, "
                 f"val_loss {vloss:.4f}  |  spline MAE per col: "
                 + ", ".join(f"{int(c['r']*100)}%:{c['spline']:.3f}" for c in cols),
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0, 1, 0.93])
    save = out / f"gallery_multimodal_{names[s]}.png"
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"wrote {save}")


if __name__ == '__main__':
    main()
