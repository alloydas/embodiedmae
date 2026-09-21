"""
figures/make_crossmodal_report.py — build report-quality figures + a LaTeX/PDF report for
the cross-modal generation probe (feed only RGB/Depth, generate PC + spline params).

Reuses forward_crossmodal() from generate_crossmodal.py and the epoch-2400 model.
Produces, under <out>/:
    figures/per_plant/<name>.png   polished per-plant panel (input | GT | generated)
    figures/crossmodal_gallery.png grid of GT vs generated clouds across plants
    figures/condition_bars.png     PC-Chamfer & param-MAE bar chart over conditions
    main.tex                       article matching latex_report_v3_2400 style
    summary.json                   aggregate numbers used in the tables
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from embodied_mae_4m import (
    embodied_mae_4m_base, embodied_mae_4m_small, chamfer_distance,
    _PLANT_SCALE, _PLANT_SHIFT, _LEAF_SCALE, _LEAF_SHIFT,
)
from sorghum_dataset_4m import SorghumDataset4M
from generate_crossmodal import forward_crossmodal, MODALITIES

PLANT_FIELDS = ['stem_len', 'stem_dir_x', 'stem_dir_y', 'stem_dir_z',
                'pan_size_x', 'pan_size_y', 'pan_size_z', 'pan_seed_n', 'pan_seed_r']
LEAF_FIELDS  = ['start_pt', 'length', 'roll_ang', 'branch_ang', 'wav_freq',
                'wav_per0', 'wav_per1']

mpl_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
mpl_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_rgb(t):
    return torch.clamp(t * mpl_std + mpl_mean, 0, 1).permute(1, 2, 0).numpy()


def subsample(pc, n=1800):
    if pc.shape[0] > n:
        return pc[np.random.choice(pc.shape[0], n, replace=False)]
    return pc


# ── per-plant figure ──────────────────────────────────────────────────────────

def per_plant_figure(s, path, visible):
    cond = '+'.join(visible).upper()
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.28, wspace=0.30)

    # inputs
    axr = fig.add_subplot(gs[0, 0]); axr.imshow(denorm_rgb(s['rgb'])); axr.axis('off')
    axr.set_title(f"RGB  ({'INPUT' if 'rgb' in visible else 'unused'})",
                  fontsize=11, fontweight='bold',
                  color='black' if 'rgb' in visible else 'gray')
    axd = fig.add_subplot(gs[1, 0])
    im = axd.imshow(s['depth'][0].numpy(), cmap='viridis'); axd.axis('off')
    axd.set_title(f"Depth  ({'INPUT' if 'depth' in visible else 'unused'})",
                  fontsize=11, fontweight='bold',
                  color='black' if 'depth' in visible else 'gray')
    fig.colorbar(im, ax=axd, fraction=0.046, pad=0.02)

    # clouds
    gt, gen = subsample(s['pc'].numpy()), subsample(s['pred_pc'].numpy())
    axg = fig.add_subplot(gs[:, 1], projection='3d')
    axg.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c=gt[:, 2], cmap='viridis', s=2)
    axg.set_title("Ground-truth\npoint cloud", fontsize=11, fontweight='bold')
    axg.view_init(elev=18, azim=45); axg.set_box_aspect((1, 1, 1.3))

    axp = fig.add_subplot(gs[:, 2], projection='3d')
    axp.scatter(gen[:, 0], gen[:, 1], gen[:, 2], c=gen[:, 2], cmap='plasma', s=2)
    qual = ("plant-specific" if s['chamfer'] < 0.6 * s['chamfer_rand'] else "weak")
    axp.set_title(f"GENERATED from {cond}\nChamfer {s['chamfer']:.4f} "
                  f"({qual})", fontsize=11, fontweight='bold')
    axp.view_init(elev=18, azim=45); axp.set_box_aspect((1, 1, 1.3))

    # param table
    axt = fig.add_subplot(gs[:, 3]); axt.axis('off')
    lines = [f"SPLINE PARAMS  (generated from {cond})", "-" * 40,
             f"{'field':<11}{'GT':>9}{'gen':>9}{'|Δ|n':>7}", ""]
    pl_gt  = s['gt_params'][0] * _PLANT_SCALE - _PLANT_SHIFT
    pl_gen = s['gen_params'][0] * _PLANT_SCALE - _PLANT_SHIFT
    pl_dn  = np.abs(s['gen_params'][0] - s['gt_params'][0])
    lines.append("PLANT:")
    for k, (f, a, b, d) in enumerate(zip(PLANT_FIELDS, pl_gt, pl_gen, pl_dn)):
        lines.append(f"  {f:<9}{a:>9.3f}{b:>9.3f}{d:>7.2f}")
    lines.append("")
    n_real = int(s['text_valid'].sum())
    for ti in range(1, min(n_real, 3)):
        lf_gt  = s['gt_params'][ti][:7] * _LEAF_SCALE[:7] - _LEAF_SHIFT[:7]
        lf_gen = s['gen_params'][ti][:7] * _LEAF_SCALE[:7] - _LEAF_SHIFT[:7]
        lf_dn  = np.abs(s['gen_params'][ti][:7] - s['gt_params'][ti][:7])
        lines.append(f"LEAF {ti}:")
        for f, a, b, d in zip(LEAF_FIELDS, lf_gt, lf_gen, lf_dn):
            lines.append(f"  {f:<9}{a:>9.2f}{b:>9.2f}{d:>7.2f}")
        lines.append("")
    lines.append(f"plant MAE(norm)={s['plant_mae']:.3f}  "
                 f"leaf MAE(norm)={s['leaf_mae']:.3f}")
    axt.text(0.0, 1.0, "\n".join(lines), va='top', ha='left',
             family='monospace', fontsize=8.0, transform=axt.transAxes)

    fig.suptitle(f"Cross-modal generation  —  {s['name']}   |   visible: {cond}  "
                 f"(random-plant Chamfer baseline {s['chamfer_rand']:.4f})",
                 fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)


# ── gallery ───────────────────────────────────────────────────────────────────

def gallery_figure(samples, path, visible):
    n = len(samples)
    fig = plt.figure(figsize=(2.4 * n, 6.0))
    for i, s in enumerate(samples):
        gt, gen = subsample(s['pc'].numpy(), 1200), subsample(s['pred_pc'].numpy(), 1200)
        a1 = fig.add_subplot(2, n, i + 1, projection='3d')
        a1.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c=gt[:, 2], cmap='viridis', s=1.2)
        a1.view_init(elev=18, azim=45); a1.axis('off'); a1.set_box_aspect((1, 1, 1.3))
        a1.set_title(s['name'].replace('Sorghum_', 'S'), fontsize=8)
        a2 = fig.add_subplot(2, n, n + i + 1, projection='3d')
        a2.scatter(gen[:, 0], gen[:, 1], gen[:, 2], c=gen[:, 2], cmap='plasma', s=1.2)
        a2.view_init(elev=18, azim=45); a2.axis('off'); a2.set_box_aspect((1, 1, 1.3))
        a2.set_title(f"CD {s['chamfer']:.3f}", fontsize=8)
    fig.text(0.012, 0.74, "Ground\ntruth", fontsize=10, fontweight='bold', ha='center')
    fig.text(0.012, 0.27, f"Generated\nfrom {'+'.join(visible).upper()}",
             fontsize=10, fontweight='bold', ha='center')
    fig.suptitle(f"Cross-modal point-cloud generation across {n} validation plants "
                 f"(epoch 2400, visible = {'+'.join(visible).upper()})",
                 fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)


def bars_figure(cond_results, path):
    conds = list(cond_results.keys())
    cd  = [cond_results[c]['chamfer_mean'] for c in conds]
    rnd = [cond_results[c]['chamfer_random_baseline_mean'] for c in conds]
    pm  = [cond_results[c]['plant_mae_norm_mean'] for c in conds]
    lm  = [cond_results[c]['leaf_mae_norm_mean'] for c in conds]
    x = np.arange(len(conds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.bar(x - 0.2, cd, 0.4, label='generated vs GT', color='tab:blue')
    ax1.bar(x + 0.2, rnd, 0.4, label='random-plant baseline', color='lightgray')
    ax1.set_xticks(x); ax1.set_xticklabels(conds, rotation=15, ha='right')
    ax1.set_ylabel('PC Chamfer (lower better)'); ax1.legend()
    ax1.set_title('Point-cloud generation quality', fontweight='bold')
    ax2.bar(x - 0.2, pm, 0.4, label='plant params', color='tab:green')
    ax2.bar(x + 0.2, lm, 0.4, label='leaf params', color='tab:orange')
    ax2.axhline(0.33, ls='--', c='red', lw=1, label='chance (~0.33)')
    ax2.set_xticks(x); ax2.set_xticklabels(conds, rotation=15, ha='right')
    ax2.set_ylabel('Param MAE (norm [0,1])'); ax2.legend()
    ax2.set_title('Spline-param recovery', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)


# ── run one conditioning, collect samples + metrics ───────────────────────────

@torch.no_grad()
def run_condition(model, batch, visible, dev):
    rgb, depth, pc, params, tv, names = batch
    rgb, depth, pc = rgb.to(dev), depth.to(dev), pc.to(dev)
    params, tv = params.to(dev), tv.to(dev)
    pr, pd, ppc, ppar, _ = forward_crossmodal(model, rgb, depth, pc, params, visible)
    gt_np, gen_np = params.cpu().numpy(), ppar.cpu().numpy()
    B = rgb.shape[0]
    samples = []
    for i in range(B):
        cd = chamfer_distance(ppc[i:i+1], pc[i:i+1]).item()
        j = (i + 1) % B
        cd_rand = chamfer_distance(pc[j:j+1], pc[i:i+1]).item()
        valid = tv[i].bool()
        err = (ppar[i] - params[i]).abs().cpu().numpy()
        plant_mae = err[0, :9].mean()
        leaf_idx = valid.clone(); leaf_idx[0] = False
        leaf_mae = err[leaf_idx.cpu().numpy()][:, :7].mean() if leaf_idx.any() else np.nan
        samples.append(dict(
            name=names[i], rgb=rgb[i].cpu(), depth=depth[i].cpu(),
            pc=pc[i].cpu(), pred_pc=ppc[i].cpu(), text_valid=tv[i].cpu(),
            gt_params=gt_np[i], gen_params=gen_np[i],
            chamfer=cd, chamfer_rand=cd_rand,
            plant_mae=float(plant_mae), leaf_mae=float(leaf_mae)))
    agg = dict(
        chamfer_mean=float(np.mean([s['chamfer'] for s in samples])),
        chamfer_random_baseline_mean=float(np.mean([s['chamfer_rand'] for s in samples])),
        plant_mae_norm_mean=float(np.nanmean([s['plant_mae'] for s in samples])),
        leaf_mae_norm_mean=float(np.nanmean([s['leaf_mae'] for s in samples])))
    return samples, agg


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def esc(s):
    return s.replace('_', r'\_')


def build_tex(out, headline, cond_aggs, gallery_names, epoch):
    h = cond_aggs[headline]
    rows = "\n".join(
        f"{esc(c)} & {a['chamfer_mean']:.4f} & {a['chamfer_random_baseline_mean']:.4f} "
        f"& {a['plant_mae_norm_mean']:.3f} & {a['leaf_mae_norm_mean']:.3f} \\\\"
        for c, a in cond_aggs.items())
    per_plant = "\n\\clearpage\n".join(
        f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\linewidth]{{per_plant/{n}.png}}
\\caption{{Plant {esc(n)} -- cross-modal generation from {esc(headline.upper())}: input image
modality (left), ground-truth vs.\\ generated point cloud (centre), and ground-truth
vs.\\ generated spline parameters with absolute normalised error (right).}}
\\label{{fig:xm:{n}}}
\\end{{figure}}""" for n in gallery_names)

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage[skip=4pt]{{parskip}}
\hypersetup{{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}}
\graphicspath{{{{figures/}}}}

\title{{EmbodiedMAE-4M: Zero-shot Cross-modal Generation \\[2pt]
\large Generating point clouds and spline parameters from images alone (run~v3, epoch~{epoch})}}
\author{{Alloy Das \\ Iowa State University \\ \texttt{{alloydas@iastate.edu}}}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We probe whether the trained EmbodiedMAE-4M checkpoint can perform \emph{{cross-modal
generation}}: presenting only a subset of modalities to the encoder (e.g.\ RGB and/or
depth) and asking the decoder to \emph{{generate}} the missing point cloud and procedural
spline parameters. The model was trained as a masked autoencoder with every modality
$\sim$75\% visible (\texttt{{min\_mask\_ratio}}~$=0.25$) and was never shown a fully-absent
modality, so this is an out-of-distribution masking regime. Despite that, the epoch-{epoch}
checkpoint generates plant-specific point clouds (Chamfer $\approx${h['chamfer_mean']:.4f}
vs.\ {h['chamfer_random_baseline_mean']:.4f} for a random plant) and recovers plant-level
parameters (MAE $\approx${h['plant_mae_norm_mean']:.3f} on $[0,1]$ vs.\ $\approx$0.33 chance)
when conditioned on {esc(headline.upper())}. We report results across four conditioning sets
and a per-plant gallery.
\end{{abstract}}

\section{{Setup}}
No weights were changed. The encoder's masking step is re-run deterministically with a chosen
set of \emph{{fully-visible}} modalities; the remaining modalities are given zero visible tokens,
so the decoder must reconstruct them entirely from mask tokens that attend (through the shared
decoder) to the visible image tokens. Point clouds are scored with bidirectional Chamfer distance
against the ground-truth cloud; a \emph{{random-plant}} baseline (Chamfer to a \emph{{different}}
validation plant's cloud) bounds what an average, non-specific shape would score. Spline
parameters are scored with mean absolute error in the normalised $[0,1]$ space, split into the
plant token and the per-leaf tokens; chance for $\mathcal{{U}}[0,1]$ targets is $\approx$0.33.
Checkpoint: \texttt{{outputs/4m\_run\_v3/checkpoints/checkpoint\_epoch\_{epoch}.pth}} (best
\texttt{{val\_loss}} 0.0204).

\section{{Results across conditioning sets}}
Table~\ref{{tab:cond}} sweeps which modalities are visible. Two clear trends: (i) a single image
modality already generates a plant-specific cloud and recovers plant parameters; (ii) quality
\emph{{degrades}} as more modalities are made fully visible -- ``everything visible'' collapses to
the random baseline -- a direct consequence of the narrow $\sim$75\%-visible training regime.
Leaf parameters stay near a learned prior in all settings (they are weakly image-determined).

\begin{{table}}[H]
\centering
\caption{{Cross-modal generation metrics by conditioning set (mean over the gallery plants).
Lower Chamfer / MAE is better. ``rand'' is the random-plant Chamfer baseline.}}
\label{{tab:cond}}
\begin{{tabular}}{{lrrrr}}
\toprule
Visible (conditioning) & Chamfer & rand & Plant MAE & Leaf MAE \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{condition_bars.png}}
\caption{{Left: generated-vs-ground-truth Chamfer against the random-plant baseline per
conditioning set. Right: plant- and leaf-parameter MAE (normalised) with the
$\mathcal{{U}}[0,1]$ chance line.}}
\label{{fig:bars}}
\end{{figure}}

\section{{Generation gallery}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{crossmodal_gallery.png}}
\caption{{Top row: ground-truth point clouds. Bottom row: clouds \emph{{generated}} from
{esc(headline.upper())} alone (per-plant Chamfer annotated). The generated clouds follow each
plant's overall stature and panicle, confirming the generation is plant-specific rather than an
average shape.}}
\label{{fig:gallery}}
\end{{figure}}
\clearpage

\section{{Per-plant cross-modal generation}}
Each figure conditions on {esc(headline.upper())} and shows the input image modality (left), the
ground-truth vs.\ generated point cloud (centre), and a table of ground-truth vs.\ generated
spline parameters with absolute normalised error (right).

{per_plant}

\end{{document}}
"""
    (out / "main.tex").write_text(tex)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='outputs/4m_run_v3/checkpoints/checkpoint_epoch_2400.pth')
    ap.add_argument('--config', default='config_4m.yaml')
    ap.add_argument('--output_dir', default='latex_report_crossmodal')
    ap.add_argument('--num_plants', type=int, default=8)
    ap.add_argument('--headline', default='rgb_depth',
                    help='conditioning set used for the gallery & per-plant figures')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(0); torch.manual_seed(0)

    ds = SorghumDataset4M(cfg['data']['data_root'], img_size=cfg['data']['img_size'],
                          num_points=cfg['data']['num_points'], split='val',
                          max_leaves=cfg['model']['max_leaves'])
    batch = next(iter(DataLoader(ds, batch_size=args.num_plants, shuffle=False, num_workers=4)))

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
    epoch = ckpt.get('epoch')
    print(f"Loaded epoch {epoch}")

    conditions = ['depth', 'rgb', 'rgb_depth', 'rgb_depth_pc']
    cond_samples, cond_aggs = {}, {}
    for c in conditions:
        vis = [m for m in c.split('_') if m in MODALITIES]
        samples, agg = run_condition(model, batch, vis, dev)
        cond_samples[c], cond_aggs[c] = samples, agg
        print(f"  {c:<14} chamfer={agg['chamfer_mean']:.4f} (rand "
              f"{agg['chamfer_random_baseline_mean']:.4f})  "
              f"plantMAE={agg['plant_mae_norm_mean']:.3f}  "
              f"leafMAE={agg['leaf_mae_norm_mean']:.3f}")

    out = Path(args.output_dir)
    figdir = out / "figures"; pp = figdir / "per_plant"
    pp.mkdir(parents=True, exist_ok=True)

    head_vis = [m for m in args.headline.split('_') if m in MODALITIES]
    head_samples = cond_samples[args.headline]
    for s in head_samples:
        per_plant_figure(s, pp / f"{s['name']}.png", head_vis)
    gallery_figure(head_samples, figdir / "crossmodal_gallery.png", head_vis)
    bars_figure(cond_aggs, figdir / "condition_bars.png")

    build_tex(out, args.headline, cond_aggs,
              [s['name'] for s in head_samples], epoch)
    json.dump({'epoch': epoch, 'headline': args.headline,
               'conditions': cond_aggs}, open(out / "summary.json", 'w'), indent=2)
    print(f"\nWrote {out}/main.tex + {len(head_samples)} per-plant figures + gallery + bars")


if __name__ == '__main__':
    main()
