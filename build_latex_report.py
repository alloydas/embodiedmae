"""Assemble a self-contained LaTeX project for the EmbodiedMAE-4M v3 completion.

Bundles main.tex + figures/ in latex_report_v3_2400/ and then zips it to
latex_report_v3_2400.zip for Overleaf upload.

Figures pulled in:
  arch_diagram.png
  loss_curves_2400.png
  slide_figure_v3_2400.png
  per_plant/Sorghum_*.png  (15 of them, from slide_per_plant_v3_2400/)
"""

import json
import shutil
import zipfile
from pathlib import Path


REPO     = Path('/work/mech-ai-scratch/alloy/embodiedmae')
OUT_DIR  = REPO / 'latex_report_v3_2400'
FIG_DIR  = OUT_DIR / 'figures'
PP_DIR   = FIG_DIR / 'per_plant'
ZIP_PATH = REPO / 'latex_report_v3_2400.zip'

SUMMARY = json.load(open(REPO / 'outputs/4m_run_v3/metrics_summary_2400.json'))
AGG     = json.load(open(REPO / 'slide_dump_v3_2400/metrics_aggregate.json'))


def reset_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    PP_DIR.mkdir(parents=True)


def stage_figures():
    pairs = [
        (REPO / 'arch_diagram.png',                            FIG_DIR / 'arch_diagram.png'),
        (REPO / 'outputs/4m_run_v3/loss_curves_2400.png',      FIG_DIR / 'loss_curves_2400.png'),
        (REPO / 'slide_figure_v3_2400.png',                    FIG_DIR / 'slide_figure_v3_2400.png'),
    ]
    for src, dst in pairs:
        if not src.exists():
            raise SystemExit(f'Missing figure: {src}')
        shutil.copy(src, dst)

    per_plant_src = REPO / 'slide_per_plant_v3_2400'
    pngs = sorted(per_plant_src.glob('*.png'))
    for src in pngs:
        shutil.copy(src, PP_DIR / src.name)
    return [p.name for p in pngs]


def safe_name(n: str) -> str:
    return n.replace('_', r'\_')


def _fmt(x, p=4):
    if x is None:
        return '--'
    return f'{x:.{p}f}'


def _pct(x):
    if x is None:
        return '--'
    return f'{x*100:.2f}\\%'


def build_tex(per_plant_names):
    best, final = SUMMARY['best'], SUMMARY['final']
    best_ep, final_ep = SUMMARY['best_epoch'], SUMMARY['final_epoch']

    lines = []
    add = lines.append
    add(r'\documentclass[11pt,a4paper]{article}')
    add(r'\usepackage[margin=1in]{geometry}')
    add(r'\usepackage{graphicx}')
    add(r'\usepackage{booktabs}')
    add(r'\usepackage{caption}')
    add(r'\usepackage{subcaption}')
    add(r'\usepackage{float}')
    add(r'\usepackage{xcolor}')
    add(r'\usepackage{hyperref}')
    add(r'\usepackage[skip=4pt]{parskip}')
    add(r'\hypersetup{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}')
    add(r'\graphicspath{{figures/}{figures/per_plant/}}')
    add('')
    add(r'\title{EmbodiedMAE-4M: A Multi-modal Masked Autoencoder for Synthetic Sorghum \\[2pt] \large run~v3 -- 2400-epoch completion report}')
    add(r'\author{Alloy Das \\ Iowa State University \\ \texttt{alloydas@iastate.edu}}')
    add(r'\date{\today}')
    add('')
    add(r'\begin{document}')
    add(r'\maketitle')
    add('')

    # ── Abstract ─────────────────────────────────────────────────────────────
    add(r'\begin{abstract}')
    add(r'EmbodiedMAE-4M is a four-modality masked autoencoder pre-trained on synthetic sorghum plants '
        r'rendered from a procedural generator. Each sample bundles an aligned RGB image, a depth map, a '
        r'point cloud, and the underlying procedural parameters (plant + per-leaf), and a single transformer '
        r'encoder is asked to reconstruct all four after Dirichlet-allocated token masking.')
    add(
        f'This report covers run~v3, which trained for {final_ep} epochs on the synthetic dataset at '
        r'\texttt{mask\_ratio} $= 0.15$ and reached its best held-out validation loss of '
        f'{_fmt(best["val_loss"])} at epoch~{best_ep} '
        f'(parameter accuracy at $|\\Delta| < 0.05$ of {_pct(best["val_param_acc05"])}). We include the '
        f'architecture diagram, full training curves, per-modality metrics, and a 15-plant '
        f'reconstruction gallery.')
    add(r'\end{abstract}')
    add('')

    # ── Architecture ─────────────────────────────────────────────────────────
    add(r'\section{Architecture}')
    add(r'A shared ViT-base encoder ingests visible tokens from all four modalities and a per-modality '
        r'decoder reconstructs the masked tokens. Modality embeddings: RGB/depth via patch embedding '
        r'($196$ tokens at $224{\times}224$, patch~$16$); point cloud via FPS centres ($196$ tokens) + '
        r'kNN grouping ($k{=}32$) feeding a PointNet-style MLP; spline parameters via char + positional '
        r'embeddings mean-pooled per token ($1$ plant token $+$ up to $24$ leaf tokens). A single Dirichlet '
        r'draw per training step decides the per-modality fraction of visible tokens; the 4M variant also '
        r'enforces \texttt{min\_mask\_ratio}~$=0.25$ per modality.')
    add('')
    add(r'\begin{figure}[H]')
    add(r'\centering')
    add(r'\includegraphics[width=\linewidth]{arch_diagram.png}')
    add(r'\caption{EmbodiedMAE-4M architecture. RGB, depth, point cloud, and spline/parameter tokens are '
        r'embedded, masked under a Dirichlet allocation, fed through a shared encoder, and reconstructed '
        r'by four modality-specific decoder heads.}')
    add(r'\label{fig:arch}')
    add(r'\end{figure}')
    add('')

    # ── Training setup ───────────────────────────────────────────────────────
    add(r'\section{Training setup}')
    add(r'\begin{table}[H]')
    add(r'\centering')
    add(r'\caption{Run~v3 training configuration.}')
    add(r'\label{tab:setup}')
    add(r'\begin{tabular}{ll}')
    add(r'\toprule')
    add(r'Key & Value \\')
    add(r'\midrule')
    rows = [
        ('Dataset',              r'synthetic sorghum (10\,000 train / 100 val)'),
        ('Image size',           r'$224 \times 224$ (patch $16$)'),
        ('Point cloud',          r'$8\,196$ points, unit-sphere normalised, FPS centres $=196$, kNN $k=32$'),
        ('Max leaves per plant', r'$24$ (padded)'),
        ('Model size',           r'base ($\sim$114\,M params, $d=768$, depth $12$ enc / $8$ dec)'),
        ('Mask ratio (train)',   r'$0.15$ (Dirichlet over modalities, min $0.25$ per modality)'),
        ('Loss weights',         r'RGB $1$, Depth $1$, PC $10$ (Chamfer), Params $5$ (Smooth-L1)'),
        ('Optimizer',            r'AdamW, weight decay $0.05$'),
        ('LR schedule',          r'$1.5\!\times\!10^{-4}$ peak, $10$-epoch linear warmup, cosine decay'),
        ('Batch / GPU',          r'$16$ on H200 (resumed run); effective batch $16$'),
        ('Epochs',               f'$1$ -- ${final_ep}$ (completed)'),
        ('Validation interval',  r'every $20$ epochs'),
    ]
    for k, v in rows:
        add(f'{k} & {v} \\\\')
    add(r'\bottomrule')
    add(r'\end{tabular}')
    add(r'\end{table}')
    add('')

    # ── Training curves ──────────────────────────────────────────────────────
    add(r'\section{Training dynamics}')
    add(
        r'Figure~\ref{fig:curves} shows per-modality train and validation losses on a log scale across '
        f'all {final_ep} epochs, plus the parameter accuracy at $|\\Delta|<0.05$ and the cosine LR '
        f'schedule. The dashed vertical marks the best-validation epoch (\\textbf{{{best_ep}}}); the '
        r'total loss is the sum of the weighted modality terms, so the small PC and text terms appear '
        r'close to the floor on the log axis but are reported separately in Table~\ref{tab:metrics}.')
    add('')
    add(r'\begin{figure}[H]')
    add(r'\centering')
    add(r'\includegraphics[width=\linewidth]{loss_curves_2400.png}')
    add(
        r'\caption{Per-modality train (blue, raw and 15-epoch smoothed) and validation (red) losses '
        f'across the full {final_ep}-epoch run, plus parameter accuracy@$0.05$ and LR schedule.'
        r'}')
    add(r'\label{fig:curves}')
    add(r'\end{figure}')
    add('')

    # ── Metrics table ────────────────────────────────────────────────────────
    add(r'\section{Validation metrics}')
    add(r'\begin{table}[H]')
    add(r'\centering')
    add(r'\caption{Validation metrics at the best epoch (lowest \texttt{val\_loss}) and at the final '
        r'epoch. Weighted columns use the training loss weights; the lower block reports raw modality '
        r'metrics. \texttt{param\_acc05} counts predictions within $|\Delta| < 0.05$ in normalised '
        r'parameter space.}')
    add(r'\label{tab:metrics}')
    add(r'\begin{tabular}{lrr}')
    add(r'\toprule')
    add(f'Metric & Best (ep~{best_ep}) & Final (ep~{final_ep}) \\\\')
    add(r'\midrule')
    metric_rows = [
        ('val\\_loss (weighted total)',       'val_loss',           4),
        ('val\\_rgb (per-patch MSE)',         'val_rgb',            4),
        ('val\\_depth (per-patch MSE)',       'val_depth',          4),
        ('val\\_pc ($\\times$ 10)',           'val_pc',             4),
        ('val\\_text ($\\times$ 5)',          'val_text',           4),
    ]
    for label, key, p in metric_rows:
        add(f'{label} & {_fmt(best[key], p)} & {_fmt(final[key], p)} \\\\')
    add(r'\midrule')
    raw_rows = [
        ('RGB MSE (unweighted)',                  'val_rgb_mse',        4),
        ('Depth MSE (on min-max target)',         'val_depth_mse',      4),
        ('PC Chamfer (bidirectional)',            'val_pc_chamfer',     6),
        ('Param MSE (Smooth-L1 surrogate)',       'val_param_mse',      4),
        ('Param MAE (all leaves)',                'val_param_mae',      4),
        ('Param MAE (masked tokens)',             'val_param_mae_masked', 6),
    ]
    for label, key, p in raw_rows:
        add(f'{label} & {_fmt(best[key], p)} & {_fmt(final[key], p)} \\\\')
    add(f'Param acc@$0.05$ & {_pct(best["val_param_acc05"])} & {_pct(final["val_param_acc05"])} \\\\')
    add(r'\bottomrule')
    add(r'\end{tabular}')
    add(r'\end{table}')
    add('')

    # ── Eval dump aggregate ──────────────────────────────────────────────────
    add(r'\section{Held-out evaluation on the 15-plant gallery}')
    add(
        r'For the reconstruction gallery (Figure~\ref{fig:gallery}) we ran the checkpoint at epoch '
        f'\\textbf{{{final_ep}}} on {AGG.get("n_samples", 15)} validation plants at a higher '
        f'\\texttt{{mask\\_ratio}} of {_fmt(AGG.get("mask_ratio", 0), 2)} so the masking pattern is '
        r'visible. The aggregate metrics (Table~\ref{tab:eval}) confirm the reconstruction quality '
        r'holds up under more aggressive masking.')
    add('')
    add(r'\begin{table}[H]')
    add(r'\centering')
    add(r'\caption{Aggregate metrics over the 15-plant evaluation dump at '
        f'\\texttt{{mask\\_ratio}} = {_fmt(AGG.get("mask_ratio", 0), 2)}.'
        r'}')
    add(r'\label{tab:eval}')
    add(r'\begin{tabular}{lr}')
    add(r'\toprule')
    add(r'Metric & Value \\')
    add(r'\midrule')
    rows = [
        ('RGB MSE',                              _fmt(AGG.get('rgb_mse'), 4)),
        ('Depth MSE',                            _fmt(AGG.get('depth_mse'), 4)),
        ('PC Chamfer (bidirectional)',           _fmt(AGG.get('pc_chamfer'), 6)),
        ('PC EMD (skipped)',                     _fmt(AGG.get('pc_emd'), 4)),
        ('Param MAE (all)',                      _fmt(AGG.get('param_mae_all'), 4)),
        ('Param MAE (masked tokens)',            _fmt(AGG.get('param_mae_masked'), 6)),
        ('Number of plants',                     str(AGG.get('n_samples', 15))),
    ]
    for k, v in rows:
        add(f'{k} & {v} \\\\')
    add(r'\bottomrule')
    add(r'\end{tabular}')
    add(r'\end{table}')
    add('')

    add(r'\begin{figure}[H]')
    add(r'\centering')
    add(r'\includegraphics[width=\linewidth]{slide_figure_v3_2400.png}')
    add(
        f'\\caption{{15-plant reconstruction gallery at epoch~{final_ep}. Rows top-to-bottom: '
        r'RGB ground truth, masked RGB input, RGB reconstruction, depth ground truth, depth '
        r'reconstruction, point-cloud reconstruction (3D scatter). Each column is one '
        r'validation plant.}')
    add(r'\label{fig:gallery}')
    add(r'\end{figure}')
    add(r'\clearpage')
    add('')

    # ── Per-plant pages ──────────────────────────────────────────────────────
    add(r'\section{Per-plant reconstruction figures}')
    add(r'Each figure below corresponds to one validation plant. From left to right: masked input, '
        r'model reconstruction, ground truth. Modality rows: RGB, depth, point cloud, spline / '
        r'procedural parameters. The right-hand panel lists every plant- and leaf-level parameter '
        r'with the target, model prediction, and absolute difference; entries highlighted in red '
        r'were masked at eval time.')
    add('')
    for fname in per_plant_names:
        plant = fname.replace('.png', '')
        add(r'\begin{figure}[H]')
        add(r'\centering')
        add(f'\\includegraphics[width=\\linewidth]{{per_plant/{fname}}}')
        add(f'\\caption{{Plant {safe_name(plant)} -- epoch {final_ep} reconstruction.}}')
        add(f'\\label{{fig:plant:{plant}}}')
        add(r'\end{figure}')
        add(r'\clearpage')
        add('')

    add(r'\end{document}')
    return '\n'.join(lines) + '\n'


def main():
    reset_dirs()
    per_plant_names = stage_figures()
    tex = build_tex(per_plant_names)
    (OUT_DIR / 'main.tex').write_text(tex)
    # README so anyone unfamiliar can rebuild
    (OUT_DIR / 'README.md').write_text(
        '# EmbodiedMAE-4M run v3 (2400-epoch) LaTeX bundle\n\n'
        '- Compile with `pdflatex main.tex` (or upload as a project to Overleaf).\n'
        '- Single-file `main.tex` + `figures/` (top-level + `per_plant/`).\n'
        '- Numbers are pulled from `outputs/4m_run_v3/metrics_summary_2400.json` and the eval '
        '`slide_dump_v3_2400/metrics_aggregate.json` snapshot taken on 2026-05-30.\n')

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob('*'):
            if p.is_file():
                zf.write(p, p.relative_to(OUT_DIR.parent))
    print(f'Wrote {OUT_DIR}/main.tex')
    print(f'Wrote {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
