"""PDF report: every result on the new E10 data (the 3,000-row upload, 2026-11-13), maize and sorghum.
User, 2026-11-13: "Give me a pdf of all the results on new maize and sorghum".

Sections: the data; the fine-tuning runs (settings, validation curves); test tables per species and window for every
model and input stream (reports/*.json of eval/eval_real_rows.py and train_real_rows.py); the summary charts and the
percentile examples of figures/make_e10_r3000_final.py; caveats; what is still running.
Reads whatever reports exist, so a re-run after the last jobs finish picks them up (maize 40 epochs: job 17437008;
sorghum full strip 40 epochs: 17437009).
  python figures/make_e10_r3000_report.py      (CPU, seconds; needs tectonic in the tex env)
-> reports/E10_rows3000_results.pdf (build files in reports/_e10_r3000_report/)
"""
import sys, pathlib, json, glob, subprocess, shutil, datetime
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _scene                                                     # Liberation Sans for the plots
import ugg_style as U

B = R / 'reports/_e10_r3000_report'; B.mkdir(parents=True, exist_ok=True)
PDF = R / 'reports/E10_rows3000_results.pdf'
IMG = R / 'figures/pages/e10_r3000'; DATA = R / 'figures/pages/rows3000/img'
TEX_BIN = '/work/mech-ai-scratch/alloy/.conda/envs/tex/bin'
JOBS = {'maize': [17436546, 17436544, 17437008], 'sorghum zoom': [17444049, 17444047, 17444048], 'sorghum full': [17436547, 17443808, 17437009]}
LABEL = {'maize_4m_pretrain_15k_d2': 'Headline model, zero-shot', '4m_pretrain_15k_d2_sink': 'Headline model, zero-shot',
         'maize_row_iso_e1': 'Isolated control, zero-shot', 'sorghum_row_iso': 'Isolated control, zero-shot',
         'maize_row_ft_e1': 'Synthetic-row fine-tune, zero-shot', 'sorghum_row_ft': 'Synthetic-row fine-tune, zero-shot',
         'maize_real_ft_r3000': 'Real-row fine-tune, 15 ep', 'maize_real_ft_r3000_e40': 'Real-row fine-tune, 40 ep',
         'sorghum_real_ft_r3000': 'Real-row fine-tune, 15 ep', 'sorghum_real_ft_r3000_e40': 'Real-row fine-tune, 40 ep',
         'sorghum_real_ft_r3000_zoom': 'Real-row fine-tune + crops, 15 ep', 'sorghum_real_ft_r3000_zoom_e40': 'Real-row fine-tune + crops, 40 ep'}
ORDER = list(LABEL)
SRC = [('all', 'all three'), ('rgb', 'RGB'), ('depth', 'depth'), ('pc', 'cloud')]
RUNS = [('maize', 'maize_real_ft_r3000'), ('maize', 'maize_real_ft_r3000_e40'), ('sorghum', 'sorghum_real_ft_r3000'),
        ('sorghum', 'sorghum_real_ft_r3000_e40'), ('sorghum', 'sorghum_real_ft_r3000_zoom'), ('sorghum', 'sorghum_real_ft_r3000_zoom_e40')]


def groups(key):
    """{(model, source): group} from the species' reports, first report wins, layout rows3000 only"""
    out, meta, missing = {}, None, []
    for j in JOBS[key]:
        f = [p for p in glob.glob(str(R / f'reports/e10_*_{j}.json'))]
        if not f: missing.append(j); continue
        d = json.load(open(f[0])); meta = meta or d
        for g in d['groups']:
            if g['layout'] == 'rows3000': out.setdefault((g['model'], g['source']), g)
    return out, meta, missing


def f1(x, d=1): return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:,.{d}f}'


def table(key):
    G, meta, missing = groups(key); maize = key == 'maize'
    models = sorted({m for m, _ in G}, key=lambda m: ORDER.index(m) if m in ORDER else 99)
    best = {s: min(G[(m, s)]['chamfer_m2']['mean'] for m in models if (m, s) in G) for s, _ in SRC}
    if maize:
        cols = r'll r r r r r r r r'
        head = (r'Model & Input & \multicolumn{1}{c}{CD [95\,\% CI]} & \multicolumn{1}{c}{Miss@5mm} & \multicolumn{1}{c}{Height err.} & \multicolumn{1}{c}{Leaves $\pm$1} '
                r'& \multicolumn{1}{c}{Internode} & \multicolumn{1}{c}{Length} & \multicolumn{1}{c}{Width} & \multicolumn{1}{c}{Curv.} \\'
                r' & & \multicolumn{1}{c}{$10^{-4}$\,m$^2$} & & median & & cm & cm & cm & 1/m \\')
        cols = 'llrrrrrrrr'
    else:
        cols = 'llrrrr'
        head = (r'Model & Input & \multicolumn{1}{c}{CD mean [95\,\% CI]} & \multicolumn{1}{c}{CD median} & \multicolumn{1}{c}{Miss@5mm} & \multicolumn{1}{c}{Height err.$^\ast$} \\'
                r' & & \multicolumn{1}{c}{$10^{-4}$\,m$^2$} & $10^{-4}$\,m$^2$ & & median \\')
    L = [r'\begin{tabular}{' + cols + '}', r'\toprule', head, r'\midrule']
    for i, m in enumerate(models):
        first = True
        for s, sl in SRC:
            if (m, s) not in G: continue
            g = G[(m, s)]; c = g['chamfer_m2']; cd = f"{c['mean'] * 1e4:,.1f} [{c['lo'] * 1e4:,.1f}, {c['hi'] * 1e4:,.1f}]"
            if abs(c['mean'] - best[s]) < 1e-12: cd = r'\textbf{' + cd + '}'
            name = (r'\textcolor{ours}{' + LABEL.get(m, m) + '}' if 'real_ft' in m else LABEL.get(m, m)) if first else ''
            row = [name, sl, cd]
            if maize:
                row += [f1(g['miss_5mm']['mean'], 3), f1(100 * g.get('height_rel_err_median', np.nan)) + r'\,\%', f1(100 * g.get('leaves_acc_pm1', np.nan), 0) + r'\,\%',
                        f1(100 * g.get('internode_mae', np.nan)), f1(100 * g.get('leaf_len_mae', np.nan)), f1(100 * g.get('leaf_width_mae', np.nan)), f1(g.get('curvature_mae', np.nan), 2)]
            else:
                row += [f"{c['median'] * 1e4:,.1f}", f1(g['miss_5mm']['mean'], 3), f1(100 * g.get('height_rel_err_median', np.nan)) + r'\,\%']
            L.append(' & '.join(row) + r' \\'); first = False
        if i < len(models) - 1: L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(L), meta, missing, G


def curves():
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.0), gridspec_kw=dict(wspace=0.35))
    col = {'maize_real_ft_r3000': '#3f5f8f', 'maize_real_ft_r3000_e40': U.OURS, 'sorghum_real_ft_r3000': '#b3b3b3',
           'sorghum_real_ft_r3000_e40': '#7a7a7a', 'sorghum_real_ft_r3000_zoom': '#3f5f8f', 'sorghum_real_ft_r3000_zoom_e40': U.OURS}
    st = {}
    for sp, run in RUNS:
        p = R / 'outputs' / run / 'training_history.json'
        if not p.exists(): continue
        h = json.load(open(p)); v = h['val']; ep = [x['epoch'] for x in v]; st[run] = (len(v) - 1, v)
        lab = LABEL[run].replace('Real-row fine-tune', 'full strip' if run.startswith('sorghum') and 'zoom' not in run else 'FT').replace('FT + crops', 'zoom + crops')
        if sp == 'maize':
            axs[0].plot(ep, [x['val_chamfer_m2'] * 1e4 for x in v], color=col[run], marker='o', ms=3, mfc='white', lw=1.4, label=lab)
            axs[1].plot(ep, [100 * x['val_height_rel_err'] for x in v], color=col[run], marker='o', ms=3, mfc='white', lw=1.4, label=lab)
            axs[2].plot(ep, [100 * x['val_leaves_pm1'] for x in v], color=col[run], marker='o', ms=3, mfc='white', lw=1.4, label=lab)
        else:
            axs[3].plot(ep, [x['val_chamfer_m2'] * 1e4 for x in v], color=col[run], marker='o', ms=3, mfc='white', lw=1.4, label=lab)
    for ax, t, yl in zip(axs, ('Maize: validation CD', 'Maize: height error', 'Maize: leaf count within $\\pm$1', 'Sorghum: validation CD'),
                         ('CD (10$^{-4}$ m$^2$, log)', 'median rel. error (%)', 'stops (%)', 'CD (10$^{-4}$ m$^2$, log)')):
        ax.set_title(t, fontsize=10, loc='left'); ax.set_xlabel('epoch', fontsize=9); ax.set_ylabel(yl, fontsize=9); ax.tick_params(labelsize=8.5)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axs[0].set_yscale('log'); axs[3].set_yscale('log')
    fig.savefig(B / 'curves.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    return st


def data_fig():
    fig, axs = plt.subplots(1, 8, figsize=(13.5, 3.4), gridspec_kw=dict(wspace=0.05))
    names = ['maize_0', 'maize_1', 'maize_2', 'maize_3', 'sorghum_0', 'sorghum_0_zoom', 'sorghum_1', 'sorghum_1_zoom']
    tit = ['Maize', 'Maize', 'Maize', 'Maize', 'Sorghum, full strip', 'Sorghum, zoomed', 'Sorghum, full strip', 'Sorghum, zoomed']
    for ax, n, t in zip(axs, names, tit):
        p = DATA / f'{n}.jpg'
        if p.exists(): ax.imshow(plt.imread(p))
        ax.set_title(t, fontsize=8.5); ax.set_axis_off()
    fig.savefig(B / 'data.png', dpi=200, bbox_inches='tight'); plt.close(fig)


def main():
    curves_st = curves(); data_fig()
    stats = json.load(open(R / 'figures/pages/rows3000/stats.json'))
    tabs = {k: table(k) for k in JOBS}
    ex = json.load(open(IMG / 'final_examples.json'))
    Gm, Gz, Gf = tabs['maize'][3], tabs['sorghum zoom'][3], tabs['sorghum full'][3]
    g = lambda G, m, s='all': G.get((m, s))
    cd = lambda G, m, s='all': (G[(m, s)]['chamfer_m2']['mean'] * 1e4) if (m, s) in G else float('nan')
    m40 = g(Gm, 'maize_real_ft_r3000_e40'); m15 = g(Gm, 'maize_real_ft_r3000'); mzs = g(Gm, 'maize_4m_pretrain_15k_d2')
    mbest = m40 if (m40 and m40['chamfer_m2']['mean'] < m15['chamfer_m2']['mean']) else m15
    mbest_lab = LABEL['maize_real_ft_r3000_e40' if mbest is m40 else 'maize_real_ft_r3000'].replace('Real-row fine-tune, ', '')
    run_rows = []
    for sp, run in RUNS:
        c = json.load(open(R / 'outputs' / run / 'config.json')).get('real_finetune', {})
        n, v = curves_st.get(run, (0, [{}]))
        bv = min(v[1:], key=lambda x: x['val_chamfer_m2']) if len(v) > 1 else None
        win = c.get('window', 'square' if sp == 'maize' else 'full strip') if sp == 'sorghum' else 'square'
        run_rows.append(' & '.join([sp.capitalize(), run.replace('_', r'\_'), win + (' + crops' if c.get('zoom_aug') else ''),
                                    f"{n} / {c.get('epochs', '?')}", ', '.join(c.get('supervised', [])).replace('_', r'\_').replace('leaf.', ''),
                                    f"{v[0]['val_chamfer_m2'] * 1e4:,.1f}" if v and 'val_chamfer_m2' in v[0] else '--',
                                    (f"{bv['val_chamfer_m2'] * 1e4:,.1f} (ep {bv['epoch']})" if bv else '--')]) + r' \\')
    missing = sorted({j for k in tabs for j in tabs[k][2]})

    def exfig(sp):
        return '\n'.join(r'\begin{center}\includegraphics[width=\linewidth]{' + str(IMG / e['file']) + r'}\\[-2pt]{\footnotesize ' +
                         f"{e['pct']}th percentile: test plant {e['plant'].replace('_', r'\_')}, row {int(e['row'])}, stop {e['stop']}; "
                         f"{e['occ'] * 100:.0f}\\,\\% of the plant hidden by neighbours; true height {e['height']:.2f}\\,m"
                         + (f", {e['n_true']} leaves; written {e['hp']:.2f}\\,m, {e['n_pred']} leaves" if sp == 'maize' else f"; written {e['hp']:.2f}\\,m")
                         + r'}\end{center}' for e in ex[sp])

    today = datetime.date.today().isoformat()
    tex = rf"""\RequirePackage[T1]{{fontenc}}
\documentclass[10pt]{{article}}
\usepackage[letterpaper,margin=0.7in]{{geometry}}
\usepackage{{helvet}}\renewcommand{{\familydefault}}{{\sfdefault}}
\usepackage{{graphicx,booktabs,xcolor,microtype,hyperref,caption}}
\definecolor{{ours}}{{HTML}}{{6A2FD6}}
\hypersetup{{colorlinks,linkcolor=ours,urlcolor=ours}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\captionsetup{{font=footnotesize,labelfont=bf}}
\title{{\textbf{{E10 on the new real-plant rows (3,000-row upload)}}\\\large Maize and sorghum: zero-shot and real-row fine-tuning results}}
\author{{CropCode}}\date{{{today}}}
\begin{{document}}\maketitle

\section*{{Headline}}
\begin{{itemize}}\itemsep2pt
\item \textbf{{Maize}} (130 held-out test plants, {tabs['maize'][1]['n_stops']:,} camera stops, all three inputs): Chamfer distance (CD) {cd(Gm, 'maize_4m_pretrain_15k_d2'):.1f} (headline model, zero-shot) $\to$ \textbf{{{mbest['chamfer_m2']['mean'] * 1e4:.1f}}} after real-row fine-tuning ({mbest_lab}), $10^{{-4}}$\,m$^2$. Median height error {100 * mzs['height_rel_err_median']:.0f}\,\% $\to$ \textbf{{{100 * mbest['height_rel_err_median']:.1f}\,\%}}; leaf count within $\pm$1 on {100 * mzs['leaves_acc_pm1']:.0f}\,\% $\to$ \textbf{{{100 * mbest['leaves_acc_pm1']:.0f}\,\%}} of stops; per-leaf length error {100 * mzs['leaf_len_mae']:.0f} $\to$ \textbf{{{100 * mbest['leaf_len_mae']:.1f}\,cm}}, width {100 * mzs['leaf_width_mae']:.1f} $\to$ \textbf{{{100 * mbest['leaf_width_mae']:.1f}\,cm}}, internode {100 * mzs['internode_mae']:.1f} $\to$ \textbf{{{100 * mbest['internode_mae']:.1f}\,cm}} (true means 67, 18 and 13\,cm).
\item \textbf{{Sorghum}} (13 held-out test scans, zoomed window): CD {cd(Gz, '4m_pretrain_15k_d2_sink'):.0f} (headline, zero-shot), {cd(Gz, 'sorghum_row_iso'):.0f} (isolated control) $\to$ \textbf{{{cd(Gz, 'sorghum_real_ft_r3000_zoom_e40'):.1f}}} after real-row fine-tuning with zoomed-crop augmentation, 40 epochs ({cd(Gz, 'sorghum_real_ft_r3000_zoom'):.1f} at 15 epochs). On the full tall strip the 15-epoch fine-tune reaches {cd(Gf, 'sorghum_real_ft_r3000'):.1f} (from {cd(Gf, '4m_pretrain_15k_d2_sink'):,.0f}); the zoom is the better input.
\item \textbf{{Sorghum height is not learned}} (error $\approx$72\,\% before and after): the scans carry no procedural parameters, so the sorghum fine-tune has no recipe or height loss, and the written stem length stays near the synthetic ceiling while the real plants are 2.2--5.3\,m tall. Read sorghum by its clouds.
\end{{itemize}}

\section*{{1\quad The data}}
The 2026-11-13 upload replaces the first E10 rows. Each species has {stats['maize']['rows']:,} rows $\times$ {stats['maize']['stops_per_row']} camera stops ({stats['maize']['stops']:,} stops): a photograph and a labelled point cloud (80k points per row, metres, camera frame) per stop, with the middle plant as the target.
\begin{{itemize}}\itemsep1pt
\item \textbf{{Maize}}: {stats['maize']['plants']} plant models; a per-leaf CSV for the middle plant (internode distance, leaf length, width, curvature, zenith, \dots). Split by plant: {stats['maize']['split']['train']} train / {stats['maize']['split']['val']} validation / {stats['maize']['split']['test']} test.
\item \textbf{{Sorghum}}: 52 scans, each with four tilted copies (208 plant models), heights 2.2--5.3\,m; no leaf parameters. Split by scan (the tilted copies stay together): 34 / 5 / 13. The photographs are tall strips; the \emph{{zoomed}} window is a square of side $\min(H, 3W)$ centred on the median image row of the middle plant's leaves, so the plant appears at maize-like scale. Training adds random zoomed crops (side $\times U(0.75, 1.25)$, centre between the 10th and 90th percentile of leaf rows, $\pm$10\,\% of the width sideways).
\end{{itemize}}
\begin{{center}}\includegraphics[width=\linewidth]{{{B / 'data.png'}}}\end{{center}}

\section*{{2\quad Fine-tuning runs}}
All runs start from the synthetic-row fine-tune (epoch 100) of their species and train on the training plants' stops (lr $2\times10^{{-5}}$, batch 16, 80\,\% of input tokens hidden); validation every epoch on the validation plants; the best validation checkpoint is tested.
\begin{{center}}\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lllrp{{5.2cm}}rr}}\toprule
Species & Run & Window & Epochs & Supervised & Val CD start & Best val CD \\\midrule
{chr(10).join(run_rows)}
\bottomrule\end{{tabular}}\end{{center}}
\begin{{center}}\includegraphics[width=\linewidth]{{{B / 'curves.png'}}}\end{{center}}

\clearpage
\section*{{3\quad Test results}}
Held-out test plants only. CD = squared Chamfer distance of the target plant's predicted cloud to its true cloud, $10^{{-4}}$\,m$^2$ (lower is better; bold = best per input). Miss@5mm = share of true points with no prediction within 5\,mm. Input: which streams the model sees (the others are hidden; the recipe is never an input). Per-leaf errors compare the written recipe with the leaf CSV, leaves matched by index from the base.

\subsection*{{3.1\quad Maize, {tabs['maize'][1]['plants']} test plants, {tabs['maize'][1]['n_stops']:,} stops}}
\begin{{center}}\scriptsize\setlength{{\tabcolsep}}{{3.2pt}}
{tabs['maize'][0]}
\end{{center}}

\subsection*{{3.2\quad Sorghum, zoomed window, {tabs['sorghum zoom'][1]['plants']} test scans, {tabs['sorghum zoom'][1]['n_stops']:,} stops}}
\begin{{center}}\scriptsize\setlength{{\tabcolsep}}{{4pt}}
{tabs['sorghum zoom'][0]}
\end{{center}}

\subsection*{{3.3\quad Sorghum, full tall strip, {tabs['sorghum full'][1]['plants']} test scans}}
\begin{{center}}\scriptsize\setlength{{\tabcolsep}}{{4pt}}
{tabs['sorghum full'][0]}
\end{{center}}
{{\footnotesize $^\ast$Sorghum height is read from the written stem length, which no sorghum run is trained on (no recipe truth); it stays near the synthetic ceiling against 2.2--5.3\,m real plants. The zoomed and full-strip CDs are on different windows and are not comparable with each other.}}

\begin{{center}}\includegraphics[width=\linewidth]{{{IMG / 'final_charts.png'}}}\end{{center}}
{{\footnotesize Left two: test CD of every model (all three inputs, 95\,\% CI). Third: maize height per test plant (median over its stops); the zero-shot headline model writes about 0.77\,m for every plant. Right: maize per-leaf error.}}

\clearpage
\section*{{4\quad Examples}}
Test plants at the 25th, 50th and 75th percentile of the fine-tuned model's per-plant CD, each at its median-error stop. Columns: the photograph, the true plant, the synthetic-row fine-tune (before), the real-row fine-tune (after), the procedural model the real-row fine-tune writes, and (maize) its leaf lengths against the true ones.
\subsection*{{Maize (real-row fine-tune, 15 epochs)}}
{exfig('maize')}
\subsection*{{Sorghum (zoomed window, real-row fine-tune + crops, 40 epochs)}}
{exfig('sorghum')}

\section*{{5\quad Caveats and status}}
\begin{{itemize}}\itemsep1pt
\item The first E10 upload gave maize height error 1.6\,\% after fine-tuning; these rows give {100 * mbest['height_rel_err_median']:.1f}\,\% per stop (5\,\% per plant, median over its stops): a harder set (more plants, taller, more varied).
\item Sorghum has 13 test scans (52 scans in total), so its confidence intervals are wide.
\item The paper's Sec.~4.8 still reports the first upload.
\item {'All runs have a test report.' if not missing else 'Not yet in this report (still running): ' + ', '.join(f'job {j}' for j in missing) + '. Re-run figures/make\\_e10\\_r3000\\_report.py when they finish.'}
\end{{itemize}}
\end{{document}}
"""
    (B / 'report.tex').write_text(tex)
    env = dict(**__import__('os').environ); env['PATH'] = TEX_BIN + ':' + env['PATH']
    r = subprocess.run(['tectonic', '-X', 'compile', 'report.tex'], cwd=B, env=env, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
    shutil.copy(B / 'report.pdf', PDF); print('wrote', PDF, 'missing reports:', missing)


def visual():
    """--visual: the example figures only (user: "I need only the visual result no more data")."""
    ex = json.load(open(IMG / 'final_examples.json'))
    body = []
    for sp, title in (('maize', 'Maize'), ('sorghum', 'Sorghum (zoomed window)')):
        body.append(r'\section*{' + title + '}')
        for e in ex[sp]:
            body.append(r'\begin{center}\includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{' + str(IMG / e['file']) + r'}\end{center}')
    tex = (r"""\RequirePackage[T1]{fontenc}
\documentclass[10pt]{article}
\usepackage[letterpaper,landscape,margin=0.5in]{geometry}
\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{graphicx}\pagestyle{empty}\setlength{\parindent}{0pt}
\begin{document}
""" + '\n'.join(body) + '\n\\end{document}\n')
    (B / 'visual.tex').write_text(tex)
    env = dict(**__import__('os').environ); env['PATH'] = TEX_BIN + ':' + env['PATH']
    r = subprocess.run(['tectonic', '-X', 'compile', 'visual.tex'], cwd=B, env=env, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
    out = R / 'reports/E10_rows3000_visual.pdf'; shutil.copy(B / 'visual.pdf', out); print('wrote', out)


if __name__ == '__main__':
    visual() if '--visual' in sys.argv else main()
