"""PREVIEW variants of the six result plots of make_paper_figs_v2.py, restyled after the UGG paper's figure
conventions (figures/ugg_style.py). 2026-11-13.

  python figures/make_ugg_plots_v2.py      # writes paper/figs/ugg_preview/<name>_ugg.{pdf,png} only

Data, numbers, panels, panel letters/titles, fonts and font sizes are the original script's (its rcParams and
data tables are imported, so the two can never drift apart). What changes: colours (E2 arms take the colour of the
stream they add, target properties the generic SERIES order, our model OURS violet, baselines greys), hollow
markers on solid lines, dashed grey references, one frameless legend row above the panels, no top/right spines,
light grid. Each PDF is saved at exactly the page size of the paper's current figure, so swapping a preview in
would not move the layout. The original script and paper/figs/<name>.* / paper/overleaf/Figures/ are untouched.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figs_v2 as V                       # sets rcParams (font + sizes) and loads P, FS
import ugg_style as U

PREV = V.REPO / 'paper' / 'figs' / 'ugg_preview'; PREV.mkdir(parents=True, exist_ok=True)
# page sizes (pt) of the figures as used in the paper (paper/overleaf/Figures/<name>.pdf, pdfinfo 2026-11-13)
TARGET = {'fig_e2_streams': (452.958, 193.334), 'fig_scaling': (432.153, 188.099), 'fig_fscore': (430.332, 185.815),
          'fig_partial_clouds': (228.415, 181.08), 'fig_param_controls': (429.295, 182.13),
          'fig_rows': (461.764, 307.846), 'fig_rows_maize': (461.764, 180.334), 'fig_rows_sorghum': (461.764, 180.334)}
PAD = 0.1                                             # savefig's default pad_inches
ARMLAB = {r: lab for r, (lab, _) in V.ARM.items()}    # run -> 'PC', 'PC+RGB', ...
TCOL = {'height': U.SERIES[0], 'leaf_angle_roll': U.SERIES[1], 'leaf_len_mean': U.SERIES[2], 'leaf_width_mean': U.SERIES[3]}
TMK = {'height': 'o', 'leaf_angle_roll': 's', 'leaf_len_mean': '^', 'leaf_width_mean': 'D'}


def sty(ax, grid='y'):
    """ugg_style.style_axes, and switch off the other grid axis (the original rcParams turn both on)."""
    U.style_axes(ax, grid=grid)
    ax.grid(False, axis={'y': 'x', 'x': 'y'}[grid])
    return ax


def _tight(fig):
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer())


def fit(fig, name, bottom, top=None, iters=8):
    """Move the subplots' bottom edge until the tight bbox (+ pad) is exactly as tall as the paper's figure."""
    th = TARGET[name][1] / 72; fh = fig.get_figheight()
    for _ in range(iters):
        fig.subplots_adjust(bottom=bottom, **({'top': top} if top is not None else {}))
        d = _tight(fig).height + 2 * PAD - th
        if abs(d) < 0.004: break
        bottom += d / fh
    return bottom


def save(fig, name):
    tb = _tight(fig); tw, th = (v / 72 for v in TARGET[name])
    w, h = tb.width + 2 * PAD, tb.height + 2 * PAD
    flag = '' if (w <= tw + 0.01 and h <= th + 0.01) else '   WARNING: content larger than the paper figure'
    print(f'{name}: content {w * 72:.1f} x {h * 72:.1f} pt, page {tw * 72:.1f} x {th * 72:.1f} pt{flag}')
    cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
    bb = Bbox.from_bounds(cx - tw / 2, cy - th / 2, tw, th)
    for ext in ('pdf', 'png'):
        fig.savefig(PREV / f'{name}_ugg.{ext}', dpi=300, bbox_inches=bb, facecolor='white', transparent=False)
    plt.close(fig); print('wrote', PREV / f'{name}_ugg.pdf')


def arm_handle(run):
    lab = ARMLAB[run]
    return Line2D([], [], color=U.ARM[lab], marker=U.ARM_MARKER[lab], mfc='white', mec=U.ARM[lab], mew=1.0, lw=1.4, ms=4.0)


def top_legend(fig, handles, labels, y, ncol=None, **kw):
    """ugg_style.top_legend (one frameless row centred above the panels) with its spacing overridable."""
    opts = dict(handlelength=1.8, columnspacing=1.4); opts.update(kw)
    return fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
                      frameon=False, **opts)


def ours_legend_text(leg, label):
    for t in leg.get_texts():
        if t.get_text() == label:
            t.set_color(U.OURS); t.set_fontweight('bold')


# ---------------- E2 streams: probe dot plot + F1 bars ----------------
def fig_e2():
    T = ['height', 'leaf_angle', 'leaf_angle_roll', 'leaf_len_mean', 'leaf_width_mean']
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={'width_ratios': [2.6, 1]})
    sty(a, 'x'); sty(b, 'y')
    y = np.arange(len(T))[::-1]
    for k, run in enumerate(V.ARM):
        lab = ARMLAB[run]; col = U.ARM[lab]
        a.scatter([V.P.loc[run, t] for t in T], y + (1.5 - k) * 0.17, s=22, facecolor='white', edgecolor=col,
                  linewidth=1.1, marker=U.ARM_MARKER[lab], zorder=3)
    a.set_yticks(y); a.set_yticklabels([V.TLAB[t] for t in T]); a.set_xlim(0, 1.0)
    a.set_xlabel('frozen-probe R² (val)'); a.set_title('a  What each stream adds to the latent', loc='left')
    a.tick_params(axis='y', length=0)
    f1 = V.FS[(V.FS.split == 'test')].set_index('run')
    runs = list(V.ARM); vals = [f1.loc[r, 'f1_0.01_mean'] for r in runs]
    b.bar(range(len(runs)), vals, color=[U.ARM[ARMLAB[r]] for r in runs], width=0.7, edgecolor='none')
    for i, v in enumerate(vals): b.text(i, v + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=7, color=V.INK)
    b.set_xticks(range(len(runs))); b.set_xticklabels([ARMLAB[r] for r in runs], rotation=30, ha='right')
    b.set_ylabel('cloud F1 @ τ=0.01 (test)'); b.set_ylim(0, 0.36)
    b.set_title('b  Reconstruction quality', loc='left')
    top_legend(fig, [arm_handle(r) for r in runs], [ARMLAB[r] for r in runs], y=0.985)
    fit(fig, 'fig_e2_streams', bottom=0.11, top=0.885)
    save(fig, 'fig_e2_streams')


# ---------------- data and model scaling ----------------
def fig_scaling():
    T = ['height', 'leaf_angle_roll', 'leaf_len_mean', 'leaf_width_mean']
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3))
    for ax in (a, b): sty(ax, 'y')
    e3 = [('e3_1k_d2', 1000), ('e3_3k_d2', 3000), ('e3_10k_d2', 10000)]
    for t in T:
        U.line(a, [n for _, n in e3], [V.P.loc[r, t] for r, _ in e3], TCOL[t], label=V.TLAB[t], marker=TMK[t])
    a.set_xscale('log'); a.set_xticks([1000, 3000, 10000]); a.set_xticklabels(['1k', '3k', '10k']); a.minorticks_off()
    a.set_xlabel('training plants (197,400 steps each)'); a.set_ylabel('frozen-probe R² (val)')
    a.set_ylim(0, 1.0); a.set_title('a  Data scaling', loc='left')
    e4 = [('e4_small_d2', 25.2), ('e2_pcrgbdt_d2', 114.3), ('e4_large_d2', 332.2)]
    for t in T:
        U.line(b, [s for _, s in e4], [V.P.loc[r, t] for r, _ in e4], TCOL[t], label=V.TLAB[t], marker=TMK[t])
    b.set_xscale('log'); b.set_xticks([25.2, 114.3, 332.2]); b.set_xticklabels(['small\n25 M', 'base\n114 M', 'large\n332 M'])
    b.minorticks_off(); b.set_ylim(0, 1.0); b.set_title('b  Model scaling', loc='left'); b.set_xlabel('encoder size')
    h, l = a.get_legend_handles_labels(); top_legend(fig, h, l, y=0.985)
    fit(fig, 'fig_scaling', bottom=0.11, top=0.885)
    save(fig, 'fig_scaling')


# ---------------- point-cloud P / R / F1 ----------------
def fig_fscore():
    E1 = '4m_pretrain_15k_d2_sink'; E1LAB = 'E1 (1,000 ep)'
    M = [(E1, E1LAB, U.OURS, 'o'), ('e2_pcrgbd_d2', 'PC+RGB+D', None, None), ('e2_pcrgb_d2', 'PC+RGB', None, None),
         ('e2_pcrgbdt_d2', 'PC+RGB+D+θ', None, None), ('e2_pc_d2', 'PC', None, None)]
    M = [(r, l, c or U.ARM[l], m or U.ARM_MARKER[l]) for r, l, c, m in M]
    te = V.FS[V.FS.split == 'test'].set_index('run'); taus = [0.005, 0.01, 0.02]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.4))
    for ax in (a, b): sty(ax, 'y')
    for run, lab, col, mk in M:
        ys = [te.loc[run, f'f1_{t:g}_mean'] for t in taus]
        if run == E1:
            U.line(a, taus, ys, col, label=lab, marker=mk, hollow=False, lw=1.8, zorder=4)
        else:
            U.line(a, taus, ys, col, label=lab, marker=mk)
    a.set_xscale('log'); a.set_xticks(taus); a.set_xticklabels(['0.005', '0.01', '0.02']); a.minorticks_off()
    a.set_xlabel('distance threshold τ (unit-sphere frame; 0.01 ≈ 1 cm)'); a.set_ylabel('F1 (test, per-plant mean)')
    a.set_ylim(0, 0.8); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); a.set_title('a  F1 against threshold', loc='left')
    e1y = [te.loc[E1, f'f1_{t:g}_mean'] for t in taus]
    U.label_ends(a, e1y[0], e1y[-1], U.OURS, fmt='{:.2f}')
    a.yaxis.labelpad += 2
    for run, lab, col, mk in M:
        x, y = te.loc[run, 'recall_0.01_mean'], te.loc[run, 'precision_0.01_mean']
        ours = run == E1
        b.scatter(x, y, marker=mk, s=28, zorder=3, facecolor=col if ours else 'white', edgecolor=col, linewidth=1.1)
        left = lab == 'PC+RGB'                        # it sits 0.03 below PC+RGB+D: label it on the other side
        b.annotate(lab, (x, y), xytext=(-4 if left else 4, -3), textcoords='offset points', fontsize=6.8,
                   ha='right' if left else 'left', color=U.OURS if ours else V.INK)
    b.plot([0, 0.7], [0, 0.7], color=U.REF, lw=0.9, ls=(0, (4, 3))); b.text(0.30, 0.25, 'P = R', fontsize=6.5, color=U.REF, rotation=33)
    b.set_xlim(0, 0.45); b.set_ylim(0, 0.75); b.set_xlabel('recall @ τ=0.01'); b.set_ylabel('precision @ τ=0.01')
    b.set_title('b  Every model under-covers the plant', loc='left')
    h, l = a.get_legend_handles_labels(); leg = top_legend(fig, h, l, y=0.985); ours_legend_text(leg, E1LAB)
    fit(fig, 'fig_fscore', bottom=0.11, top=0.885)
    save(fig, 'fig_fscore')


# ---------------- partial clouds ----------------
def fig_pcview():
    pv = V.pd.read_csv(V.R / 'pcview_16930426_probe.csv')
    cov = V.pd.read_csv(V.R / 'pcview_16930426_coverage.csv').groupby('condition')['mean'].mean()
    conds = ['1view', '3view', 'full']; x = [cov[c] * 100 for c in conds]
    fig, a = plt.subplots(figsize=(3.3, 2.4)); sty(a, 'y')
    for run in V.ARM:
        lab = ARMLAB[run]; col = U.ARM[lab]; mk = U.ARM_MARKER[lab]
        g = lambda fit: [float(pv[(pv.run == run) & (pv.fit == fit) & (pv.condition == c) & (pv.target == 'height')].test_r2.iloc[0]) for c in conds]
        U.line(a, x, g('matched'), col, label=lab, marker=mk)
        U.line(a, x, g('transfer'), col, marker=mk, ls=(0, (4, 2.5)), lw=1.1, ms=3.6, alpha=0.85)
    a.set_xlabel('share of the plant the cameras see (%)'); a.set_ylabel('height R² (test)'); a.set_ylim(-0.1, 1.02)
    a.set_xticks(x); a.set_xticklabels(['1 cam\n49', '3 cams\n78', 'full\n100'])
    kw = dict(fontsize=6.3, handlelength=1.5, columnspacing=0.9, handletextpad=0.4)
    h, l = a.get_legend_handles_labels()
    top_legend(fig, h, l, y=1.045, **kw)
    st = [Line2D([], [], color=V.INK2, lw=1.4, marker='o', mfc='white', mec=V.INK2, ms=4.0),
          Line2D([], [], color=V.INK2, lw=1.1, ls=(0, (4, 2.5)), marker='o', mfc='white', mec=V.INK2, ms=3.6)]
    top_legend(fig, st, ['refit per condition', 'full-cloud probe transferred'], y=0.975, **kw)
    fit(fig, 'fig_partial_clouds', bottom=0.11, top=0.94)
    save(fig, 'fig_partial_clouds')


# ---------------- parameter-stream controls ----------------
def fig_controls():
    arms = [('e2_pcrgbd_d2', 'none\n(PC+RGB+D)'), ('e2_pcrgbdt_tg100_d2', 'target\nonly'), ('e2_pcrgbdt_tg_d2', '20 %\nvisible'),
            ('e2_pcrgbdt_d2', '58 %\n(as trained)'), ('e2_pcrgbdt_tg40_d2', '60 %\nvisible')]
    T = ['height', 'leaf_angle_roll', 'leaf_len_mean', 'leaf_width_mean']
    fig, a = plt.subplots(figsize=(6.9, 2.2)); sty(a, 'y'); w = 0.19; xs = np.arange(len(arms))
    for j, t in enumerate(T):
        a.bar(xs + (j - 1.5) * w, [V.P.loc[r, t] for r, _ in arms], width=w * 0.92, color=TCOL[t], label=V.TLAB[t], edgecolor='none')
    a.set_xticks(xs); a.set_xticklabels([l for _, l in arms]); a.set_ylabel('frozen-probe R² (val)'); a.set_ylim(0, 1.0)
    a.set_xlabel('recipe (θ) stream during pretraining, same 117 visible sensor tokens'); a.tick_params(axis='x', length=0)
    h, l = a.get_legend_handles_labels(); top_legend(fig, h, l, y=0.985, handlelength=1.2)
    fit(fig, 'fig_param_controls', bottom=0.11, top=0.94)
    save(fig, 'fig_param_controls')


# ---------------- rows (Regime III), both species ----------------
ROWCOL = {'row-trained': (U.OURS, 'o'), 'isolated-trained': (U.GREYS[1], 's'), 'zero-shot': (U.GREYS[0], 'D')}


def _rows_panels(a, b, species, letters):
    load = lambda p: json.load(open(sorted(glob.glob(str(V.R / p)))[-1]))
    ft, iso, zs = (load(p) for p in V.ROWS[species])
    m = lambda d, k: d['by_mode_source'][k]['chamfer_gt_m2']['mean'] * 1e4
    rows = [('isolated-trained,\nclean input', m(iso, 'isolated/all'), U.GREYS[3]), ('row-trained,\nrow input', m(ft, 'row/all'), U.OURS),
            ('isolated-trained,\nrow input', m(iso, 'row/all'), U.GREYS[1]), ('zero-shot,\nrow input', m(zs, 'row/all'), U.GREYS[0])]
    sty(a, 'x'); sty(b, 'y')
    vmax = max(v for _, v, _ in rows)
    a.barh(range(len(rows))[::-1], [v for _, v, _ in rows], color=[c for *_, c in rows], height=0.62, edgecolor='none')
    for i, (lab, v, c) in enumerate(rows):
        a.text(v + 0.015 * vmax * 1.15, len(rows) - 1 - i, f'{v:.1f}', va='center', fontsize=7,
               color=U.OURS if c == U.OURS else V.INK, fontweight='bold' if c == U.OURS else 'normal')
    a.set_yticks(range(len(rows))[::-1]); a.set_yticklabels([l for l, *_ in rows], fontsize=6.8)
    for tl, (_, _, c) in zip(a.get_yticklabels(), rows):
        if c == U.OURS: tl.set_color(U.OURS)
    a.tick_params(axis='y', length=0)
    a.set_xlim(0, vmax * 1.15); a.set_xlabel('Chamfer vs target plant (10⁻⁴ m², test)')
    a.set_title(f'{letters[0]}  {species.capitalize()}: Chamfer by training and input', loc='left')
    th = sorted(ft['by_theta']['row/all'], key=int); xs = [int(t) for t in th]
    for d, lab in ((ft, 'row-trained'), (iso, 'isolated-trained'), (zs, 'zero-shot')):
        col, mk = ROWCOL[lab]
        U.line(b, xs, [d['by_theta']['row/all'][t]['chamfer_gt_m2']['mean'] * 1e4 for t in th], col, label=lab, marker=mk,
               **({'zorder': 4} if lab == 'row-trained' else {}))
    b.set_xlabel('row yaw θ (°; 0 broadside, 90 end-on)'); b.set_ylabel('Chamfer (10⁻⁴ m²)'); b.set_xticks([0, 30, 60, 90])
    b.set_title(f'{letters[1]}  {species.capitalize()}: by row orientation', loc='left')
    return b


def _rows_legend(fig, b, y):
    h, l = b.get_legend_handles_labels(); leg = top_legend(fig, h, l, y=y); ours_legend_text(leg, 'row-trained')


def fig_rows():
    fig, ax = plt.subplots(2, 2, figsize=(6.9, 4.6), gridspec_kw={'width_ratios': [1.1, 1], 'wspace': 0.42, 'hspace': 0.62})
    b = _rows_panels(ax[0, 0], ax[0, 1], 'maize', 'ab'); _rows_panels(ax[1, 0], ax[1, 1], 'sorghum', 'cd')
    _rows_legend(fig, b, 0.99)
    fit(fig, 'fig_rows', bottom=0.11, top=0.935)
    save(fig, 'fig_rows')
    for sp in ('maize', 'sorghum'):
        f, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={'width_ratios': [1.1, 1], 'wspace': 0.42})
        b = _rows_panels(a, b, sp, 'ab'); _rows_legend(f, b, 0.985)
        fit(f, f'fig_rows_{sp}', bottom=0.11, top=0.885)
        save(f, f'fig_rows_{sp}')


if __name__ == '__main__':
    fig_e2(); fig_scaling(); fig_fscore(); fig_pcview(); fig_controls(); fig_rows()
