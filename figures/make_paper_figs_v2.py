"""Paper figures on the final sorghum data (and the maize row study), white background.

  python figures/make_paper_figs_v2.py            # writes paper/figs/fig_*.{pdf,png}

Every number is read from reports/ and outputs/ at run time (same sources as the ledger page).
Colours: the validated categorical order (blue, orange, aqua, yellow, magenta, green), assigned
to ENTITIES in a fixed order, never by rank; identity is also carried by direct labels or
markers, never colour alone. Sized for CVPR: single column 3.3 in, double 6.9 in, 8 pt text.
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; R = REPO / 'reports'; O = REPO / 'outputs'
OUT = REPO / 'paper' / 'figs'; OUT.mkdir(parents=True, exist_ok=True)
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'lines.linewidth': 1.6, 'lines.markersize': 4.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'DejaVu Sans'})
ARM = {'e2_pc_d2': ('PC', C['orange']), 'e2_pcrgb_d2': ('PC+RGB', C['aqua']),
       'e2_pcrgbd_d2': ('PC+RGB+D', C['blue']), 'e2_pcrgbdt_d2': ('PC+RGB+D+θ', C['yellow'])}
TLAB = {'height': 'height', 'leaf_count': 'leaf count', 'biomass': 'biomass', 'leaf_angle': 'leaf angle',
        'leaf_angle_roll': 'roll', 'leaf_len_mean': 'leaf length', 'leaf_width_mean': 'leaf width'}

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig); print('wrote', OUT / f'{name}.pdf')

def probe(files, feature='cls', split='val'):
    d = pd.concat([pd.read_csv(f) for f in files])
    if feature and 'feature' in d.columns: d = d[d.feature == feature]
    return d.pivot_table(index='run', columns='target', values=f'{split}_r2')

P = probe([R/'probe_pr_600_d2_16930422.csv', R/'probe_pr_6168_d2_16928969.csv', R/'probe_pr_2024_d2_16928970.csv', R/'probe_pr_624_d2_16928971.csv'])
FS = pd.read_csv(R/'pcfscore_17071405_summary.csv')

# ---------------- Fig 1: E2 streams, probe dot plot + F1 ----------------
def fig_e2():
    T = ['height', 'leaf_angle', 'leaf_angle_roll', 'leaf_len_mean', 'leaf_width_mean']
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={'width_ratios': [2.6, 1]})
    y = np.arange(len(T))[::-1]
    for k, (run, (lab, col)) in enumerate(ARM.items()):
        a.scatter([P.loc[run, t] for t in T], y + (1.5 - k) * 0.17, s=22, color=col, label=lab, zorder=3,
                  marker='os^D'[k], edgecolor='white', linewidth=0.6)
    a.set_yticks(y); a.set_yticklabels([TLAB[t] for t in T]); a.set_xlim(0, 1.0)
    a.set_xlabel('frozen-probe R² (val)'); a.legend(loc='center left', bbox_to_anchor=(0.42, 0.5), handletextpad=0.2)
    a.set_title('a  What each stream adds to the latent', loc='left')
    a.grid(axis='y', visible=False)
    f1 = FS[(FS.split == 'test')].set_index('run')
    runs = list(ARM); vals = [f1.loc[r, 'f1_0.01_mean'] for r in runs]
    bars = b.bar(range(len(runs)), vals, color=[ARM[r][1] for r in runs], width=0.7, edgecolor='white', linewidth=1)
    for i, v in enumerate(vals): b.text(i, v + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=7, color=INK)
    b.set_xticks(range(len(runs))); b.set_xticklabels([ARM[r][0] for r in runs], rotation=30, ha='right')
    b.set_ylabel('cloud F1 @ τ=0.01 (test)'); b.set_ylim(0, 0.36); b.grid(axis='x', visible=False)
    b.set_title('b  Reconstruction quality', loc='left')
    save(fig, 'fig_e2_streams')

# ---------------- Fig 2: data and model scaling ----------------
def fig_scaling():
    T = [('height', C['blue'], 'o'), ('leaf_angle_roll', C['orange'], 's'), ('leaf_len_mean', C['aqua'], '^'), ('leaf_width_mean', C['magenta'], 'D')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3))
    e3 = [('e3_1k_d2', 1000), ('e3_3k_d2', 3000), ('e3_10k_d2', 10000)]   # full data (10.5k) is base in panel b
    for t, col, mk in T:
        a.plot([n for _, n in e3], [P.loc[r, t] for r, _ in e3], color=col, marker=mk, label=TLAB[t])
    a.set_xscale('log'); a.set_xticks([1000, 3000, 10000]); a.set_xticklabels(['1k', '3k', '10k']); a.minorticks_off()
    a.set_xlabel('training plants (197,400 steps each)'); a.set_ylabel('frozen-probe R² (val)')
    a.set_ylim(0, 1.0); a.set_title('a  Data scaling', loc='left'); a.legend(loc='center right', bbox_to_anchor=(1.0, 0.42))
    e4 = [('e4_small_d2', 25.2), ('e2_pcrgbdt_d2', 114.3), ('e4_large_d2', 332.2)]
    for t, col, mk in T:
        b.plot([s for _, s in e4], [P.loc[r, t] for r, _ in e4], color=col, marker=mk, label=TLAB[t])
    b.set_xscale('log'); b.set_xticks([25.2, 114.3, 332.2]); b.set_xticklabels(['small\n25 M', 'base\n114 M', 'large\n332 M'])
    b.minorticks_off(); b.set_ylim(0, 1.0); b.set_title('b  Model scaling', loc='left'); b.set_xlabel('encoder size')
    save(fig, 'fig_scaling')

# ---------------- Fig 3: point-cloud P / R / F1 ----------------
def fig_fscore():
    M = [('4m_pretrain_15k_d2_sink', 'E1 (1,000 ep)', INK, 'o'), ('e2_pcrgbd_d2', 'PC+RGB+D', C['blue'], 's'),
         ('e2_pcrgb_d2', 'PC+RGB', C['aqua'], '^'), ('e2_pcrgbdt_d2', 'PC+RGB+D+θ', C['yellow'], 'D'), ('e2_pc_d2', 'PC', C['orange'], 'v')]
    te = FS[FS.split == 'test'].set_index('run'); taus = [0.005, 0.01, 0.02]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.4))
    for run, lab, col, mk in M:
        a.plot(taus, [te.loc[run, f'f1_{t:g}_mean'] for t in taus], color=col, marker=mk, label=lab)
    a.set_xscale('log'); a.set_xticks(taus); a.set_xticklabels(['0.005', '0.01', '0.02']); a.minorticks_off()
    a.set_xlabel('distance threshold τ (unit-sphere frame; 0.01 ≈ 1 cm)'); a.set_ylabel('F1 (test, per-plant mean)')
    a.set_ylim(0, 0.8); a.legend(loc='upper left'); a.set_title('a  F1 against threshold', loc='left')
    for run, lab, col, mk in M:
        b.scatter(te.loc[run, 'recall_0.01_mean'], te.loc[run, 'precision_0.01_mean'], color=col, marker=mk, s=28, zorder=3, edgecolor='white', linewidth=0.6)
        b.annotate(lab, (te.loc[run, 'recall_0.01_mean'], te.loc[run, 'precision_0.01_mean']), xytext=(5, -3), textcoords='offset points', fontsize=6.8, color=INK)
    b.plot([0, 0.7], [0, 0.7], color=INK2, lw=0.8, ls='--'); b.text(0.30, 0.25, 'P = R', fontsize=6.5, color=INK2, rotation=33)
    b.set_xlim(0, 0.45); b.set_ylim(0, 0.75); b.set_xlabel('recall @ τ=0.01'); b.set_ylabel('precision @ τ=0.01')
    b.set_title('b  Every model under-covers the plant', loc='left')
    save(fig, 'fig_fscore')

# ---------------- Fig 4: partial clouds ----------------
def fig_pcview():
    pv = pd.read_csv(R/'pcview_16930426_probe.csv'); cov = pd.read_csv(R/'pcview_16930426_coverage.csv').groupby('condition')['mean'].mean()
    conds = ['1view', '3view', 'full']; x = [cov[c] * 100 for c in conds]
    fig, a = plt.subplots(figsize=(3.3, 2.4))
    for k, (run, (lab, col)) in enumerate(ARM.items()):
        g = lambda fit: [float(pv[(pv.run == run) & (pv.fit == fit) & (pv.condition == c) & (pv.target == 'height')].test_r2.iloc[0]) for c in conds]
        a.plot(x, g('matched'), color=col, marker='os^D'[k], label=lab)
        a.plot(x, g('transfer'), color=col, marker='os^D'[k], ls='--', lw=1.1, mfc='white')
    a.set_xlabel('share of the plant the cameras see (%)'); a.set_ylabel('height R² (test)'); a.set_ylim(-0.1, 1.02)
    a.set_xticks(x); a.set_xticklabels(['1 cam\n49', '3 cams\n78', 'full\n100'])
    a.plot([], [], color=INK2, label='refit per condition'); a.plot([], [], color=INK2, ls='--', mfc='white', marker='o', label='full-cloud probe transferred')
    a.legend(loc='lower right', fontsize=6.3)
    save(fig, 'fig_partial_clouds')

# ---------------- Fig 5: parameter-stream controls ----------------
def fig_controls():
    arms = [('e2_pcrgbd_d2', 'none\n(PC+RGB+D)'), ('e2_pcrgbdt_tg100_d2', 'target\nonly'), ('e2_pcrgbdt_tg_d2', '20 %\nvisible'),
            ('e2_pcrgbdt_d2', '58 %\n(as trained)'), ('e2_pcrgbdt_tg40_d2', '60 %\nvisible')]
    T = [('height', C['blue']), ('leaf_angle_roll', C['orange']), ('leaf_len_mean', C['aqua']), ('leaf_width_mean', C['magenta'])]
    fig, a = plt.subplots(figsize=(6.9, 2.2)); w = 0.19; xs = np.arange(len(arms))
    for j, (t, col) in enumerate(T):
        a.bar(xs + (j - 1.5) * w, [P.loc[r, t] for r, _ in arms], width=w * 0.92, color=col, label=TLAB[t], edgecolor='white', linewidth=0.8)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in arms]); a.set_ylabel('frozen-probe R² (val)'); a.set_ylim(0, 1.0)
    a.set_xlabel('recipe (θ) stream during pretraining, same 117 visible sensor tokens'); a.grid(axis='x', visible=False)
    a.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.0, 1.13))
    save(fig, 'fig_param_controls')

# ---------------- Fig 6: maize rows (Regime III) ----------------
def fig_rows():
    load = lambda p: json.load(open(sorted(glob.glob(str(R/p)))[-1]))
    ft, iso, zs = load('row_maize_maize_row_ft_*_test_17059822.json'), load('row_maize_maize_row_iso_*_test_17064411.json'), load('row_maize_maize_4m_*_test_17059823.json')
    m = lambda d, k: d['by_mode_source'][k]['chamfer_gt_m2']['mean'] * 1e4
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={'width_ratios': [1.1, 1], 'wspace': 0.42})
    rows = [('isolated-trained,\nclean input', m(iso, 'isolated/all'), C['aqua']), ('row-trained,\nrow input', m(ft, 'row/all'), C['blue']),
            ('isolated-trained,\nrow input', m(iso, 'row/all'), C['orange']), ('zero-shot,\nrow input', m(zs, 'row/all'), C['yellow'])]
    a.barh(range(len(rows))[::-1], [v for _, v, _ in rows], color=[c for *_, c in rows], height=0.62, edgecolor='white')
    for i, (lab, v, _) in enumerate(rows): a.text(v + 0.6, len(rows) - 1 - i, f'{v:.1f}', va='center', fontsize=7, color=INK)
    a.set_yticks(range(len(rows))[::-1]); a.set_yticklabels([l for l, *_ in rows], fontsize=6.8)
    a.set_xlim(0, 38); a.set_xlabel('Chamfer vs target plant (10⁻⁴ m², test)'); a.grid(axis='y', visible=False)
    a.set_title('a  Neighbours cost little once trained for', loc='left')
    th = sorted(ft['by_theta']['row/all'], key=int); xs = [int(t) for t in th]
    for d, lab, col, mk in ((ft, 'row-trained', C['blue'], 'o'), (iso, 'isolated-trained', C['orange'], 's'), (zs, 'zero-shot', C['yellow'], 'D')):
        b.plot(xs, [d['by_theta']['row/all'][t]['chamfer_gt_m2']['mean'] * 1e4 for t in th], color=col, marker=mk, label=lab)
    b.set_xlabel('row yaw θ (°; 0 broadside, 90 end-on)'); b.set_ylabel('Chamfer (10⁻⁴ m²)'); b.set_xticks([0, 30, 60, 90])
    b.legend(loc='upper left'); b.set_title('b  By row orientation', loc='left')
    save(fig, 'fig_rows_maize')

if __name__ == '__main__':
    fig_e2(); fig_scaling(); fig_fscore(); fig_pcview(); fig_controls(); fig_rows()
