"""E10 on the 3000-row upload, fine-tuning IN PROGRESS: a section for the E10 review page (figures/pages/e10_page.html).
  * status of the six runs and their validation Chamfer curves (outputs/<run>/training_history.json);
  * held-out TEST stops (plants from the run's split.csv 'test') scored by the row fine-tune it starts from ('before')
    and by the run's current best checkpoint ('after', the epoch is stated). Maize: maize_real_ft_r3000; sorghum: the
    zoomed run with crop augmentation (sorghum_real_ft_r3000_zoom_e40), on the zoomed window.
Writes figures/pages/e10_r3000/{maize,sorghum}_<k>.png and e10_r3000_section.html (spliced into the page by hand).
CPU: sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=48G --time=00:45:00 \
       --output=logs/%x_%j.out --wrap "python figures/make_e10_r3000_preview.py"
"""
import sys, pathlib, json; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'eval'), str(R / 'figures')]
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from eval_row import build_model, predict
from real_rows import RealRowDataset
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2
import _scene                                                     # Liberation Sans
import ugg_style as U

torch.set_num_threads(8)
OUT = R / 'figures/pages/e10_r3000'; OUT.mkdir(parents=True, exist_ok=True)
RUNS = [('maize', 'maize_real_ft_r3000', '15 epochs'), ('maize', 'maize_real_ft_r3000_e40', '40 epochs'),
        ('sorghum', 'sorghum_real_ft_r3000', 'full strip, 15 epochs'), ('sorghum', 'sorghum_real_ft_r3000_e40', 'full strip, 40 epochs'),
        ('sorghum', 'sorghum_real_ft_r3000_zoom', 'zoom + crops, 15 epochs'), ('sorghum', 'sorghum_real_ft_r3000_zoom_e40', 'zoom + crops, 40 epochs')]
TOTAL = {'maize_real_ft_r3000': 15, 'maize_real_ft_r3000_e40': 40, 'sorghum_real_ft_r3000': 15, 'sorghum_real_ft_r3000_e40': 40,
         'sorghum_real_ft_r3000_zoom': 15, 'sorghum_real_ft_r3000_zoom_e40': 40}
SHOW = {'maize': ('maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth', 'maize_real_ft_r3000'),
        'sorghum': ('sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth', 'sorghum_real_ft_r3000_zoom_e40')}
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
H_IDX = PLANT_FIELDS_V2.index('height')
HALO = [pe.withStroke(linewidth=2.0, foreground='white')]


def hist(run):
    h = json.load(open(R / 'outputs' / run / 'training_history.json'))
    return [(v['epoch'], v['val_chamfer_m2'] * 1e4) for v in h['val']]


def cloud(ax, P):
    P = np.asarray(P, np.float64); h = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9)
    near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9); o = np.argsort(P[:, 2])
    c = GREENS(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]
    ax.scatter(P[o, 0], P[o, 1], s=0.35, c=c[o], linewidths=0, rasterized=True); ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)


def tile_row(r, path, after_label):
    W, t, g, m = 6.875, 1.55, 0.06, 0.03; H = t + 0.42
    fig = plt.figure(figsize=(W, H)); x0 = (W - (4 * t + 3 * g)) / 2
    heads = ['Image', 'Ground truth', 'Row fine-tune', after_label]; colors = ['#111111', '#111111', U.STREAM_INK['pc'], U.OURS]
    kinds = [('rgb', True), (None, False), ('pc', False), ('pc', False)]
    for k in range(4):
        x = x0 + k * (t + g)
        U.pastel_box(fig, x, 0.04, t, t, kinds[k][0], dashed=kinds[k][1], r=0.05, transform=fig.dpi_scale_trans, z=-1)
        fig.text((x + t / 2) / W, (H - 0.04) / H, heads[k], fontsize=8.5, ha='center', va='top', color=colors[k])
        ax = fig.add_axes([(x + m) / W, (0.04 + m) / H, (t - 2 * m) / W, (t - 2 * m) / H]); ax.set_axis_off(); ax.patch.set_alpha(0)
        if k == 0: ax.imshow(r['img'])
        elif k == 1:
            cloud(ax, r['gt'])
            if r['species'] == 'maize': ax.text(0.04, 0.03, f"height {r['height']:.2f} m", transform=ax.transAxes, fontsize=7.5, path_effects=HALO)
        else:
            j = k - 2; cloud(ax, r['preds'][j])
            lab = f"CD = {r['ch'][j]:.0f}" + (f"\nheight {r['hp'][j]:.2f} m" if r['hp'][j] is not None else '')
            ax.text(0.04, 0.03, lab, transform=ax.transAxes, fontsize=7.5, color=colors[k], path_effects=HALO, linespacing=1.15)
    fig.savefig(path, dpi=200); plt.close(fig)


def examples(sp, n=3):
    before, run = SHOW[sp]
    split = pd.read_csv(R / 'outputs' / run / 'split.csv', dtype=str); test = sorted(split[split.split == 'test'].plant)
    picks = list(np.random.default_rng(5).choice(test, n, replace=False))
    hv = hist(run); best_ep = min(hv[1:], key=lambda e: e[1])[0]
    M = []
    for spec in (before, f'{run}:best_model.pth'):
        r_, ck = spec.split(':', 1); m, cfg, _ = build_model(R / 'outputs' / r_, ck, torch.device('cpu'), sp)
        M.append((m, float(cfg.get('mask_ratio', 0.8)), float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None))
    out = []
    for i, pl in enumerate(picks):
        ds = RealRowDataset(species=sp, source='rows3000', plants=[pl], stops={9}, seed=0, rgb_gain=M[1][2])
        rgb, dep, pin, par, valid, pgt, info = ds[0]
        r = dict(species=sp, plant=info['plant'], row=info['row'], stop=info['stop'], height=info['height'], occ=info['occ_frac'],
                 img=(rgb.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1), gt=pgt.numpy(), preds=[], ch=[], hp=[])
        for j, (m, mr, _) in enumerate(M):
            with torch.no_grad():
                pp, ppar = predict(m, 'all', rgb[None], dep[None], pin[None], par[None], mr, 1000003 * i)
            P = pp[0].float(); d2 = torch.cdist(P[None], pgt[None])[0]
            r['ch'].append(1e4 * float(d2.min(1).values.pow(2).mean() + d2.min(0).values.pow(2).mean()) * info['scale'] ** 2)
            r['preds'].append(P.numpy())
            r['hp'].append(float(ppar[0, 0, H_IDX].clamp(0, 1)) * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX] if sp == 'maize' else None)
        name = f'{sp}_{i}.png'; tile_row(r, OUT / name, f'Real-row fine-tune (epoch {best_ep})')
        out.append(dict(file=name, plant=r['plant'], row=r['row'], stop=r['stop'], height=r['height'], occ=r['occ'], ch=r['ch'], hp=r['hp']))
        print(sp, pl, [round(c, 1) for c in r['ch']], flush=True)
    return out, best_ep


def curves_svg(sp):
    """validation Chamfer per epoch for the species' runs; log y; colours from ugg_style; text in currentColor"""
    runs = [(r, lab) for s, r, lab in RUNS if s == sp]; Wd, Hd, L, Rr, T, B = 560, 230, 52, 150, 12, 34
    data = {r: hist(r) for r, _ in runs}
    ys = [y for d in data.values() for _, y in d]; lo, hi = np.log10(min(ys)) - 0.05, np.log10(max(ys)) + 0.05
    xmax = max(TOTAL[r] for r, _ in runs)
    X = lambda e: L + e / xmax * (Wd - L - Rr); Y = lambda v: T + (hi - np.log10(v)) / (hi - lo) * (Hd - T - B)
    cols = {'maize_real_ft_r3000': '#3f5f8f', 'maize_real_ft_r3000_e40': U.OURS, 'sorghum_real_ft_r3000': '#b3b3b3',
            'sorghum_real_ft_r3000_e40': '#8c8c8c', 'sorghum_real_ft_r3000_zoom': '#3f5f8f', 'sorghum_real_ft_r3000_zoom_e40': U.OURS}
    s = [f'<svg viewBox="0 0 {Wd} {Hd}" role="img" aria-label="Validation Chamfer per epoch, {sp}" style="width:100%;height:auto;max-width:{Wd}px">']
    for v in [10 ** k for k in range(0, 5)] + [2 * 10 ** k for k in range(0, 4)] + [5 * 10 ** k for k in range(0, 4)]:
        if lo <= np.log10(v) <= hi:
            s.append(f'<line x1="{L}" x2="{Wd - Rr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="currentColor" stroke-opacity=".12"/>'
                     f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="currentColor" fill-opacity=".7">{v:g}</text>')
    for e in range(0, xmax + 1, 5 if xmax <= 15 else 10):
        s.append(f'<text x="{X(e):.1f}" y="{Hd - 14}" font-size="11" text-anchor="middle" fill="currentColor" fill-opacity=".7">{e}</text>')
    s.append(f'<text x="{(L + Wd - Rr) / 2}" y="{Hd - 1}" font-size="11.5" text-anchor="middle" fill="currentColor">epoch (validation Chamfer, 10⁻⁴ m², log scale)</text>')
    for k, (r, lab) in enumerate(runs):
        d = data[r]; c = cols[r]
        s.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{" ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in d)}"/>')
        s += [f'<circle cx="{X(e):.1f}" cy="{Y(v):.1f}" r="2.6" fill="white" stroke="{c}" stroke-width="1.4"/>' for e, v in d]
        e, v = d[-1]; s.append(f'<text x="{Wd - Rr + 8}" y="{T + 14 + 18 * k}" font-size="11.5" fill="{c}">■ {lab}: {v:.1f}</text>')
    s.append('</svg>'); return ''.join(s)


if __name__ == '__main__':
    ex = {sp: examples(sp) for sp in ('maize', 'sorghum')}
    zs = {sp: json.load(open(R / f'reports/e10_real_rows_{j}.json')) for sp, j in (('maize', 17436546), ('sorghum', 17444049))}
    zrow = lambda sp: {e['model']: e['chamfer_m2']['mean'] * 1e4 for e in zs[sp]['groups'] if e['layout'] == 'both' and e['source'] == 'all'}
    rows_html = []
    for sp, r, lab in RUNS:
        hv = hist(r); best = min(hv[1:], key=lambda e: e[1])
        rows_html.append(f'<tr><td>{sp.capitalize()}</td><td>{lab}</td><td class="num">{len(hv) - 1} / {TOTAL[r]}</td>'
                         f'<td class="num">{hv[0][1]:,.0f}</td><td class="num after">{best[1]:,.1f}</td><td class="num">{best[0]}</td></tr>')
    zm, zsg = zrow('maize'), zrow('sorghum')
    tiles = lambda sp: '\n'.join(f'<figure class="tile"><img loading="lazy" src="e10_r3000/{e["file"]}" alt="{sp} test plant {e["plant"]}, row {e["row"]}, stop {e["stop"]}: image, ground truth, before and after"><figcaption><b>Test plant {e["plant"]}, row {int(e["row"])}, stop {e["stop"]}</b>'
                                 f'<span>{"true height %.2f m · " % e["height"] if sp == "maize" else ""}{e["occ"] * 100:.0f}% hidden by neighbors</span>'
                                 f'<span>CD {e["ch"][0]:,.0f} → <em>{e["ch"][1]:,.0f}</em>' + (f' · height {e["hp"][0]:.2f} → <em>{e["hp"][1]:.2f} m</em>' if e["hp"][0] is not None else '') + '</span></figcaption></figure>' for e in ex[sp][0])
    html = f'''
  <section id="rows3000">
    <h2>New data · 3,000-row upload, fine-tuning in progress</h2>
    <p class="lede">The 2026-11-13 upload: 3,000 rows per species built from the 520 scanned maize plants and 52 sorghum scans (with four tilted copies each), 18 camera stops per row; split by plant, 75 % for fine-tuning. Maize is also trained on the per-leaf parameters. Sorghum is shown zoomed (a square window of maize-like proportions centred on the middle plant's leaves) and trained with random zoomed crops; its Chamfer is on that window, so it is not comparable with the full-strip runs or with the first upload's numbers above. All numbers here are interim: validation every epoch, and test examples from the current best checkpoint.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>Species</th><th>Run</th><th style="text-align:right;padding-right:22px">Epochs done</th><th style="text-align:right;padding-right:22px">Val CD at start</th><th style="text-align:right;padding-right:22px">Best val CD</th><th style="text-align:right;padding-right:22px">at epoch</th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody></table></div>
    <p class="lede">Zero-shot on the held-out test plants (all streams, CD 10⁻⁴ m²): maize headline model {zm.get("maize_4m_pretrain_15k_d2", float("nan")):.0f}, isolated-plant control {zm.get("maize_row_iso_e1", float("nan")):.0f}; sorghum on the zoomed window: headline {zsg.get("4m_pretrain_15k_d2_sink", float("nan")):.0f}, isolated-plant control {zsg.get("sorghum_row_iso", float("nan")):.0f}, row fine-tune {zsg.get("sorghum_row_ft", float("nan")):.0f}.</p>
    <div class="curves"><div><h3>Maize · validation Chamfer</h3>{curves_svg("maize")}</div><div><h3>Sorghum · validation Chamfer</h3>{curves_svg("sorghum")}</div></div>
    <h3>Maize · held-out test plants (real-row fine-tune at epoch {ex["maize"][1]})</h3>
    <div class="gallery maize">{tiles("maize")}</div>
    <h3>Sorghum · held-out test scans, zoomed (zoom + crops run at epoch {ex["sorghum"][1]})</h3>
    <div class="gallery maize">{tiles("sorghum")}</div>
  </section>
'''
    (R / 'figures/pages/e10_r3000_section.html').write_text(html); print('wrote section')
