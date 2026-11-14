"""E10 on the 3000-row upload: final visual summary for the E10 review page (user, 2026-11-13: "Can you give me some
visualization of the E10?").

  * test-set summary charts from the finished reports: Chamfer per model (both species), maize per-plant height
    (median over the plant's stops, zero-shot headline vs real-row fine-tune) and maize per-leaf parameter error;
  * test examples at the 25th / 50th / 75th percentile of the fine-tuned model's per-plant Chamfer, camera stop 9:
    image | true centre plant | synthetic-row fine-tune | real-row fine-tune | the procedural model the real-row
    fine-tune writes | (maize) per-leaf length, truth vs both models (the rows3000 leaf CSV; leaves matched by index
    from the base, as eval/eval_real_rows.py scores them).
Runs: maize_real_ft_r3000 (15 epochs, test report 17436544), sorghum_real_ft_r3000_zoom_e40 (40 epochs, zoomed window,
report 17444048); zero-shot reports 17436546 (maize) / 17444049 (sorghum zoom).
Writes figures/pages/e10_r3000/final_*.png and e10_r3000_final_section.html.
CPU: sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=48G --time=01:00:00 \
       --output=logs/%x_%j.out --wrap "python figures/make_e10_r3000_final.py"
"""
import sys, pathlib, json; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'eval'), str(R / 'figures')]
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import make_e10_r3000_preview as P                                 # cloud(), constants, Liberation Sans
import _fig12_kit as K
import _plantviz as V
import ugg_style as U
from eval_row import build_model, predict
from real_rows import RealRowDataset

torch.set_num_threads(8)
OUT = P.OUT
FT = {'maize': ('maize_real_ft_r3000', 17436544, '15 epochs'), 'sorghum': ('sorghum_real_ft_r3000_zoom_e40', 17444048, '40 epochs')}
ZS = {'maize': 17436546, 'sorghum': 17444049}
BEFORE = {'maize': 'maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth', 'sorghum': 'sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth'}
LABEL = {'maize_4m_pretrain_15k_d2': 'Headline, zero-shot', '4m_pretrain_15k_d2_sink': 'Headline, zero-shot',
         'maize_row_iso_e1': 'Isolated control', 'sorghum_row_iso': 'Isolated control',
         'maize_row_ft_e1': 'Synthetic-row fine-tune', 'sorghum_row_ft': 'Synthetic-row fine-tune'}
GREY, PINK, VIOLET, INK = '#9a9a9a', U.STREAM_INK['pc'], U.OURS, '#1a1a1a'
FS = 9


def summary(sp):
    """test Chamfer (mean, 95 % CI) of every model on the species' test stops, all three inputs"""
    zs = json.load(open(R / f'reports/e10_real_rows_{ZS[sp]}.json')); ft = json.load(open(R / f'reports/e10_real_ft*_{FT[sp][1]}.json'.replace('*', '_r3000_test' if sp == 'maize' else '_sorghum_r3000_zoom_e40_test')))
    seen, rows = set(), []
    for g in zs['groups'] + ft['groups']:
        if g['source'] != 'all' or g['model'] in seen: continue
        seen.add(g['model']); c = g['chamfer_m2']
        rows.append((LABEL.get(g['model'], f'Real-row fine-tune ({FT[sp][2]})'), c['mean'] * 1e4, c['lo'] * 1e4, c['hi'] * 1e4, g))
    order = ['Headline, zero-shot', 'Isolated control', 'Synthetic-row fine-tune']
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 9)
    return rows


def charts():
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.3), gridspec_kw=dict(width_ratios=[1.05, 1.25, 1.0, 1.0], wspace=0.42))
    for ax, sp in zip(axs[:2], ('maize', 'sorghum')):
        rows = summary(sp); y = np.arange(len(rows))[::-1]
        cols = [VIOLET if r[0].startswith('Real') else GREY for r in rows]
        ax.barh(y, [r[1] for r in rows], color=cols, height=0.62)
        ax.errorbar([r[1] for r in rows], y, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]], fmt='none', ecolor=INK, lw=0.8, capsize=2)
        for yy, r in zip(y, rows): ax.text(r[3] * 1.15, yy, f'{r[1]:.1f}' if r[1] < 100 else f'{r[1]:.0f}', va='center', fontsize=FS, color=VIOLET if r[0].startswith('Real') else INK)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=FS); ax.set_xscale('log')
        ax.set_xlabel('test Chamfer (10⁻⁴ m², log)', fontsize=FS); ax.set_title(f"{sp.capitalize()} ({rows[0][4]['n_plants']} test plants{', zoomed window' if sp == 'sorghum' else ''})", fontsize=FS + 1, loc='left')
        ax.set_xlim(right=max(r[3] for r in rows) * 3)
    # maize height per plant
    ax = axs[2]
    z = pd.read_csv(R / f'reports/e10_real_rows_{ZS["maize"]}.csv', dtype={'plant': str}); f = pd.read_csv(R / f'reports/e10_real_ft_r3000_test_{FT["maize"][1]}.csv', dtype={'plant': str})
    for d, mdl, c, lab in ((z, 'maize_4m_pretrain_15k_d2', GREY, 'Headline, zero-shot'), (f, 'maize_real_ft_r3000', VIOLET, 'Real-row fine-tune')):
        g = d[(d.model == mdl) & (d.source == 'all')].groupby('plant')[['height', 'height_pred']].median()
        ax.scatter(g.height, g.height_pred, s=14, facecolors='none', edgecolors=c, lw=1.0, label=f'{lab} ({(abs(g.height_pred - g.height) / g.height).median() * 100:.0f} %)')
    lim = [0.5, 2.6]; ax.plot(lim, lim, color=INK, lw=0.7, ls=(0, (3, 2))); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('true height (m)', fontsize=FS); ax.set_ylabel('predicted height (m)', fontsize=FS)
    ax.set_title('Maize height per test plant', fontsize=FS + 1, loc='left'); ax.legend(fontsize=FS - 1, frameon=False, loc='upper left', title='median rel. error', title_fontsize=FS - 1)
    # maize per-leaf error
    ax = axs[3]
    zs = {g['model']: g for g in json.load(open(R / f'reports/e10_real_rows_{ZS["maize"]}.json'))['groups'] if g['source'] == 'all'}
    ft = {g['model']: g for g in json.load(open(R / f'reports/e10_real_ft_r3000_test_{FT["maize"][1]}.json'))['groups'] if g['source'] == 'all'}
    keys = [('internode_mae', 'internode'), ('leaf_len_mae', 'leaf length'), ('leaf_width_mae', 'leaf width')]
    M = [(zs['maize_4m_pretrain_15k_d2'], GREY, 'Headline, zero-shot'), (ft['maize_row_ft_e1'], PINK, 'Synthetic-row fine-tune'), (ft['maize_real_ft_r3000'], VIOLET, 'Real-row fine-tune')]
    x = np.arange(len(keys)); w = 0.26
    for j, (g, c, lab) in enumerate(M):
        v = [g[k] * 100 for k, _ in keys]; ax.bar(x + (j - 1) * w, v, w, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys], fontsize=FS); ax.set_ylabel('MAE per leaf (cm)', fontsize=FS)
    ax.set_title('Maize per-leaf parameters', fontsize=FS + 1, loc='left'); ax.legend(fontsize=FS - 1, frameon=False, loc='upper right')
    ax.set_ylim(0, max(g[k] for g, _, _ in M for k, _ in keys) * 100 * 1.6)
    for a in axs:
        a.tick_params(labelsize=FS - 0.5)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    fig.savefig(OUT / 'final_charts.png', dpi=200, bbox_inches='tight'); plt.close(fig)


def glyph_ax(ax, sp, rec, ref_h=None):
    K.glyph(ax, sp, rec, U.STREAM_INK['text'], stem='#4a3b2a', lw_scale=0.55)
    bb = ax.dataLim; w = max(abs(bb.x0), abs(bb.x1)); y0 = min(bb.y0, 0.0); st = max(2 * w, bb.y1 - y0) * 1.08 * 1.15
    ax.set_xlim(-st / 2, st / 2); ax.set_ylim(y0 - 0.13 * st, y0 + 0.87 * st); ax.set_aspect('equal'); ax.set_axis_off()


def example(sp, pl, row, stop, i, M, best_ep):
    ds = RealRowDataset(species=sp, source='rows3000', plants=[pl], rows=[row], stops={stop}, seed=0, rgb_gain=M[1][2],
                        **({'sorghum_window': 'zoom'} if sp == 'sorghum' else {}))
    rgb, dep, pin, par, valid, pgt, info = ds[0]
    img = (rgb.permute(1, 2, 0).numpy() * P.STD + P.MEAN).clip(0, 1)
    preds, ch, recs, leafs = [], [], [], []
    for j, (m, mr, _) in enumerate(M):
        with torch.no_grad():
            pp, ppar = predict(m, 'all', rgb[None], dep[None], pin[None], par[None], mr, 1000003 * i)
        Pc = pp[0].float(); d2 = torch.cdist(Pc[None], pgt[None])[0]
        ch.append(1e4 * float(d2.min(1).values.pow(2).mean() + d2.min(0).values.pow(2).mean()) * info['scale'] ** 2)
        preds.append(Pc.numpy()); pr = ppar[0].clamp(0, 1).float().numpy()
        rec = V.decode_recipe(sp, pr, None, predicted=True); recs.append(rec)
        if sp == 'maize': leafs.append([l['leafLength'] for l in rec['leaves']])
    ncol = 6 if sp == 'maize' else 5
    W, t, g, m_ = 2.0 * ncol + 0.1 * (ncol - 1), 2.0, 0.1, 0.04; H = t + 0.5
    fig = plt.figure(figsize=(W, H))
    heads = ['Image', 'Ground truth', 'Synthetic-row fine-tune', f'Real-row fine-tune (ep {best_ep})', 'Its procedural model'] + (['Leaf length per leaf'] if sp == 'maize' else [])
    hc = [INK, INK, PINK, VIOLET, U.STREAM_INK['text'], INK]
    kinds = [('rgb', True), (None, False), ('pc', False), ('pc', False), ('text', False), (None, False)]
    for k in range(ncol):
        x = k * (t + g)
        U.pastel_box(fig, x, 0.04, t, t, kinds[k][0], dashed=kinds[k][1], r=0.05, transform=fig.dpi_scale_trans, z=-1)
        fig.text((x + t / 2) / W, (H - 0.05) / H, heads[k], fontsize=FS + 1, ha='center', va='top', color=hc[k])
        if k == 5:
            ax = fig.add_axes([(x + 0.42) / W, (0.04 + 0.36) / H, (t - 0.55) / W, (t - 0.5) / H])
        else:
            ax = fig.add_axes([(x + m_) / W, (0.04 + m_) / H, (t - 2 * m_) / W, (t - 2 * m_) / H]); ax.set_axis_off(); ax.patch.set_alpha(0)
        if k == 0: ax.imshow(img)
        elif k == 1:
            P.cloud(ax, pgt.numpy()); ax.text(0.04, 0.03, f"true height {info['height']:.2f} m", transform=ax.transAxes, fontsize=FS, path_effects=P.HALO)
        elif k in (2, 3):
            P.cloud(ax, preds[k - 2]); ax.text(0.04, 0.03, f'CD = {ch[k - 2]:.0f}' if ch[k - 2] >= 10 else f'CD = {ch[k - 2]:.1f}', transform=ax.transAxes, fontsize=FS, color=hc[k], path_effects=P.HALO)
        elif k == 4:
            glyph_ax(ax, sp, recs[1], max(recs[1]['height'], 0.3))
            ax.text(0.04, 0.03, f"height {recs[1]['height']:.2f} m · {recs[1]['n']} leaves", transform=ax.transAxes, fontsize=FS, color=hc[k], path_effects=P.HALO)
        else:
            tr = info['leaf_raw'][:, 1]; n = len(tr)
            ax.plot(np.arange(1, n + 1), tr, color=INK, lw=1.6, marker='o', ms=3, label='true')
            for lv, c, lab in ((leafs[0], PINK, 'synthetic-row'), (leafs[1], VIOLET, 'real-row')):
                ax.plot(np.arange(1, len(lv) + 1), lv, color=c, lw=1.2, marker='o', ms=2.5, mfc='white', label=lab)
            ax.set_xlabel('leaf (base → top)', fontsize=FS - 1); ax.set_ylabel('length (m)', fontsize=FS - 1); ax.tick_params(labelsize=FS - 1.5)
            ax.set_ylim(min(min(tr), *[min(l) for l in leafs if l]) * 0.85, max(max(tr), *[max(l) for l in leafs if l]) * 1.25)
            ax.legend(fontsize=FS - 2, frameon=False, loc='upper right', ncol=1, handlelength=1.2, borderaxespad=0.1); ax.patch.set_alpha(0)
            for s in ('top', 'right'): ax.spines[s].set_visible(False)
    name = f'final_{sp}_{i}.png'; fig.savefig(OUT / name, dpi=170); plt.close(fig)
    return dict(file=name, plant=info['plant'], row=info['row'], stop=info['stop'], height=info['height'], occ=info['occ_frac'], ch=ch,
                hp=recs[1]['height'], n_true=info['leaves'], n_pred=recs[1]['n'])


def examples(sp):
    run, job, _ = FT[sp]
    f = pd.read_csv(R / f"reports/{'e10_real_ft_r3000_test' if sp == 'maize' else 'e10_real_ft_sorghum_r3000_zoom_e40_test'}_{job}.csv", dtype={'plant': str, 'row': str})
    f = f[(f.model == run) & (f.source == 'all')]
    g = f.groupby('plant').chamfer_m2.mean().sort_values()
    picks = []
    for q in (0.25, 0.5, 0.75):                      # the plant at that percentile, at its median-error stop
        pl = g.index[int(round(q * (len(g) - 1)))]; s_ = f[f.plant == pl].sort_values('chamfer_m2'); r_ = s_.iloc[len(s_) // 2]
        picks.append((pl, r_['row'], int(r_['stop'])))
    h = json.load(open(R / 'outputs' / run / 'training_history.json')); best_ep = min(h['val'][1:], key=lambda v: v['val_chamfer_m2'])['epoch']
    M = []
    for spec in (BEFORE[sp], f'{run}:best_model.pth'):
        r_, ck = spec.split(':', 1); m, cfg, _ = build_model(R / 'outputs' / r_, ck, torch.device('cpu'), sp)
        M.append((m, float(cfg.get('mask_ratio', 0.8)), float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None))
    out = []
    for i, (pl, row, stop) in enumerate(picks):
        e = example(sp, pl, row, stop, i, M, best_ep); e['pct'] = (25, 50, 75)[i]; out.append(e)
        print(sp, pl, [round(c, 1) for c in e['ch']], flush=True)
    return out


if __name__ == '__main__':
    charts()
    ex = {sp: examples(sp) for sp in ('maize', 'sorghum')}
    json.dump(ex, open(OUT / 'final_examples.json', 'w'), indent=1, default=float)
    print('wrote', OUT / 'final_charts.png', flush=True)
