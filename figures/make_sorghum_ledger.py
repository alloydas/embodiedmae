"""Build the sorghum (final data) results page: one HTML file with the numbers embedded as JSON
and rendered client-side (tables, inline-SVG charts, t-SNE and reconstruction images), in the
style of the maize d2 page.

  python figures/make_sorghum_ledger.py --out /path/to/cropcode_ledger.html

Sources: outputs/<run>/{training_history,config}.json, reports/probe_pr_*_d2_*.csv,
reports/*_delta_d2_16950816.csv, reports/e9_pr_*_d2_*/e9_latent_summary_{val,test}.csv,
reports/pcfscore_17071405_items.csv, reports/pcview_16930426_*.csv, reports/quant_*_d2_*.json,
reports/row_*.json, the E9 t-SNE figures and the E1 visualisations. Queue rows are passed in
--queue (a JSON file) or left as written below.
"""
import argparse, base64, glob, io, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'reports'; O = REPO / 'outputs'

RUNS = {  # slug -> short label
 'e2_pc_d2':'PC', 'e2_pcrgb_d2':'PC+RGB', 'e2_pcrgbd_d2':'PC+RGB+D', 'e2_pcrgbdt_d2':'all four',
 'e2_pcrgbdt_tg_d2':'tg 20%', 'e2_pcrgbdt_tg40_d2':'tg 60%', 'e2_pcrgbdt_tg100_d2':'tg target-only',
 'e3_1k_d2':'1k', 'e3_3k_d2':'3k', 'e3_10k_d2':'10k', 'e4_small_d2':'small', 'e4_large_d2':'large',
 'e5_topdown_d2':'E5 top-down', 'e5_horizon_d2':'E5 horizon', 'e5_single_d2':'E5 single',
 '4m_pretrain_15k_d2_sink':'E1', 'e8_supervised_d2':'supervised',
 'e2_pcd_d2':'PC+D', 'e2_pct_d2':'PC+recipe', 'e2_pcrgbt_d2':'PC+RGB+recipe', 'e2_pcdt_d2':'PC+D+recipe',
 'sorghum_xfer_warm_d2':'sorghum, warm from maize', 'sorghum_xfer_scratch_d2':'sorghum, scratch',
 'maize_xfer_warm':'maize, warm from sorghum E1', 'maize_xfer_scratch':'maize, scratch',
}
CKPT = {'e3_1k_d2':6168, 'e3_3k_d2':2024, 'e3_10k_d2':624, '4m_pretrain_15k_d2_sink':1000}
TARGETS = ['height','leaf_count','biomass','leaf_angle','leaf_angle_roll','leaf_len_mean','leaf_len_max','leaf_width_mean','leaf_width_max']

def val_epochs(n_val, epochs, val_freq):
    ep = [1] + [k*val_freq for k in range(1, epochs//val_freq + 1)]
    if ep[-1] != epochs: ep.append(epochs)
    ep = [e for e in ep if e <= epochs]
    if len(ep) != n_val:          # resumes can duplicate the last point: pad/trim to the recorded length
        ep = (ep + [epochs]*n_val)[:n_val]
    return ep

def curve(run):
    h = json.load(open(O/run/'training_history.json')); c = json.load(open(O/run/'config.json'))
    tl = h['train_loss']; n = len(tl); epochs = int(c['epochs']); vf = int(c.get('val_freq', 25))
    step = max(1, n // 120)
    train = [[i+1, round(tl[i], 5)] for i in range(0, n, step)] + ([[n, round(tl[-1], 5)]] if (n-1) % step else [])
    ve = val_epochs(len(h['val_loss']), epochs, vf)
    pts = lambda k: [[e, round(v, 6)] for e, v in zip(ve, h[k])] if h.get(k) else []
    fin = {'val_loss': h['val_loss'][-1] if h['val_loss'] else None, 'val_pc_chamfer': h['val_pc_chamfer'][-1] if h['val_pc_chamfer'] else None,
           'val_rgb_mse': h['val_rgb_mse'][-1] if h.get('val_rgb_mse') else None, 'val_depth_mse': h['val_depth_mse'][-1] if h.get('val_depth_mse') else None,
           'val_param_mae': h['val_param_mae_masked'][-1] if h.get('val_param_mae_masked') else None, 'train_loss': tl[-1],
           'test_pc_chamfer': h['test_pc_chamfer'][-1] if h.get('test_pc_chamfer') else None, 'test_param_mae': h['test_param_mae_masked'][-1] if h.get('test_param_mae_masked') else None}
    cfg = {k: c.get(k) for k in ('epochs','batch_size','world_size','max_plants','active_modalities','model_size','mask_ratio','lr','rgb_file','rgb_brighten_gain','pc_loss_name','text_mask_ratio','accum_steps','pc_sinkhorn_weight','train_views')}
    return {'epochs': epochs, 'done': n, 'val_freq': vf, 'train_loss': train, 'val_loss': pts('val_loss'), 'val_chamfer': pts('val_pc_chamfer'), 'val_param_mae': pts('val_param_mae_masked'), 'final': fin, 'cfg': cfg}

def probe_rows(files, feature='cls'):
    d = pd.concat([pd.read_csv(f) for f in files]); d = d[d.feature == feature] if (feature and 'feature' in d.columns) else d
    out = {}
    for run, g in d.groupby('run'):
        g = g.drop_duplicates('target')
        out[run] = {'epoch': int(g.epoch.iloc[0]), 'targets': {r.target: {'unit': r.unit, 'src': r.source_col, 'val_r2': r.val_r2, 'test_r2': r.test_r2, 'val_mae': r.val_mae, 'test_mae': r.test_mae, 'val_base_mae': r.val_base_mae} for r in g.itertuples() if r.target in TARGETS}}
    return out

def img_b64(path, width=900, quality=80):
    im = Image.open(path).convert('RGB')
    if im.width > width: im = im.resize((width, int(im.height*width/im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=quality, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--no-images', action='store_true'); a = ap.parse_args()
    D = {}
    D['curves'] = {r: curve(r) for r in RUNS if (O/r/'training_history.json').exists() and r != 'e8_supervised_d2'}
    # supervised row (different history layout)
    hs = json.load(open(O/'e8_supervised_d2'/'training_history.json'))
    D['e8sup'] = {'done': len(hs['epoch']), 'val_epochs': hs['val_epoch'], 'val_loss': hs['val_loss'], 'r2': {t: hs[f'val_r2_{t}'][-1] for t in TARGETS if f'val_r2_{t}' in hs}}
    # generalist distillation (in progress)
    hd = json.load(open(O/'4m_distill_15k_all_d2'/'training_history.json'))
    D['distill'] = {'train': [[e['epoch'], round(e['loss'], 5)] for e in hd['train']], 'val': [{'epoch': v['epoch'], 'mean_gen': v['mean_gen'], 'per_source': {s: {k: v['per_source'][s][k] for k in ('pc_chamfer','param_mae_masked','total')} for s in v['per_source']}} for v in hd['val']], 'epochs': 100}
    D['students'] = {}
    for r in ('rgb2pc_full_d2', 'rgb2pc_smr50_d2', 'rgb2pc_smr50b_d2', 'rgbd2pc_d2', 'rgbt2pc_d2', 'dt2pc_d2', 'pairs_d2'):
        f = O/f'4m_distill_15k_{r}'/'training_history.json'
        if f.exists():
            h = json.load(open(f)); D['students'][r] = {'done': len(h['train']), 'train': [[e['epoch'], round(e['loss'], 5)] for e in h['train']], 'val': [[v['epoch'], v['mean_gen']] for v in h['val']], 'val_pc_chamfer_rgb': [[v['epoch'], v['per_source']['rgb']['pc_chamfer']] for v in h['val'] if 'rgb' in v['per_source']], 'val_pc_chamfer': {src: [[v['epoch'], v['per_source'][src]['pc_chamfer']] for v in h['val'] if src in v['per_source']] for src in (h['val'][-1]['per_source'] if h['val'] else {})}}
    D['probe'] = probe_rows([R/'probe_pr_600_d2_16930422.csv', R/'probe_pr_6168_d2_16928969.csv', R/'probe_pr_2024_d2_16928970.csv', R/'probe_pr_624_d2_16928971.csv']
                            + sorted(R.glob('probe_pr_e1_d2_*.csv')) + sorted(R.glob('probe_pr_e2more*_d2_*.csv')) + sorted(R.glob('probe_pr_e5_d2_*.csv')))
    # T1 transfer arms: probe at epoch 150 (sorghum on the sorghum targets, maize on its own)
    D['probe_t1'] = {}
    for sp, pat in (('sorghum', 'probe_pr_xfer_sorghum_*.csv'), ('maize', 'probe_pr_xfer_maize_*.csv')):
        fs1 = sorted(R.glob(pat))
        if fs1:
            d1 = pd.read_csv(fs1[-1]); d1 = d1[d1.feature == 'cls'] if 'feature' in d1.columns else d1
            D['probe_t1'][sp] = {'job': fs1[-1].stem.rsplit('_', 1)[-1], 'rows': [{k: r[k] for k in ('run', 'target', 'val_r2', 'test_r2')} for r in d1.to_dict('records')]}
    D['probe_mean'] = probe_rows([R/'probe_pr_mean600_d2_16930423.csv'], 'mean')
    D['e8'] = probe_rows([R/'probe_e8_base_sorghum_d2_16930425.csv'], 'mean')
    D['e8']['e8_supervised_d2 (mean pool)'] = probe_rows([R/'probe_e8_score_d2_16930424_mean.csv'], 'mean')['e8_supervised_d2']
    D['e8']['e8_supervised_d2 (own head)'] = probe_rows([R/'probe_e8_score_d2_16930424_head.csv'], None)['e8_supervised_d2_head']
    bootfiles = sorted(glob.glob(str(R/'*_delta_d2_16950816.csv'))) + sorted(glob.glob(str(R/'e5_delta_d2_*.csv'))) + sorted(glob.glob(str(R/'e2more_*_d2.csv'))) + sorted(glob.glob(str(R/'t1_warm_vs_scratch_sorghum.csv')))
    boot = pd.concat([pd.read_csv(f) for f in bootfiles])
    boot = boot[boot.target.isin(TARGETS)]
    D['boot'] = [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in boot[['a','b','feature','target','split','r2_a','r2_b','delta','lo','hi']].to_dict('records')]
    # epoch-0 warm start of the final-data distillation (eval/eval_warmstart.py logs), per source
    D['warmstart'] = {}
    for f in sorted(glob.glob('logs/evalwarm_*.out')):
        txt = open(f).read()
        if 'EVAL_DONE' not in txt or '_d2.yaml' not in txt: continue
        for m in re.finditer(r'src=(\S+)\s+total ([\d.]+)\s+pc_loss [\d.]+\s+text_loss [\d.]+\s+pc_chamfer ([\d.]+)\s+param_mae ([\d.]+)', txt):
            D['warmstart'][m.group(1)] = {'pc_chamfer': float(m.group(3)), 'param_mae_masked': float(m.group(4)), 'log': Path(f).name}
        mm = re.search(r'MEAN generation metric: ([\d.]+)', txt)
        if mm and 'all_d2' in txt: D['warmstart']['_mean_all'] = float(mm.group(1))
    e9 = {}
    for d in [R/'e9_pr_600_d2_16930422', R/'e9_pr_6168_d2_16928969', R/'e9_pr_2024_d2_16928970', R/'e9_pr_624_d2_16928971'] + sorted(R.glob('e9_pr_e2more*_d2_*')) + sorted(R.glob('e9_pr_e5_d2_*')):
        for sp in ('val','test'):
            for r in pd.read_csv(d/f'e9_latent_summary_{sp}.csv').to_dict('records'):
                e9.setdefault(r['run'], {})[sp] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
    D['e9'] = e9
    it = pd.read_csv(R/'pcfscore_17071405_items.csv'); it['chamfer'] = it.chamfer_pred_to_gt + it.chamfer_gt_to_pred
    fs = []
    for (run, split), g in it.groupby(['run','split']):
        for t in (0.005, 0.01, 0.02):
            f1 = g[f'f1_{t:g}']; fs.append({'run': run, 'split': split, 'epoch': int(g.epoch.iloc[0]), 'n_plants': len(g), 'tau': t, 'precision': round(g[f'precision_{t:g}'].mean(), 5), 'recall': round(g[f'recall_{t:g}'].mean(), 5), 'f1': round(f1.mean(), 5), 'f1_median': round(float(f1.median()), 5), 'f1_p10': round(float(f1.quantile(0.1)), 5), 'chamfer': round(g.chamfer.mean(), 6), 'scale_m': round(float(g.scale_m.median()), 3)})
    D['fscore'] = fs
    pv = pd.concat([pd.read_csv(R/'pcview_16930426_probe.csv')] + [pd.read_csv(f) for f in sorted(R.glob('pcview_1708282*_probe.csv'))])
    rc = pd.concat([pd.read_csv(R/'pcview_16930426_recon.csv')] + [pd.read_csv(f) for f in sorted(R.glob('pcview_1708282*_recon.csv'))])
    cov = pd.read_csv(R/'pcview_16930426_coverage.csv')
    D['pcview'] = {'probe': [{k: r[k] for k in ('run','condition','fit','target','val_r2','test_r2')} for r in pv[pv.target.isin(['height','leaf_angle_roll','leaf_len_mean'])].to_dict('records')],
                   'recon': [{k: r[k] for k in ('run','condition','chamfer','echo_chamfer')} for r in rc.to_dict('records')],
                   'coverage': {r['condition']: r['mean'] for r in cov.groupby('condition')['mean'].mean().reset_index().to_dict('records')}}
    q = {}
    for f in sorted(glob.glob(str(R/'quant_*_d2_*.json'))):
        d = json.load(open(f)); q[f'{d["label"]}|{d["split"]}'] = {'label': d['label'], 'split': d['split'], 'epoch': d['epoch'], 'chamfer': d['summary']['model']['symmetric']['mean'], 'skill': d['skill_vs_mean_plant']}
    D['quant'] = list(q.values())
    rows = {}
    for lab, pat in (('zs', 'row_sorghum_4m_pretrain_15k_d2_sink_checkpoint_epoch_1000_test_*.json'), ('ft', 'row_sorghum_sorghum_row_ft_checkpoint_epoch_100_test_*.json'), ('iso', 'row_sorghum_sorghum_row_iso_checkpoint_epoch_100_test_*.json')):
        f = sorted(glob.glob(str(R/pat)))
        if f:
            d = json.load(open(f[-1])); rows[lab] = {'job': Path(f[-1]).stem.split('_')[-1], 'by_mode_source': {k: {m: v[m]['mean'] for m in v if v[m]} for k, v in d['by_mode_source'].items()},
                                                   'by_theta': {t: v['chamfer_gt_m2']['mean'] for t, v in d['by_theta']['row/all'].items()}}
    pair = sorted(glob.glob(str(R/'row_sorghum_pair_*.csv')))
    if pair: rows['pair'] = pd.read_csv(pair[-1]).to_dict('records')
    for run in ('sorghum_row_ft', 'sorghum_row_iso'):
        if (O/run/'training_history.json').exists():
            h = json.load(open(O/run/'training_history.json')); rows[run] = {'epochs': len(h['epoch']), 'val_chamfer': [[e, v] for e, v in zip(h['val_epoch'], h['val_pc_chamfer'])], 'val_chamfer_vs_input': [[e, v] for e, v in zip(h['val_epoch'], h['val_pc_chamfer_vs_input'])], 'val_leaf_mae': [[e, v] for e, v in zip(h['val_epoch'], h['val_leaf_mae_masked'])]}
    D['rows'] = rows
    # cross-species probes (PLAN_cross_species.md Z1) + native anchors
    xp = []
    for f in sorted(R.glob('xprobe_*.csv')):
        d = pd.read_csv(f); tag = Path(f).stem.rsplit('_', 1)[0].replace('xprobe_', '')
        for r in d.to_dict('records'):
            xp.append({'set': tag, 'run': r['run'], 'model_species': r['model_species'], 'data_species': r['data_species'],
                       'target': r['target'], 'val_r2': r['val_r2'], 'test_r2': r['test_r2']})
    nat = pd.read_csv(R/'probe_pm_e2_600_16669530.csv'); nat = nat[nat.feature == 'cls'] if 'feature' in nat.columns else nat
    for r in nat.to_dict('records'):
        xp.append({'set': 'native_maize_v1', 'run': r['run'], 'model_species': 'maize', 'data_species': 'maize', 'target': r['target'], 'val_r2': r['val_r2'], 'test_r2': r['test_r2']})
    D['xprobe'] = xp
    # cross-species rows (Z3) and single-photo generation (Z2), with native anchors
    def rowsum(f):
        b = json.load(open(f))['by_mode_source']
        return {k: round(v['chamfer_gt_m2']['mean'] * 1e4, 2) for k, v in b.items() if 'chamfer_gt_m2' in v}
    xr = []
    for lab, model, data, pat in (('maize row model on sorghum rows', 'maize', 'sorghum', 'row_sorghum_maize_row_ft_checkpoint_epoch_100_test_*.json'),
                                  ('sorghum row model on sorghum rows (native)', 'sorghum', 'sorghum', 'row_sorghum_sorghum_row_ft_checkpoint_epoch_100_test_*.json'),
                                  ('sorghum row model on maize rows', 'sorghum', 'maize', 'row_maize_sorghum_row_ft_checkpoint_epoch_100_test_*.json'),
                                  ('maize row model on maize rows (native)', 'maize', 'maize', 'row_maize_maize_row_ft_checkpoint_epoch_100_test_*.json')):
        rf = sorted(R.glob(pat))
        if rf: xr.append({'label': lab, 'model': model, 'data': data, 'job': rf[-1].stem.rsplit('_', 1)[-1], 'c': rowsum(rf[-1])})
    D['xrows'] = xr
    xq = []
    for f in sorted(R.glob('xquant_*.json')):
        d = json.load(open(f)); xq.append({'tag': f.stem.rsplit('_', 1)[0].replace('xquant_', ''), 'chamfer': d['summary']['model']['symmetric']['mean'], 'skill': d['skill_vs_mean_plant'], 'epoch': d['epoch']})
    D['xquant'] = xq
    D['images'] = {}
    if not a.no_images:
        for run in list(RUNS)[:12]:
            p = sorted(glob.glob(str(R/'e9_pr_*_d2_*'/'figures'/f'e9_latent_{run}_val_tsne.png')))
            if p: D['images'][f'tsne_{run}'] = img_b64(p[0], 760, 72)
        for i, p in enumerate(sorted(glob.glob(str(O/'4m_pretrain_15k_d2_sink'/'visualizations'/'epoch_980_sample_*_00.png')))[:3]):
            D['images'][f'recon{i}'] = img_b64(p, 700, 72); D['images'][f'recon{i}_name'] = Path(p).stem.split('_', 4)[-1]
        gv = sorted((O/'4m_distill_15k_all_d2'/'visualizations').glob('epoch_*_src-pc_sample_1_*.png'))
        if gv:
            D['images']['gen_v2_ep1'] = img_b64(gv[0], 1100, 80); D['images']['gen_v2_ep1_name'] = gv[0].stem
            D['images']['gen_v2_last'] = img_b64(gv[-1], 1100, 80); D['images']['gen_v2_last_name'] = gv[-1].stem
        for f in sorted((REPO/'paper'/'figs').glob('fig_*.png')):
            if f.stem in ('fig_rows_maize', 'fig_rows_sorghum'): continue   # single-species copies of fig_rows
            D['images']['paper_' + f.stem] = img_b64(f, 1600 if f.stem == 'fig_rows_examples' else 1400, 85)
    tpl = open(REPO/'figures'/'sorghum_ledger_template.html').read()
    html = tpl.replace('/*DATA*/', json.dumps(D, separators=(',', ':')))
    Path(a.out).write_text(html); print('wrote', a.out, len(html)//1024, 'KB; images', len(D['images']), '; fscore rows', len(fs), '; boot rows', len(D['boot']))

if __name__ == '__main__':
    main()
