"""Patch the published "Maize D2 Results" page (https://claude.ai/artifact/PqNcyvGKEcznLdm5n9z8fY)
with the maize results that ran on Nova (v7).

The page has no template in the repo (Delta builds it with figures/update_maize_d2_page.py by
patching the published copy). Read the artifact (Artifact read), then on Nova
    python figures/update_maize_d2_page_nova.py <saved artifact .html> <out .html>
and publish <out> with url set. Merges: the maize E1 epoch-1000 probe (17189722) into the probe
table, E1 point-cloud F1 (17189723) into the F-score table, and a new "Nova" section with the
distillation before/after per source, student F1 (17189743), the E8 frozen baselines on d2
(17189733), the rows from E1 (17189621 zero-shot, 17189701 isolated control), the E1 partial
clouds (17189727) and E5 progress. Every text replacement asserts it matched exactly once.
Re-running on a page that already has the Nova section replaces that section's data.
"""
import csv, json, re, sys
from pathlib import Path

SRC, OUT = sys.argv[1], sys.argv[2]
REP = Path('reports'); OUTD = Path('outputs')
s = open(SRC).read()
if '<body>' in s:
    s = s[s.index('<body>') + len('<body>'):]
    s = s[:s.rindex('</body>')].strip('\n') + '\n'
m = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', s, re.S)
D = json.loads(m.group(2))
E1 = 'maize_4m_pretrain_15k_d2'

def rows(f): return list(csv.DictReader(open(f)))
def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80]); s = s.replace(a, b)

# ---- E1 probe row (CLS, epoch 1000), same format as the Delta rows
ref = D['probe']['maize_4m_d2']['targets']
e = {'version': 'd2', 'epoch': '1000', 'targets': {}}
for r in rows(REP / 'probe_pm_e1_d2_17189722.csv'):
    if r.get('feature', 'cls') != 'cls': continue
    e['targets'][r['target']] = {'unit': r['unit'], 'val_r2': float(r['val_r2']), 'test_r2': float(r['test_r2']),
                                 'val_mae': float(r['val_mae']), 'test_mae': float(r['test_mae']),
                                 'val_base_mae': float(r['val_base_mae']), 'src': r['source_col']}
assert set(e['targets']) == set(ref), set(ref) ^ set(e['targets'])
D['probe'][E1] = e

# ---- E1 point-cloud F1 rows
keep = [x for x in D['fscore'] if x['run'] != E1]
for r in rows(REP / 'pc_fscore_maize_nova_17189723.csv'):
    x = {k: (float(v) if k not in ('run', 'split', 'active') else v) for k, v in r.items()}
    x['epoch'] = int(x['epoch']); x['n_plants'] = int(x['n_plants']); keep.append(x)
D['fscore'] = keep

# ---- Nova section data
N = {}
def f(v, d=3): return None if v is None else round(float(v), d)
# distillation: per source, epoch-0 warm start vs final epoch (val, masked-regime generation)
dist = []
LAB = {'maize_distill_e1_d2': 'generalist (all four sources)', 'maize_distill_rgb2pc_e1_d2': 'RGB -> cloud, source visible',
       'maize_distill_rgb2pc_smr50_e1_d2': 'RGB -> cloud, 50 % hidden (draw 1)', 'maize_distill_rgb2pc_smr50b_e1_d2': 'RGB -> cloud, 50 % hidden (draw 2)',
       'maize_distill_rgbd2pc_e1_d2': 'RGB+depth -> cloud', 'maize_distill_rgbt2pc_e1_d2': 'RGB+recipe -> cloud',
       'maize_distill_dt2pc_e1_d2': 'depth+recipe -> cloud', 'maize_distill_pairs2pc_e1_d2': 'three pairs -> cloud'}
for run, lab in LAB.items():
    h = OUTD / run / 'training_history.json'
    if not h.exists(): dist.append([lab, '', 'not started', '', '', '', '']); continue
    v = json.load(open(h))['val']; v0, vl = v[0], v[-1]; ep = vl['epoch']
    state = 'done' if ep >= 100 else f'epoch {ep} / 100'
    for src in v0['per_source']:
        a, b = v0['per_source'][src], vl['per_source'][src]
        dist.append([lab if src == list(v0['per_source'])[0] else '', src, state,
                     f(a['pc_chamfer'] * 1e3, 2), f(b['pc_chamfer'] * 1e3, 2) if ep > 0 else None,
                     None if 'text' in src else f(a.get('param_mae_masked'), 4),
                     None if 'text' in src or ep == 0 else f(b.get('param_mae_masked'), 4)])  # recipe visible: no parameter error to report
N['dist'] = dist
# student F1 at tau 0.01 (+ E1)
fs = {}
for fn in ('pc_fscore_maize_nova_17189723.csv', 'pc_fscore_maize_nova_17189743.csv'):
    for r in rows(REP / fn):
        if abs(float(r['tau']) - 0.01) < 1e-9:
            fs.setdefault(r['run'], {})[r['split']] = (float(r['precision']), float(r['recall']), float(r['f1']), float(r['chamfer']))
N['sfs'] = [[{E1: 'E1 teacher (epoch 1000)'}.get(k, LAB.get(k, k))] + [f(x, 3) for x in fs[k]['val'][:3]] + [f(fs[k]['test'][2], 3)]
            for k in [E1, 'maize_distill_e1_d2', 'maize_distill_rgb2pc_e1_d2', 'maize_distill_rgb2pc_smr50_e1_d2', 'maize_distill_rgb2pc_smr50b_e1_d2'] if k in fs]
# E8 frozen baselines on d2 (best pooling per model, val / test R2), with E1 and maize_4m_d2 CLS for reference
TG = ['height', 'leaf_count', 'leaf_angle', 'stem_radius', 'leaf_curl', 'tassel_droop']
base = {}
for r in rows(REP / 'probe_e8_base_maize_d2_17189733.csv'):
    base.setdefault((r['run'], r['feature']), {})[r['target']] = (float(r['val_r2']), float(r['test_r2']))
BL = {'random_init': 'random init', 'dinov2_vitb14': 'DINOv2 ViT-B/14', 'multimae': 'MultiMAE-B', 'embodiedmae': 'EmbodiedMAE-B',
      'pointmae': 'Point-MAE', 'pointmae_gravity': 'Point-MAE + gravity', 'pointmae_upright': 'Point-MAE + full pose'}
e8 = []
for run, lab in BL.items():
    for feat in ('cls', 'mean', 'max+mean'):
        if (run, feat) in base:
            t = base[(run, feat)]; e8.append([lab, feat] + [f(t[k][0]) if k in t else None for k in TG] + [f(t[k][1]) if k in t else None for k in TG])
for run, lab in [(E1, 'ours: maize E1 (frozen CLS)'), ('maize_4m_d2', 'ours: maize_4m_d2 (frozen CLS)')]:
    t = D['probe'][run]['targets']; e8.append([lab, 'cls'] + [f(t[k]['val_r2']) for k in TG] + [f(t[k]['test_r2']) for k in TG])
N['e8'] = e8; N['e8_targets'] = TG
# rows from E1 (test, chamfer vs target, 1e-4 m^2)
def rj(fn): return json.load(open(REP / fn))['by_mode_source']
zs, iso = rj('row_maize_maize_4m_pretrain_15k_d2_checkpoint_epoch_1000_test_17189621.json'), rj('row_maize_maize_row_iso_e1_checkpoint_epoch_100_test_17189701.json')
ftj = sorted(REP.glob('row_maize_maize_row_ft_e1_*_test_*.json'))
ft = json.load(open(ftj[-1]))['by_mode_source'] if ftj else None
rowt = []
for mode in ('isolated', 'row'):
    for src in ('all', 'rgb', 'depth', 'pc'):
        k = f'{mode}/{src}'
        g = lambda d: None if d is None or k not in d else f(d[k]['chamfer_gt_m2']['mean'] * 1e4, 2)
        rowt.append([mode, src, g(zs), g(iso), g(ft)])
N['rows'] = rowt; N['rows_ft'] = bool(ft)
# partial clouds (E1, matched probe val R2; coverage; recon chamfer)
pv = {}
for r in rows(REP / 'pcview_maize_nova_17189727_probe.csv'):
    if r['fit'] == 'matched': pv.setdefault(r['condition'], {})[r['target']] = float(r['val_r2'])
cov = {r['condition']: float(r['mean']) for r in rows(REP / 'pcview_maize_nova_17189727_coverage.csv') if r['split'] == 'val'}
rec = {r['condition']: float(r['chamfer']) for r in rows(REP / 'pcview_maize_nova_17189727_recon.csv')}
N['pcview'] = [[c, f(cov.get(c)), f(rec.get(c) * 1e3, 2)] + [f(pv[c].get(t)) for t in TG] for c in ('1view', '3view', 'full') if c in pv]
# E5 progress
e5 = []
for arm, band in [('topdown', 'top:2 (two highest cameras)'), ('horizon', 'horizon:2'), ('single', 'view 0 (random camera)')]:
    d = OUTD / f'maize_e5_{arm}_d2_nova' / 'checkpoints'
    ck = sorted(d.glob('checkpoint_epoch_*.pth'), key=lambda p: int(p.stem.split('_')[-1])) if d.exists() else []
    e5.append([f'maize_e5_{arm}_d2', band, int(ck[-1].stem.split('_')[-1]) if ck else 0])
N['e5'] = e5
# ---- visual results: trainer grids, cropped and JPEG-encoded into D.images
import base64, io
from PIL import Image
def enc(path, crop=None, w=1100):
    im = Image.open(path).convert('RGB')
    if crop: im = im.crop((0, 0, im.width, int(im.height * crop)))
    if im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=70, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()
def latest(d, plant, prefix='epoch_'):
    fs = sorted((OUTD / d / 'visualizations').glob(f'{prefix}*{plant}.png'), key=lambda p: int(p.name.split('_')[1]))
    return fs[-1] if fs else None
GAL = []
def add(key, path, cap, crop=None, group=''):
    if path and Path(path).exists():
        D['images'][key] = enc(path, crop); GAL.append({'key': key, 'cap': cap, 'group': group})
V = OUTD / E1 / 'visualizations'
for ep in ('001', '100', '1000'):
    f = sorted(V.glob(f'epoch_{ep}_sample_*_plant_0003_00.png'))
    add(f'nv_e1_{ep}', f[0] if f else None, f'E1 at epoch {int(ep)}, plant_0003 view 00: RGB, depth and point cloud, input / masked / reconstructed at mask ratio 0.80.', 0.61, 'E1 reconstructions over training (same plant)')
for pl in ('plant_0000_00', 'plant_0034_00'):
    f = sorted(V.glob(f'epoch_1000_sample_*_{pl}.png'))
    add(f'nv_e1_1000_{pl}', f[0] if f else None, f'E1 at epoch 1000, {pl}, full grid including the decoded recipe (masked tokens predicted).', None, 'E1 at epoch 1000, other plants')
for run, src, lab in [('maize_distill_rgb2pc_e1_d2', 'rgb', 'RGB student (photograph only)'), ('maize_distill_e1_d2', 'pc', 'generalist, from the point cloud only')]:
    for ep in ('001', '100'):
        f = sorted((OUTD / run / 'visualizations').glob(f'epoch_{ep}_src-{src}_sample_*_plant_0000_02.png'))
        add(f'nv_dist_{src}_{ep}', f[0] if f else None, f'{lab}, epoch {int(ep)}: everything generated from one source, plant_0000 view 02 (val). Chamfer in the cloud panel title.', None, 'Distillation from E1: one source in, all four out (epoch 1 vs 100)')
for arm, lab in [('topdown', 'top-down (two highest cameras)'), ('horizon', 'horizon (two most level cameras)'), ('single', 'single (view 0)')]:
    f = latest(f'maize_e5_{arm}_d2_nova', 'plant_0003_00')
    if f: add(f'nv_e5_{arm}', f, f'E5 {lab}, epoch {int(f.name.split("_")[1])}, plant_0003 view 00. Each grid has its own masking draw, so compare shapes, not the losses in the title.', 0.61, 'E5 arms on the same plant')
N['gallery'] = GAL
D['nova'] = N

s = s[:m.start(2)] + json.dumps(D, separators=(',', ':')) + s[m.end(2):]

# ---- markup: Nova section before the queue (or replace an earlier one)
SEC = '''  <section id="nova">
    <div class="sechead"><span class="eyebrow">Nova</span><h2>Maize results that ran on Nova</h2></div>
    <p>E1 finished on Nova (job 17147020, 1000/1000 on 2026-11-01), so everything derived from it ran there on the same d2 data (<span class="mono">/work/mech-ai-scratch/alloy/Maize_1</span>): the distillation, the rows, the partial clouds, the E8 frozen baselines and E5. E1's probe row is in the probe table above and its F1 in the F-score table.</p>
    <h3>Distillation from the E1 teacher (val, before = epoch-0 warm start)</h3>
    <p class="small muted">Generation from the named source with the other streams masked. Chamfer x1e-3 of the generated cloud; parameter MAE on masked tokens.</p>
    <div class="tablewrap"><table id="nv_dist"></table></div>
    <h3 style="margin-top:18px">Student point-cloud F1 at τ = 0.01 (17189743)</h3>
    <div class="tablewrap"><table id="nv_sfs"></table></div>
    <h3 style="margin-top:18px">E8 frozen baselines on d2 (17189733), R²</h3>
    <p class="small muted">Each baseline at each pooling it supports. Our two rows are the frozen CLS probe. Pose rows hand Point-MAE the camera rotation, which our arms never get.</p>
    <div class="toggle" role="group" aria-label="E8 split"><button id="nv_e8v" aria-pressed="true">val</button><button id="nv_e8t" aria-pressed="false">test</button></div>
    <div class="tablewrap"><table id="nv_e8"></table></div>
    <h3 style="margin-top:18px">Regime III rows, from E1 (test, Chamfer vs target, x1e-4 m²)</h3>
    <div class="tablewrap"><table id="nv_rows"></table></div>
    <h3 style="margin-top:18px">Partial clouds, E1 (17189727, val)</h3>
    <p class="small muted">A maize cloud is what one camera sees, so the question is inverted: 1view = the cloud E1 trained on, 3view / full = 3 or 10 cameras fused into camera 00's frame. Coverage = share of the ten-camera cloud within 5 mm; probe refitted per condition.</p>
    <div class="tablewrap"><table id="nv_pcv"></table></div>
    <h3 style="margin-top:18px">E5 view regime (training)</h3>
    <div class="tablewrap"><table id="nv_e5"></table></div>
    <h3 style="margin-top:22px">Visual results</h3>
    <p class="small muted">The trainers' own grids on val plants. In the reconstruction grids the depth panel's silhouette comes from the ground truth (only the values inside it are predicted).</p>
    <div id="nv_gallery"></div>
  </section>
'''
if '<section id="nova">' in s:
    s = re.sub(r'  <section id="nova">.*?</section>\n', lambda _: SEC, s, flags=re.S)
else:
    once('  <section id="queue">', SEC + '  <section id="queue">')

JS = '''
// ---------- Nova section
(function(){
  const N=D.nova; if(!N) return;
  const c=v=>v==null?'<td class="blankc"></td>':`<td class="n">${typeof v==='number'?v:esc(String(v))}</td>`;
  const tbl=(id,head,rows)=>{document.getElementById(id).innerHTML='<thead><tr>'+head.map(x=>`<th${x[1]?' class="n"':''}>${esc(x[0])}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((v,i)=>i<2&&typeof v==='string'?`<td class="${i?'':'lead'}">${esc(v)}</td>`:c(v)).join('')+'</tr>').join('')+'</tbody>';};
  tbl('nv_dist',[['student'],['source'],['state'],['chamfer before',1],['chamfer after',1],['param MAE before',1],['param MAE after',1]],N.dist.map(r=>[r[0],r[1],r[2],r[3],r[4],r[5],r[6]]));
  tbl('nv_sfs',[['model'],['precision (val)',1],['recall (val)',1],['F1 (val)',1],['F1 (test)',1]],N.sfs);
  const TL={height:'height',leaf_count:'leaves',leaf_angle:'leaf angle',stem_radius:'stem r.',leaf_curl:'curl',tassel_droop:'tassel droop'};
  function e8(split){ const off=split==='val'?2:2+N.e8_targets.length; const best={}; N.e8_targets.forEach((t,i)=>{best[t]=Math.max(...N.e8.map(r=>r[off+i]==null?-9:r[off+i]));});
    document.getElementById('nv_e8').innerHTML='<thead><tr><th>model</th><th>pool</th>'+N.e8_targets.map(t=>`<th class="n">${TL[t]}</th>`).join('')+'</tr></thead><tbody>'+N.e8.map(r=>`<tr><td class="lead">${esc(r[0])}</td><td>${esc(r[1])}</td>`+N.e8_targets.map((t,i)=>{const v=r[off+i]; return v==null?'<td class="blankc"></td>':`<td class="n"${v>=best[t]-1e-9?' style="font-weight:600"':''}>${v.toFixed(3)}</td>`;}).join('')+'</tr>').join('')+'</tbody>';
    document.getElementById('nv_e8v').setAttribute('aria-pressed',split==='val'); document.getElementById('nv_e8t').setAttribute('aria-pressed',split==='test'); }
  e8('val'); document.getElementById('nv_e8v').addEventListener('click',()=>e8('val')); document.getElementById('nv_e8t').addEventListener('click',()=>e8('test'));
  tbl('nv_rows',[['input'],['sources'],['zero-shot E1',1],['isolated-trained',1],['row-trained'+(N.rows_ft?'':' (training)'),1]],N.rows);
  tbl('nv_pcv',[['condition'],['coverage',1],['recon chamfer x1e-3',1],...N.e8_targets.map(t=>[TL[t]+' R²',1])],N.pcview);
  tbl('nv_e5',[['run'],['train views'],['epoch / 600',1]],N.e5);
  if(N.gallery){ const groups=[...new Set(N.gallery.map(g=>g.group))];
    document.getElementById('nv_gallery').innerHTML=groups.map(gr=>`<h3 style="margin-top:14px;font-size:.92rem">${esc(gr)}</h3><div class="imggrid">`+N.gallery.filter(g=>g.group===gr).map(g=>`<figure><img loading="lazy" src="${D.images[g.key]}" alt="${esc(g.cap)}"><figcaption>${esc(g.cap)}</figcaption></figure>`).join('')+'</div>').join(''); }
})();
'''
if '// ---------- Nova section' in s:
    s = re.sub(r'\n// ---------- Nova section.*?\}\)\(\);\n', lambda _: JS, s, flags=re.S)
else:
    once('\n// ---------- queue', JS + '\n// ---------- queue')

# ---- probe table: E1 becomes a real row; F-score order includes E1
if f"'{E1}'" not in s.split('const PROBED = [', 1)[1].split(']', 1)[0]:
    once("const PROBED = [", f"const PROBED = ['{E1}',")
s = re.sub(r"\n  \[\['maize_4m_pretrain_15k_d2','E1 · finished on Nova.*?\}\);", '', s, count=1, flags=re.S)
if "const order = DONE.concat(['maize_4m_pretrain_15k_d2'])" not in s:
    once("const order = DONE.filter(r=>FS.some(x=>x.run===r));", "const order = DONE.concat(['maize_4m_pretrain_15k_d2']).filter(r=>FS.some(x=>x.run===r));")

# ---- matrix text
def setrow(key, state, text):
    global s
    pat = re.compile(r"\n    \['" + re.escape(key) + r"','([^']*)','[a-z]+',(`[^`]*`|'[^']*')\],")
    mm = pat.search(s); assert mm, key
    s = s[:mm.start()] + f"\n    ['{key}','{mm.group(1)}','{state}','{text}']," + s[mm.end():]
setrow('E5', 'running', 'maize_e5_{topdown,horizon,single}_d2 on Nova (views ranked by camera elevation: top:2 / horizon:2 / view 0), epochs ' + ' / '.join(str(r[2]) for r in e5) + ' of 600 (Nova section). Their reference, maize_4m_d2, exists only on Delta: its probe cache must be copied to Nova, or the E5 arms probed on Delta.')
setrow('E8', 'done', 'Supervised from scratch: maize_e8_supervised_d2 scored on Delta (23344948; head row in the probe table). Frozen baselines on d2 (Nova 17189733): Point-MAE, + gravity, + full pose, EmbodiedMAE-B, MultiMAE-B, DINOv2, random init (Nova section).')
setrow('Distill', 'running', 'From the maize E1 epoch-1000 teacher, on Nova: generalist and three RGB students 100/100 (val chamfer from RGB 11.28e-3 before -> 0.77-0.80e-3 after; F1 0.49-0.51 vs the teacher 0.50); four two-stream students training or pending (Nova section).')
setrow('Partial clouds', 'done', 'eval/pc_view_eval_maize.py on maize E1 (Nova 17189727): 1 / 3 / 10 cameras fused (Nova section). The 600-epoch E2 arms were not run.')

# ---- lede, queue
s = re.sub(r'State as of [0-9-]+ [0-9:]+ CST\.', 'State as of 2026-11-04 11:30 CST.', s, count=1)
s = re.sub(r"\n    \['[^']*','[^']*\(Nova\)',[^\n]*\],", '', s)   # drop Nova rows of an earlier build
once("    ['23475549','pm_600_e2more',", """    ['17162333-35, 17213246','mdist_* (Nova)','nova','three running, pairs2pc pending','maize two-stream distillation students from E1: rgbt2pc ~68, dt2pc ~39, rgbd2pc ~35 of 100 (11-04 11:15); rgbt2pc ends ~11-04 22:00, rgbd2pc ~11-06'],
    ['17233173 -> 17189700','maize_row_ft_e1 (Nova)','scavenger','pending','row fine-tune from E1, epochs 91-100 on 1 GPU (accum 2 = global 32), then its test eval'],
    ['17189739-41','maize_e5_* (Nova)','nova / scavenger','top-down done, two running','E5 arms, 600 epochs each; single and horizon finish 11-04 afternoon'],
    ['23475549','pm_600_e2more',""")
open(OUT, 'w').write(s)
print('wrote', OUT, len(s) // 1024, 'KB')
