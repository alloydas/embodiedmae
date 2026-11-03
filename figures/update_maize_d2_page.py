"""Patch the published "Maize D2 Results" page (https://claude.ai/artifact/PqNcyvGKEcznLdm5n9z8fY) to v6.

The page has no template in the repo: it is rebuilt from the published copy. Read the artifact
(Artifact read), then
    python figures/update_maize_d2_page.py <saved artifact .html> <out .html>
and publish <out> with url set. It merges the epoch-600 probes 23475548/9, their E9 folders and
t-SNE images, the E8 scoring pass 23344948 and the 2026-11-03 paired bootstraps into the embedded
JSON, then rewrites the matrix, strip, probe charts and queue. Every text replacement asserts it
matched exactly once, so a page that has drifted fails loudly instead of half-updating.
Needs PIL (the det env). A later update should copy this file and edit the run lists and texts.
"""
import base64, csv, io, json, re, sys
from pathlib import Path
from PIL import Image

SRC = sys.argv[1]; OUT = sys.argv[2]
REPO = Path('/work/hdd/bepk/adas14/embodiedmae'); REP = REPO / 'reports'
s = open(SRC).read()
s = s[s.index('<body>') + len('<body>'):]
s = s[:s.rindex('</body>')].strip('\n') + '\n'

m = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', s, re.S)
D = json.loads(m.group(2))

NEW = ['maize_e4_small_d2', 'maize_e4_large_d2', 'maize_4m_tg_d2', 'maize_4m_tg40_d2', 'maize_4m_tg100_d2',
       'maize_e2_pcd_d2', 'maize_e2_pct_d2', 'maize_e2_pcrgbt_d2', 'maize_e2_pcdt_d2']
PROBE_CSV = [REP / 'probe_pm_600_e4ctrl_23475548.csv', REP / 'probe_pm_600_e2more_23475549.csv']
E9_DIR = [REP / 'e9_pm_600_e4ctrl_23475548', REP / 'e9_pm_600_e2more_23475549']

# ---- probe rows (the reference rows in these CSVs are cache hits of probe 22751615)
ref = D['probe']['maize_4m_d2']['targets']
for f in PROBE_CSV:
    for r in csv.DictReader(open(f)):
        run = r['run']
        if run in ('maize_4m_d2', 'maize_e2_pcrgbd_d2'):
            assert abs(float(r['val_r2']) - D['probe'][run]['targets'][r['target']]['val_r2']) < 1e-9, (run, r['target'])
            continue
        e = D['probe'].setdefault(run, {'version': 'd2', 'epoch': r['epoch'], 'targets': {}})
        e['targets'][r['target']] = {'unit': r['unit'], 'val_r2': float(r['val_r2']), 'test_r2': float(r['test_r2']),
                                     'val_mae': float(r['val_mae']), 'test_mae': float(r['test_mae']),
                                     'val_base_mae': float(r['val_base_mae']), 'src': r['source_col']}
for r in NEW:
    assert set(D['probe'][r]['targets']) == set(ref), r

# ---- E9 summaries + val t-SNE images
def num(v):
    try: return float(v)
    except ValueError: return v
for d in E9_DIR:
    for split in ('val', 'test'):
        for r in csv.DictReader(open(d / f'e9_maize_latent_summary_{split}.csv')):
            if r['run'] in NEW:
                D['e9'].setdefault(r['run'], {})[split] = {k: num(v) for k, v in r.items()}
    for run in NEW:
        png = d / 'figures' / f'e9_maize_{run}_val_tsne.png'
        if png.exists():
            im = Image.open(png).convert('RGB'); w = 1000
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            b = io.BytesIO(); im.save(b, 'JPEG', quality=72, optimize=True)
            D['images']['tsne_' + run] = 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()
for r in NEW:
    assert 'val' in D['e9'][r] and 'test' in D['e9'][r] and 'tsne_' + r in D['images'], r

# ---- E8: test split of its own head + the frozen CLS / mean probes of it
e8 = {}
for k in ('head', 'cls', 'mean'):
    for r in csv.DictReader(open(REP / f'probe_e8_score_maize_23344948_{k}.csv')):
        e8.setdefault(k, {})[r['target']] = [round(float(r['val_r2']), 4), round(float(r['test_r2']), 4)]
# the page's old val row is the trainer's own epoch-600 val; the scoring pass agrees within 0.004
assert all(abs(e8['head'][t][0] - v) < 4e-3 for t, v in D['e8']['r2'].items())
D['e8']['r2'] = {t: v[0] for t, v in e8['head'].items()}
D['e8']['test_r2'] = {t: v[1] for t, v in e8['head'].items()}
D['e8']['cls'] = e8['cls']; D['e8']['mean'] = e8['mean']; D['e8']['job'] = '23344948'

# ---- new paired bootstraps (only those finished when this runs)
BOOT = ['maize_ctrl_e4_delta_d2_23475548.csv', 'maize_ctrl_vs_pcrgbd_delta_d2_23475548.csv',
        'maize_e2more_vs_pc_delta_d2_23475549.csv', 'maize_e2_pcrgbt_vs_pcrgb_delta_d2_23475549.csv',
        'maize_e2_pcdt_vs_pcd_delta_d2_23475549.csv']
have = {(x['a'], x['b']) for x in D['boot']}
nboot = 0
for f in BOOT:
    p = REP / f
    if not p.exists(): continue
    for r in csv.DictReader(open(p)):
        if (r['a'], r['b']) in have: continue
        D['boot'].append({'a': r['a'], 'b': r['b'], 'target': r['target'], 'split': r['split'],
                          **{k: round(float(r[k]), 4) for k in ('r2_a', 'r2_b', 'delta', 'lo', 'hi')}})
        nboot += 1
print('new boot rows', nboot)
s = s[:m.start(2)] + json.dumps(D, separators=(',', ':')) + s[m.end(2):]

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)

def rep_line(start, new):
    """Replace the whole line that begins with `start` (exactly one)."""
    global s
    assert s.count(start) == 1, (s.count(start), start[:90])
    i = s.index(start); j = s.index('\n', i)
    s = s[:i] + new + s[j:]

rep('State as of 2026-10-30 11:00 CDT.', 'State as of 2026-11-03 13:00 CST.')
rep("const PROBED = DONE.slice(0,7);",
    "const PROBED = ['maize_e2_pc_d2','maize_e2_pcd_d2','maize_e2_pct_d2','maize_e2_pcdt_d2','maize_e2_pcrgb_d2','maize_e2_pcrgbt_d2','maize_e2_pcrgbd_d2','maize_4m_d2','maize_4m_tg40_d2','maize_4m_tg_d2','maize_4m_tg100_d2','maize_e4_small_d2','maize_e4_large_d2','maize_e3_1k_d2','maize_e3_3k_d2','maize_e3_10k_d2'];")

# ---- summary strip
rep("['probed + E9','7','10 finished runs not probed yet'],", "['probed + E9','16','every finished Delta run, epoch 600 cut'],")
rep("['E1 epochs', c.maize_4m_pretrain_15k_d2.done+' / 1000', 'paused (next segment held); best = epoch 220'],",
    "['E1 epochs','1000 / 1000','finished on Nova 11-01; Delta copy stopped at '+c.maize_4m_pretrain_15k_d2.done],\n"
    "    ['best arm @ 600 (height)', f3(D.probe.maize_4m_tg100_d2.targets.height.val_r2), 'tg 1.0 (text as target only); 4m '+f3(D.probe.maize_4m_d2.targets.height.val_r2)],")
rep("['SU balance','1000','2026-10-30'],", "['SU balance','801','2026-11-03'],")

# ---- programme matrix
rep_line("    ['E1','headline pretrain','running',",
    "    ['E1','headline pretrain','done',`maize_4m_pretrain_15k_d2: 1000 epochs over all 105,000 views, global batch 256, QAL + 0.5 x Sinkhorn. "
    "Delta trained it to epoch ${c.maize_4m_pretrain_15k_d2.done} (best val loss epoch 220); the epoch-365 state was copied to Nova, where job 17147020 "
    "finished 1000/1000 on 2026-11-01 (val chamfer 0.00067, best_model epoch 960). Its probe, F1 and partial clouds ran on Nova "
    "(val R² height 0.739, leaf angle 0.958, stem radius 0.830, curl 0.698; F1 @ 0.01 = 0.504) and are not in this page's caches. "
    "The held Delta segment 23294091 was cancelled on 2026-11-03.`],")
rep_line("    ['E2 more arms',",
    "    ['E2 more arms','every stream set with the cloud','done',`PC+D / PC+T / PC+RGB+T / PC+D+T (= PC+RGB+D with only active_modalities changed), 600/600, "
    "probed and E9 (job 23475549). Val R² height ${f3(D.probe.maize_e2_pcd_d2.targets.height.val_r2)} / ${f3(D.probe.maize_e2_pct_d2.targets.height.val_r2)} / "
    "${f3(D.probe.maize_e2_pcrgbt_d2.targets.height.val_r2)} / ${f3(D.probe.maize_e2_pcdt_d2.targets.height.val_r2)} against PC ${f3(D.probe.maize_e2_pc_d2.targets.height.val_r2)} "
    "and PC+RGB ${f3(D.probe.maize_e2_pcrgb_d2.targets.height.val_r2)}: depth lifts the cloud a lot, nothing without RGB comes close to the RGB arms.`],")
rep_line("    ['E2 control','text gating','done',",
    "    ['E2 control','text gating','done',`maize_4m_tg_d2 / tg40_d2 / tg100_d2 (text_mask_ratio 0.80 / 0.40 / 1.0), 600/600, probed and E9 (job 23475548). "
    "Val R² height ${f3(D.probe.maize_4m_tg_d2.targets.height.val_r2)} / ${f3(D.probe.maize_4m_tg40_d2.targets.height.val_r2)} / ${f3(D.probe.maize_4m_tg100_d2.targets.height.val_r2)} "
    "against ${f3(D.probe.maize_4m_d2.targets.height.val_r2)} for the ungated four-stream arm. Gating the text at 0.80 or using it as a target only beats the ungated arm on "
    "every target; 0.40 is worse. The opposite of sorghum, where text as target only lost to PC+RGB+D.`],")
rep_line("    ['E4','model scaling','done',",
    "    ['E4','model scaling','done',`E4 small and E4 large 600/600, probed and E9 (job 23475548). Val chamfer small / base / large = ${c.maize_e4_small_d2.final.val_pc_chamfer.toFixed(5)} / "
    "${c.maize_4m_d2.final.val_pc_chamfer.toFixed(5)} / ${c.maize_e4_large_d2.final.val_pc_chamfer.toFixed(5)}; val R² height ${f3(D.probe.maize_e4_small_d2.targets.height.val_r2)} / "
    "${f3(D.probe.maize_4m_d2.targets.height.val_r2)} / ${f3(D.probe.maize_e4_large_d2.targets.height.val_r2)}. Base is best on both. F1 for small only.`],")
rep_line("    ['E8','baselines','running',",
    "    ['E8','baselines','running',`Supervised from scratch maize_e8_supervised_d2 600/600 and scored (job 23344948): its own head reaches val / test R² height "
    "${D.e8.r2.height.toFixed(3)} / ${D.e8.test_r2.height.toFixed(3)}, tassel droop ${D.e8.r2.tassel_droop.toFixed(3)} / ${D.e8.test_r2.tassel_droop.toFixed(3)}, ahead of every frozen "
    "600-epoch probe on most targets (rows in the probe table). Frozen baselines on d2 are running on Nova (17189733).`],")
rep_line("    ['E9','latent analysis','done',", "    ['E9','latent analysis','done','All sixteen probed runs, val and test.'],")
rep_line("    ['Distill','cross-modal distillation','blank',",
    "    ['Distill','cross-modal distillation','running','From the maize E1 epoch-1000 teacher, on Nova: generalist and three RGB students 100/100 (2026-11-03; val chamfer from RGB "
    "11.28e-3 before -> 0.77-0.80e-3 after); four two-stream students (rgbd2pc, rgbt2pc, dt2pc, pairs2pc) training. Results are in the Nova ledger, not on this page.'],")
rep_line("    ['E5','view regime','built',",
    "    ['E5','view regime','running','maize_e5_{topdown,horizon,single}_d2 (views ranked by camera elevation: top:2 / horizon:2 / view 0) training on Nova (17189739-41, scavenger). "
    "Their reference, maize_4m_d2, exists only on Delta.'],")
rep_line("    ['Partial clouds','1 vs 3 vs 10 cameras','built',",
    "    ['Partial clouds','1 vs 3 vs 10 cameras','done','eval/pc_view_eval_maize.py on maize E1 (Nova 17189727). The 600-epoch E2 arms were not run.'],")

# ---- curves section: E1 is no longer paused
rep('Sixteen self-supervised runs are finished; E1 is paused at epoch 368 of 1000.',
    'Sixteen self-supervised runs are finished on Delta. E1 stopped here at epoch 368; Nova carried the same run to 1000.')
rep('Every 20 epochs, to epoch 360. Loss is lowest at epoch 220 (the best_model.pth); chamfer keeps improving to 320.',
    'Delta part of the run, every 20 epochs to 360 (Nova continued from 365 to 1000). Loss is lowest at epoch 220 here; chamfer keeps improving to 320.')

# ---- probe section
rep('E4 small, E4 large, the three text-gating controls and the four new E2 arms are finished but not probed yet. The E8 supervised row is that network\'s own regression head on val (trained end to end), not a frozen probe.',
    'All sixteen finished Delta runs are probed at their matched epoch-600 cut (E3 at 6168 / 2024 / 624). The E8 supervised row is that network\'s own regression head (trained end to end on these targets), not a frozen probe; its frozen CLS probe matches the head within 0.004.')
rep("if(mode!=='mae' && D.e8 && D.e8.r2){ h+=`<tr><td class=\"lead mono\">maize_e8_supervised_d2 <span class=\"pill done\">head, val</span></td><td class=\"n\">${D.e8.val_epoch||''}</td>`+TARGETS.map(t=>{const v=D.e8.r2[t]; return v==null?'<td class=\"blankc\"></td>':`<td class=\"n muted\">${mode==='val'?v.toFixed(3):''}</td>`;}).join('')+'</tr>'; }",
    "if(mode!=='mae' && D.e8 && D.e8.r2){ const E8=mode==='val'?D.e8.r2:D.e8.test_r2; h+=`<tr><td class=\"lead mono\">maize_e8_supervised_d2 <span class=\"pill done\">head, ${mode}</span></td><td class=\"n\">${D.e8.val_epoch||''}</td>`+TARGETS.map(t=>{const v=E8[t]; return v==null?'<td class=\"blankc\"></td>':`<td class=\"n muted\">${v.toFixed(3)}</td>`;}).join('')+'</tr>'; }")
rep_line("  [['maize_e4_small_d2 / e4_large_d2',",
    "  [['maize_4m_pretrain_15k_d2','E1 · finished on Nova (val R² height 0.739, leaf angle 0.958, stem radius 0.830, curl 0.698); Delta copy stopped at epoch '+D.curves.maize_4m_pretrain_15k_d2.done]].forEach(p=>{ h+=`<tr><td class=\"lead mono\">${p[0]}</td><td class=\"blankc\" colspan=\"${TARGETS.length+1}\">${esc(p[1])}</td></tr>`; });")
rep('<div class="chart"><h3>E2: what each stream adds (val R²)</h3><div class="sub">PC → +RGB → +depth → +params.</div><div id="ch_e2_probe"></div></div>',
    '<div class="chart"><h3>E2: all eight stream sets (val R²)</h3><div class="sub">Every stream set that contains the point cloud. 4m = all four.</div><div id="ch_e2_probe"></div></div>\n'
    '      <div class="chart"><h3>Text gating and model size (val R²)</h3><div class="sub">tg = text_mask_ratio 0.40 / 0.80 / 1.0 (1.0 = target only); E4 small / large against base (4m).</div><div id="ch_ctrl_probe"></div></div>')
i = s.index("  const e2=['maize_e2_pc_d2','maize_e2_pcrgb_d2','maize_e2_pcrgbd_d2','maize_4m_d2'];")
j = s.index("  document.getElementById('ch_e2_probe').innerHTML=g;", i)
j = s.index('\n', j)
s = s[:i] + """  const tg=['height','leaf_count','biomass','leaf_angle','stem_radius','leaf_curl'];
  const tcol=['var(--c1)','var(--c2)','var(--c3)','var(--c4)','var(--c5)','var(--c6)'];
  function bars(elId, runs, lab, xlabel){
    const W=640,H=300,m={l:48,r:10,t:10,b:44}; const gw=(W-m.l-m.r)/runs.length, bw=gw/(tg.length+1);
    const sy=v=>H-m.b-Math.max(0,v)*(H-m.t-m.b);
    let g=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(xlabel)}">`;
    [0,0.2,0.4,0.6,0.8,1].forEach(v=>{ g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--line)" stroke-dasharray="2 3"/><text x="${m.l-6}" y="${sy(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)" font-family="var(--font-mono)">${v}</text>`; });
    runs.forEach((r,i)=>{ tg.forEach((t,j)=>{ const v=P[r].targets[t].val_r2; const x=m.l+i*gw+bw*0.5+j*bw; g+=`<rect x="${x.toFixed(1)}" y="${sy(v).toFixed(1)}" width="${(bw*0.9).toFixed(1)}" height="${(H-m.b-sy(v)).toFixed(1)}" fill="${tcol[j]}"><title>${esc(lab(r))} · ${TLAB[t]}: ${v.toFixed(3)}</title></rect>`; }); g+=`<text x="${m.l+i*gw+gw/2}" y="${H-m.b+16}" text-anchor="middle" font-size="${runs.length>6?10:11}" fill="var(--fg)">${esc(lab(r))}</text>`; });
    g+=`<text x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)">${esc(xlabel)}</text></svg>`;
    g+='<div class="legend">'+tg.map((t,j)=>`<span><i style="background:${tcol[j]}"></i>${TLAB[t]}</span>`).join('')+'</div>';
    document.getElementById(elId).innerHTML=g;
  }
  bars('ch_e2_probe', ['maize_e2_pc_d2','maize_e2_pcd_d2','maize_e2_pct_d2','maize_e2_pcdt_d2','maize_e2_pcrgb_d2','maize_e2_pcrgbt_d2','maize_e2_pcrgbd_d2','maize_4m_d2'],
       r=>r==='maize_4m_d2'?'4m':SHORT[r], 'E2 arm (val R², CLS probe, epoch 600)');
  bars('ch_ctrl_probe', ['maize_4m_d2','maize_4m_tg40_d2','maize_4m_tg_d2','maize_4m_tg100_d2','maize_e4_small_d2','maize_e4_large_d2'],
       r=>r==='maize_4m_d2'?'4m (base)':SHORT[r], 'control or model size (val R², CLS probe, epoch 600)');""" + s[j:]

# ---- bootstrap reading
rep('Every reduced-data arm loses 10 of 11 targets to full data.</div>',
    'Every reduced-data arm loses 10 of 11 targets to full data.</div>\n'
    '    <div class="note"><strong>New pairs (2026-11-03).</strong> Gating the parameter stream wins: tg 0.80 and tg 1.0 (text as a target only) beat the ungated four-stream arm '
    'on all 11 targets, and beat PC+RGB+D on all 11 too; tg 0.40 loses 9 and wins none. Both E4 sizes lose 10 of 11 to the base model. Depth is the biggest single addition to the '
    'point cloud (PC+D over PC: 9 wins, loses curl); the parameter stream also helps the cloud alone (PC+T over PC: 7 wins, no loss). Added to PC+RGB it trades: '
    'PC+RGB+T wins twist, curl and tassel droop and loses height, leaf angle, leaf count, droop and leaf length; PC+D+T over PC+D wins twist, curl and width and loses height and stem radius.</div>')

# ---- queue
i = s.index("  const rows=[\n    ['23294091'"); j = s.index('  ];', i)
s = s[:i] + """  const rows=[
    ['23475549','pm_600_e2more','gpuA40x4','done 11-03 ~11:40','probe + E9 at 600: PC+D, PC+T, PC+RGB+T, PC+D+T (+ PC+RGB+D, 4m as cache-hit references)'],
    ['23475548','pm_600_e4ctrl','gpuA40x4','done 11-03 11:35','probe + E9 at 600: E4 small, E4 large, tg 0.80 / 0.40 / 1.0 (+ 4m)'],
    ['23475302/03','pm_600_*','gpuA40x4','failed 11-03 10:35 (launcher bug, fixed)','FEATURE echoed before its default under set -u; resubmitted as 23475548/49'],
    ['23344948','e8_score_maize','gpuA100x8','done 10-30 19:59','E8 supervised scoring: head, CLS and mean probes, val + test'],
    ['23294091','tr_maize_e1_d2','gpuA100x4','cancelled 11-03 (user)','next E1 segment; redundant once Nova finished E1'],
    ['23135087','tr_maize_e1_d2','gpuA100x4','timeout 10-29 07:24 (planned)','E1 4-GPU segment, epochs 236 to 368'],
    ['23160460-63','tr_maize_e2_{pcd,pct,pcrgbt,pcdt}_d2','gpuA100x4','done 10-26','four new E2 arms, 600/600'],
    ['22751641/43/45, 22752739, 22752835','tg*, e8_supervised, e4_large','gpuA100x4','done 10-24 / 10-25','600/600 each'],
""" + s[j:]
rep('Balance 1000 SU on bepk-delta-gpu (2026-10-30). Nothing of ours is running. Not yet queued: probes and E9 at epoch 600 for E4 small, E4 large, the three text-gating controls and the four new E2 arms; the E8 supervised scoring pass; point-cloud F1 for the new runs; the maize E5 arms and partial clouds. The distillations wait for E1 at epoch 1000.',
    'Balance 801 SU on bepk-delta-gpu (2026-11-03; ours since 10-30 was one 14-minute and two ~45-minute 1-GPU jobs, the rest is other project members). Nothing of ours is running on Delta. Not run here: point-cloud F1 for the runs after E4 small, partial clouds of the 600-epoch arms. E1, the distillations, E5 and the E8 frozen baselines run on Nova.')
open(OUT, 'w').write(s)
print('ok', len(s), 'bytes; boot rows', len(D['boot']))
