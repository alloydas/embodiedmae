"""Visualize ALL cross-modal distillation results into one dashboard PNG."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = ['rgb', 'depth', 'pc', 'text']
COL = {'rgb': '#e74c3c', 'depth': '#2980b9', 'pc': '#27ae60', 'text': '#8e44ad'}

base = json.load(open('crossmodal_baseline.json'))['per_source']
base_total = {s: base[s]['total'] for s in SRC}
base_mean = float(np.mean([base_total[s] for s in SRC]))

# all-source long run (resumed to 100 ep)
H = json.load(open('outputs/4m_distill_v1/training_history.json'))
val = sorted(H['val'], key=lambda e: e['epoch'])
ep = [e['epoch'] for e in val]
mean_gen = [e['mean_gen'] for e in val]
per_src = {s: [e['per_source'][s]['total'] for e in val] for s in SRC}
chamfer = {s: [e['per_source'][s]['pc_chamfer'] for e in val] for s in SRC}
pmae = {s: [e['per_source'][s]['param_mae_masked'] for e in val] for s in SRC}

def at_epoch(target):
    i = min(range(len(ep)), key=lambda k: abs(ep[k] - target))
    return ep[i], mean_gen[i], {s: per_src[s][i] for s in SRC}

_, m25, ps25 = at_epoch(25)
_, m100, ps100 = at_epoch(100)

# specialists
spec = {}
for s in SRC:
    p = Path(f'outputs/4m_distill_src_{s}/training_history.json')
    if p.exists():
        h = json.load(open(p))
        v = sorted(h['val'], key=lambda e: e['epoch'])
        if v:
            spec[s] = {'ep': [e['epoch'] for e in v],
                       'mean': [e['mean_gen'] for e in v],
                       'final': v[-1]['mean_gen'], 'done': v[-1]['epoch'] >= 25}

fig = plt.figure(figsize=(19, 11))
fig.suptitle('EmbodiedMAE-4M  ·  Cross-modal Distillation Results  (one modality → generate all four)',
             fontsize=16, fontweight='bold')

# (1) all-source MEAN vs epoch
ax = plt.subplot(2, 3, 1)
ax.plot(ep, mean_gen, '-o', color='#222', lw=2, ms=4, label='all-source student')
ax.axhline(base_mean, ls='--', color='gray', label=f'zero-shot baseline ({base_mean:.3f})')
ax.scatter([25, 100], [m25, m100], color='#d35400', zorder=5, s=60)
ax.annotate(f'{m25:.3f}', (25, m25), textcoords='offset points', xytext=(4, 8), fontsize=9)
ax.annotate(f'{m100:.3f}', (100, m100), textcoords='offset points', xytext=(-30, 8), fontsize=9)
ax.set_xlabel('epoch'); ax.set_ylabel('mean gen loss (4 sources)')
ax.set_title('(1) All-source: mean cross-modal loss', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=.3)

# (2) all-source per-source total vs epoch
ax = plt.subplot(2, 3, 2)
for s in SRC:
    ax.plot(ep, per_src[s], '-o', color=COL[s], ms=3, label=f'{s}→all')
    ax.axhline(base_total[s], ls=':', color=COL[s], alpha=.5)
ax.set_xlabel('epoch'); ax.set_ylabel('gen loss (total)')
ax.set_title('(2) All-source: per-source generation loss\n(dotted = each source baseline)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=.3)

# (3) bar comparison per source: baseline / all-25 / all-100 / specialist
ax = plt.subplot(2, 3, 3)
x = np.arange(len(SRC)); w = 0.2
ax.bar(x - 1.5*w, [base_total[s] for s in SRC], w, label='baseline (0 ep)', color='#bbb')
ax.bar(x - 0.5*w, [ps25[s] for s in SRC], w, label='all-source 25 ep', color='#f39c12')
ax.bar(x + 0.5*w, [ps100[s] for s in SRC], w, label='all-source 100 ep', color='#16a085')
spec_vals = [spec[s]['final'] if s in spec else np.nan for s in SRC]
ax.bar(x + 1.5*w, spec_vals, w, label='specialist 25 ep', color='#c0392b')
for i, s in enumerate(SRC):
    if s in spec:
        tag = '' if spec[s]['done'] else '*'
        ax.text(i + 1.5*w, spec[s]['final'] + .01, f"{spec[s]['final']:.3f}{tag}",
                ha='center', fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([f'{s}→all' for s in SRC])
ax.set_ylabel('gen loss (total)')
ax.set_title('(3) Per-source: baseline vs all-source vs specialist\n(* = still running)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')

# (4) specialist mean trajectories
ax = plt.subplot(2, 3, 4)
for s in SRC:
    if s in spec:
        style = '-o' if spec[s]['done'] else '--o'
        lbl = f'{s} specialist' + ('' if spec[s]['done'] else ' (running)')
        ax.plot(spec[s]['ep'], spec[s]['mean'], style, color=COL[s], ms=4, label=lbl)
        ax.axhline(base_total[s], ls=':', color=COL[s], alpha=.4)
ax.set_xlabel('epoch'); ax.set_ylabel(f'{"source"}→all gen loss')
ax.set_title('(4) Per-source specialists: convergence\n(dotted = that source baseline)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

# (5) all-source generated-PC chamfer vs epoch
ax = plt.subplot(2, 3, 5)
for s in SRC:
    ax.plot(ep, chamfer[s], '-o', color=COL[s], ms=3, label=f'{s}→PC')
ax.set_xlabel('epoch'); ax.set_ylabel('chamfer (generated PC vs GT)')
ax.set_title('(5) All-source: generated point-cloud quality', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

# (6) all-source generated-param MAE vs epoch
ax = plt.subplot(2, 3, 6)
for s in SRC:
    if s == 'text':
        continue  # text-as-source => params not masked (trivially 0)
    ax.plot(ep, pmae[s], '-o', color=COL[s], ms=3, label=f'{s}→params')
ax.set_xlabel('epoch'); ax.set_ylabel('param MAE (masked, normalized)')
ax.set_title('(6) All-source: generated spline-param accuracy', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = 'crossmodal_distill_results.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print('saved', out)

# also print a compact text summary
print('\n=== SUMMARY ===')
print(f"{'source':8} {'baseline':>9} {'all-25ep':>9} {'all-100ep':>10} {'specialist':>11}")
for s in SRC:
    sv = f"{spec[s]['final']:.4f}" + ('' if spec.get(s, {}).get('done') else '*') if s in spec else '   --'
    print(f"{s+'→all':8} {base_total[s]:9.4f} {ps25[s]:9.4f} {ps100[s]:10.4f} {sv:>11}")
print(f"{'MEAN':8} {base_mean:9.4f} {m25:9.4f} {m100:10.4f}")
