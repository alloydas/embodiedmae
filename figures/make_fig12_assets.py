"""Image assets for the SAGA-style Figures 1 and 2 (figures/make_fig12_saga.py lays them out).

Writes paper/figs/fig12_assets/:
  radar.png            trait-probe radar (paper Table 10, mean-pooled validation R^2): CropCode headline vs the
                       published encoders and the supervised reference; each axis scaled from the lowest to the
                       highest model on it, CropCode's R^2 printed at its axis end.
  tsne_sorghum.png     t-SNE of the headline models' mean-pooled validation features (2,250 plants each), coloured
  tsne_maize.png       by leaf count (sorghum) and plant height (maize).
  s1_*.png             one held-out sorghum test plant, view 00: the four streams, what the headline model sees under
                       its pretraining masking (mask ratio 0.8, Dirichlet split), and what it reconstructs.
  s2_*.png             the distilled generalist reading the photograph alone: its cloud.
  bars_distill.png     single-stream cloud Chamfer before / after distillation (paper Table 6).
  bars_rows.png        Chamfer on rows: zero-shot / isolated-trained / row fine-tune (paper Table 5).
  text.json            recipe and trait text for the panels (true and predicted).
CPU, a few minutes (two forward passes, two t-SNEs).
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import json, io
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import _plantviz as V
from _scene import png_array, png_cloud
from train_sorghum_4m import unpatchify, _unnorm_pix

OUT = R / 'paper/figs/fig12_assets'; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = '#1d221c', '#5f685c'
CS = {'rgb': '#c0392b', 'depth': '#b8860b', 'pc': '#2a9d8f', 'text': '#30638e', 'ours': '#2b3a67'}
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name, dpi=300):
    fig.savefig(OUT / name, dpi=dpi, transparent=False, facecolor='white', bbox_inches='tight', pad_inches=0.02); plt.close(fig)


# ---------------------------------------------------------------- radar (Table 10)
AXES = ['Sorghum\nheight', 'Sorghum\nleaf count', 'Sorghum\nbranch', 'Sorghum\nroll', 'Sorghum\nleaf length',
        'Maize\nheight', 'Maize\nleaf angle', 'Maize\ncurl', 'Maize\nstem radius']
TAB = {  # mean-pooled validation R^2, paper Table 10 (tab:e8)
    'DINOv2 ViT-B/14': [0.924, 0.928, 0.840, 0.573, 0.235, 0.561, 0.816, 0.072, 0.588],
    'MultiMAE-B': [0.922, 0.925, 0.833, 0.679, 0.232, 0.570, 0.783, 0.069, 0.442],
    'Point-MAE': [0.958, 0.963, 0.863, 0.716, 0.188, 0.509, 0.863, 0.199, 0.426],
    'EmbodiedMAE-B': [0.976, 0.979, 0.871, 0.765, 0.265, 0.528, 0.823, 0.139, 0.428],
    'Supervised (labels)': [0.973, 0.978, 0.924, 0.634, 0.271, 0.785, 0.935, 0.618, 0.860],
    'CropCode (no labels)': [0.983, 0.988, 0.912, 0.798, 0.371, 0.747, 0.967, 0.739, 0.841],
}
STYLE = {'DINOv2 ViT-B/14': ('#9aa5b1', '-'), 'MultiMAE-B': ('#c49a6c', '-'), 'Point-MAE': ('#7fb08a', '-'),
         'EmbodiedMAE-B': ('#b48ead', '-'), 'Supervised (labels)': ('#555555', '--'), 'CropCode (no labels)': (CS['ours'], '-')}
A = np.array(list(TAB.values())); lo, hi = A.min(0), A.max(0)
ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False); angc = np.r_[ang, ang[:1]]
fig = plt.figure(figsize=(2.6, 2.6)); ax = fig.add_subplot(projection='polar')
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
for name, v in TAB.items():
    r = 0.25 + 0.75 * (np.array(v) - lo) / np.maximum(hi - lo, 1e-9); r = np.r_[r, r[:1]]
    c, ls = STYLE[name]; ours = name.startswith('CropCode')
    ax.plot(angc, r, color=c, lw=1.8 if ours else 0.9, ls=ls, zorder=5 if ours else 2, label=name)
    if ours: ax.fill(angc, r, color=c, alpha=0.12, zorder=1)
ours_v = TAB['CropCode (no labels)']
ax.set_xticks(ang); ax.set_xticklabels([f'{a}\n{v:.2f}' for a, v in zip(AXES, ours_v)], fontsize=5.2, color=INK)
ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
ax.grid(color='#dddddd', lw=0.5); ax.spines['polar'].set_color('#cccccc')
ax.tick_params(axis='x', pad=1)
fig.legend(loc='lower center', ncol=2, fontsize=5.0, frameon=False, bbox_to_anchor=(0.5, -0.13), handlelength=1.8, columnspacing=0.8)
save(fig, 'radar.png')

# ---------------------------------------------------------------- t-SNE of the headline latents
from sklearn.manifold import TSNE
feat = pd.read_csv(R / 'data_split/features.csv')
for sp, f, tgt, lab, cmap in (('sorghum', 'outputs/_probe_cache/4m_pretrain_15k_d2_sink__checkpoint_epoch_1000__val__mean__seed0__rep1__g1.85.npz',
                               None, 'leaf count', 'Blues'),
                              ('maize', 'outputs/_probe_cache_maize/maize_maize_4m_pretrain_15k_d2__checkpoint_epoch_1000__val__mean__seed0__rep1.npz',
                               None, 'height (m)', 'Greens')):
    z = np.load(R / f, allow_pickle=True); X, plants = z['feats'], [str(p) for p in z['plants']]
    if sp == 'sorghum':
        m = dict(zip(feat.plant.astype(int), feat.n_leaves)); y = np.array([m[int(p.split('_')[-1])] for p in plants], float)
    else:
        h = {}
        for line in open('/work/mech-ai-scratch/alloy/Maize_1/plant_params.jsonl'):
            r = json.loads(line)
            if r['split'] == 'val': h[r['plant']] = r['params']['height']
        y = np.array([h[p] for p in plants], float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    E = TSNE(n_components=2, perplexity=40, init='pca', random_state=0).fit_transform(Xs)
    np.savez(OUT / f'tsne_{sp}.npz', E=E, y=y)
    fig, ax = plt.subplots(figsize=(0.95, 0.9))
    o = np.argsort(y); sc = ax.scatter(E[o, 0], E[o, 1], c=y[o], cmap=cmap, s=0.8, linewidths=0, vmin=np.percentile(y, 2), vmax=np.percentile(y, 98))
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=4.5, length=1.5); cb.outline.set_linewidth(0.3)
    cb.set_label(lab, fontsize=5, labelpad=1)
    ax.set_title(sp, fontsize=6.5, pad=1, color=INK)
    save(fig, f'tsne_{sp}.png')
    print('t-SNE', sp, len(y), flush=True)

# ---------------------------------------------------------------- Stage 1 / Stage 2: one sorghum validation plant
PLANT = 'Sorghum_10036_00'
ds = V.dataset('sorghum', 'test'); item = V.item_by_name(ds, PLANT, seed=0)   # held-out test plant (as the isolated teaser)
rgb, depth, pc, par, tv = (item[k][None] for k in range(5))
in_rgb = V.tensor_rgb(rgb[0]); in_dep = depth[0, 0].numpy()
fr = V.frame_of(pc[0].numpy())
(OUT / 's1_rgb.png').write_bytes(png_array(in_rgb)); (OUT / 's1_depth.png').write_bytes(png_array(V.depth_rgb(in_dep)))
(OUT / 's1_pc.png').write_bytes(png_cloud(pc[0].numpy(), 1.2, frame=fr))
model, cfg, ep = V.load_model('sorghum', 'headline')
torch.manual_seed(3)
with torch.no_grad():
    _, _, (pr, pd_, ppc, ppar), (mr, md, mp, mt) = model(rgb, depth, pc, par, tv, mask_ratio=0.8)
vis = {k: int((1 - m[0]).sum()) for k, m in (('rgb', mr), ('depth', md), ('pc', mp), ('text', mt))}
print('visible tokens', vis, flush=True)


def masked(img, m, grey=0.82):
    out = img.copy(); hid = m[0].view(14, 14).numpy() > 0.5
    for i in range(14):
        for j in range(14):
            if hid[i, j]: out[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = grey
    return out


(OUT / 's1_rgb_masked.png').write_bytes(png_array(masked(in_rgb, mr)))
(OUT / 's1_depth_masked.png').write_bytes(png_array(masked(V.depth_rgb(in_dep), md)))
# the visible cloud tokens, drawn with display groups of the same count (farthest points, nearest-centre groups)
P = pc[0].numpy(); rng = np.random.default_rng(0); cent = [rng.integers(len(P))]; dmin = np.linalg.norm(P - P[cent[0]], axis=1)
for _ in range(195):
    cent.append(int(dmin.argmax())); dmin = np.minimum(dmin, np.linalg.norm(P - P[cent[-1]], axis=1))
grp = np.linalg.norm(P[:, None] - P[cent][None], axis=2).argmin(1)
keep = rng.choice(196, vis['pc'], replace=False)
(OUT / 's1_pc_masked.png').write_bytes(png_cloud(P[np.isin(grp, keep)], 1.2, frame=fr))
rec_rgb = V.tensor_rgb(unpatchify(_unnorm_pix(model, pr, rgb), model.patch_size, 3, model.img_size)[0])
rec_dep = unpatchify(pd_, model.patch_size, 1, model.img_size)[0, 0].float().numpy()
(OUT / 's1_rec_rgb.png').write_bytes(png_array(rec_rgb))
def pred_depth_img(pn):
    """predicted min-max depth -> display; background where the PREDICTION is near 0 (not the true silhouette)"""
    d = pn * (in_dep.max() - in_dep.min()) + in_dep.min(); d[pn < 0.05] = 0.0; return V.depth_rgb(d)


(OUT / 's1_rec_depth.png').write_bytes(png_array(pred_depth_img(rec_dep)))
(OUT / 's1_rec_pc.png').write_bytes(png_cloud(ppc[0].float().numpy(), 1.2, frame=fr))
rec_t = V.decode_recipe('sorghum', par[0].numpy(), tv[0].numpy()); rec_p = V.decode_recipe('sorghum', ppar[0].clamp(0, 1).numpy(), tv[0].numpy())
FLD = ('start', 'len', 'roll', 'width')
text = dict(plant=PLANT, visible=vis, recipe_true=V.recipe_lines('sorghum', rec_t, max_leaves=3, fields=FLD),
            recipe_rec=V.recipe_lines('sorghum', rec_p, max_leaves=3, fields=FLD),
            chamfer_rec=V.chamfer(ppc[0], pc[0]))
gen, gcfg, gep = V.load_model('sorghum', 'generalist')
g = V.generate(gen, item, ['rgb'], seed=0)
(OUT / 's2_gen_pc.png').write_bytes(png_cloud(g['pc'], 1.2, frame=fr))
(OUT / 's2_gen_depth.png').write_bytes(png_array(pred_depth_img(g['depth'])))
text['recipe_from_rgb'] = V.recipe_lines('sorghum', V.decode_recipe('sorghum', g['par'], tv[0].numpy(), predicted=True), max_leaves=3, fields=FLD)
text['chamfer_from_rgb'] = V.chamfer(g['pc'], pc[0]); text['chamfer_from_rgb_headline'] = None
json.dump(text, open(OUT / 'text.json', 'w'), indent=1, default=float)
print('stage assets', text['chamfer_rec'], text['chamfer_from_rgb'], flush=True)

# ---------------------------------------------------------------- small result charts
fig, axs = plt.subplots(1, 2, figsize=(1.5, 0.82))
for ax, (sp, before, after) in zip(axs, (('sorghum', [2.27, 0.90, 5.48], [0.88, 0.63, 1.39]), ('maize', [11.3, 1.26, 6.43], [0.77, 0.58, 0.83]))):
    x = np.arange(3)
    ax.bar(x - 0.19, before, 0.36, color='#d9d4c7', edgecolor='none', label='before')
    ax.bar(x + 0.19, after, 0.36, color=CS['ours'], edgecolor='none', label='after')
    ax.set_xticks(x); ax.set_xticklabels(['RGB', 'depth', 'cloud'], fontsize=5); ax.set_yscale('log'); ax.tick_params(labelsize=4.5, length=1.5)
    ax.set_title(sp, fontsize=5.5, pad=1); ax.set_ylim(0.3, 20)
axs[0].set_ylabel('Chamfer ×1e-3', fontsize=5); axs[1].legend(fontsize=4.5, frameon=False, loc='upper right', handlelength=1)
fig.tight_layout(pad=0.2, w_pad=0.4); save(fig, 'bars_distill.png')
fig, axs = plt.subplots(1, 2, figsize=(1.5, 0.82))
for ax, (sp, v) in zip(axs, (('sorghum', [230.5, 133.8, 30.7]), ('maize', [29.6, 29.0, 4.0]))):
    ax.bar(np.arange(3), v, 0.6, color=['#d9d4c7', '#a7b3c9', CS['ours']], edgecolor='none')
    ax.set_xticks(range(3)); ax.set_xticklabels(['zero-\nshot', 'isolated\ntrained', 'row\nfine-tune'], fontsize=4.5)
    ax.tick_params(labelsize=4.5, length=1.5); ax.set_title(sp, fontsize=5.5, pad=1)
    for i, vv in enumerate(v): ax.text(i, vv, f'{vv:g}', ha='center', va='bottom', fontsize=4.3)
    ax.set_ylim(0, max(v) * 1.22)
axs[0].set_ylabel('Chamfer on rows ×1e-4 m²', fontsize=5)
fig.tight_layout(pad=0.2, w_pad=0.4); save(fig, 'bars_rows.png')
print('done', OUT)
