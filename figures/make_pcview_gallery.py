"""Partial-cloud gallery: for a few val plants, the input cloud under each camera condition,
the species' E1 reconstruction from it (masked regime, mask ratio 0.80, as the evals score
it) and the reference cloud. One PNG per plant -> reports/vis_pcview_<species>/.
  maize   (eval/pc_view_eval_maize.py): clouds of 1, 3 or 10 cameras fused in camera 00's frame;
          reference = the 10-camera cloud.
  sorghum (eval/pc_view_eval.py): the complete cloud masked to the points cameras 00 / 00-02
          can see (outputs/_pcview_masks); reference = the complete cloud.

usage: python figures/make_pcview_maize_gallery.py [--species maize|sorghum] [--n 3] [--device cpu]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'eval'))
import importlib.util
def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'eval' / f'{name}.py')
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument('--species', default='maize', choices=['maize', 'sorghum'])
ap.add_argument('--run', default=None); ap.add_argument('--ckpt', default='checkpoints/checkpoint_epoch_1000.pth')
ap.add_argument('--data-root', default=None)
ap.add_argument('--split', default='val'); ap.add_argument('--n', type=int, default=6); ap.add_argument('--pool', type=int, default=40)
ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=None)
a = ap.parse_args()
out = Path(a.out or REPO / 'reports' / f'vis_pcview_{a.species}'); out.mkdir(parents=True, exist_ok=True)
if a.species == 'maize':
    PV = _mod('pc_view_eval_maize')
    from embodied_mae_4m_maize import N_PARAMS, MAX_LEAVES
    run = a.run or 'maize_4m_pretrain_15k_d2'
    model, cfg, epoch = PV.LP.build_model(REPO / 'outputs' / run, a.ckpt, a.device)
    rgb_file, schema = PV.run_data(run)
    ds = PV.PCViewSetMaize(a.data_root or '/work/mech-ai-scratch/alloy/Maize_1', a.split, 8192, 224, MAX_LEAVES,
                           a.seed, limit=a.pool, rgb_file=rgb_file, param_schema=schema)
    NAMES = {'1view': '1 camera', '3view': '3 cameras', 'full': '10 cameras'}; REF = '10-camera cloud'
else:
    PV = _mod('pc_view_eval')
    from embodied_mae_4m import N_PARAMS
    MAX_LEAVES = 24
    run = a.run or '4m_pretrain_15k_d2_sink'
    model, cfg, epoch = PV.LP.build_model(REPO / 'outputs' / run, a.ckpt, a.device)
    rgb_file, gain = PV.run_rgb(run)
    masks = PV.PM.Masks(a.split, PV.MASK_VIEWS)
    ds = PV.PCViewSet(a.data_root or PV.PM.DATA_ROOT, a.split, masks, int(cfg.get('num_points', 8196)), 224, 24,
                      a.seed, limit=a.pool, rgb_file=rgb_file, rgb_gain=gain)
    NAMES = {'1view': 'seen by camera 00', '3view': 'seen by cameras 00-02', 'full': 'complete cloud'}; REF = 'complete cloud'
model.eval(); mr = float(cfg.get('mask_ratio', 0.8))
n_text = 1 + int(cfg.get('max_leaves', MAX_LEAVES))
visible = tuple(m for m in model.active_modalities if m != 'text')

def cham(p, g):
    d = torch.cdist(p[None], g[None])[0]
    return float(d.min(1).values.pow(2).mean() + d.min(0).values.pow(2).mean())

LIM = {}
def scat(ax, P, title, col):
    P = P.cpu().numpy()
    lo, hi = LIM['lo'], LIM['hi']; c = (lo + hi) / 2; h = (hi - lo).max() / 2 * 1.05
    ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=0.8, c=col, alpha=0.75, linewidths=0)
    ax.view_init(elev=18, azim=-35); ax.set_title(title, fontsize=8)
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[2] - h, c[2] + h); ax.set_zlim(c[1] - h, c[1] + h)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()

def to_full(b, c, X):
    """Map points from condition c's normalised frame into the complete/10-camera frame
    (both frames normalise the same raw ground truth: an exact affine map)."""
    g, f = b[f'gt_{c}'].numpy(), b['gt_full'].numpy()
    k = f.std() / g.std(); return X * k + (f.mean(0) - k * g.mean(0))

def view(ax, P, col, title, azim, elev=12, cmap=None, vmin=None, vmax=None, s=0.9):
    lo, hi = LIM['lo'], LIM['hi']; c = (lo + hi) / 2; h = (hi - lo).max() / 2 * 1.05
    sc = ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=s, c=col, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, linewidths=0)
    ax.view_init(elev=elev, azim=azim); ax.set_title(title, fontsize=8)
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[2] - h, c[2] + h); ax.set_zlim(c[1] - h, c[1] + h)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); return sc

# choose plants spanning the single-camera coverage range of a pool
pool = [ds[i] for i in range(len(ds))]
order = sorted(range(len(pool)), key=lambda i: pool[i]['cov_1view'])
pick = [order[round(q * (len(order) - 1))] for q in np.linspace(0, 1, a.n)]
print('picked', [(pool[i]['plant'], round(pool[i]['cov_1view'], 2)) for i in pick])
from scipy.spatial import cKDTree
TAU = 0.02          # normalised units: "near" for the coverage and miss maps
CATC = np.array(['#2f6b2a', '#b8860b', '#c0392b'])
for i in pick:
    b = pool[i]
    rgb, depth = b['rgb'][None].to(a.device), b['depth'][None].to(a.device)
    params = torch.zeros(1, n_text, N_PARAMS, device=a.device)
    G = b['gt_full'].numpy(); LIM['lo'], LIM['hi'] = G.min(0), G.max(0)
    preds, chs = {}, {}
    for c in ['1view', '3view', 'full']:
        pc, gt = b[f'pc_{c}'].to(a.device), b[f'gt_{c}'].to(a.device)
        torch.manual_seed(a.seed + 1)
        with torch.no_grad():
            enc = model.forward_encoder_select(rgb, depth, pc[None], params, visible=visible, source_mask_ratio=mr)
            preds[c] = model.forward_decoder(enc[0], *enc[5:9], *enc[9:13])[2][0].float()
        chs[c] = cham(preds[c], gt)
    tag = f'{a.species}_{b["plant"]}'
    # (1) inputs / reconstructions / reference
    fig = plt.figure(figsize=(10.5, 10.5))
    for r, c in enumerate(['1view', '3view', 'full']):
        LIM['lo'], LIM['hi'] = G.min(0), G.max(0)
        view(fig.add_subplot(3, 3, 3 * r + 1, projection='3d'), to_full(b, c, b[f'pc_{c}'].numpy()), '#2f6b2a', f'input: {NAMES[c]}\n{b[f"cov_{c}"]:.0%} of the {REF}', -35, 18)
        view(fig.add_subplot(3, 3, 3 * r + 2, projection='3d'), to_full(b, c, preds[c].numpy()), '#b8860b', f'E1 reconstruction from {NAMES[c]}\nChamfer to the {REF} {chs[c] * 1e3:.2f}e-3', -35, 18)
        view(fig.add_subplot(3, 3, 3 * r + 3, projection='3d'), G, '#1f6f8b', f'reference: {REF}', -35, 18)
    fig.suptitle(f'{a.species} {b["plant"]} ({a.split}), E1 epoch {epoch}: input cloud = {" / ".join(NAMES.values())}; RGB and depth of camera 00; mask ratio {mr}', fontsize=9)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93, wspace=0, hspace=0.08); f = out / f'pcview_{tag}.png'; fig.savefig(f, dpi=150); plt.close(fig); print('wrote', f)
    # (2) coverage map (two angles) + miss maps (reference coloured by distance to each reconstruction)
    p1, p3 = to_full(b, '1view', b['pc_1view'].numpy()), to_full(b, '3view', b['pc_3view'].numpy())
    cat = np.where(cKDTree(p1).query(G)[0] < TAU, 0, np.where(cKDTree(p3).query(G)[0] < TAU, 1, 2))
    fig = plt.figure(figsize=(12, 8.4))
    lab = {'maize': ['in camera 00\'s cloud', 'added by cameras 01-02', 'only in cameras 03-09'],
           'sorghum': ['seen by camera 00', 'seen by camera 01 or 02 only', 'hidden from all three']}[a.species]
    share = f'{NAMES["1view"]}: {b["cov_1view"]:.0%} of the {REF}; {NAMES["3view"]}: {b["cov_3view"]:.0%}'
    for j, (az, nm) in enumerate([(-35, 'oblique'), (55, 'turned 90°'), (-125, 'turned 180°')]):
        view(fig.add_subplot(2, 3, 1 + j, projection='3d'), G, CATC[cat], f'{REF}, {nm}', az, 15)
    vmax = 0.08
    for j, c in enumerate(['1view', '3view', 'full']):
        d = cKDTree(to_full(b, c, preds[c].numpy())).query(G)[0]
        sc = view(fig.add_subplot(2, 3, 4 + j, projection='3d'), G, d, f'miss map, E1 from {NAMES[c]}\n{np.mean(d > TAU):.0%} of the {REF} farther than {TAU} from it', -35, 15, cmap='inferno_r', vmin=0, vmax=vmax)
    cax = fig.add_axes([0.35, 0.05, 0.3, 0.012]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.set_label('distance to the nearest reconstructed point (normalised units)', fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.suptitle(f'{a.species} {b["plant"]} ({a.split}); {share}\n'
                 f'Top: {REF} coloured green = {lab[0]}, gold = {lab[1]}, red = {lab[2]} (nearest input point within {TAU})\n'
                 f'Bottom: {REF} coloured by distance to E1\'s reconstruction from each input (dark = missed)', fontsize=8.5)
    fig.subplots_adjust(left=0, right=1, bottom=0.09, top=0.88, wspace=0, hspace=0.12); f = out / f'pcmap_{tag}.png'; fig.savefig(f, dpi=140); plt.close(fig); print('wrote', f)
