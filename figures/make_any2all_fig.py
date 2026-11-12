"""Any-to-all figures (replace the paper's fig:any2any): one held-out test plant per species, the distilled generalist
given the IMAGE alone, a 2 x 2 grid per species: the input image | the generated point cloud / depth image /
procedural parameters, with a drawn divider between input and outputs. Outputs are drawn over the ground truth (grey);
the parameters as the stem-and-leaves schematic plus height and leaf count against the truth. 9 pt labels, Okabe-Ito
colours, column width (3.25 in). Cherry-picked, and the caption says so:
  candidates = test plants whose evaluated view looks within 25 deg of the horizon (readable silhouettes), ranked by
  the photograph-to-cloud Chamfer of the full test evaluation (reports/quant_*generalist_rgb_d2_test.json, the same
  model, view and seed); the best --n per species are regenerated (--dump, CPU job) and one is picked by eye (--sheet).
  python figures/make_any2all_fig.py --dump
  python figures/make_any2all_fig.py --sheet
  python figures/make_any2all_fig.py --render --sorghum Sorghum_X_vv --maize plant_Y_vv
      -> paper/figs/fig_any2all_{sorghum,maize}.{pdf,png} (+ copies in paper/overleaf/Figures/)
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import argparse, csv, gzip, json, pickle, shutil
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import _scene
from _scene import png_points, cloud_frame
import _fig12_kit as K

OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/any2all_dump.pkl')
REPORT = {'sorghum': 'reports/quant_generalist_rgb_d2_test.json', 'maize': 'reports/quant_maize_generalist_rgb_d2_test.json'}
FS = 9.0
INK, MUTED, GREY = '#1d221c', '#5f685c', '#cfd0cc'
C_IN, C_OUT = '#E69F00', '#0072B2'          # Okabe-Ito orange (input) and blue (outputs)
VIEW = dict(elev=5.0, azim=15.0)


def elevation(sp, name):
    plant, view = name.rsplit('_', 1); view = int(view)
    if sp == 'sorghum': return float(np.degrees(np.arcsin(0.9 - 0.2 * view)))
    return ELEV.get((plant, view), 99.0)


ELEV = {}
for r in csv.DictReader(gzip.open(R / 'data_split/maize_view_elevation.csv.gz', 'rt')):
    if r['split'] == 'test': ELEV[(r['plant'], int(r['view']))] = float(r['elev_deg'])


def candidates(sp, n):
    d = json.load(open(R / REPORT[sp])); ch = np.array(d['per_sample']['model']); names = d['plants']
    ok = np.array([abs(elevation(sp, nm)) <= 25 for nm in names])
    order = [i for i in np.argsort(ch) if ok[i]][:n]
    print(sp, f'{ok.sum()} of {len(names)} test plants within 25 deg of the horizon; median Chamfer {np.median(ch) * 1e3:.2f}e-3; '
          f'picked range {ch[order[0]] * 1e3:.2f}-{ch[order[-1]] * 1e3:.2f}e-3', flush=True)
    return [names[i] for i in order], float(np.median(ch))


def dump(n):
    import torch, _plantviz as V
    torch.set_num_threads(8)
    D = {}
    for sp in ('sorghum', 'maize'):
        names, med = candidates(sp, n)
        ds = V.dataset(sp, 'test'); m, cfg, ep = V.load_model(sp, 'generalist')
        D[sp] = dict(median=med, items={})
        for nm in names:
            it = V.item_by_name(ds, nm, seed=0)
            g = V.generate(m, it, ['rgb'], seed=0)
            rec_t = V.decode_recipe(sp, it[3].numpy(), it[4].numpy()); rec_p = V.decode_recipe(sp, g['par'].numpy(), it[4].numpy(), predicted=True)
            pack = lambda r: dict(height=float(r['height']), n=int(r['n']), leaves=[{k: float(v) for k, v in l.items()} for l in r['leaves']])
            D[sp]['items'][nm] = dict(rgb=V.tensor_rgb(it[0]).astype(np.float32), depth=it[1][0].numpy(), pc=it[2].numpy(), gen_pc=g['pc'],
                                      gen_depth=g['depth'], rec_true=pack(rec_t), rec_pred=pack(rec_p), chamfer=V.chamfer(g['pc'], it[2]),
                                      elev=elevation(sp, nm), epoch=ep)
            print(sp, nm, f"elev {elevation(sp, nm):+.0f}", f"chamfer {D[sp]['items'][nm]['chamfer'] * 1e3:.2f}e-3",
                  f"height {rec_p['height']:.2f} / {rec_t['height']:.2f}", f"leaves {rec_p['n']} / {rec_t['n']}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(D, open(str(OUT) + '.tmp', 'wb')); pathlib.Path(str(OUT) + '.tmp').replace(OUT)


# ====================================================================== drawing
def img_ax(fig, box, W, H):
    ax = fig.add_axes([box[0] / W, box[1] / H, box[2] / W, box[3] / H]); ax.set_xticks([]); ax.set_yticks([])
    return ax


def depth_display(it):
    """the generated depth (min-max normalised by the model) mapped back to the true depth's range for display; background
    where the model predicts near zero; the same colour scale as the truth"""
    d = it['depth']; fg = d > 0; lo, hi = np.percentile(d[fg], 1), np.percentile(d[fg], 99)
    z = it['gen_depth'] * (d.max() - d.min()) + d.min(); z = z.copy(); z[it['gen_depth'] < 0.05] = 0
    return K.depth_rgb(z, lo, hi)


def crop_square(arr, fgmask, pad=6):
    ys, xs = np.where(fgmask)
    if not len(ys): return arr
    c0, c1, d0, d1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    side = max(c1 - c0, d1 - d0); cy, cx = (c0 + c1) // 2, (d0 + d1) // 2
    y0 = int(np.clip(cy - side // 2, 0, max(0, arr.shape[0] - side))); x0 = int(np.clip(cx - side // 2, 0, max(0, arr.shape[1] - side)))
    return arr[y0:y0 + side, x0:x0 + side]


def render(sp, it, out_stem):
    W = 3.25; c = 1.43; gx = 0.30; gy = 0.14; tag = 0.20; ttl = 0.19
    x0 = (W - (2 * c + gx)) / 2
    H = tag + ttl + c + gy + ttl + c + 0.02
    fig = plt.figure(figsize=(W, H))
    xs = (x0, x0 + c + gx)
    ys = (H - tag - ttl - c, H - tag - ttl - c - gy - ttl - c)
    cells = {(0, 0): 'in', (0, 1): 'pc', (1, 0): 'depth', (1, 1): 'par'}
    titles = {'in': 'Image', 'pc': 'Point cloud', 'depth': 'Depth image', 'par': 'Procedural parameters'}
    fg = it['depth'] > 0
    halo = [pe.withStroke(linewidth=2.6, foreground='white')]
    for (r, cc), key in cells.items():
        x, y = xs[cc], ys[r]
        col = C_IN if key == 'in' else C_OUT
        fig.text((x + c / 2) / W, (y + c + 0.04) / H, titles[key], fontsize=FS, ha='center', va='bottom', color=INK)
        ax = img_ax(fig, (x, y, c, c), W, H)
        if key == 'in':
            ax.imshow(crop_square(it['rgb'], fg))
        elif key == 'depth':
            ax.imshow(crop_square(depth_display(it), fg))
        elif key == 'pc':
            fr = cloud_frame(it['pc'], **VIEW)
            P, Q = it['gen_pc'], it['pc']; r_, u_, d_ = _scene.view_basis(**VIEW)
            ax.scatter(Q @ r_, Q @ u_, s=0.35, c=GREY, lw=0, rasterized=True)
            h = (P @ u_ - (P @ u_).min()) / max(np.ptp(P @ u_), 1e-9); o = np.argsort(P @ d_)
            ax.scatter((P @ r_)[o], (P @ u_)[o], s=0.30, c=_scene.PLANT_CMAP(0.15 + 0.85 * h[o]), lw=0, rasterized=True)
            (cu, cv), hw = fr; hw *= 1.06; ax.set_xlim(cu - hw, cu + hw); ax.set_ylim(cv - hw * 0.92, cv + hw * 1.20); ax.set_aspect('equal')
            ax.text(0.04, 0.97, f"Chamfer {it['chamfer'] * 1e3:.2f}", transform=ax.transAxes, fontsize=FS, ha='left', va='top', color=INK, path_effects=halo)
        else:
            Ht, Hp = it['rec_true']['height'], it['rec_pred']['height']
            K.glyph(ax, sp, it['rec_true'], GREY, stem=GREY, lw_scale=0.9)
            K.glyph(ax, sp, it['rec_pred'], C_OUT, lw_scale=0.5)
            bb = ax.dataLim; w = max(abs(bb.x0), abs(bb.x1)); y0_ = min(bb.y0, 0.0)
            st = max(2 * w, bb.y1 - y0_) * 1.36                # square cell; the top ~quarter stays free for the text
            ax.set_xlim(-st / 2, st / 2); ax.set_ylim(y0_ - 0.02 * st, y0_ + 0.98 * st); ax.set_aspect('equal')
            txt = f"height {Hp:.2f} m ({Ht:.2f})\nleaves {it['rec_pred']['n']} ({it['rec_true']['n']})"
            ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=FS, ha='left', va='top', color=INK, linespacing=1.1, path_effects=halo)
        for s_ in ax.spines.values(): s_.set_edgecolor(col); s_.set_linewidth(1.8 if key == 'in' else 1.1)
    # the divider: a dashed L that closes off the input cell; tags above each side
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.set_axis_off(); ov.patch.set_alpha(0)
    xm = x0 + c + gx / 2; ym = ys[0] - gy / 2 - 0.01
    ov.plot([xm, xm, 0.0], [H, ym, ym], color=INK, lw=1.2, ls=(0, (4, 2.5)))
    ov.text(x0 + c / 2, H - tag / 2, 'INPUT', fontsize=FS, color=C_IN, fontweight='bold', ha='center', va='center')
    ov.text(xs[1] + c / 2, H - tag / 2, 'OUTPUT', fontsize=FS, color=C_OUT, fontweight='bold', ha='center', va='center')
    out = R / 'paper/figs' / out_stem
    fig.savefig(f'{out}.pdf', dpi=300); fig.savefig(f'{out}.png', dpi=220); plt.close(fig)
    shutil.copy(f'{out}.pdf', R / 'paper/overleaf/Figures' / f'{out_stem}.pdf')
    print('wrote', f'{out}.pdf', f'{W} x {H:.2f} in', flush=True)


def sheet(D):
    for sp in ('sorghum', 'maize'):
        its = D[sp]['items']; n = len(its)
        fig, axs = plt.subplots(n, 4, figsize=(6.4, 1.55 * n))
        for r, (nm, it) in enumerate(its.items()):
            fg = it['depth'] > 0
            axs[r, 0].imshow(crop_square(it['rgb'], fg)); axs[r, 0].set_ylabel(f"{nm}\n{it['elev']:+.0f} deg", fontsize=7)
            axs[r, 1].imshow(crop_square(depth_display(it), fg))
            fr = cloud_frame(it['pc'], **VIEW); rr, uu, dd = _scene.view_basis(**VIEW)
            axs[r, 2].scatter(it['pc'] @ rr, it['pc'] @ uu, s=0.2, c=GREY, lw=0); axs[r, 2].scatter(it['gen_pc'] @ rr, it['gen_pc'] @ uu, s=0.15, c='#2f7d32', lw=0)
            (cu, cv), hw = fr; axs[r, 2].set_xlim(cu - hw, cu + hw); axs[r, 2].set_ylim(cv - hw, cv + hw); axs[r, 2].set_aspect('equal')
            axs[r, 2].set_title(f"{it['chamfer'] * 1e3:.2f}e-3", fontsize=7)
            K.glyph(axs[r, 3], sp, it['rec_true'], GREY, stem=GREY, lw_scale=0.9); K.glyph(axs[r, 3], sp, it['rec_pred'], C_OUT, lw_scale=0.5)
            axs[r, 3].set_aspect('equal'); axs[r, 3].set_title(f"h {it['rec_pred']['height']:.2f}/{it['rec_true']['height']:.2f} n {it['rec_pred']['n']}/{it['rec_true']['n']}", fontsize=7)
            for a in axs[r]: a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(pad=0.2)
        fig.savefig(f'/tmp/claude-490224/-work-mech-ai-scratch-alloy-embodiedmae/08a82cbc-d3db-4182-8667-cb84feb249c4/scratchpad/any2all_{sp}.png', dpi=90)
        plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', action='store_true'); ap.add_argument('--sheet', action='store_true'); ap.add_argument('--render', action='store_true')
    ap.add_argument('--n', type=int, default=15); ap.add_argument('--sorghum'); ap.add_argument('--maize')
    a = ap.parse_args()
    if a.dump: dump(a.n)
    if a.sheet or a.render:
        D = pickle.load(open(OUT, 'rb'))
        if a.sheet: sheet(D)
        if a.render:
            for sp in ('sorghum', 'maize'):
                pick = getattr(a, sp)
                if pick: render(sp, D[sp]['items'][pick], f'fig_any2all_{sp}')
