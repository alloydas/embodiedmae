"""Fig. 1 teaser, maize only, four streams in -> four streams out (user, 2026-11-14: "Make it only for maize and use 4
modality to 4 modality all"; chosen setup: "Masked, all 4 in").

One isolated held-out maize test plant (plant_0106, view 01, the plant of the current teaser, broadside). All four streams
go in with 80 % of their tokens hidden (Dirichlet split, as in Stage 1); the maize headline model (maize_4m_pretrain_15k_d2,
epoch 1000) rebuilds all four. Data: the Stage-1 items of the _fig12_kit dump (the same pass as Fig. 2's Stage 1 panel).
UGG flow layout: the four masked inputs in a column (visible-token counts beside them), one wire per stream (paper
palette) into one CropCode box, four wires out to the rebuilt streams with their numbers. The procedural parameters are
drawn in 3D (figures/_proc3d.py): input = the visible leaf tokens only (purple); output = the written plant, textured.
No ground truth is drawn in the images (user, 2026-11-14); the true height and leaf count are in the text.
Built on _scene.Scene like Figs. 2-5. main.tex is NOT touched.
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=4 --mem=24G --output=logs/%x_%j.out \\
         --wrap "python figures/make_teaser_4to4_tikz.py"
-> paper/overleaf/Figures/fig_teaser_4to4_tikz.tex + Figures/fig1b/, preview paper/figs/ugg_preview/fig_teaser_4to4.{pdf,png}
"""
import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, matplotlib
matplotlib.use('Agg')
from _scene import Scene, cloud_frame
import _fig12_kit as K
import _proc3d as P3
import ugg_style as U

FIG = R / 'paper/overleaf/Figures'
ITEM = 'plant_0106_01'
E, Fi, I = U.STREAM, U.STREAM_FILL, U.STREAM_INK
INK, MUTED, GREY = '#1a1a1a', '#5f5f5f', '#8c8c8c'
F = 8.5
V = dict(elev=10.0, azim=20.0)


def pred_depth(d):
    pn = d['rec']['depth']; z = pn * (d['depth'].max() - d['depth'].min()) + d['depth'].min(); z = z.copy(); z[pn < 0.05] = 0; return z


if __name__ == '__main__':
    d = K.load_dump()['stage']['maize'][ITEM]
    rt, rp = d['recipe_true'], d['rec']['recipe']
    vis_t = [m < 0.5 for m in d['mask']['text']]                        # token 0 = plant, 1.. = leaves
    rv = dict(rt, leaves=[l for i, l in enumerate(rt['leaves']) if 1 + i < len(vis_t) and vis_t[1 + i]])
    W, H = 6.875, 2.50
    S = Scene(W, H, 'Fig 1 teaser maize 4 to 4')
    t, g = 0.45, 0.05
    x_lab, x_in = 1.46, 1.50
    x_mod, w_mod = 2.58, 1.50
    x_out = 4.66; x_ann = x_out + t + 0.10
    ys = [H - 0.30 - t - k * (t + g) for k in range(4)]                  # bottom tile ends 0.25 above the figure's foot                  # top to bottom: image, depth, cloud, parameters
    yc = [y + t / 2 for y in ys]
    fr = cloud_frame(d['pc'], **V); inner = t - 0.04
    n_tok = len(vis_t)
    streams = ['rgb', 'depth', 'pc', 'text']
    names = {'rgb': 'image', 'depth': 'depth', 'pc': 'point cloud', 'text': 'parameters'}
    vis_txt = {'rgb': f"{d['visible']['rgb']} of 196 patches visible", 'depth': f"{d['visible']['depth']} of 196 patches visible",
               'pc': f"{d['visible']['pc']} of 196 groups visible", 'text': f"{d['visible']['text']} of {n_tok} tokens visible"}
    ins = {'rgb': K.tile_img(K.masked(d['rgb'], d['mask']['rgb'])),
           'depth': K.tile_img(K.masked(K.depth_rgb(d['depth']), d['mask']['depth'])),
           'pc': K.tile_masked_cloud(d['pc'], d['pc_centers'], d['mask']['pc'], fr, inner, s=0.30, **V),
           'text': P3.png([(rv, P3.LEAF_C, 1.0)], 'maize', side=inner, dpi=600)}   # the visible tokens only: no ground truth drawn
    outs = {'rgb': K.tile_img(np.clip(d['rec']['rgb'], 0, 1)), 'depth': K.tile_depth(pred_depth(d)),
            'pc': K.tile_cloud(d['rec']['pc'], fr, inner, s=0.30, **V),
            'text': P3.png([(rp, P3.TEXTURE, 1.0)], 'maize', side=inner, dpi=600)}   # textured, no ground truth in the image
    ann = {'rgb': 'image rebuilt', 'depth': 'depth rebuilt', 'pc': f"cloud rebuilt\nChamfer {d['rec']['chamfer'] * 1e3:.2f}×10⁻³",
           'text': f"height {rp['height']:.2f} m (true {rt['height']:.2f})\n{rp['n']} leaves (true {rt['n']})"}
    # headers
    S.text(0.08, H - 0.12, 'INPUT', F, color=INK, bold=True, va='center')
    S.text(0.62, H - 0.12, 'all four streams, 80 % of tokens hidden', F, color=MUTED, va='center')
    S.text(x_out, H - 0.12, 'OUTPUT', F, color=INK, bold=True, va='center')
    S.text(x_out + 0.66, H - 0.12, 'every stream rebuilt', F, color=MUTED, va='center')
    # model
    yb, yt = ys[-1], ys[0] + t
    import make_ugg_figs as UG
    S.image(x_mod, yb, w_mod, yt - yb, UG._band_png(), 'model band'); S.rect(x_mod, yb, w_mod, yt - yb, GREY, lw=1.0, r=0.08)
    cx = x_mod + w_mod / 2; cy = (yb + yt) / 2
    S.text(cx, cy + 0.36, 'CropCode', F + 2.5, color=INK, ha='center', va='center', bold=True)
    S.text(cx, cy + 0.10, 'one model, four streams', F, color=MUTED, ha='center', va='center')
    S.text(cx, cy - 0.20, 'any subset in', F, color=INK, ha='center', va='center')
    S.text(cx, cy - 0.40, 'every stream out', F, color=INK, ha='center', va='center')
    S.text(0.08, 0.07, 'held-out maize test plant · the headline model, 80 % of all tokens hidden',
           F - 1.5, color=MUTED, va='center')
    S.text(W - 0.04, 0.07, 'kilobytes  ·  editable  ·  re-renderable', F - 1.0, color=I['text'], ha='right', va='center')
    for k, st in enumerate(streams):
        y = ys[k]
        S.text(x_lab, yc[k] + 0.07, names[st], F, color=I[st], ha='right', va='center', bold=True)
        S.text(x_lab, yc[k] - 0.09, vis_txt[st], F - 1.5, color=MUTED, ha='right', va='center')
        S.rect(x_in, y, t, t, E[st], lw=0.9, fill=Fi[st], r=0.04, dash=True)
        S.image(x_in + 0.02, y + 0.02, inner, inner, ins[st], f'input {st}')
        S.rect(x_out, y, t, t, E[st], lw=0.9, fill=Fi[st], r=0.04)
        S.image(x_out + 0.02, y + 0.02, inner, inner, outs[st], f'output {st}')
        ym = yb + 0.25 + (3 - k) * (yt - yb - 0.50) / 3                  # wires fan in and out of the box
        S.arrow((x_in + t + 0.02, yc[k]), (x_mod - 0.02, ym), color=E[st], lw=1.6, curve=0.0 if abs(ym - yc[k]) < 0.05 else (0.15 if ym < yc[k] else -0.15), ms=7)
        S.arrow((x_mod + w_mod + 0.02, ym), (x_out - 0.02, yc[k]), color=E[st], lw=1.6, curve=0.0 if abs(ym - yc[k]) < 0.05 else (0.15 if ym > yc[k] else -0.15), ms=7)
        S.text(x_ann, yc[k], ann[st], F - 0.5 if st != 'text' else F, color=I[st] if st == 'text' else MUTED, va='center', bold=st == 'text', ls=1.15)
    S.render(R / 'paper/figs/ugg_preview/fig_teaser_4to4')
    S.render_tikz(FIG / 'fig_teaser_4to4_tikz.tex', FIG / 'fig1b', 'Figures/fig1b')
    print('wrote', FIG / 'fig_teaser_4to4_tikz.tex', 'height', H, 'in =', round(H * 2.54, 2), 'cm;', ITEM, 'rec height', round(rp['height'], 3),
          'true', round(rt['height'], 3), 'leaves', rp['n'], rt['n'], 'chamfer e-3', round(d['rec']['chamfer'] * 1e3, 2), flush=True)
