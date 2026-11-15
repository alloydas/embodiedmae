"""Fig. 1 teaser as a UGG-style flow diagram (spec: figures/fig_teaser_flow_SPEC.md, requested 2026-11-14).

Three zones, two species rows (sorghum on top, maize below) sharing one model box:
  INPUT   per species the occluded row as the rover captures it: the row photograph (the current teaser's tile, with
          the dashed target box), the row depth frame and the 24,576-point scene cloud. The photograph's wire is solid
          (the input used here), the depth and cloud wires dimmed and dashed: any one of these suffices.
  MODEL   one CropCode box (UGG's stream-colour gradient), its three training stages.
  OUTPUT  per species the recipe CropCode writes, re-rendered as a textured 3D plant (figures/_proc3d.py TEXTURE;
          user 2026-11-14: "the output procedural model need to be textured", "dont use groundtruth in the image"), with written vs true height and leaf count.
Wires are coloured by stream with the paper palette of the any-to-all figure (image blue, depth yellow, cloud pink,
recipe purple). Held-out test rows (test split, yaw 0, seed 0): maize plant_0106 as the current teaser; sorghum
Sorghum_10780 (figures/pick_teaser_sorghum.py: leaf count written = true),
the same recipes as the current teaser (teaser_row_dump.pkl: the row fine-tune given ALL THREE streams), so all three
wires are solid. --photo-only draws the spec's variant (photo solid, depth/cloud dimmed) with the photo-only recipe; on
these rows that recipe is far off (sorghum 1.79 m / 24 leaves vs 1.10 / 21; maize 0.79 m / 6 vs 1.06 / 13).
Built on _scene.Scene like Figs. 2-5. main.tex is NOT touched (the Mac session swaps the \\input).
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=4 --mem=24G --output=logs/%x_%j.out \\
         --wrap "python figures/make_teaser_flow_tikz.py"
-> paper/overleaf/Figures/fig_teaser_flow_tikz.tex + Figures/fig1/, preview paper/figs/ugg_preview/fig_teaser_flow.{pdf,png}
"""
import sys, pathlib, pickle, json, io
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from _scene import Scene, cloud_frame, text_size
import _fig12_kit as K
import _plantviz as V
import _proc3d as P3
import ugg_style as U

FIG = R / 'paper/overleaf/Figures'
PL = {'sorghum': 'Sorghum_10780', 'maize': 'plant_0106'}        # sorghum: figures/pick_teaser_sorghum.py (user, 2026-11-14:
# "Pick one best sorghum plant where leave number matches with ground truth": 19 leaves written = 19 true, height 0.3 % off)
ROW_RUNS = {'sorghum': 'sorghum_row_ft', 'maize': 'maize_row_ft_e1'}
CACHE = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12') / f"teaser_flow_rows_{PL['sorghum']}_{PL['maize']}.pkl"
E, Fi, I = U.STREAM, U.STREAM_FILL, U.STREAM_INK
INK, MUTED, GREY = '#1a1a1a', '#5f5f5f', '#8c8c8c'
F = 8.5


def rows():
    """per species, the held-out test row at yaw 0 (shot 0, RowDataset seed 0, the teaser protocol): the row photograph as
    captured, the target's box (from the target-only render's alpha), the hidden fraction, the depth frame, the scene cloud,
    and the recipe the row fine-tune writes from all three streams ('all') and from the photograph alone ('rgb')"""
    if CACHE.exists(): return pickle.load(open(CACHE, 'rb'))
    from row_dataset import RowDataset
    from eval_row import build_model, predict, DEFAULT_ROOT
    D = {}
    for sp in ('sorghum', 'maize'):
        model, cfg, ep = build_model(R / 'outputs' / ROW_RUNS[sp], 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        ds = RowDataset(DEFAULT_ROOT[sp], 'test', sp, 'row', cfg.get('img_size', 224), cfg['num_points'], cfg['max_leaves'],
                        thetas=[0], rgb_gain=gain, gt_points=cfg['gt_points'], plants=[PL[sp]])
        np.random.seed(0)
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = ds[0]
        folder = pathlib.Path(ds.items[ds._resolve(0)][3])
        gen = {}
        for src in ('all', 'rgb'):
            _, ppar = predict(model, src, rgb[None], depth[None], pc[None], params[None], float(cfg.get('mask_ratio', 0.8)), 0)
            gen[src] = ppar[0].clamp(0, 1).float().numpy()
        a = np.asarray(Image.open(folder / 'rgba.png').convert('RGBA')).astype(np.float32) / 255
        photo = np.clip(a[..., :3] * (gain or 1.0), 0, 1) * a[..., 3:]
        ga = np.asarray(Image.open(folder / 'gt_rgba.png').convert('RGBA'))[..., 3] > 127
        ys_, xs_ = np.where(ga); h_, w_ = ga.shape; pad = 0.015
        box = (max(0, xs_.min() / w_ - pad), max(0, ys_.min() / h_ - pad), min(1, xs_.max() / w_ + pad), min(1, ys_.max() / h_ + pad))
        D[sp] = dict(plant=PL[sp], photo=photo, box=box, occ=float(info['occ_frac']), depth=depth[0].numpy(), scene=pc.numpy(),
                     par=params.numpy(), tv=valid.numpy(), gen=gen, epoch=ep)
    pickle.dump(D, open(CACHE, 'wb')); return D


def dim(png, f=0.45):
    """blend a tile toward white (the inputs not used here)"""
    im = Image.open(io.BytesIO(png)).convert('RGBA'); a = np.asarray(im).astype(np.float32)
    a[..., :3] = a[..., :3] * (1 - f) + 255 * f
    b = io.BytesIO(); Image.fromarray(a.astype(np.uint8)).save(b, 'PNG'); return b.getvalue()


if __name__ == '__main__':
    PHOTO_ONLY = '--photo-only' in sys.argv          # off: all three streams in, as the reported numbers were made
    X = rows()
    W, H = 6.875, 2.50
    S = Scene(W, H, 'Fig 1 teaser flow')
    s_ph, s_sm, m = 0.98, 0.45, 0.03                  # photo tile, small tiles, inset
    x_ph, x_sm = 0.24, 1.27
    x_mod, w_mod = 2.30, 1.18
    x_card, w_card = 3.86, W - 3.86 - 0.04
    y0s = {'sorghum': 1.30, 'maize': 0.23}
    # ---- headers
    S.text(x_ph, H - 0.10, 'INPUT', F, color=I['rgb'], bold=True, va='center')
    S.text(x_ph + 0.50, H - 0.10, 'any subset, from an occluded row', F, color=MUTED, va='center')
    S.text(x_card, H - 0.10, 'OUTPUT', F, color=I['text'], bold=True, va='center')
    S.text(x_card + 0.72, H - 0.10, 'procedural plant', F, color=MUTED, va='center')
    # ---- the model, spanning both rows (user, 2026-11-14: "make the crop code box more flashy, like put some icons for
    # each of 3 methods"): shadow, stream-gradient body, a dark title pill, one white card per training stage with an icon
    yb, yt = y0s['maize'], y0s['sorghum'] + s_ph
    import make_ugg_figs as UG
    S.rect(x_mod + 0.035, yb - 0.035, w_mod, yt - yb, '#cfd0d8', lw=0.0, fill='#cfd0d8', r=0.08, z=-1)      # drop shadow
    S.image(x_mod, yb, w_mod, yt - yb, UG._band_png(stops=('#d3deee', '#f6e7b4', '#f0d3de', '#dbd2f0')), 'model band')   # a deeper stream gradient
    S.rect(x_mod, yb, w_mod, yt - yb, '#5b5f6b', lw=1.3, r=0.08)
    cx = x_mod + w_mod / 2
    S.rect(x_mod + 0.10, yt - 0.42, w_mod - 0.20, 0.30, '#2b2f3a', lw=0.0, fill='#2b2f3a', r=0.15, z=1.5)  # title pill
    S.text(cx, yt - 0.27, 'CropCode', F + 2.5, color='#ffffff', ha='center', va='center', bold=True)
    S.text(cx, yt - 0.55, 'one model, three stages', F - 1.0, color=MUTED, ha='center', va='center')

    def lift(z):                                   # raise the last op above the gradient image (Scene images sit at z 1)
        S.ops[-1]['z'] = z

    def icon_mask(ix, cy):                         # masked pretraining: a token grid, some hidden
        a_, g_ = 0.062, 0.012; cols = [E['rgb'], E['depth'], E['pc'], E['text']]
        hidden = {1, 3, 5, 6, 8}
        for r_ in range(3):
            for c_ in range(3):
                k = r_ * 3 + c_; x_ = ix + c_ * (a_ + g_); y_ = cy + 0.105 - (r_ + 1) * a_ - r_ * g_
                if k in hidden: S.rect(x_, y_, a_, a_, '#9a9aa6', lw=0.5, fill='#ffffff', r=0.008, z=1.7)
                else: S.rect(x_, y_, a_, a_, cols[k % 4], lw=0.5, fill=cols[k % 4], r=0.008, z=1.7)

    def icon_distill(ix, cy):                      # cross-modal distillation: frozen teacher -> trained student
        K.icon(S, 'snow', ix + 0.055, cy, 0.125); lift(1.7)
        S.arrow((ix + 0.12, cy), (ix + 0.155, cy), color='#5b5f6b', lw=1.0, ms=5)
        K.icon(S, 'fire', ix + 0.212, cy, 0.125); lift(1.7)

    def icon_row(ix, cy):                          # row fine-tuning: three plants in a row, the centre one the target
        for k, px in enumerate((ix + 0.035, ix + 0.112, ix + 0.189)):
            c_ = '#2f7d32' if k == 1 else '#b4b4b4'
            S.arrow((px, cy - 0.10), (px, cy + 0.10), color=c_, lw=1.0, head=False)
            for dy, sgn in ((-0.045, 1), (-0.005, -1), (0.035, 1)):
                S.arrow((px, cy + dy), (px + sgn * 0.034, cy + dy + 0.045), color=c_, lw=0.9, head=False, curve=0.35 * sgn)
        S.rect(ix + 0.077, cy - 0.115, 0.07, 0.235, '#2f7d32', lw=0.6, dash=True, r=0.01)

    stages = [('masked\npretraining', icon_mask), ('cross-modal\ndistillation', icon_distill), ('row\nfine-tuning', icon_row)]
    for k, (lab, ic) in enumerate(stages):
        ccy = yt - 0.86 - k * 0.45; ch = 0.38
        S.rect(x_mod + 0.07, ccy - ch / 2, w_mod - 0.14, ch, '#c4c6d2', lw=0.7, fill='#ffffff', r=0.05, z=1.5)
        ic(x_mod + 0.13, ccy)
        S.text(x_mod + 0.45, ccy, lab, F - 1.0, color=INK, va='center', ls=1.05)
        S.ellipse(x_mod + 0.07, ccy + ch / 2 - 0.05, 0.05, stroke='#2b2f3a', lw=0.0, fill='#2b2f3a', z=1.8)   # step badge
        S.text(x_mod + 0.07, ccy + ch / 2 - 0.05, f'{k + 1}', F - 2.0, color='#ffffff', ha='center', va='center', bold=True)
    # ---- per species
    for sp in ('sorghum', 'maize'):
        y0 = y0s[sp]; yc = y0 + s_ph / 2; d = X[sp]
        S.text(0.10, yc, sp.capitalize(), F, color=INK, ha='center', va='center', bold=True, rot=90)
        # photograph (current teaser tile) with the dashed target box
        S.rect(x_ph, y0, s_ph, s_ph, E['rgb'], lw=1.0, fill=Fi['rgb'], r=0.05, dash=True)
        inner = s_ph - 2 * m
        S.image(x_ph + m, y0 + m, inner, inner, K.png_array(d['photo']), f'{sp} row photograph')
        bx0, btop, bx1, bbot = d['box']
        S.rect(x_ph + m + inner * bx0, y0 + m + inner * (1 - bbot), inner * (bx1 - bx0), inner * (bbot - btop), '#ffffff', lw=0.7, r=0.01, dash=True)
        S.text(x_ph + m + 0.04, y0 + m + 0.05, f"{d['occ'] * 100:.0f} % hidden", F - 1.5, color='#ffffff', va='bottom')
        # depth (top) and scene cloud (bottom), dimmed: not used here
        dimf = (lambda png, f: dim(png, f)) if PHOTO_ONLY else (lambda png, f: png)
        tiles = [('depth', y0 + s_ph - s_sm, dimf(K.tile_depth(d['depth']), 0.30)),
                 ('pc', y0, dimf(K.tile_cloud(d['scene'], cloud_frame(d['scene'], elev=10.0, azim=20.0), s_sm - 2 * m, s=0.30, elev=10.0, azim=20.0), 0.25))]
        for st, yy, png in tiles:
            S.rect(x_sm, yy, s_sm, s_sm, E[st], lw=0.8, fill='#ffffff', r=0.04, dash=True)
            S.image(x_sm + m, yy + m, s_sm - 2 * m, s_sm - 2 * m, png, f'{sp} {st}')
            S.text(x_sm + s_sm + 0.05, yy + s_sm - 0.02 if st == 'depth' else yy + 0.02, {'depth': 'depth', 'pc': 'cloud'}[st], F - 1.5, color=I[st],
                   va='top' if st == 'depth' else 'bottom')
        # wires into the model: the photograph solid through the gap between the small tiles, depth and cloud dashed
        S.arrow((x_ph + s_ph + 0.01, yc), (x_mod - 0.02, yc), color=E['rgb'], lw=1.8, ms=8)
        lw2, ms2 = (1.1, 6) if PHOTO_ONLY else (1.8, 8)
        S.arrow((x_sm + s_sm + 0.02, y0 + s_ph - s_sm / 2), (x_mod - 0.02, yc + 0.20), color=E['depth'], lw=lw2, dash=PHOTO_ONLY, curve=0.25, ms=ms2)
        S.arrow((x_sm + s_sm + 0.02, y0 + s_sm / 2), (x_mod - 0.02, yc - 0.20), color=E['pc'], lw=lw2, dash=PHOTO_ONLY, curve=-0.25, ms=ms2)
        # output card: the written recipe as a 3D plant over the true one
        S.rect(x_card, y0, w_card, s_ph, E['text'], lw=1.0, fill=Fi['text'], r=0.06)
        rt = V.decode_recipe(sp, d['par'], d['tv']); rp = V.decode_recipe(sp, d['gen']['rgb' if PHOTO_ONLY else 'all'], d['tv'], predicted=True)
        S.image(x_card + m, y0 + m, inner, inner, P3.png([(rp, P3.TEXTURE, 1.0)], sp, side=inner, dpi=600), f'{sp} procedural plant')   # no ground truth in the image (user)
        ll = (lambda r: np.mean([l['length'] for l in r['leaves']])) if sp == 'sorghum' else (lambda r: np.mean([l['leafLength'] for l in r['leaves']]))
        L = {'height': (rp['height'], rt['height']), 'leaves': (rp['n'], rt['n']), 'leaf length': (ll(rp), ll(rt))}
        print(sp, 'photo only' if PHOTO_ONLY else 'all three streams', {k: (round(float(a_), 3), round(float(b_), 3)) for k, (a_, b_) in L.items()}, flush=True)
        xt = x_card + inner + 0.16
        S.text(xt, y0 + s_ph - 0.18, 'written by CropCode', F, color=I['text'], bold=True, va='center')
        S.text(xt + 1.38, y0 + s_ph - 0.18, 'true', F, color=MUTED, bold=True, va='center')
        rows = [('height', f"{L['height'][0]:.2f} m", f"{L['height'][1]:.2f} m"), ('leaves', f"{L['leaves'][0]:.0f}", f"{L['leaves'][1]:.0f}"),
                ('leaf length', f"{L['leaf length'][0]:.2f} m", f"{L['leaf length'][1]:.2f} m")]
        for k, (nm, pv, tv) in enumerate(rows):
            yy = y0 + s_ph - 0.42 - k * 0.19
            S.text(xt, yy, nm, F, color=MUTED, va='center'); S.text(xt + 0.80, yy, pv, F, color=I['text'], va='center', bold=True)
            S.text(xt + 1.38, yy, tv, F, color=MUTED, va='center')
        S.arrow((x_mod + w_mod + 0.02, yc), (x_card - 0.02, yc), color=E['text'], lw=1.8, ms=8)
    # ---- legend line
    # the paper's font sets text ~5 % wider than the preview measures: keep the two footer notes far apart
    S.text(x_ph, 0.07, ('no mask or bounding box given  ·  solid: used here  ·  dashed: either also suffices' if PHOTO_ONLY else
            'no mask or bounding box given  ·  here: all three streams (any subset is accepted)'),
           F - 1.5, color=MUTED, va='center')
    S.text(x_card + w_card, 0.07, 'kilobytes  ·  editable  ·  re-renderable', F - 1.0, color=I['text'], ha='right', va='center')
    S.render(R / 'paper/figs/ugg_preview/fig_teaser_flow')
    S.render_tikz(FIG / 'fig_teaser_flow_tikz.tex', FIG / 'fig1', 'Figures/fig1')
    print('wrote', FIG / 'fig_teaser_flow_tikz.tex', 'height', H, 'in =', round(H * 2.54, 2), 'cm', flush=True)
