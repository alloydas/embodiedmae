"""Fig. 1 teaser as a UGG-style flow diagram (spec: figures/fig_teaser_flow_SPEC.md, requested 2026-11-14).

Three zones, two species rows (sorghum on top, maize below) sharing one model box:
  INPUT   per species the occluded row as the rover captures it: the row photograph (the current teaser's tile, with
          the dashed target box), the row depth frame and the 24,576-point scene cloud. The photograph's wire is solid
          (the input used here), the depth and cloud wires dimmed and dashed: any one of these suffices.
  MODEL   one CropCode box (UGG's stream-colour gradient), its three training stages.
  OUTPUT  per species the recipe CropCode writes, re-rendered as a 3D plant (figures/_proc3d.py: purple = written,
          translucent grey = true), with written vs true height and leaf count.
Wires are coloured by stream with the paper palette of the any-to-all figure (image blue, depth yellow, cloud pink,
recipe purple). Same held-out test rows as the current teaser (Sorghum_10113, plant_0106; test split, yaw 0, seed 0),
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
TEASER = FIG / 'teaser'
PL = {'sorghum': 'Sorghum_10113', 'maize': 'plant_0106'}
ROW_RUNS = {'sorghum': 'sorghum_row_ft', 'maize': 'maize_row_ft_e1'}
DUMP = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_row_dump.pkl')
CACHE = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_flow_inputs.pkl')
BOX = {'sorghum': (0.355, 0.064, 0.645, 0.955), 'maize': (0.364, 0.045, 0.709, 0.982)}   # target box in the photo tile (x0, top, x1, bottom)
OCC = {'sorghum': 25, 'maize': 22}
E, Fi, I = U.STREAM, U.STREAM_FILL, U.STREAM_INK
INK, MUTED, GREY = '#1a1a1a', '#5f5f5f', '#8c8c8c'
F = 8.5


def inputs():
    """the row depth frame and scene cloud of the teaser rows (the RowDataset item of make_teaser_v2.py --mode row)"""
    if CACHE.exists(): return pickle.load(open(CACHE, 'rb'))
    from row_dataset import RowDataset
    from eval_row import build_model, DEFAULT_ROOT
    D = {}
    for sp in ('sorghum', 'maize'):
        _, cfg, _ = build_model(R / 'outputs' / ROW_RUNS[sp], 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        ds = RowDataset(DEFAULT_ROOT[sp], 'test', sp, 'row', cfg.get('img_size', 224), cfg['num_points'], cfg['max_leaves'],
                        thetas=[0], rgb_gain=gain, gt_points=cfg['gt_points'], plants=[PL[sp]])
        np.random.seed(0)
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = ds[0]
        D[sp] = dict(depth=depth[0].numpy(), scene=pc.numpy())
    pickle.dump(D, open(CACHE, 'wb')); return D


PHOTO = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_flow_photo_only.pkl')


def photo_only():
    """the recipe the row fine-tune writes from the ROW PHOTOGRAPH ALONE (the figure's solid wire; eval_row.predict with
    src='rgb': the photograph fully visible, depth / cloud / recipe hidden), same rows and seed as the teaser"""
    if PHOTO.exists(): return pickle.load(open(PHOTO, 'rb'))
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
        _, ppar = predict(model, 'rgb', rgb[None], depth[None], pc[None], params[None], float(cfg.get('mask_ratio', 0.8)), 0)
        D[sp] = dict(par=params.numpy(), tv=valid.numpy(), gen_par=ppar[0].clamp(0, 1).float().numpy(), epoch=ep)
    pickle.dump(D, open(PHOTO, 'wb')); return D


def dim(png, f=0.45):
    """blend a tile toward white (the inputs not used here)"""
    im = Image.open(io.BytesIO(png)).convert('RGBA'); a = np.asarray(im).astype(np.float32)
    a[..., :3] = a[..., :3] * (1 - f) + 255 * f
    b = io.BytesIO(); Image.fromarray(a.astype(np.uint8)).save(b, 'PNG'); return b.getvalue()


if __name__ == '__main__':
    PHOTO_ONLY = '--photo-only' in sys.argv          # off: all three streams in, as the reported numbers were made
    X = inputs(); T = photo_only() if PHOTO_ONLY else pickle.load(open(DUMP, 'rb'))
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
    # ---- the model, spanning both rows
    yb, yt = y0s['maize'], y0s['sorghum'] + s_ph
    import make_ugg_figs as UG
    S.image(x_mod, yb, w_mod, yt - yb, UG._band_png(), 'model band')
    S.rect(x_mod, yb, w_mod, yt - yb, GREY, lw=1.0, r=0.08)
    cx = x_mod + w_mod / 2
    S.text(cx, yt - 0.30, 'CropCode', F + 2.5, color=INK, ha='center', va='center', bold=True)
    S.text(cx, yt - 0.52, 'one model', F, color=MUTED, ha='center', va='center')
    for k, st in enumerate(('masked pretraining', 'cross-modal\ndistillation', 'row fine-tuning')):
        yy = yt - (0.80, 1.20, 1.70)[k]                       # the two-line label needs more room
        S.text(cx, yy, f'{k + 1}', F - 0.5, color=MUTED, ha='center', va='center', bold=True)
        S.text(cx, yy - 0.13, st, F, color=INK, ha='center', va='top', ls=1.05)
    # ---- per species
    for sp in ('sorghum', 'maize'):
        y0 = y0s[sp]; yc = y0 + s_ph / 2; d = X[sp]
        S.text(0.10, yc, sp.capitalize(), F, color=INK, ha='center', va='center', bold=True, rot=90)
        # photograph (current teaser tile) with the dashed target box
        S.rect(x_ph, y0, s_ph, s_ph, E['rgb'], lw=1.0, fill=Fi['rgb'], r=0.05, dash=True)
        inner = s_ph - 2 * m
        S.image(x_ph + m, y0 + m, inner, inner, (TEASER / f"{sp[0]}_rgb.png").read_bytes(), f'{sp} row photograph')
        bx0, btop, bx1, bbot = BOX[sp]
        S.rect(x_ph + m + inner * bx0, y0 + m + inner * (1 - bbot), inner * (bx1 - bx0), inner * (bbot - btop), '#ffffff', lw=0.7, r=0.01, dash=True)
        S.text(x_ph + m + 0.04, y0 + m + 0.05, f'{OCC[sp]} % hidden', F - 1.5, color='#ffffff', va='bottom')
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
        t = T[sp]; rt = V.decode_recipe(sp, t['par'], t['tv']); rp = V.decode_recipe(sp, t['gen_par'], t['tv'], predicted=True)
        S.image(x_card + m, y0 + m, inner, inner, P3.png([(rt, P3.TRUTH_C, 0.45), (rp, P3.LEAF_C, 1.0)], sp, side=inner, dpi=600), f'{sp} procedural plant')
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
