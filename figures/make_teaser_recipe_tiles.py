"""Fig. 1 (teaser), UGG variant: the output shown as procedural parameters only, drawn as in the any-input-to-all
figure (figures/make_any2all_fig.py _glyph: the recipe as a stem-and-leaves schematic, the one the model writes in
purple over the true one in grey). User, 2026-11-13: "in the fig 1 teaser show only procedural parameter as you
shown in fig 4 any to any".

Same plants and protocol as figures/make_teaser_v2.py --mode row (test split, yaw 0, seed 0; the row fine-tunes
sorghum_row_ft / maize_row_ft_e1 at epoch 100 read the photograph, depth and scene cloud of the whole row with 80 % of
tokens hidden, the recipe never an input), so the numbers equal the paper's teaser.
Writes paper/overleaf/Figures/teaser/{s,m}_recipe.png and prints the parameter lines used in
paper/overleaf/Figures/fig_teaser_tikz_ugg.tex. CPU: sbatch ... --wrap "python figures/make_teaser_recipe_tiles.py"
"""
import sys, pathlib, pickle, json
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _plantviz as V
import _fig12_kit as K

PL = {'sorghum': 'Sorghum_10113', 'maize': 'plant_0106'}
ROW_RUNS = {'sorghum': 'sorghum_row_ft', 'maize': 'maize_row_ft_e1'}
CACHE = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/teaser_row_dump.pkl')
TILES = R / 'paper/overleaf/Figures/teaser'
PURPLE, GREY = '#5e4a96', '#c9c9c9'               # ugg_style STREAM_INK['text'], the any2all truth grey
SEED, YAW, SPLIT = 0, 0, 'test'


def compute():
    """make_teaser_v2.compute_row, verbatim protocol."""
    from row_dataset import RowDataset
    from eval_row import build_model, predict, DEFAULT_ROOT
    D = {}
    for sp in ('sorghum', 'maize'):
        model, cfg, ep = build_model(R / 'outputs' / ROW_RUNS[sp], 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        ds = RowDataset(DEFAULT_ROOT[sp], SPLIT, sp, 'row', cfg.get('img_size', 224), cfg['num_points'], cfg['max_leaves'],
                        thetas=[YAW], rgb_gain=gain, gt_points=cfg['gt_points'], plants=[PL[sp]])
        np.random.seed(SEED)
        rgb, depth, pc, params, valid, g_rgb, g_depth, g_pc, info = ds[0]
        ppc, ppar = predict(model, 'all', rgb[None], depth[None], pc[None], params[None], float(cfg.get('mask_ratio', 0.8)), SEED)
        D[sp] = dict(plant=PL[sp], par=params.numpy(), tv=valid.numpy(), gen_par=ppar[0].clamp(0, 1).float().numpy(), epoch=ep)
    return D


def glyph_png(sp, rec, truth, side=1.1, px=600):
    """make_any2all_fig._glyph: truth in grey under the written recipe."""
    fig = plt.figure(figsize=(side, side), dpi=px); ax = fig.add_axes([0.03, 0.03, 0.94, 0.94]); ax.set_axis_off()
    K.glyph(ax, sp, truth, GREY, stem=GREY, lw_scale=0.75 * side / 0.735)
    K.glyph(ax, sp, rec, PURPLE, stem='#4a3b2a', lw_scale=0.42 * side / 0.735)
    bb = ax.dataLim; w = max(abs(bb.x0), abs(bb.x1)); y0 = min(bb.y0, 0.0); st = max(2 * w, bb.y1 - y0) * 1.06
    ax.set_xlim(-st / 2, st / 2); ax.set_ylim(y0 - 0.02 * st, y0 + 0.98 * st); ax.set_aspect('equal')
    return fig


if __name__ == '__main__':
    D = pickle.load(open(CACHE, 'rb')) if CACHE.exists() else None
    if D is None or any(D[s]['plant'] != PL[s] for s in PL):
        D = compute(); pickle.dump(D, open(CACHE, 'wb'))
    lines = {}
    for sp, tag in (('sorghum', 's'), ('maize', 'm')):
        d = D[sp]
        rt = V.decode_recipe(sp, d['par'], d['tv']); rp = V.decode_recipe(sp, d['gen_par'], d['tv'], predicted=True)
        fig = glyph_png(sp, rp, rt); fig.savefig(TILES / f'{tag}_recipe.png', dpi=600, transparent=True); plt.close(fig)
        L = rp['leaves']; Lt = rt['leaves']
        if sp == 'sorghum':
            extra = [('leaf length', np.mean([l['length'] for l in L]), np.mean([l['length'] for l in Lt]), 'm', 2),
                     ('leaf width', 100 * np.mean([l['width'] for l in L]), 100 * np.mean([l['width'] for l in Lt]), 'cm', 1)]
        else:
            extra = [('leaf length', np.mean([l['leafLength'] for l in L]), np.mean([l['leafLength'] for l in Lt]), 'm', 2),
                     ('leaf angle', np.mean([l['leafAngle'] for l in L]), np.mean([l['leafAngle'] for l in Lt]), '°', 0)]
        lines[sp] = [('height', rp['height'], rt['height'], 'm', 2), ('leaves', rp['n'], rt['n'], '', 0)] + extra
        print(sp, PL[sp], 'epoch', d['epoch'], [(k, round(float(p), 3), round(float(t), 3)) for k, p, t, _, _ in lines[sp]], flush=True)
    json.dump({sp: [(k, float(p), float(t), u, n) for k, p, t, u, n in v] for sp, v in lines.items()},
              open(TILES / 'recipe_lines.json', 'w'), indent=1)
    print('wrote', TILES / 's_recipe.png', TILES / 'm_recipe.png', TILES / 'recipe_lines.json')
