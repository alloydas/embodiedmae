"""Paper Fig. 4 (any input to all, maize) and its sorghum supplement twin as TikZ (user, 2026-11-14: "Make fig 4 as tikz").

Same figure as make_ugg_figs.render4_ugg (fig_any2all_{maize,sorghum}_ugg.pdf): four blocks, one per input stream, each
the input tile (dashed box) and the three streams the distilled generalist writes from it (solid boxes), with the
Chamfer of the generated cloud (x1e-3) and the height and leaf count of the written recipe under their tiles. Same data
(any2all4_dump.pkl), layout, colours and numbers; built on _scene.Scene like Figs. 2, 3 and 5: boxes, labels and arrows
are TikZ, each tile is its own PNG drawn by make_any2all_fig.tile at the tile's printed size (Figures/fig4/).
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=2 --mem=12G --output=logs/%x_%j.out \
         --wrap "python figures/make_fig4_tikz.py"
-> paper/overleaf/Figures/fig_any2all_{maize,sorghum}_tikz.tex + Figures/fig4/
"""
import sys, pathlib, pickle, io
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _scene import Scene
import make_ugg_figs as UG                       # palette and the any2all module (A), as render4_ugg uses them
A = UG.A

FIGS = R / 'paper/overleaf/Figures'


def tile_png(sp, e, st, src, side):
    fig = plt.figure(figsize=(side, side))
    ax = A.tile(fig, sp, e, st, src, (0, 0, side, side), side, side)
    ax.patch.set_alpha(0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=600, transparent=True); plt.close(fig); return b.getvalue()


def build(sp, nm, D):
    e = D[sp][nm]
    W = 6.875; t = 0.72; g = 0.06; gd = 0.24; gb = 0.24; tag = 0.24; lab = 0.31; gr = 0.10; m = 0.035; num = 0.19
    block = t + gd + 3 * t + 2 * g
    x0 = (W - (2 * block + gb)) / 2
    rowh = lab + t + num
    H = tag + 2 * rowh + gr + 0.01
    S = Scene(W, H, f'Fig 4 any2all {sp}')
    F = A.FS
    for k, src in enumerate(A.SRCS):
        r, c = divmod(k, 2)
        bx = x0 + c * (block + gb); ytop = H - tag - r * (rowh + gr); ty = ytop - lab - t
        outs = [st for st in A.SRCS if st != src]
        xs = [bx] + [bx + t + gd + j * (t + g) for j in range(3)]
        if r == 0:
            S.text(bx + t / 2, H - tag / 2, 'Input', F + 1, ha='center', va='center', bold=True)
            S.text(xs[1] + (3 * t + 2 * g) / 2, H - tag / 2, 'Output', F + 1, ha='center', va='center', bold=True)
        for j, (st, x) in enumerate(zip([src] + outs, xs)):
            S.rect(x, ty, t, t + lab, UG.EDGE[st], lw=1.0, fill=UG.FILL[st], r=0.05, dash=j == 0)
            S.text(x + t / 2, ty + t + lab / 2 - 0.01, A.NAME2[st], F, ha='center', va='center', ls=0.95)
            S.image(x + m, ty + m, t - 2 * m, t - 2 * m, tile_png(sp, e, st, None if j == 0 else src, t - 2 * m), f'{sp} {src} {st}')
            if j and st == 'pc':
                S.text(x + t / 2, ty - 0.03, f"{e['out'][src]['chamfer'] * 1e3:.2f}", F, ha='center', va='top', color=UG.INK)
            if j and st == 'text':
                rp = e['out'][src]['rec']
                S.text(x + t / 2, ty - 0.03, f"{rp['height']:.2f} m, {rp['n']}", F, ha='center', va='top', color=UG.INK)
        S.arrow((bx + t + 0.045, ty + t / 2), (bx + t + gd - 0.045, ty + t / 2), color=UG.ARROW, lw=2.2, ms=9)
    return S


if __name__ == '__main__':
    D = pickle.load(open(A.OUT4, 'rb'))
    for sp, nm in (('maize', 'plant_1875_09'), ('sorghum', 'Sorghum_7712_05')):
        S = build(sp, nm, D)
        S.render(R / f'paper/figs/ugg_preview/fig_any2all_{sp}_tikz')
        S.render_tikz(FIGS / f'fig_any2all_{sp}_tikz.tex', FIGS / 'fig4', 'Figures/fig4')
        print('wrote', FIGS / f'fig_any2all_{sp}_tikz.tex', flush=True)
