"""Figures 1 and 2, second version (layout of SAGA, Kundu et al., CVPR 2026; devices from the CVPR figure study).

Figure 1 (paper/figs/fig1_overview.*): (a) three crop rows in (image, depth image, point cloud), CropCode, the
  center plant out: its procedural parameters drawn as a stem-and-leaves schematic, traits beside their true values,
  the generated point cloud beside the true plant. Rows 1-2: held-out simulated test rows (sorghum, maize; the row
  fine-tune). Row 3: a held-out real maize plant after the real-row fine-tune. (b) t-SNE of the headline latents.
  (c) trait-probe radar against the published encoders and the supervised reference.
Figure 2 (paper/figs/fig2_stages.*): three stacked stages (masked pretraining, cross-modal distillation, row
  fine-tuning), each drawn with the real tiles of one example and ending in its output, the truth and the output
  without that stage; a zoom-in on the network at the right.
Data: figures/make_fig12_dump.py (one CPU job) and paper/figs/fig12_assets/tsne_*.npz (figures/make_fig12_assets.py).
Each figure is drawn by figures/_scene.py to PDF / PNG (matplotlib) and to an editable .drawio file.
usage: python figures/make_fig12_v2.py [--only 1|2] [--out DIR] [--no-drawio]
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures')]
import argparse
import numpy as np
from _scene import Scene, cloud_frame, text_size
import _fig12_kit as K
from _fig12_kit import INK, MUTED, FAINT, NAVY, LOSS, CS, T_TITLE, T_HEAD, T_LAB, T_MIN

VIEW = dict(elev=6.0, azim=18.0)
ROW_PLANT = {'sorghum': 'Sorghum_10113', 'maize': 'plant_0106'}
REAL = 'parallel:0148:14'


def traits(sp, rec):
    L = rec['leaves']
    if sp == 'sorghum':
        return [('height', rec['height'], '{:.2f}', 'm'), ('leaves', rec['n'], '{:.0f}', ''),
                ('leaf length', np.mean([l['length'] for l in L]) if L else np.nan, '{:.2f}', 'm'),
                ('leaf width', 100 * np.mean([l['width'] for l in L]) if L else np.nan, '{:.1f}', 'cm')]
    return [('height', rec['height'], '{:.2f}', 'm'), ('leaves', rec['n'], '{:.0f}', ''),
            ('leaf length', np.mean([l['leafLength'] for l in L]) if L else np.nan, '{:.2f}', 'm'),
            ('leaf angle', np.mean([l['leafAngle'] for l in L]) if L else np.nan, '{:.0f}', '°')]


# ======================================================================= Figure 1
def figure1(D):
    W, H = 7.0, 2.78
    S = Scene(W, H, 'Fig 1 overview'); T = 0.50
    Y = {'sorghum': 1.76, 'maize': 1.04, 'real': 0.30}
    X_IN = [0.22, 0.74, 1.26]; XB, WB = 1.89, 0.50; XG = 2.52; XT, WT = 3.06, 1.04; XP, XQ = 4.15, 4.68
    XA1 = XQ + T                                                      # right edge of panel (a)
    C_OURS, C_TRUE = XT + 0.80, XT + WT - 0.03
    hy = Y['sorghum'] + T + 0.05
    # ---- headers
    S.text(X_IN[0], H - 0.10, 'Occluded crop rows, as a rover sees them', T_TITLE - 0.5, bold=True)
    S.text(XG, H - 0.10, 'The center plant, written out', T_TITLE - 0.5, bold=True)
    for lab, x, c in (('Image', X_IN[0], CS['rgb']), ('Depth\nimage', X_IN[1], CS['depth']), ('Point\ncloud', X_IN[2], CS['pc'])):
        S.text(x + T / 2, hy, lab, T_LAB, color=c, ha='center', va='bottom', bold=True, ls=1.0)
    S.text(XB + WB / 2, hy, 'CropCode', T_HEAD, color=NAVY, ha='center', va='bottom', bold=True)
    S.text(XG + T / 2, hy, 'Procedural\nparameters', T_LAB, color=CS['text'], ha='center', va='bottom', bold=True, ls=1.0)
    S.text(XT + 0.04, hy, 'Traits', T_LAB, va='bottom', bold=True)
    S.text(C_OURS, hy, 'ours', T_MIN, color=NAVY, ha='right', va='bottom', bold=True)
    S.text(C_TRUE, hy, 'true', T_MIN, color=MUTED, ha='right', va='bottom')
    S.text(XP + T / 2, hy, 'Generated\npoint cloud', T_LAB, color=CS['pc'], ha='center', va='bottom', bold=True, ls=1.0)
    S.text(XQ + T / 2, hy, 'True\nplant', T_LAB, color=MUTED, ha='center', va='bottom', bold=True, ls=1.0)
    # ---- the real-plant row sits on a tinted band
    S.rect(0.02, Y['real'] - 0.20, XA1 + 0.04, T + 0.27, '#eadfc8', lw=0.6, fill='#fbf7ee', r=0.06)
    S.text(0.065, Y['sorghum'] - 0.11, 'simulated test rows', T_MIN, color=MUTED, ha='center', rot=90)
    S.text(0.065, Y['real'] + T / 2 - 0.03, 'real plants', T_MIN, color='#8a6a1f', ha='center', rot=90, bold=True)
    # ---- the model: one box for the simulated rows, one for the real-row fine-tune
    by0, by1 = Y['maize'], Y['sorghum'] + T
    S.rect(XB, by0, WB, by1 - by0, NAVY, lw=1.1, fill=K.BOX_FILL, r=0.07)
    K.model(S, XB + 0.03, (by0 + by1) / 2 - 0.10, WB - 0.06, 0.44, size=T_MIN)
    S.text(XB + WB / 2, (by0 + by1) / 2 - 0.26, 'row\nfine-tuned', T_MIN, color=NAVY, ha='center', ls=1.05)
    for k, c in enumerate(CS.values()):                               # the four token streams it was pretrained on
        K.tok_row(S, XB + 0.07 + k * 0.095, (by0 + by1) / 2 + 0.42, 2, a=0.062, gap=0.02, color=c, hidden=(k % 2,), vertical=True, lw=0.45)
    ry0, ry1 = Y['real'], Y['real'] + T
    S.rect(XB, ry0, WB, ry1 - ry0, NAVY, lw=1.1, fill=K.BOX_FILL, r=0.07)
    K.model(S, XB + 0.07, ry0 + 0.20, WB - 0.14, 0.25, labels=False)
    S.text(XB + WB / 2, ry0 + 0.10, '+ real rows', T_MIN, color=NAVY, ha='center')

    def row(key, sp, y, rgb_png, dep_png, scene, gt_pc, pred_pc, sub_in, sub_out):
        S.text(0.155, y + T / 2, sp, T_LAB, ha='center', bold=True, rot=90)
        K.tile(S, X_IN[0], y, T, rgb_png, CS['rgb'], name=f'{key} image')
        K.tile(S, X_IN[1], y, T, dep_png, CS['depth'], name=f'{key} depth image')
        K.tile(S, X_IN[2], y, T, K.tile_cloud(scene, cloud_frame(scene, **VIEW), T, s=0.22, **VIEW), CS['pc'], name=f'{key} point cloud')
        S.text(X_IN[0], y - 0.045, sub_in, T_MIN, color=MUTED, va='top')
        S.arrow((X_IN[2] + T + 0.025, y + T / 2), (XB - 0.02, y + T / 2))
        S.arrow((XB + WB + 0.02, y + T / 2), (XG - 0.025, y + T / 2))
        fr = cloud_frame(gt_pc, **VIEW)
        K.tile(S, XP, y, T, K.tile_cloud(pred_pc, fr, T, **VIEW), CS['pc'], name=f'{key} generated point cloud')
        K.tile(S, XQ, y, T, K.tile_cloud(gt_pc, fr, T, **VIEW), MUTED, dash=True, lw=0.7, name=f'{key} true plant')
        S.text((XP + XQ + T) / 2, y - 0.045, sub_out, T_MIN, color=MUTED, ha='center', va='top')

    for sp in ('sorghum', 'maize'):
        d = D['rows'][sp][ROW_PLANT[sp]]; p = d['pred']['ft:all']; y = Y[sp]
        row(sp, sp, y, K.tile_rgba(d['rgba'], d['gain']), K.tile_depth(d['depth448']), d['scene'], d['gt_pc'], p['pc'],
            f"{d['occ'] * 100:.0f} % of the center plant is hidden", f"Chamfer {p['chamfer_m2e4']:.1f} ×10⁻⁴ m²")
        K.tile(S, XG, y, T, K.tile_glyph(sp, p['recipe'], side=T, lw_scale=0.5), CS['text'], name=f'{sp} parameters')
        S.rect(XT, y, WT, T, FAINT, lw=0.6, r=0.03)
        for i, ((name, vp, f, u), (_, vt, _, _)) in enumerate(zip(traits(sp, p['recipe']), traits(sp, d['recipe_true']))):
            yy = y + T - 0.075 - i * 0.117
            S.text(XT + 0.04, yy, name, T_MIN); S.text(C_OURS, yy, f.format(vp) + (' ' + u if u in ('m', 'cm') else u), T_MIN, color=NAVY, ha='right', bold=True)
            S.text(C_TRUE, yy, f.format(vt), T_MIN, color=MUTED, ha='right')
    e = D['e10'][REAL]; pr, pb = e['pred']['real'], e['pred']['row']; y = Y['real']
    row('real', 'maize', y, K.tile_img(e['rgb']), K.tile_depth(e['depth']), e['scene'], e['gt_pc'], pr['pc'],
        'a held-out real plant, taller than any simulated one', f"Chamfer {pr['chamfer_m2e4']:.1f} ×10⁻⁴ m²")
    S.rect(XG, y, XT + WT - XG, T, FAINT, lw=0.6, r=0.03)                # no true recipe for a real plant: the fine-tuning effect instead
    cb, ca, ct = XG + 0.96, XG + 1.26, XT + WT - 0.03
    S.text(XG + 0.04, y + T - 0.075, 'fine-tuning', T_MIN, color=MUTED, italic=True)
    S.text(cb, y + T - 0.075, 'before', T_MIN, color=MUTED, ha='right'); S.text(ca, y + T - 0.075, 'after', T_MIN, color=NAVY, ha='right', bold=True)
    S.text(ct, y + T - 0.075, 'true', T_MIN, color=MUTED, ha='right')
    for i, (name, vb, va_, vt) in enumerate((('height', f"{pb['height']:.2f} m", f"{pr['height']:.2f} m", f"{e['info']['height']:.2f}"),
                                             ('leaves', f"{pb['recipe']['n']}", f"{pr['recipe']['n']}", f"{e['info']['leaves']:.0f}"),
                                             ('Chamfer', f"{pb['chamfer_m2e4']:.1f}", f"{pr['chamfer_m2e4']:.1f}", ''))):
        yy = y + T - 0.075 - (i + 1) * 0.117
        S.text(XG + 0.04, yy, name, T_MIN); S.text(cb, yy, vb, T_MIN, color=MUTED, ha='right')
        S.text(ca, yy, va_, T_MIN, color=NAVY, ha='right', bold=True); S.text(ct, yy, vt, T_MIN, color=MUTED, ha='right')
    S.text((X_IN[0] + XA1) / 2, 0.035, '(a) One model reads an occluded row and writes the center plant', T_HEAD, ha='center', va='bottom')
    # ---- (b) t-SNE  (c) radar
    XR = XA1 + 0.19; WR = W - 0.02 - XR
    S.arrow((XA1 + 0.095, 0.10), (XA1 + 0.095, H - 0.06), color='#d0d3cc', lw=0.6, head=False)
    hb = 0.90; yb = H - 0.03 - hb
    S.plot(XR, yb, WR / 2, hb, K.tsne_draw('sorghum', 'leaf count', 'Blues', title='sorghum'), 't-SNE sorghum')
    S.plot(XR + WR / 2, yb, WR / 2, hb, K.tsne_draw('maize', 'height (m)', 'Greens', title='maize', fmt='{:.1f}'), 't-SNE maize')
    S.text(XR + WR / 2, yb - 0.02, '(b) Latents order plants by trait', T_HEAD, ha='center', va='top')
    yc = 0.17; hc = yb - 0.17 - yc
    S.plot(XR, yc, WR, hc, K.radar_draw(), 'radar')
    S.text(XR + WR / 2, 0.035, '(c) Trait probes vs. other encoders', T_HEAD, ha='center', va='bottom')
    return S


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='12'); ap.add_argument('--out', default=str(R / 'paper/figs')); ap.add_argument('--no-drawio', action='store_true')
    a = ap.parse_args()
    D = K.load_dump(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if '1' in a.only:
        S = figure1(D); S.lint(); S.render(out / 'fig1_overview', drawio=not a.no_drawio)
    if '2' in a.only:
        S = figure2(D); S.lint(); S.render(out / 'fig2_stages', drawio=not a.no_drawio)
