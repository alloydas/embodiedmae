#!/usr/bin/env python3
"""Build the results deck as a .pptx, figures included.

Reads the JSON exports the eval scripts already produce -- no torch, no GPU, so
this runs in `det` (which has python-pptx) rather than `det_cu128`:

    vis_unpredicted/clouds.json   <- vis_pc_unpredicted.py --export_json
    <gallery>/views.json          <- eval_views_one_plant.py --export_gallery

Figures are written to vis_deck/ and then placed; regenerating a figure and
re-running picks it up. Numbers on the chart slides are typed from
reports/RESULTS_DECK_2026-09-20.md, which is the text version of this deck --
change them in one place and they disagree, so change both.

    conda activate det
    python make_results_pptx.py --gallery <dir-with-views.json>
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor

# One palette for figures and slides, so a chart never disagrees with the text
# beside it about what "after" is coloured.
INK      = RGBColor(0x10, 0x17, 0x1C)
INK2     = RGBColor(0x3D, 0x4C, 0x55)
INK3     = RGBColor(0x6B, 0x7C, 0x86)
ACCENT   = RGBColor(0x0C, 0x6F, 0x7D)
BEFORE   = RGBColor(0x9A, 0x5A, 0x16)
CRITICAL = RGBColor(0xA0, 0x3D, 0x33)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
PAPER    = RGBColor(0xF6, 0xF8, 0xF9)

M_ACCENT, M_BEFORE, M_CRIT = '#0c6f7d', '#9a5a16', '#a03d33'
M_GREEN, M_RED = '#34a35c', '#e0493c'

W, H = Inches(13.333), Inches(7.5)          # 16:9


# ── figures ──────────────────────────────────────────────────────────────

def fig_coverage(clouds_json, out, thr=0.03, plants=4):
    """GT clouds, green where the model covered the point, red where it missed.

    Threshold is 0.03 rather than the 0.01 qal_threshold default: at 0.01 about
    80% of every cloud is red in BOTH models, which is true but shows nothing.
    0.03 is where the two models visibly separate. The caption says so.
    """
    d = json.loads(Path(clouds_json).read_text())
    S = d['samples'][:plants]
    fig, ax = plt.subplots(2, len(S), figsize=(3.4 * len(S), 7.2),
                           subplot_kw={'projection': '3d'})
    for c, s in enumerate(S):
        for r, cl in enumerate(s['clouds']):            # 0 before, 1 after
            p = np.asarray(cl['xyz'], dtype=np.float32).reshape(-1, 3)
            nn = np.asarray(cl['nn'], dtype=np.float32)
            bad = nn > thr
            a = ax[r, c]
            a.scatter(p[~bad, 0], p[~bad, 2], p[~bad, 1], c=M_GREEN, s=1.1,
                      linewidths=0, alpha=.75)
            a.scatter(p[bad, 0], p[bad, 2], p[bad, 1], c=M_RED, s=1.8,
                      linewidths=0)
            a.view_init(18, 42)
            a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
            a.grid(False)
            for pane in (a.xaxis, a.yaxis, a.zaxis):
                pane.pane.fill = False
                pane.pane.set_edgecolor('#dfe6e9')
            a.set_title(f"{cl['label']}\n{100*bad.mean():.0f}% missed",
                        fontsize=9, fontweight='bold',
                        color=M_BEFORE if r == 0 else M_ACCENT, pad=-2)
        ax[0, c].text2D(.5, 1.10, s['name'].replace('Sorghum_', 'plant '),
                        transform=ax[0, c].transAxes, ha='center',
                        fontsize=10, fontweight='bold')
    fig.legend(handles=[Patch(facecolor=M_GREEN, label='predicted — model put a point within %.2f' % thr),
                        Patch(facecolor=M_RED, label='missed — nothing within %.2f' % thr)],
               loc='lower center', ncol=2, frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.07, wspace=.02, hspace=.10)
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)
    return out


def fig_views(views_json, gallery_dir, out, picks=(0, 3, 6, 9), thr=0.03):
    """Input render, the cloud generated from it, and GT coloured by coverage.

    Clouds are drawn in the CAMERA's own frame -- the frame is x right, y up,
    forward -z (established from worldToCamera, not from pixels), so screen
    coordinates are simply (x, y) and the cloud stands in the same pose as the
    photograph beside it.
    """
    d = json.loads(Path(views_json).read_text())
    g = Path(gallery_dir)
    V = [d['views'][i] for i in picks]
    # Views across, panel types down. A 3x3 of square panels is square and
    # letterboxes badly into a slide; this is landscape and fits a fourth view
    # into the same width.
    fig, ax = plt.subplots(3, len(V), figsize=(2.9 * len(V), 9.1))
    for c, v in enumerate(V):
        ax[0, c].imshow(Image.open(g / v['img']))
        ax[0, c].set_title(f"view {v['view']}   sin(elev) {-0.9 + 0.2*picks[c]:+.1f}",
                           fontsize=10.5, fontweight='bold', pad=5)
        pred = np.asarray(v['pred'], dtype=np.float32).reshape(-1, 3)
        gt   = np.asarray(v['gt'],   dtype=np.float32).reshape(-1, 3)
        nn   = np.asarray(v['nn'],   dtype=np.float32)
        ax[1, c].scatter(pred[:, 0], pred[:, 1], s=1.0, c=pred[:, 2],
                         cmap='viridis', linewidths=0)
        ax[1, c].set_title(f"chamfer {v['chamfer']:.5f}", fontsize=9, pad=3)
        bad = nn > thr
        ax[2, c].scatter(gt[~bad, 0], gt[~bad, 1], s=1.0, c=M_GREEN, linewidths=0, alpha=.8)
        ax[2, c].scatter(gt[bad, 0], gt[bad, 1], s=1.6, c=M_RED, linewidths=0)
        ax[2, c].set_title(f"{100*bad.mean():.0f}% missed", fontsize=9.5,
                           color=M_CRIT, fontweight='bold', pad=3)
        for r in (1, 2):
            ax[r, c].set_aspect('equal')
        for r in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    for r, t in enumerate(['input render (RGB)', 'generated from RGB',
                           'ground truth\ngreen covered / red missed']):
        ax[r, 0].set_ylabel(t, fontsize=10, fontweight='bold')
    fig.subplots_adjust(left=.075, right=.995, top=.955, bottom=.005,
                        wspace=.03, hspace=.10)
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)
    return out


def fig_charts(out):
    """E2 arms, data scaling, model scaling — one strip, one scale each."""
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

    arms = ['PC', 'PC+RGB', 'PC+RGB\n+D', 'PC+RGB\n+D+params']
    vals = [0.004950, 0.002504, 0.001706, 0.002478]
    cols = [INK3, M_ACCENT, M_ACCENT, M_CRIT]
    cols = ['#6b7c86', M_ACCENT, M_ACCENT, M_CRIT]
    b = ax[0].bar(arms, vals, color=cols, width=.62)
    for r, v in zip(b, vals):
        ax[0].text(r.get_x() + r.get_width()/2, v, f'{v:.5f}', ha='center',
                   va='bottom', fontsize=8.5, fontweight='bold')
    ax[0].set_title('E2 — modality value-add', fontsize=11, fontweight='bold')
    ax[0].set_ylabel('val PC chamfer  (lower better)', fontsize=9)
    ax[0].set_ylim(0, 0.0060)
    ax[0].tick_params(labelsize=8.5)

    xs, ys = [1000, 3000, 10000], [0.002994, 0.002582, 0.002405]
    ax[1].plot(xs, ys, '-o', color=M_ACCENT, lw=2.2, ms=7)
    ax[1].plot([10500], [0.002478], 'o', color=M_CRIT, ms=9)
    ax[1].annotate('10,500 plants scores 3.0% WORSE\nthan 10,000 → ≈3% noise floor',
                   xy=(10500, 0.002478), xytext=(2200, 0.00272),
                   fontsize=8.2, color=M_CRIT,
                   arrowprops=dict(arrowstyle='->', color=M_CRIT, lw=1.2))
    for x, y in zip(xs, ys):
        ax[1].annotate(f'{y:.5f}', (x, y), textcoords='offset points',
                       xytext=(0, -14), ha='center', fontsize=8)
    ax[1].set_xscale('log'); ax[1].set_xticks([1000, 3000, 10000])
    ax[1].set_xticklabels(['1k', '3k', '10k'])
    ax[1].set_title('E3 — data scaling, equal steps', fontsize=11, fontweight='bold')
    ax[1].set_xlabel('train plants', fontsize=9)
    ax[1].set_ylim(0.00225, 0.00320)
    ax[1].tick_params(labelsize=8.5)

    mx, my = [25.1, 114.3], [0.002989, 0.002478]
    ax[2].plot(mx, my, '-o', color=M_ACCENT, lw=2.2, ms=7)
    # No y-value for `large`: it is at 42% of schedule and its current number is
    # not comparable. Plotting it at ANY height asserts a result we do not have
    # -- an earlier draft put it at base's value, which read as "large ~= base".
    ax[2].axvline(332.2, color=M_BEFORE, ls='--', lw=1.4, alpha=.75)
    ax[2].annotate('large: still running\n(42% of schedule,\nno result yet)',
                   xy=(332.2, 0.00268), xytext=(120, 0.00292), fontsize=8.2,
                   color=M_BEFORE, ha='left',
                   arrowprops=dict(arrowstyle='->', color=M_BEFORE, lw=1.2))
    for x, y in zip(mx, my):
        ax[2].annotate(f'{y:.5f}', (x, y), textcoords='offset points',
                       xytext=(0, -14), ha='center', fontsize=8)
    ax[2].set_xscale('log'); ax[2].set_xticks([25.1, 114.3, 332.2])
    ax[2].set_xticklabels(['25M', '114M', '332M'])
    ax[2].set_title('E4 — model scaling, equal steps', fontsize=11, fontweight='bold')
    ax[2].set_xlabel('parameters', fontsize=9)
    ax[2].set_ylim(0.00225, 0.00320)
    ax[2].tick_params(labelsize=8.5)

    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
        a.grid(axis='y', color='#e4eaec', lw=.9)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)
    return out


# ── slide helpers ────────────────────────────────────────────────────────

def tb(slide, l, t, w, h, lines, *, size=16, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space=4, font='Calibri'):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    # A pptx text box does not clip: text longer than the box renders straight
    # over whatever sits below it. Declaring autofit asks the renderer to shrink
    # instead. It is a request, not a guarantee -- fonts substitute and metrics
    # differ between PowerPoint, Keynote and Slides -- so the layout below also
    # leaves real slack rather than relying on this alone.
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        txt, sz, bd, cl = (line if isinstance(line, tuple) else (line, size, bold, color))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        r = p.add_run(); r.text = txt
        r.font.size = Pt(sz); r.font.bold = bd
        r.font.color.rgb = cl; r.font.name = font
    return box


# Title box right edge and subtitle left edge, with a gutter between them.
# These must not be chosen independently: the title was 9.60 wide from L1.35,
# reaching R10.95, while the subtitle started at L9.90 -- so every title long
# enough to fill its box ran straight through the subtitle.
_T_L, _S_L, _GUT = 1.35, 9.90, 0.30


def head(slide, num, title, sub=None):
    # Title gets room for two lines at 22pt WITH slack, and the rule sits below
    # that. Earlier versions gave the title 0.86in and put the rule at 1.06in,
    # which is fine until a substituted font wraps the title to three lines and
    # it runs through the rule and into the figure.
    tb(slide, Inches(.55), Inches(.30), Inches(_T_L - .55 - .10), Inches(.4), num,
       size=13, bold=True, color=ACCENT, font='Consolas')
    tb(slide, Inches(_T_L), Inches(.18), Inches(_S_L - _T_L - _GUT), Inches(1.02),
       title, size=22, bold=True, color=INK)
    if sub:
        tb(slide, Inches(_S_L), Inches(.32), Inches(2.95), Inches(.58), sub,
           size=11, color=INK3, align=PP_ALIGN.RIGHT, font='Consolas')
    ln = slide.shapes.add_shape(1, Inches(.55), Inches(1.30), Inches(12.2), Pt(2.2))
    ln.fill.solid(); ln.fill.fore_color.rgb = INK
    ln.line.fill.background(); ln.shadow.inherit = False


def pic(slide, path, l, t, max_w, max_h):
    """Place an image letterboxed inside a box, never stretched."""
    iw, ih = Image.open(path).size
    s = min(max_w / iw, max_h / ih)
    w, h = int(iw * s), int(ih * s)
    return slide.shapes.add_picture(str(path), int(l + (max_w - w) / 2),
                                    int(t + (max_h - h) / 2),
                                    width=w, height=h)


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    return s


# ── deck ─────────────────────────────────────────────────────────────────

def build(figs, out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 — title + headline
    s = blank(prs)
    tb(s, Inches(.8), Inches(1.5), Inches(11.5), Inches(1.0),
       'EmbodiedMAE for Sorghum', size=40, bold=True, color=INK)
    tb(s, Inches(.8), Inches(2.5), Inches(11.5), Inches(.7),
       'Cross-modal generation, modality value-add, and scaling',
       size=19, color=INK2)
    tb(s, Inches(.8), Inches(3.5), Inches(11.5), Inches(.5),
       'Sorghum_15K validation split · held-out plants · 20 September 2026',
       size=13, color=INK3, font='Consolas')
    for i, (v, lab, col) in enumerate([
            ('0.3439 → 0.2781', 'cross-modal generation after distillation', ACCENT),
            ('−65.5%', 'PC chamfer from adding RGB + depth', ACCENT),
            ('≈3%', 'run-to-run noise floor, measured', BEFORE)]):
        x = Inches(.8 + i * 4.05)
        tb(s, x, Inches(4.6), Inches(3.8), Inches(.6), v, size=26, bold=True,
           color=col, font='Consolas')
        tb(s, x, Inches(5.3), Inches(3.8), Inches(.8), lab, size=12, color=INK2)

    # 2 — the three charts
    s = blank(prs)
    head(s, '01', 'Results at equal compute', 'every arm = 197,400 steps')
    pic(s, figs['charts'], Inches(.45), Inches(1.52), Inches(12.4), Inches(3.95))
    tb(s, Inches(.55), Inches(5.65), Inches(12.2), Inches(1.5), [
        ('RGB buys −49.4% and depth a further 16 points — but adding the parametric stream '
         'gives back everything depth gained.', 14, True, INK),
        ('10,000 and 10,500 plants differ by 5% in data, yet the larger arm scores 3.0% worse: those two runs are '
         'effectively a duplicate, so ≈3% is this pipeline\'s noise floor. That makes 3k→10k (−6.9%) only ~2.3× noise, '
         'while 25M→114M parameters buys −17.1%. At this budget, capacity pays better than data.', 12.5, False, INK2)])

    # 3 — coverage, green/red
    s = blank(prs)
    head(s, '02', 'What the model cannot predict',
         'ground truth, coloured by coverage')
    pic(s, figs['coverage'], Inches(.45), Inches(1.50), Inches(8.5), Inches(5.45))
    tb(s, Inches(9.25), Inches(1.55), Inches(3.6), Inches(5.35), [
        ('Green', 20, True, RGBColor(0x34, 0xA3, 0x5C)),
        ('a prediction lands within 0.03 of this ground-truth point.', 12.5, False, INK2),
        ('', 8, False, INK2),
        ('Red', 20, True, RGBColor(0xE0, 0x49, 0x3C)),
        ('nothing does — geometry the reconstruction never reached.', 12.5, False, INK2),
        ('', 10, False, INK2),
        ('Distillation closes the interior of the plant first. Thin leaf tips '
         'stay red longest in both models — they are the hardest geometry in '
         'the dataset.', 12.5, False, INK2),
        ('', 8, False, INK2),
        ('Threshold 0.03, not the 0.01 training default: at 0.01 about 80% of '
         'every cloud is red in both models, which is true but shows nothing. '
         'Chamfer reports a SQUARED distance, so 0.0028 is an RMS error near '
         '0.053.', 10.5, False, INK3)])

    # 4 — per view
    s = blank(prs)
    head(s, '03', 'Ten cameras, one plant',
         'generated from RGB alone')
    pic(s, figs['views'], Inches(.45), Inches(1.48), Inches(8.3), Inches(5.50))
    tb(s, Inches(9.05), Inches(1.55), Inches(3.8), Inches(5.35), [
        ('The view index is an exact elevation ladder', 14, True, INK),
        ('−0.9 + 0.2·index, from the camera matrix. All ten views share one '
         'point cloud and one parameter vector, so the answer never moves.', 12, False, INK2),
        ('', 8, False, INK2),
        ('Chamfer traces a U', 14, True, INK),
        ('best side-on, worst looking steeply up or down — in both models.', 12, False, INK2),
        ('', 8, False, INK2),
        ('Distillation improves 9 of 10 views, mean −28.5%', 13, True, ACCENT),
        ('…but best/worst spread widens 2.35× → 2.49×. It lowers the curve '
         'without flattening it, so it is not the lever for view robustness.', 12, False, INK2),
        ('', 8, False, INK2),
        ('', 9, False, INK2),
        ('Clouds are in the camera\'s own frame — same pose as the photograph.',
         10.5, False, INK3)])

    # 5 — parameters + next
    s = blank(prs)
    head(s, '04', 'Parameter head, and next steps',
         '142 leaf tokens · 8 plants')
    rows = [('starting_point', 0.95, 0.90), ('branching_angle', 0.93, 0.85),
            ('length', 0.67, 0.66), ('roll_angle', 0.50, 0.33),
            ('waviness_frequency', -0.02, -0.01),
            ('waviness_period_start_0', -0.01, -0.04),
            ('waviness_period_start_1', -0.01, 0.02)]
    tb(s, Inches(.55), Inches(1.46), Inches(6.2), Inches(.50),
       'Skill per leaf field   (1 = perfect, ≤ 0 = no better than the mean)',
       size=12, bold=True, color=INK)
    tbl = s.shapes.add_table(len(rows) + 1, 3, Inches(.55), Inches(2.04),
                             Inches(6.2), Inches(2.9)).table
    # Every column set explicitly, summing to the requested 6.2in. Setting only
    # column 0 leaves the other two at their default share of the original
    # width, so the table silently grows to 7.3in and runs into the right-hand
    # column of the slide.
    for i_c, w_c in enumerate([3.2, 1.5, 1.5]):
        tbl.columns[i_c].width = Inches(w_c)
    for j, t in enumerate(['leaf field', 'from PC', 'from RGB']):
        c = tbl.cell(0, j); c.text = t
        c.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        c.text_frame.paragraphs[0].runs[0].font.bold = True
    for i, (n, a, b) in enumerate(rows, start=1):
        dead = a <= 0
        for j, t in enumerate([n, f'{a:+.2f}', f'{b:+.2f}']):
            c = tbl.cell(i, j); c.text = t
            r = c.text_frame.paragraphs[0].runs[0]
            r.font.size = Pt(11)
            r.font.bold = (j > 0 and not dead)
            r.font.color.rgb = CRITICAL if dead else (ACCENT if j > 0 else INK)
    tb(s, Inches(.55), Inches(5.00), Inches(6.2), Inches(1.9), [
        ('Four of seven fields carry the whole result. The three waviness fields '
         'emit the dataset average and nothing more, from either source — so ~40% '
         'of the leaf vector dilutes a real result with a constant.', 12, False, INK2),
        ('PC beats RGB on every field with skill, and on 7 of 8 plants.', 12.5, True, INK)])

    tb(s, Inches(7.1), Inches(1.52), Inches(5.7), Inches(5.25), [
        ('Two decisions needed', 17, True, INK),
        ('', 6, False, INK2),
        ('1 · Build the downstream linear probe', 14, True, ACCENT),
        ('Plan decision 6.4 requires it, it does not exist, and it is the only '
         'thing that can tell us whether the E2 parameter result is a '
         'token-budget artefact or real. Three of four targets are already in '
         'features.csv; biomass needs a definition and “leaf angle” is ambiguous '
         'between roll and branching.', 12, False, INK2),
        ('', 6, False, INK2),
        ('2 · Choose the E8 baseline set', 14, True, ACCENT),
        ('Ranked the largest reject risk; nothing in the repo addresses it. Each '
         'candidate is a training run that must fit before the 24 Oct freeze, so '
         'the decision is more time-critical than the runs.', 12, False, INK2),
        ('', 8, False, INK2),
        ('Status: E2 complete (4/4) · E3 2 of 3 · E4 1 of 2. e3_1k and e4_large '
         'were preempted at 94% and 42% and are resuming from checkpoint.',
         10.5, False, INK3)])

    prs.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clouds', default='vis_unpredicted/clouds.json')
    ap.add_argument('--gallery', required=True,
                    help='directory holding views.json and the view_NN.jpg renders')
    ap.add_argument('--figdir', default='vis_deck')
    ap.add_argument('--out', default='sorghum_results_2026-09-20.pptx')
    ap.add_argument('--thr', type=float, default=0.03)
    a = ap.parse_args()

    fd = Path(a.figdir); fd.mkdir(parents=True, exist_ok=True)
    figs = {
        'coverage': fig_coverage(a.clouds, fd / 'coverage.png', thr=a.thr),
        'views':    fig_views(Path(a.gallery) / 'views.json', a.gallery,
                              fd / 'views.png', thr=a.thr),
        'charts':   fig_charts(fd / 'charts.png'),
    }
    for k, v in figs.items():
        print(f"  figure {k:<9} -> {v}")
    out = build(figs, a.out)
    print(f"\nwrote {out}  ({Path(out).stat().st_size/1e6:.1f} MB, 5 slides)")


if __name__ == '__main__':
    main()
