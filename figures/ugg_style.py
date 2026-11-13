"""Shared figure conventions taken from the UGG paper (Unified Generative Geometry Model, arXiv:2611.14440,
https://ugg-model.github.io/assets/ugg-paper.pdf), for the CropCode paper figures. 2026-11-13, user: "update my
figures getting motivation from this paper" and earlier "for previous diagrams the font was good so keep that".

FONTS ARE NOT TOUCHED HERE: every script keeps the font it already uses. Only colour, marks and layout change.

What UGG's figures do, and the helper that does it here:
  * one pastel colour per data stream, used for boxes, token squares and series alike  -> STREAM, STREAM_INK, STREAM_FILL
    our streams: image = blue, depth image = yellow, point cloud = pink, procedural parameters (recipe) = purple
  * a stream SET (E2 arms) takes the colour of the stream it adds: PC pink, +RGB blue, +D yellow, +recipe purple
                                                                                       -> ARM
  * our model highlighted (UGG writes "Ours" in bold violet); baselines / references in greys     -> OURS, GREYS
  * small-multiple line plots (UGG Fig. 7): hollow round markers on a solid line, the reference
    (single sample / no adaptation) as a dashed grey line, the first and last values written in bold on the
    y axis in the series colour, a one-row legend above the panels, light horizontal grid, no top/right spines
                                                                                       -> style_axes, label_ends, top_legend
  * qualitative grids (UGG Figs. 4 and 6): bold column headers and row labels, "Ours" in OURS colour,
    thin white gaps, white zoom boxes with zoom-in crops                                -> header, row_label, zoom_box
  * tables: best / second best shaded lavender (in LaTeX, \\cellcolor{uggbest} / {uggsecond})   -> TABLE_BEST, TABLE_SECOND
"""
import matplotlib.patches as mpatches
import numpy as np

# stream colours: edge (UGG's sampled pastel), ink (darker, for lines, points and text), fill (box background)
STREAM = {'rgb': '#809ec2', 'depth': '#e0b020', 'pc': '#d091a7', 'text': '#9d8cc4'}
STREAM_INK = {'rgb': '#3f5f8f', 'depth': '#a07800', 'pc': '#a8456e', 'text': '#5e4a96'}
STREAM_FILL = {'rgb': '#e6ecf3', 'depth': '#faf2d4', 'pc': '#f6e9ed', 'text': '#eae7f2'}
# E2 stream sets: the colour of the stream each arm adds (markers differ too, so the series survive greyscale)
ARM = {'PC': STREAM_INK['pc'], 'PC+RGB': STREAM_INK['rgb'], 'PC+RGB+D': STREAM_INK['depth'],
       'PC+RGB+D+θ': STREAM_INK['text'], 'PC+RGB+D+recipe': STREAM_INK['text']}
ARM_MARKER = {'PC': 'o', 'PC+RGB': 's', 'PC+RGB+D': 'D', 'PC+RGB+D+θ': '^', 'PC+RGB+D+recipe': '^'}
OURS = '#6a2fd6'                       # UGG's "Ours" violet
SERIES = ['#3f5f8f', '#a8456e', '#a07800', '#5e4a96', '#2e7d6b', '#8c8c8c']   # generic ordered series
GREYS = ['#4d4d4d', '#8c8c8c', '#b3b3b3', '#d0d0d0']
REF = '#9a9a9a'                        # dashed reference lines
GRID = '#e6e6e6'
ARROW = '#8c8c8c'
TABLE_BEST, TABLE_SECOND = '#dcdcf7', '#eeeefb'   # \definecolor{uggbest}{HTML}{DCDCF7}, {uggsecond}{HTML}{EEEEFB}


def style_axes(ax, grid='y'):
    """UGG plot axes: no top/right spines, thin grey left/bottom, light grid on `grid` ('y', 'x', 'both' or None)."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#666666'); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors='#333333', width=0.6, length=2.5)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


def line(ax, x, y, color, label=None, marker='o', hollow=True, lw=1.4, ms=4.0, **kw):
    """Solid line with hollow round markers (UGG Fig. 7)."""
    return ax.plot(x, y, color=color, lw=lw, marker=marker, ms=ms, mew=1.0,
                   mfc='white' if hollow else color, mec=color, label=label, **kw)


def reference(ax, y=None, x=None, label=None, color=REF):
    """Dashed grey reference (UGG: the single-sample result)."""
    if y is not None:
        return ax.axhline(y, color=color, ls=(0, (4, 3)), lw=0.9, label=label, zorder=1)
    return ax.axvline(x, color=color, ls=(0, (4, 3)), lw=0.9, label=label, zorder=1)


def label_ends(ax, y_first, y_last, color, fmt='{:.3g}', first_color=REF, fontsize=None):
    """Write the first and last values in bold on the y axis (UGG Fig. 7): the start in grey, the end in the
    series colour, as extra tick labels at those heights."""
    tl = ax.yaxis.get_ticklabels()
    fs = fontsize or (tl[0].get_fontsize() if tl else 7)
    tr = ax.get_yaxis_transform()
    for y, c in ((y_first, first_color), (y_last, color)):
        ax.text(-0.02, y, fmt.format(y), transform=tr, ha='right', va='center', fontsize=fs, fontweight='bold', color=c,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))


def top_legend(fig_or_ax, handles=None, labels=None, ncol=None, y=1.0, fontsize=None, **kw):
    """One-row legend centred above the panels, no frame (UGG Fig. 7)."""
    if handles is None:
        handles, labels = fig_or_ax.get_legend_handles_labels() if hasattr(fig_or_ax, 'get_legend_handles_labels') else ([], [])
    ncol = ncol or max(1, len(handles))
    tgt = fig_or_ax
    return tgt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      handlelength=1.8, columnspacing=1.4, fontsize=fontsize, **kw)


def header(fig, x, y, text, color='#111111', size=None, **kw):
    """Bold column / dataset header of a qualitative grid (figure coordinates)."""
    return fig.text(x, y, text, ha='center', va='bottom', fontweight='bold', color=color, fontsize=size, **kw)


def row_label(fig, x, y, text, ours=False, size=None, **kw):
    """Bold row label of a qualitative grid; our model's row in OURS violet."""
    return fig.text(x, y, text, ha='right', va='center', fontweight='bold', color=OURS if ours else '#111111', fontsize=size, **kw)


def zoom_box(ax, x0, y0, w, h, color='white', lw=1.2):
    """White square marking the region shown in a zoom-in crop (UGG Fig. 6)."""
    ax.add_patch(mpatches.Rectangle((x0, y0), w, h, fill=False, ec=color, lw=lw, zorder=10))


def pastel_box(ax, x, y, w, h, stream=None, dashed=False, r=0.05, lw=1.0, z=-1, transform=None):
    """Rounded pastel box in a stream's colour; dashed = an input, solid = an output."""
    fc, ec = (STREAM_FILL[stream], STREAM[stream]) if stream else ('#f4f4f5', '#9a9a9a')
    p = mpatches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                ls=(0, (3.2, 2.0)) if dashed else '-', zorder=z,
                                **({'transform': transform} if transform is not None else {}))
    (ax.add_artist if transform is not None else ax.add_patch)(p)
    return p
