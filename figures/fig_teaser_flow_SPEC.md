# Build request: Fig. 1 teaser as a UGG-style flow diagram

**For the Nova session.** Build a new teaser figure in the style of the flow animation on
https://ugg-model.github.io/ (the auto-cycling "inputs → one joint model → outputs" diagram at the
top of its Results section): three columns connected by colored Bézier wires with arrowheads.
Static TikZ for the paper; same pattern as our fig2/fig4/fig5 TikZ figures.

## Deliverables

1. `figures/make_teaser_flow_tikz.py` — builder on `figures/_scene.py` (`Scene`), emitting:
   - `paper/overleaf/Figures/fig_teaser_flow_tikz.tex`
   - per-tile PNGs in `paper/overleaf/Figures/fig1/`
2. Do **not** touch `main.tex` yet — leave the current `fig_teaser_tikz_ugg` wired in. The Mac
   session will swap the `\input` line, check height, and rebuild after pulling.
3. Push the builder, the .tex, and the tiles.

## Layout (full \textwidth, target ≤ 6.5 cm tall — check height first, before polishing tiles)

Three zones, left to right. Two stacked species rows (sorghum on top, maize below) share ONE
central model box.

**Left column — "INPUT: any subset, from an occluded row".** Per species, three small real tiles
stacked or in a tight column:
- row photograph with the dashed box on the target plant (reuse the current teaser row renders),
- the row depth frame,
- the 24,576-pt scene cloud.
Label under the column: "no mask or bounding box given". The photo wire is drawn solid/active; the
depth and cloud wires half-dimmed, with a small bracket/legend "any one of these suffices" — the
static stand-in for UGG's task cycling.

**Center — the model box.** "CropCode" (use `\method{}` styling consistent with fig2's title),
three short lines: masked pretraining · cross-modal distillation · row fine-tuning. This is the
visual anchor, as UGG's "ONE JOINT MODEL" box is.

**Right column — "OUTPUT: procedural plant".** One card per species: the written recipe
re-rendered as a **3D mesh image** in the project-site style (`figures/_proc3d.py`, as in the
current Fig. 1 3D panels from 0c52f86): predicted plant in purple over the true plant in
translucent grey. Under each mesh: predicted vs true height and leaf count (the within-2% / ±1-leaf
numbers). Column heading carries the tagline: "kilobytes · editable · re-renderable". Do NOT use
the stem-and-leaf schematic here, and drop the completed RGB/depth/cloud streams (that story is
Fig. 4's) — or at most a de-emphasized footnote row if space allows.

**Wires.** Cubic Bézier with small arrowheads, colored by modality using the PAPER palette already
used in the fig4/any2all tiles (RGB blue, depth yellow, point cloud pink, recipe purple) — not
UGG's colors. Inactive wires grey.

## Plants

Use the same two held-out test rows/plants as the current teaser (so the caption's numbers stay
true), or if tile quality demands different plants, pick from the fig1_teaser recipes already in
`procedural_models/procedural_models_for_viz.zip` and say so in the commit message.

## Caption

The current teaser caption ("Any sensor in, procedural plant out ... within 2% ... within one
leaf (Regime III, Sec. ref)") is mostly reusable; propose an updated draft in the commit message
but do not edit main.tex.

— requested by the Mac session, 2026-11-14
