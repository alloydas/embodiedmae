# Plan from the 2026-11-10 group meeting (CVPR 28592)

Written on the Mac session at commit `4f17a53`; the live copy for the team is the
Claude doc "CropCode CVPR Plan — 2026-11-10 Meeting Actions"
(https://claude.ai/artifact/HWUgfSPooyWDkziFpuXG5G).

## FOR THE NOVA SESSION — in priority order

1. **E10 real-world evaluation (highest value).** Run the real maize field data
   Anthirik shares through the headline / distilled models (`eval_rgb2pc_*`,
   protocol in `OOD_EVAL_rgb2pc.md`). This clears the last three red todos in
   `paper/overleaf/main.tex`: the abstract's `\todo{matches}`, Sec. 4.8
   (`sec:exp-real`), and the "Field scans" row of `tab:fieldproc`. Results also
   feed a real-data teaser panel.
2. **Beautify the generated point cloud in Fig. 1** — make it read as a plant,
   not a z-colored scatter (`figures/make_teaser_v2.py`, the cloud tile). The
   teaser is now native TikZ: only the tile PNG changes; after regenerating the
   drawio, rerun `python figures/teaser_drawio_to_tikz.py` (it extracts the
   tiles to `paper/overleaf/Figures/teaser/` and rewrites
   `Figures/fig_teaser_tikz.tex`; the meeting's column labels are pinned in its
   `LABEL_MAP`).
3. **Image-based comparison metrics** (MultiMAE and the other frozen baselines)
   plus point-cloud metrics (Chamfer, precision/recall/F1) for the supplement,
   per Adarsh.
3b. **Regime I Chamfer in millimetres (added 2026-11-10, after the meeting).**
   `eval/regime1_eval.py` and `eval/regime1_extra.py` now compute per-plant
   `scale_m` (metres per unit-sphere unit, from the raw camera-frame cloud) and
   report `rms_mm_mean` / `rms_mm_median` = sqrt(chamfer_i) x scale_m_i x 1000,
   plus a per-plant `regime1_items_<job>.csv` so unit changes never need another
   model rerun. Rerun both on val (1 GPU, same shape as jobs 17265134 /
   17276693: `python eval/regime1_eval.py` and `python eval/regime1_extra.py`),
   commit the `reports/regime1*_<job>.csv` files and push; the Mac session then
   puts the mm numbers into `tab:regime1`. Do NOT convert the aggregate
   chamfer_e3 by a single scale: the mean of sqrt(c_i)*s_i is not
   sqrt(mean c)*median s.
4. **High-quality field teaser with Anthirik**: complex field photo → one
   occluded modality → final 3D reconstruction.
5. Optional, settles an open label question: render `G(theta_hat)` meshes for
   the two teaser plants so the "Generated point cloud" panel could become a
   true re-render.

## Done on the Mac today

- Teaser converted draw.io → TikZ (`figures/teaser_drawio_to_tikz.py` →
  `paper/overleaf/Figures/fig_teaser_tikz.tex` + `Figures/teaser/*.png`);
  `main.tex` and `main_clean.tex` include it instead of `fig_teaser.pdf`.
- Meeting terminology in the diagram: Image, Depth image, Point cloud,
  Procedural parameters, Generated point cloud, Traits.
- Clean marker-free single-line copy: `paper/overleaf/main_clean.tex`.
- Labmate's supplement ablations merged earlier (B.9–B.12 + their figures);
  Sec. 4.6's E7 todo resolved by the loss ablation.

## Queued on the Mac (main.tex text edits)

- `tab:fieldproc`: "MaizeField3D" → "field scans".
- Regime I table: Chamfer units in header/caption; leaf-count ±1 definition
  with denominator (2,250 val plants/species); height error clarified as
  relative; rule/brace separating end-to-end rows from the modular two-stage
  row.

## Decisions pending (do not act without the user)

- Title: Overleaf has "CropCode: Inverse Procedural Modeling of Occluded Row
  Crop Geometry from Sparse Multimodal Field Observations"; the meeting title
  has no "CropCode:" prefix. OpenReview must match; the user updates it.
- "Generated point cloud" vs "re-rendered point cloud" label (meeting notes
  conflict; current figure says Generated, which matches what is plotted).
- Main-vs-supplement cut: decided Thursday 11:00 with Adarsh — do not trim
  pages beforehand.
