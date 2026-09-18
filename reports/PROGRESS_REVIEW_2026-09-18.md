# Progress review — 2026-09-18

Cross-modal generation before and after distillation, the E2 modality ablation,
and E3/E4 status. Presentation version:
<https://claude.ai/artifact/A6nkyfGFfnpy31dq9AdYWw>

All figures below are on the **Sorghum_15K validation split** (held-out plants,
70/15/15 by plant, seed 42).

---

## 1. Cross-modal generation: one modality in, four out

**Before** = the warm-started pretrain checkpoint at **epoch 0**, before a single
distillation step. **After** = `4m_distill_15k_all` at **epoch 100**.

| source | total before | total after | Δ total | pc_chamfer before | pc_chamfer after | Δ chamfer |
|---|---|---|---|---|---|---|
| RGB           | 0.1046 | 0.0931 | −11.0 % | 0.00104 | 0.00088 | −15.4 % |
| Depth         | 0.1091 | 0.0976 | −10.5 % | 0.00080 | 0.00062 | −22.5 % |
| Point cloud   | 0.6029 | 0.4374 | −27.5 % | 0.00281 | 0.00122 | −56.6 % |
| Spline params | 0.5590 | 0.4844 | −13.3 % | 0.01662 | 0.00634 | −61.9 % |
| **mean_gen**  | **0.3439** | **0.2781** | **−19.1 %** | — | — | — |

`param_mae_masked` moves the same way: from a point cloud it falls **43.8 %**
(0.0617 → 0.0347), against ~9–10 % from RGB or depth.

The shape of the result is that distillation helps most where the model was
weakest. RGB and depth were already decent conditioning signals and gain ~11 %;
the two weak sources gain two to six times that. That is what the frozen
full-modal teacher is for.

### PC → params on its own

The cell the paper's claim rests on — geometry in, procedural parameters out.
Two runs do it, both warm-started from the same checkpoint, so they share the
0.0617 baseline.

| run | distilled targets | param MAE from PC | vs warm start | lr |
|---|---|---|---|---|
| warm start | — | 0.0617 | baseline | — |
| `4m_distill_15k_pc2text` | params only | 0.0389 | −36.9 % | 3e−5 |
| `4m_distill_15k_all` | all four | **0.0347** | **−43.8 %** | 1e−4 |

**The generalist beats the parameter specialist at its own job** by 11 %. The
other three reconstruction targets act as useful auxiliary supervision for
parameter regression rather than competing with it.

Caveat before this goes in a slide: the two runs used different learning rates,
so part of the gap could be LR rather than the target set. A matched-LR
`pc2text` run settles it and is cheap — 100 epochs on a single source.

### Provenance — read this before re-quoting the numbers

| number | comes from |
|---|---|
| before (epoch 0) | `logs/evalwarm_12085946.out`, produced by `eval_warmstart.py`, which loads `outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth` and evaluates without training |
| after (epoch 100) | last `val` entry in `outputs/4m_distill_15k_all/training_history.json` |
| figures | `outputs/4m_distill_15k_all/visualizations/epoch_{001,100}_src-pc_sample_1_Sorghum_10001_00.png` |
| PC→params specialist | `outputs/4m_distill_15k_pc2text/training_history.json` |
| per-point miss maps | `vis_pc_unpredicted.py` (see below) |

**Two traps in the older numbers.**

1. **The `4m_distill_src_*` runs show ~−74 % and must not be quoted.** Their
   `config.json` gives `data_root: ./Dataset/new_data` — the old 10k set, whose
   val split is leaked (100 folders covering 10 plants, all also in train; see
   `OOD_EVAL_rgb2pc.md`). Their teacher is also the old `4m_run_v3` checkpoint.
   The −74 % is measuring memorisation, not generalisation.
2. **The previously published −5.9 % is not wrong, it is a different baseline.**
   It measures epoch 1 → epoch 100 and so excludes everything the first
   distillation epoch bought. Against the true epoch-0 warm start the same run
   gives −19.1 %. `eval_warmstart.py` exists precisely to close that gap — its
   docstring says so.

### The figures

Same plant and view, point cloud as the **input**. Its reconstruction therefore
does not change between the two, and the per-sample chamfer printed on the grids
(0.00150 → 0.00157) moves with the unseeded point-sampling RNG — quote the
validation aggregate, not the number on the image. What genuinely improves are
the **generated** panels: at epoch 1 the depth map is one blurred mass, and by
epoch 100 individual leaf blades separate and the generated RGB resolves leaf
structure rather than a blob.

---

## 1b. What the parameter head actually recovers

PC → params and RGB → params from the same model (`4m_distill_15k_all`), the same 8
plants as the point-cloud viewer, every other modality masked. **Skill** = 1 − MAE ÷ (error from always
predicting the field's mean). 1.0 is perfect; **≤ 0 means nothing learned beyond
the average plant.** 142 leaf tokens over 8 plants.

| leaf field | GT spread (sd) | MAE from PC | skill PC | MAE from RGB | skill RGB |
|---|---|---|---|---|---|
| starting_point          | 0.248  | 0.0097 | **0.95** | 0.0205 | **0.90** |
| branching_angle         | 5.31°  | 0.32°  | **0.93** | 0.65°  | **0.85** |
| length                  | 0.176  | 0.0432 | **0.67** | 0.0440 | **0.66** |
| roll_angle              | 107.1° | 45.8°  | **0.50** | 61.4°  | **0.33** |
| waviness_frequency      | 0.0039 | 0.0032 | −0.02 | 0.0032 | −0.01 |
| waviness_period_start_0 | 27.0°  | 23.5°  | −0.01 | 24.3°  | −0.04 |
| waviness_period_start_1 | 29.2°  | 25.9°  | −0.01 | 24.9°  | 0.02 |

### Per plant

Normalised leaf MAE, all 8 plants, same model, every other modality masked.

| plant | leaves | from PC | from RGB | better |
|---|---|---|---|---|
| Sorghum_10001_00 | 24 | 0.0488 | 0.0604 | PC |
| Sorghum_10016_00 | 24 | 0.0589 | 0.0628 | PC |
| Sorghum_10017_00 | 18 | 0.0621 | 0.0554 | **RGB** |
| Sorghum_1001_00  | 20 | 0.0488 | 0.0522 | PC |
| Sorghum_10037_00 | 14 | 0.0334 | 0.0523 | PC |
| Sorghum_10065_00 | 10 | 0.0358 | 0.0540 | PC |
| Sorghum_10069_00 | 10 | 0.0302 | 0.0380 | PC |
| Sorghum_1007_00  | 22 | 0.0601 | 0.0737 | PC |
| **mean** | 142 | **0.0473** | **0.0561** | PC on 7/8 |

`Sorghum_10017_00` is the one plant RGB wins, and it is also the plant with the
worst PC score of the eight — worth a look before claiming PC dominance without
qualification. n = 8, so this is an illustration, not a significance test.

**Four of seven leaf fields carry the whole result; three are at zero skill.**
The three waviness fields sit at zero from both sources — the head emits the
dataset average and nothing more. A single aggregate `param_mae` averages these
in, so ~40 % of the leaf vector dilutes a real result with a constant. (This
corroborates the earlier per-parameter finding independently.)

**PC beats RGB on every field that has any skill**, widest on `roll_angle`
(0.55 vs 0.36) — geometry in, geometry out. Caveat on `branching_angle`: both
look strong partly because the field barely varies, 5.1° of spread against
roll_angle's 108°.

### One leaf, end to end

Leaf 1 of `Sorghum_10001_00`, unseen plant, leaf token masked so it is generated.

| field | ground truth | from PC | from RGB |
|---|---|---|---|
| starting_point          | 0.199  | 0.198  | 0.194  |
| length                  | 0.243  | 0.266  | 0.239  |
| roll_angle              | 65.06° | 70.71° | 74.96° |
| branching_angle         | 20.30° | 20.24° | 20.10° |
| waviness_frequency      | 0.054  | 0.054  | 0.055  |
| waviness_period_start_0 | 60.14° | 36.28° | 53.61° |
| waviness_period_start_1 | 91.16° | 37.42° | 44.69° |
| stem_length (plant)     | 1.344  | 1.468  | 1.251  |

Position, length and the two angles land close; the two waviness phases are off
by 24–54°, most of their range.

```bash
python dump_param_examples.py --config configs/config_4m_distill_15k_all.yaml \
  --checkpoint outputs/4m_distill_15k_all/best_model.pth \
  --source pc --n 8 --indices 0,10,20,30,40,50,60,70 --out vis_params
```

---

## 1c. Ten images of one plant — the elevation experiment

The ten folders of a plant are a natural experiment. The view index is an
**exact elevation ladder**: carrying world-up through each view's
`worldToCamera` gives a component along the camera's forward axis of precisely
0.9 − 0.2·index, so the viewing direction's elevation is **−0.9 + 0.2·index** —
steeply down at view 00, side-on in the middle, steeply up at view 09 — while
azimuth is randomised. (This reproduces the formula already recorded in
`plot_view_elevation_effect.py`; an earlier draft of this file reported the
camera-*position* elevation instead, a different and non-exact quantity.) The
camera frame is OpenGL-style: x right, y **up**, forward −z. All ten views share one point cloud
(`_nc_cam.ply` is the same geometry rotated into the view frame, and the loader
centres and unit-scales before Chamfer) and one set of spline parameters.
**The answer never moves; only the input image does.**

Generated from **RGB alone**, every other modality masked.

| view | sin(elev) | chamfer before | chamfer after | change |
|---|---|---|---|---|
| 00 | −0.90 | 0.001456 | 0.001016 | −30.2 % |
| 01 | −0.70 | 0.000970 | 0.000674 | −30.5 % |
| 02 | −0.50 | 0.000854 | 0.000653 | −23.5 % |
| 03 | −0.30 | 0.001328 | 0.000819 | −38.3 % |
| 04 | −0.10 | 0.000986 | 0.000608 | −38.3 % |
| 05 | +0.10 | 0.000873 | 0.000654 | −25.1 % |
| 06 | +0.30 | **0.000775** | **0.000553** | −28.6 % |
| 07 | +0.50 | 0.000836 | 0.000863 | +3.2 % |
| 08 | +0.70 | 0.001820 | 0.001150 | −36.8 % |
| 09 | +0.90 | 0.001804 | 0.001380 | −23.5 % |
| **mean** | | **0.001170** | **0.000837** | **−28.5 %** |
| best/worst spread | | 2.35× | 2.49× | |

**Both models trace the same U** — best near side-on (view 06), worst at both
extremes. The target is identical from every view, so that shape is caused
entirely by what the camera can see.

**Distillation lowers the curve without flattening it.** 9 of 10 views improve,
mean −28.5 %, but the best-to-worst ratio *widens* 2.35× → 2.49×. If view
robustness is the goal, distillation as configured is not the lever.

### Parameters from ten different images

All ten views share one parameter ground truth, so the spread of the ten
predictions is a view-invariance measure needing no extra labels. Reported as a
fraction of how much each field varies between leaves.

| leaf field | MAE before | MAE after | Δ | view spread before | after |
|---|---|---|---|---|---|
| starting_point          | 0.0380 | 0.0155 | −59.2 % | 0.10 | 0.08 |
| branching_angle         | 1.217° | 0.487° | −60.0 % | 0.16 | 0.12 |
| roll_angle              | 49.7°  | 35.6°  | −28.4 % | 0.37 | 0.36 |
| length                  | 0.0482 | 0.0407 | −15.6 % | 0.17 | 0.16 |
| waviness_frequency      | 0.0028 | 0.0028 | −2.2 %  | 0.11 | **0.15** |
| waviness_period_start_0 | 22.04° | 22.48° | **+2.0 %** | 0.20 | **0.26** |
| waviness_period_start_1 | 29.88° | 31.30° | **+4.8 %** | 0.18 | **0.26** |

**The zero-skill fields got noisier, not better.** The four fields with real
skill improved on both counts. The three waviness fields did the opposite:
accuracy flat or slightly worse while view spread rose from 0.18–0.20 to 0.26.
Before distillation the head emitted something near a constant for these, which
is at least stable; after, it varies more with the input image while being no
more correct — it has started tracking image noise. A field nothing can predict
is not free to carry: it absorbs capacity and adds variance.

Caveat: chamfer here is a single point-cloud draw and `load_pointcloud`
resamples 8,196 points unseeded on every read, so per-view figures move ~5 %
between runs. The U-shape and the 28.5 % gap are far larger than that; the
+3.2 % on view 07 is not, so treat that view as a tie rather than a regression.

### Coverage per view

`--export_gallery` also writes the ten input renders beside the cloud each one
produced, plus the per-GT-point nearest-neighbour distance so the ground truth
can be coloured green (covered) / red (missed).

**Camera-frame convention.** `_nc_cam.ply` is already in the camera frame, so
one orientation reproduces the photograph, and the gallery opens there. The
frame is **x right, y up, forward −z** (OpenGL-style), established from the
camera matrix: world-up carried through `worldToCamera` lands on camera **+y**
in all ten views (component 0.436–0.995). A Y-up viewer sitting on +Z already
looks down −Z with +Y up and +X right, so no remapping is needed.

An earlier version inferred this by scoring point-cloud silhouettes against the
renders instead. That picked (x, −y) and rendered every cloud **upside down**.
Silhouette IoU cannot separate a plant from its vertical mirror on a
near-radially-symmetric subject — 0.383 against 0.326 is not a margin — and the
geometric check settles in one line what the pixel check could not settle at
all. Prefer the camera matrix over pixel agreement for any frame question.

Coverage tracks the same U as chamfer:

| view | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|---|
| % missed @ 0.01 | 67.6 | 66.4 | 65.9 | 65.8 | **58.7** | 67.5 | 67.2 | 72.4 | **77.2** | 76.4 |
| mean NN | 0.0198 | 0.0160 | 0.0149 | 0.0177 | **0.0147** | 0.0170 | 0.0149 | 0.0173 | 0.0201 | **0.0215** |

The steep-up end (views 08–09) is worst on both measures. Note the best view
differs by measure — 04 on coverage, 06 on chamfer — which is the single-draw
resampling noise, not a real disagreement.

```bash
python eval_views_one_plant.py --plant Sorghum_10001 --source rgb \
  --export_gallery <dir>
```

---

## 2. E2 — modality value-add

Four arms identical in every respect except which token streams exist. Compared
on `val_pc_chamfer`, the only metric computed identically in every arm.

| arm | pc_chamfer | vs PC-only | epochs |
|---|---|---|---|
| PC only                    | 0.004950 | baseline | 600/600 |
| PC + RGB                   | 0.002504 | −49.5 %  | 600/600 |
| PC + RGB + depth           | 0.001706 | −65.6 %  | 600/600 |
| PC + RGB + depth + params  | 0.002478 | −50.0 %  | 581/600 |

**The parametric arm loses to PC + RGB + depth**, giving back everything depth
gained. At 581/600 that gap will not close.

Two readings, with different consequences:

- **Mechanical.** At `mask_ratio` 0.80 with `min_mask_ratio` 0.25 per modality, a
  fourth token stream splits the visible-token budget four ways instead of
  three. The PC decoder sees less to work from, so chamfer penalises the params
  arm for *existing* rather than for being unhelpful.
- **Real.** The parametric modality does not help the representation.

Chamfer cannot separate these. The downstream linear probe can, and plan
decision 6.4 already names that probe — not reconstruction loss — as the metric
that settles modality value-add. This makes the probe the decisive missing
tooling rather than a documentation gap.

---

## 3. E3 / E4 — scaling runs in flight

Every arm runs the same **197,400 optimizer steps**, not the same epochs. See
the config-model section of `CLAUDE.md` for why, and do not normalise the epoch
counts.

| run | scale | state | progress |
|---|---|---|---|
| `e3_10k`   | 10,000 plants  | running | 436 / 631 epochs, 1d 02h |
| `e3_3k`    | 3,000 plants   | running | 100 / 2,100 epochs, 3h |
| `e3_1k`    | 1,000 plants   | queued  | 6,169 epochs, est. 19 Sep |
| `e4_small` | 25.1M params   | queued  | 600 epochs, est. 25 Sep |
| `e4_large` | 332.2M params  | queued  | 600 epochs, est. 25 Sep |

The base-model / full-data point comes from the E2 four-modality arm rather than
a sixth job. `e3_10k` (10,000 plants) and that arm (10,500) are within 5 % of
each other and double as the only run-to-run error bar in the E1–E10 matrix.

**Throughput is degrading under contention.** `e2_pcrgbd` took **28.9 h**
against `e2_pc`'s **12.2 h** on the same config, node and resources. The pipeline
is dataloader-bound and three co-resident jobs share the node's CPUs. Still
inside the 24 Oct freeze, but the margin is shrinking.

---

## 4. Open decisions

1. **Build the downstream linear probe.** Decision 6.4 makes it the value-add
   metric; it does not exist, so E2/E3/E4 are all on track to finish with
   chamfer curves and no probe number. It is also the only thing that can tell
   us whether the E2 params result is a token-budget artefact. Three of four
   targets are in `features.csv` (`stem_length`, `n_leaves`, and leaf angle —
   ambiguous between `roll_angle` and `branching_angle`); **biomass is in
   neither `features.csv` nor the spline params** and needs a definition.
2. **Decide the E8 baseline set.** Ranked the largest reject risk, nothing in
   the repo addresses it, and each candidate is a training run that must fit
   before 24 Oct — so the decision is more time-critical than the runs.
3. **Confirm protocol with whoever runs E6/E7.** Their numbers land in the same
   table as ours, so the comparison holds only if they match on the same
   70/15/15 split at seed 42, global batch 32, and a budget in optimizer steps
   rather than epochs. The last is likeliest to diverge.

---

## 5. Per-point coverage — `vis_pc_unpredicted.py`

Chamfer averages two failures that look nothing alike: geometry the model
**missed** and geometry it **invented**. Averaged into one scalar they cancel,
and neither is visible. The script colours a cloud by which points fall on the
wrong side of a nearest-neighbour threshold:

```bash
python vis_pc_unpredicted.py \
  --checkpoint outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth \
  --checkpoint outputs/4m_distill_15k_all/best_model.pth \
  --label "before distillation" --label "after distillation" \
  --source pc --num_samples 3 --export_json vis_unpredicted/clouds.json
```

`--stride 10` matters: folders are `<plant>_<view>` with ten views each, so a
stride of 1 gives ten views of ONE plant and reads as far more variety than it
shows. Stride 10 gives one view each of N distinct plants.

Over 8 distinct plants, at the default threshold (0.01, i.e. `qal_threshold`):

| | mean % missed | mean NN distance |
|---|---|---|
| before distillation | 86.3 % | 0.0334 |
| after distillation  | 76.3 % | 0.0215 |

Mean NN improves on **every one of the eight plants**. The red fraction does
not: `Sorghum_10017_00` goes 82.1 % → 82.2 % while its mean NN improves 38 %
(0.0334 → 0.0208). A single threshold collapses a continuous improvement into a
binary that can miss it entirely, so quote the distance, or sweep the threshold,
rather than resting on one percentage.

**That number is high for a real reason, not a bug.** `chamfer_distance` returns
a mean of *squared* distances, so the reported chamfer of 0.0028 is an RMS error
near 0.053 — five times the threshold. 0.01 is where the loss starts
*penalising*, not where reconstruction is visually acceptable. Sweep the
threshold to 0.03–0.05 to see where the two models actually diverge.

`--export_json` dumps positions plus the per-point NN distance (not a baked-in
boolean, so the threshold stays a free parameter) for the interactive viewer in
the artifact linked at the top.
