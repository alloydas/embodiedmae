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

PC → params and RGB → params from the same model (`4m_distill_15k_all`), same 8
plants, every other modality masked. **Skill** = 1 − MAE ÷ (error from always
predicting the field's mean). 1.0 is perfect; **≤ 0 means nothing learned beyond
the average plant.** 110 leaf tokens over 6 plants with real leaves.

| leaf field | GT spread (sd) | MAE from PC | skill PC | MAE from RGB | skill RGB |
|---|---|---|---|---|---|
| starting_point          | 0.244  | 0.0087 | **0.96** | 0.0207 | **0.90** |
| branching_angle         | 5.14°  | 0.32°  | **0.93** | 0.65°  | **0.85** |
| length                  | 0.177  | 0.0403 | **0.70** | 0.0406 | **0.69** |
| roll_angle              | 108.1° | 41.7°  | **0.55** | 59.7°  | **0.36** |
| waviness_frequency      | 0.0041 | 0.0034 | −0.03 | 0.0034 | −0.03 |
| waviness_period_start_0 | 26.3°  | 23.0°  | −0.02 | 23.2°  | −0.03 |
| waviness_period_start_1 | 29.3°  | 26.0°  | −0.01 | 25.0°  | 0.03 |

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
  --source pc --indices 0,10,20,30,40,50,60,70 --out vis_params
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
