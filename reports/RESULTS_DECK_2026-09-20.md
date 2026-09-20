# Results deck — 2026-09-20

Four slides covering the complete results to date. Presentation version:
<https://claude.ai/artifact/MvfBLvgGWpr2UR1Z9SoAy3>

All figures on the **Sorghum_15K** validation split (held-out plants, 70/15/15
by plant, seed 42). Every arm below runs **197,400 optimizer steps** at global
batch 32, so the comparisons are at equal compute.

---

## Slide 1 — Cross-modal generation improves 19.1 % after distillation

**Before** = warm-started pretrain at epoch 0, before any distillation step.
**After** = `4m_distill_15k_all` at epoch 100. Lower is better.

**mean generation metric: 0.3439 → 0.2781 (−19.1 %)**

| source | total | Δ total | Δ PC chamfer | Δ param MAE |
|---|---|---|---|---|
| RGB | 0.1046 → 0.0931 | −11.0 % | −15.4 % | −10.0 % |
| Depth | 0.1091 → 0.0976 | −10.5 % | −22.5 % | −8.9 % |
| **Point cloud** | 0.6029 → 0.4374 | **−27.5 %** | **−56.6 %** | **−43.8 %** |
| Spline params | 0.5590 → 0.4844 | −13.3 % | −61.9 % | source |

Distillation helps most where the model was weakest: RGB and depth were already
good conditioning signals and gain ~11 %; the two weak sources gain two to six
times that.

### PC → parameters

| run | distilled targets | param MAE from PC | vs warm start | lr |
|---|---|---|---|---|
| warm start | — | 0.0617 | baseline | — |
| `4m_distill_15k_pc2text` | params only | 0.0389 | −36.9 % | 3e−5 |
| `4m_distill_15k_all` | all four | **0.0347** | **−43.8 %** | 1e−4 |

**The generalist beats the parameter specialist at its own job.** The other
three reconstruction targets act as auxiliary supervision rather than competing.
Caveat: different learning rates, so a matched-LR `pc2text` run (100 epochs, one
source) is needed to separate target set from LR.

---

## Slide 2 — E2 modality value-add: RGB and depth pay, parameters do not

All four arms complete at 600/600 epochs. Compared on `val_pc_chamfer`, the only
metric computed identically in every arm.

| arm | PC chamfer | vs PC-only |
|---|---|---|
| PC only | 0.004950 | baseline |
| PC + RGB | 0.002504 | −49.4 % |
| **PC + RGB + depth** | **0.001706** | **−65.5 %** |
| PC + RGB + depth + params | 0.002478 | −49.9 % |

Adding the parametric stream gives back everything depth gained.

**Two readings, different consequences.**

- *Mechanical.* At `mask_ratio` 0.80 with `min_mask_ratio` 0.25 per modality, a
  fourth token stream splits the visible-token budget four ways instead of
  three. The PC decoder sees less to work from, so chamfer penalises the params
  arm for *existing* rather than for being unhelpful.
- *Real.* The parametric modality does not help the representation.

Chamfer cannot separate these. Plan decision 6.4 already names a downstream
linear probe as the metric that settles modality value-add; **it does not exist
yet**, and it is the one thing standing between this table and a defensible
claim.

---

## Slide 3 — E3 / E4 scaling at equal compute

### Data scaling

| train plants | epochs | PC chamfer | vs previous |
|---|---|---|---|
| 1,000 | 6,169 | 0.002994 | — |
| 3,000 | 2,100 | 0.002582 | −13.8 % |
| **10,000** | 631 | **0.002405** | −6.9 % |
| 10,500 (the E2 four-modality arm) | 600 | 0.002478 | **+3.0 %** |

Epochs differ tenfold *so that steps do not* — an epoch under view sampling is
one view per plant, so epoch size is the plant count.

### Model scaling

| model | params | PC chamfer | vs small |
|---|---|---|---|
| small | 25.1 M | 0.002989 | — |
| **base** | 114.3 M | **0.002478** | **−17.1 %** |
| large | 332.2 M | running (42 %) | — |

### The last data row is the error bar

10,000 and 10,500 plants differ by 5 % in data, yet the 10,500 arm scores
**3.0 % worse**. Those two runs are effectively a duplicate, so **≈3 % is this
pipeline's run-to-run noise floor** — the only such measurement anywhere in the
E1–E10 matrix.

That reframes the curve:

- 1k → 3k, −13.8 %, is ~4.6× the noise. Real.
- 3k → 10k, −6.9 %, is only ~2.3× the noise. Genuine but modest, and not to be
  claimed strongly from a single seed.

**Data is not the binding constraint.** Tripling plants from 3k to 10k buys less
than going from 25 M to 114 M parameters (−17.1 %). At this compute budget,
capacity is paying better than data. The `large` arm tests whether that
continues or turns over.

---

## Slide 4 — What the parameter head recovers, and what is next

Skill = 1 − MAE ÷ (error of always predicting that field's mean). 1.0 perfect;
≤ 0 means nothing learned beyond the average plant. 142 leaf tokens, 8 plants.

| leaf field | skill from PC | skill from RGB |
|---|---|---|
| starting_point | **0.95** | 0.90 |
| branching_angle | **0.93** | 0.85 |
| length | **0.67** | 0.66 |
| roll_angle | **0.50** | 0.33 |
| waviness_frequency | −0.02 | −0.01 |
| waviness_period_start_0 | −0.01 | −0.04 |
| waviness_period_start_1 | −0.01 | 0.02 |

**PC beats RGB on every field with skill**, widest on `roll_angle`. Per plant,
PC wins on 7 of 8.

**Three of seven fields are at zero skill.** The waviness fields emit the
dataset average and nothing more, from either source, so ~40 % of the leaf
vector dilutes a real result with a constant. Distillation made them *noisier*
without making them more accurate — view spread rose 0.18 → 0.26 while error
stayed flat.

### View robustness

The view index is an exact elevation ladder (−0.9 + 0.2·index) and every view
shares one target, so only the input image changes.

- Chamfer traces a **U**: best side-on, worst steeply up or down, in both models.
- Distillation improves **9 of 10 views**, mean **−28.5 %**…
- …but best/worst spread *widens* 2.35× → 2.49×. It lowers the curve without
  flattening it, so it is **not the lever for view robustness**.
- Parameters are strongly view-invariant where they have skill: ten different
  photographs move `starting_point` by only 0.08 of its natural spread.

### Two decisions needed

1. **Build the downstream linear probe.** Required by decision 6.4, does not
   exist, and is the only thing that can tell us whether the E2 params result is
   a token-budget artefact or real. Three of four targets are in `features.csv`;
   **biomass needs a definition** and "leaf angle" is ambiguous between
   `roll_angle` and `branching_angle`.
2. **Choose the E8 baseline set.** Ranked the largest reject risk, nothing in the
   repo addresses it, and each candidate is a training run that must fit before
   the 24 Oct freeze — so the decision is more time-critical than the runs.

---

## Run status behind these numbers

| run | state | note |
|---|---|---|
| `e2_pc` `e2_pcrgb` `e2_pcrgbd` `e2_pcrgbdt` | complete | 600/600 each |
| `e3_3k` `e3_10k` | complete | |
| `e4_small` | complete | |
| `e3_1k` | resuming from epoch 5654/6169 | preempted at 94 % |
| `e4_large` | resuming from epoch 250/600 | preempted at 42 % |

**`e3_1k` and `e4_large` both died at exactly `2026-09-19T15:51:27` on
`nova26-gpu-2`** — same second, same node, no traceback in either log, host RSS
well under the 320 GB request. That is a node-level event on the preemptible
scavenger partition, not a fault in either run. `e3_1k`'s
`training_history.json` is truncated mid-write at exactly 768 KB, which is what
a killed process leaves; its val series was recovered from the run log instead.

Both were resubmitted 2026-09-20 and auto-resume from their last checkpoint, so
the cost of the preemption is bounded by `save_freq`, not by the elapsed run.

The `e4_large` figure above is **not comparable** — it is a mid-training value at
42 % of schedule, shown as pending rather than omitted.
