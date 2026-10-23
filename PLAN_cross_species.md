# Cross-species plan: one CropCode model across maize and sorghum

Drafted 2026-10-23. Scope: everything "cross" between the two species — zero-shot transfer,
cross-species warm starts, joint pretraining, and a joint field (row) model. The results freeze
for the current paper is 2026-10-24, so Phase 0 is sized to still land as supplement /
rebuttal numbers; Phases 1-3 are camera-ready / next-paper work.

## What makes it feasible, measured

- **99.9 % of the weights are shared.** The two base models differ only in the recipe stream:
  `pos_embed_text` / `decoder_pos_embed_text` (25 vs 29 tokens), `param_embed.proj` (9 vs 14
  fields) and the last layer of `decoder_pred_params`. 114.2 of 114.3 M parameters are
  shape-identical (checked 2026-10-23). A cross-species warm start is `strict=False` load;
  the probe never shows the recipe stream, so **cross-species probing needs no new weights at all**.
- **Clouds are already in a shared frame** (centred, unit sphere).
- **DINOv2 / MultiMAE / Point-MAE baselines are species-agnostic**, so the E8 harness gives the
  reference: does crop-specific pretraining transfer across crops better than generic encoders?

## The four conventions that differ (each one a silent-failure risk; see the row-data episode)

| | sorghum (final) | maize (d2) | handling |
|---|---|---|---|
| depth encoding | ~(z − 0.024) / 50, fg 0.03-0.07 | (z − near)/(far − near), per-plant near/far, fg 0.26-0.74 | a canonical depth adapter: decode to metres with each species' convention, re-encode in the TARGET model's convention (generalise `row_dataset`'s `depth_reencode`) |
| RGB | rgb_nobg over black, ×1.85 gain | rgba over black, no gain | serve each file at its recorded gain; both black-background, ImageNet-normalised — low risk |
| cloud content | complete plant (67 % of points occluded from the camera) | single-camera capture (7 %) | not fixable by code: either accept as domain diversity, or build a matched variant by cutting sorghum clouds with the existing per-view visibility masks (`outputs/_pcview_masks`) |
| recipe stream | 9 fields, 24 leaves, YAML | 14 fields, 28 leaves, XML v2 | Phase 0-1: drop it (the E2 result says it hurts anyway). Phase 2+: pad to 14 fields / 28 leaves + a species bit in the plant token |

**Data location.** Nova: sorghum final + maize v1 (rgb.png, full XML) + both row datasets.
Delta: maize d2 (rgba, trimmed XML) + its checkpoints. Joint training on Nova needs either the
maize d2 copy (~100-150 GB: rgba.png per folder) or an explicit decision to use maize v1 there;
cross-species EVALUATION on Nova can start from the maize v1 checkpoints already here
(`outputs/maize_4m` ep 600) plus one 1.4 GB d2 checkpoint scp'd from Delta.

## Phase 0 — zero-shot transfer, evaluation only (fits before/around the freeze; ~3 GPU-hours)

- **Z1 · cross-species frozen probe (the headline question: is the representation
  crop-general?).** New `eval/cross_species_probe.py`: feed species B's RGB/depth/PC (depth
  re-encoded, RGB at its gain) through species A's frozen encoder,
  `forward_encoder_select(visible={pc,rgb,depth})`, ridge on CLS against B's targets, same
  splits/caches as the native probes. Rows: sorghum E1 and PC+RGB+D on the 11 maize targets;
  maize_4m (v1; d2 after the scp) on the 9 sorghum targets. Compare each against (a) the native
  arm, (b) DINOv2/MultiMAE/Point-MAE (already species-blind), (c) random init. Paired bootstrap
  per target. Caveat to carry: sorghum-model-on-maize also crosses the cloud convention.
- **Z2 · cross-species zero-shot generation.** `eval_rgb2pc_quant` with a `--model-species`
  flag: sorghum E1 generating a maize cloud from a maize photo and the reverse; score against
  each species' own gt with the existing shuffled / gt-vs-gt floors.
- **Z3 · cross-species rows.** `eval_row.py` already takes any checkpoint: score
  `maize_row_ft` on the sorghum test rows and `sorghum_row_ft` on the maize test rows
  (chamfer/miss only; recipe not scored across schemas). Same rover geometry, different crop:
  the cheapest "new crop, no retraining" number in the paper's story.

## Phase 1 — transfer value (one short run per direction; ~1 day each on 2 GPUs)

- **T1 · cross warm start vs scratch.** Train the maize PC+RGB+D arm initialised from sorghum
  E1 (vision weights; recipe head fresh) and from scratch, both at a matched short budget
  (150 epochs ≈ 50 k steps); probe both. Repeat in the sorghum direction from maize_4m.
  Deliverable: R² per GPU-hour with and without cross-species pretraining — the practical
  "new crop onboarding cost" number.

## Phase 2 — joint pretraining (the cross-species E2; ~1-2 days on 2 RTX PRO 6000)

- **J1 · joint PC+RGB+D arm, no recipe stream.** `MixedSpeciesDataset`: both train splits,
  species-pure batches drawn alternately (PointCloudEmbed's unsynced BatchNorm then sees
  homogeneous batches, matching every precedent), one view per plant per epoch, canonical depth.
  21,000 plants → 656 steps/epoch at global batch 32; 300 epochs matches the 197,400-step
  budget. Probe per species against that species' own PC+RGB+D arm (paired CIs): does joint
  training cost either species anything, and does it help the smaller-signal targets?
- **J2 · joint four-stream (only if J1 is neutral-or-better).** Unified recipe tokens: 14
  fields padded, 28 leaf slots, per-species scale/shift tables, a species bit in the plant
  token; trainer refuses mixed schemas per run dir unless the config says joint.

## Phase 3 — the joint field model (next paper's headline; ~1 week)

Joint E1-scale pretrain (both species, all views) → joint row fine-tune on BOTH row datasets
(the row pipeline is already species-neutral) → single-sensor distilled students that work on
either crop → evaluate per species against the species-specific chain. Claim if it holds:
one encoder + one row protocol covers two crops with no per-crop pretraining.

## Build order / prerequisites

1. canonical depth adapter + a species-conventions table in one module (reused by Z1/Z2/J1).
2. `eval/cross_species_probe.py` (Z1) — needs nothing else; launch first.
3. scp one maize d2 checkpoint Delta → Nova (`maize_4m_d2` ep 600) for the d2 rows of Z1/Z2.
4. Z2 flag, Z3 sbatch lines (exist in spirit; one command each).
5. T1 configs (2 files per direction), J1 dataset + config.
6. Decision point after Z1 + T1: if transfer is near-DINOv2, the joint model is the story;
   if it is near-native, the cross warm start alone is.
