---
name: cross-species-initiative
description: "Maize x sorghum cross-species plan (PLAN_cross_species.md), the code built for phases 0-1, and the depth-convention trap"
metadata:
  node_type: memory
  type: project
  originSessionId: 5c49c692-75ea-4443-a303-a6d89c1b614e
  modified: 2026-10-23T15:19:21.851Z
---

Cross-species (maize x sorghum) work started 2026-10-23; the plan is `PLAN_cross_species.md` in the repo. Key facts that are NOT obvious from the code:

- **99.9 % of weights are shared** between the species models (114.2 of 114.3 M); only the recipe embed, its pos tables and the recipe-head tail differ. Cross warm starts load with a shape filter; a sorghum E1 initialises a maize PC+RGB+D arm completely (0 tensors left at init).
- **Depth conventions are the silent killer** (bit twice: sorghum row renderer, and any cross feed). `species_conventions.py` is the single source: sorghum_std (z-0.0238)/50; maize_std per-plant near/far (fg band 0.26-0.74); maize_row (z-0.05)/2.95. Any new render source must be calibrated against its camera-frame cloud before a model reads it.
- **Cloud-content gap is not code-fixable**: sorghum trains on complete plants (67 % self-occluded points), maize on single-camera captures (7 %). Carry it as a caveat in every sorghum-model-on-maize number.
- Code: `eval/cross_species_probe.py` (Z1), `eval_rgb2pc_quant.py --model_species/--data_species` (Z2), `eval/eval_row.py --model-species` (Z3), `--init_checkpoint` in both trainers + `configs/config_{sorghum,maize}_xfer_{warm,scratch}*.yaml` (T1, 150 ep, not launched as of 2026-10-23).
- Data location: Nova = sorghum final + maize v1 + both row datasets; Delta = maize d2 + its checkpoints. Joint training needs the maize d2 images moved (~100-150 GB).
- Decision point after Z1 + T1: transfer near DINOv2 means the joint model is the story; near native means the cheap warm start is. See [[embodiedmae-output-policy]].
