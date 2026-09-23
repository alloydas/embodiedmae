# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyTorch implementation of **EmbodiedMAE** — a multi-modal Masked Autoencoder pre-trained on synthetic Sorghum plant data. Two model variants share the same encoder/decoder skeleton:

- **3M (`embodied_mae.py`)** — RGB + Depth + Point Cloud
- **4M (`embodied_mae_4m.py`)** — adds a parametric "spline/text" modality describing the plant's procedural generation parameters

Both use the same Dirichlet-allocation token masking, transformer encoder, and per-modality decoder heads. The 4M variant is the active line of work; 3M is kept as the baseline it was derived from.

## Repo layout

Configs and batch scripts were consolidated out of the repo root. **There are no `config.yaml` / `config_4m.yaml` at the root any more** — everything lives in:

- `configs/` — every YAML. `config.yaml` (3M), `config_4m.yaml` (4M), `config_4m_pretrain15k*.yaml` (pretrain), `config_4m_distill_*.yaml` (cross-modal distillation), `config_e2_*.yaml` (the E2 modality ablation), `config_e3_*.yaml` (data scaling) and `config_e4_*.yaml` (model scaling).
- `slurm/` — every `.sbatch` / launcher script. `e2_arm*.sbatch` launch E2 arms; `scale_arm_blackwell.sbatch` launches any E3 or E4 arm by slug (`sbatch --job-name=e3_1k slurm/scale_arm_blackwell.sbatch e3_1k`), deriving `configs/config_<slug>.yaml` and `outputs/<slug>/`.
- `data_split/` — the train/val/test split tooling (`make_split.py`, `move_split.py`, `reshuffle.py`).
- `eval/` — evaluation and analysis (`eval/linear_probe.py` = decision 6.4, `eval/latent_analysis.py` = E9, `eval/analyze_e2.py`, `eval/eval_views_one_plant.py`, `eval/vis_pc_unpredicted.py`, `eval/eval_warmstart.py`, …).
- `figures/` — anything that renders a figure, report or deck (`figures/make_results_pptx.py`, `plot_*.py`, `gen_*.py`, `regen_*.py`).
- `export/` — dumps and artefact builders (`export/dump_param_examples.py`, `export_pc_*.py`, `build_*.py`).
- `sweeps/` — masking and visualisation sweeps.
- `tools/` — one-off data-prep and diagnostic scripts.
- **The repo root holds only what other code imports**: the two models, the two datasets, the training entry points, `validate*.py` and `utils.py`. That is the rule — if a file at the root is imported by nothing, it belongs in one of the folders above.

Scripts in those folders put the repo root on `sys.path` themselves (a four-line shim at the top), so `python eval/analyze_e2.py` works from the repo root without `PYTHONPATH` set. Keep the shim when adding a script there.

If a command or script still refers to a root-level `config_4m.yaml`, it is stale — the path is `configs/config_4m.yaml`.

## Experiment programme — what is done and what is left

The programme is the E1–E10 matrix in the CVPR 2027 plan (the Google Doc is the
source of truth for scope; this section is the source of truth for *state*).
Paper deadline **Nov 13 2026**, internal results freeze **Oct 24 2026**.

**Status as of 2026-09-20.** Run state goes stale fast — the table records which
experiments have been *built*, which is durable. For live progress use
`squeue -u $USER`, `outputs/<run>/training_history.json`, and
`outputs/<run>/config.json` (which records what a run actually used).

| | experiment | state | runs |
|---|---|---|---|
| E1 | headline pretrain | **done** | `4m_pretrain_15k_v2_depthfix_qal`, 1000/1000 ep, global batch 256 |
| E2 | modality value-add | **done** | all four arms 600/600; results in `reports/RESULTS_DECK_2026-09-20.md` |
| E3 | data scaling | **done** | all three arms; `e3_1k` finished 2026-09-21 11:22 at 6169/6169 |
| E4 | model scaling | **done** | `e4_small` ✅ `e4_large` ✅ (600/600, finished 2026-09-21 21:41) |
| E5 | view regime | **not built** | — |
| E6 | masking / noise | **owned elsewhere** | a collaborator is running this — not work for this repo |
| E7 | loss study | **owned elsewhere** | same; `model.loss_name` (chamfer / qal_loss) is the switch they need |
| E8 | baselines | **not built** | — |
| E9 | latent analysis | **built, E2 arms done** | `eval/latent_analysis.py`; all four E2 arms on val. E3/E4 arms and train/test splits not run |
| E10 | real-data OOD | **partial** | `OOD_EVAL_rgb2pc.md` and the `eval_rgb2pc_*.py` scripts |

**On the two resuming arms.** `e3_1k` and `e4_large` both died at exactly
`2026-09-19T15:51:27` on `nova26-gpu-2` — same second, same node, no traceback, host RSS far
under request. That is a node-level event on the preemptible `scavenger` partition, not a fault
in either run, and the launcher's auto-resume bounds the loss at `save_freq` epochs. `e3_1k`'s
`training_history.json` was left truncated mid-write; its val series is recoverable from the run
log in `logs/`.

**On waiting for GPUs.** When these jobs sit `PENDING` for days, check whether it is your request
before shrinking it: a 1-GPU / 1-CPU / 1 GB / 10-minute test job placed at the *same* time as the
full 2-GPU / 320 GB request, which means the nodes are held by a reservation and no amount of
trimming helps. Read the *reason* in `squeue` before acting, because three different walls look
identical from the outside:

- `AssocGrpGRES` — the **whole `mech-ai` account** is capped at `gres/gpu=17` and it is full.
  Nothing about your request matters. `scavenger` runs under `mech-ai-scavenger` and bypasses it,
  and a job that asks for **no GPU at all** (`--gres=NONE`) sidesteps the cap entirely — feature
  extraction and probing need no GPU, so that is the cheap way around.
- `Priority` / `Resources` on `scavenger` — nodes carry `PLANNED`, i.e. idle GPUs held for a
  higher-priority job. `sinfo` showing free GPUs does not mean you can have one.
- Preemption — `scavenger` jobs die without warning. Pass `--requeue` and make any cache the job
  writes atomic (write to a temp file, `os.replace`), so a requeued job resumes instead of restarting.
- **A non-GPU job can block every GPU on a node from `scavenger`.** `scavenger` is
  `PriorityTier=0` with `OverSubscribe=FORCE:1`; `nova` is `PriorityTier=100`. A scavenger job
  therefore cannot co-locate with a `nova` job on the same node, so a 2-CPU job that asks for
  a large `--mem` (or `--mem=0`, which means *all* of it) and **no GPU at all** makes all eight
  GPUs on that node unreachable from scavenger. This is why `sinfo` can show
  `gpu:rtx_pro_6000:0(IDX:N/A)` — i.e. 16 idle GPUs — while your job sits `PENDING`. The only
  two Blackwell nodes are `nova26-gpu-[1-2]` (partitions `nova,scavenger,allnodes`), so a single
  such squatter on each blocks the whole generation. Diagnose with the probe below; the fix is
  patience or the `nova` partition, never a smaller request.

**The probe that separates "my request is too big" from "the nodes are held"**: submit a
deliberately tiny job — 2 GPUs, 8 CPUs, 32 GB, 5 minutes — alongside the real one and compare
`scontrol show job <id> | grep StartTime`. On 2026-09-22 the 80-CPU/320 GB/2-day maize job and
that probe returned the *same second* (`2026-09-25T16:34:09`), which proves the wall is a node
hold and that trimming CPUs or memory buys nothing. **Slurm's `StartTime` is a worst case**: it
assumes every blocking job runs to its full walltime. That estimate said Sep 25 16:34; the job
actually started **Sep 23 06:49**, more than two days early, because the squatter ended sooner.
Do not reshape a run around a pessimistic `StartTime`.

**A job landing on the wrong GPU generation fails at the first kernel launch, not at import.**
A bare `--gres=gpu:1` can place you on an RTX PRO 6000 (sm_120), where the CUDA 12.4 `det` env dies
with `no kernel image is available for execution on the device` — deep inside the PC embedder's FPS,
with the model already built and the dataset already indexed. `slurm/linear_probe.sbatch` picks the
env from `nvidia-smi --query-gpu=name` at runtime rather than pinning the GPU type; copy that
pattern instead of constraining the gres.

### The linear probe (decision 6.4) — built, and what it found

`eval/linear_probe.py` is the probe 6.4 asks for: ridge on the frozen CLS token,
one deterministic view per plant, standardiser fit on **train only**, scored on
the same held-out val plants for every arm. `slurm/linear_probe.sbatch` runs it
(1 GPU is plenty — one forward pass per plant, no backward). Features cache per
`(run, ckpt, split)` under `outputs/_probe_cache/`, so re-fitting costs nothing
and `eval/latent_analysis.py` (E9) reads the same cache and needs no GPU at all.

Five rules the script enforces; each silently produces a plausible wrong number
if dropped, and they are documented at the top of the file:

1. **The text stream is never visible, and the params are zeroed.**
   `param_floats[0][0]` is `stem_length` — the height target, verbatim. An arm
   allowed to see the spline stream at probe time is handed the answer. Verified
   `max|diff| = 0.0` between real params and zeros, so the leak is structurally
   impossible. Side effect worth knowing: `e2_pcrgbd` and `e2_pcrgbdt` are then
   probed on an identical 589-token input, so the only difference between them
   is what pretraining taught the encoder.
2. Features come from `forward_encoder_select(..., visible=active-minus-text,
   source_mask_ratio=0.0)`, **not** `forward_encoder`. Note `forward_encoder(mask_ratio=0.0)`
   does *not* raise and does keep every token, but it still permutes them per sample.
3. One deterministic view per plant (`view_sampling=True` **and**
   `deterministic_view=True` — the second alone is a silent no-op). All ten views
   share one label, so view rows inflate n tenfold and narrow every error bar ~3.2×.
4. Scaler on train only — val/test carry ~1.83× the target variance (extreme-enriched).
5. Extraction is stochastic in two places even under `eval()`/`no_grad`: FPS picks
   its first centroid with `torch.randint`, and `load_pointcloud` subsamples with an
   unseeded `np.random.choice`. Seed both or the R² will not reproduce.

**Result (val R², all four E2 arms).** Adding the parametric stream is worse than
PC+RGB+depth on **every** target — height 0.981→0.966, leaf count 0.984→0.967,
biomass 0.986→0.969, leaf length 0.362→0.278. Chamfer said the same
(0.001706→0.002478), and E9 agrees independently (shape alignment |r| 0.64 for the
params arm against 0.76 and 0.94). **Depth, meanwhile, is the big chamfer win and a
rounding error on phenotype**: PC+RGB→PC+RGB+D moves height 0.978→0.981. RGB is
where the phenotype information arrives.

**This does NOT resolve mechanical-vs-real, and do not claim it does.** The probe
removes the probe-time confound completely, but the token-budget handicap lives in
*pretraining*: at `mask_ratio` 0.80 a fourth stream splits the visible budget four
ways for 600 epochs. **One control arm separates them and is the highest-value run
left**: re-run the four-modality arm at a mask ratio giving RGB/depth/PC the same
visible-token count as the three-modality arm.

**The four targets 6.4 names have rank 2, not 4.** The generator sets
`stem_length = 0.05 × n_leaves − 0.001` (R² 0.988), so height and leaf count are
one variable (r = 0.994), and the biomass proxy `n_leaves × leaf_len_mean` is
r = 0.996 with leaf count — the same variable a third time. Neither leaf-angle
column works as specified either: `branch_mean` spans 0.27° end to end and is ~80 %
leaf count, while `roll_mean` is the mean of *n* near-uniform **circular** draws
(oracle linear R² 0.0009 from the other columns). But `roll_mean` is *not* dead —
the latent sees the leaves, and the probe recovers it at **R² 0.806**, the widest
arm spread in the table and the only target uncorrelated with size. The probe
reports all four 6.4 targets plus `leaf_len_mean`/`leaf_len_max`/`wav_mean` as the
genuinely independent axes, and prints each target's correlation with leaf count so
the redundancy is visible rather than implied.

**`best_model.pth` is not comparable across arms.** It is selected on total val
loss, which lands at 42 % of schedule for `e3_1k` and 100 % for `e3_10k`. Probing
E3 on `best_model.pth` would bill training length to data scale. Pass an explicit
`--ckpt checkpoints/checkpoint_epoch_N.pth` (last: 6168 / 2024 / 624 / 600 / 600).

### E9 — latent analysis, and the genotype problem

`eval/latent_analysis.py`. **There is no genotype variable in Sorghum_15K**: every
plant is `SorghumGenerator/Random.sg` with `Seed = plant id` — one generator, one
continuous distribution, 15 000 draws. The seed is an identifier, not a cultivar, so
the plan's "cluster by genotype" has nothing to condition on and k-means would cut a
continuum into arbitrary pieces. The answerable question is *does the latent recover
the generative factors, and is it continuous?* — under which a **near-zero silhouette
is a positive result**. Say that before quoting the number.

Findings: silhouette 0.165–0.236 with ARI ≈ 0 against shape deciles (a continuum,
not clusters); effective rank 4–7 out of 768; and **the dominant latent direction is
not the phenotype** — the size factor lives on PC3 while PC1 carries ~30 % of
variance on something unidentified (camera view is ruled out, every plant is scored
on its fixed view 00). That last one is an open question, not a loose end.

**2. No baselines exist (E8), and the plan ranks that the #1 reject risk.**
Nothing in the repo addresses it, and every baseline is itself a training run
that has to fit before the Oct 24 freeze, so the *decision* about what to
compare against is more time-critical than the runs.

### On E6 and E7 being run elsewhere

Masking/noise and the loss study are a collaborator's, so nothing in this repo
should schedule them. Their results still have to land in the same table as
everything above, which means the comparison only holds if they match this
programme on the three things that silently break it: the **same** 70/15/15
split of `Sorghum_15K` (seed 42), the **same** global batch of 32, and a budget
stated in **optimizer steps** rather than epochs. Confirm those three before
their numbers are merged, not after — `outputs/<run>/config.json` records all
three for any run that used this codebase.

That leaves **E5, E8, E10**, the E9 sweep over the E3/E4 arms, and the
mask-ratio control arm above as the work owned here.


## Environment

Conda env name is `det` (Python 3.12, PyTorch 2.5 + CUDA 12.4; see `environment.yml`). On Nova it lives at `/work/mech-ai/alloy/.conda/envs/det`. Activate with `conda activate det` before running anything.

On Blackwell / sm_120 GPUs (RTX PRO 6000) `det` will not work — those need the CUDA 12.8 build in the separate `det_cu128` env.

## Common commands

All commands assume `cwd = repo root` and the env active.

### Train 4M (the main entry point)
```bash
# Single GPU
python train_sorghum_4m.py --config configs/config_4m.yaml --world_size 1

# Multi-GPU via torchrun (preferred — sets LOCAL_RANK/RANK/WORLD_SIZE)
torchrun --standalone --nproc_per_node=4 train_sorghum_4m.py --config configs/config_4m.yaml

# Multi-GPU via mp.spawn fallback (set distributed.world_size in the YAML)
python train_sorghum_4m.py --config configs/config_4m.yaml
```

### Train 3M
```bash
python train_sorghum.py --config configs/config.yaml                  # single GPU
python train_sorghum_multi.py --config configs/config.yaml            # mp.spawn
```

### Cross-modal distillation
`train_sorghum_4m_distill.py` trains the 4M model so that **any single modality, with the other three fully masked, reconstructs all four** — a frozen full-modal teacher supplies token-aligned decoder features and a CLS latent as the privileged target, and the student is warm-started from the same checkpoint.
```bash
python train_sorghum_4m_distill.py --config configs/config_4m_distill_15k_rgb2pc.yaml
```

### Validate / dump reconstructions
```bash
python validate.py --checkpoint outputs/<run>/best_model.pth \
                   --output_dir vis_val --config configs/config.yaml --num_samples 6
```

### Downstream linear probe (decision 6.4) and E9 latent analysis
```bash
# probe — 1 GPU, ~15 min for four arms cold, seconds when the cache is warm
sbatch slurm/linear_probe.sbatch e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt
# blocked by AssocGrpGRES? scavenger bypasses the account GPU cap:
sbatch --partition=scavenger --account=mech-ai-scavenger --requeue \
       slurm/linear_probe.sbatch e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt

# E9 — no GPU needed once the probe cache exists
python eval/latent_analysis.py --runs e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt \
                               --split val --device cpu
```
Both read/write `outputs/_probe_cache/<run>__<ckpt>__<split>__cls__seed<n>__rep<n>.npz`,
written atomically (temp file + `os.replace`) so two jobs racing the cache cannot
corrupt it. Delete the cache to force re-extraction, or pass `--refresh`.

### Sanity-check a dataset folder
```bash
python sorghum_dataset.py    /path/to/Sorghum_15K    # 3M
python sorghum_dataset_4m.py /path/to/Sorghum_15K    # 4M (requires *_spline.yml)
```

### Smoke-test the model definitions
```bash
python embodied_mae.py       # builds embodied_mae_base, one forward pass on dummy tensors
python embodied_mae_4m.py    # same for 4M
```

There is no test suite, no linter, and no Makefile.

## Configuration model

Both training entry points layer config in this order: YAML → CLI flags → defaults. The YAML is the source of truth; CLI flags only override specific keys. Notable keys:

- `data.data_root` expects `<root>/train/`, `<root>/val/` and `<root>/test/` siblings, each containing one folder per sample (see "Dataset layout").
- `data.view_sampling: true` — each plant contributes **one randomly chosen view per epoch**, so an epoch over 105 000 train samples is 10 500 items, not 105 000. `view_seed` must be identical across arms of an ablation so they see the same views in the same order.
- `data.max_plants` — caps the **train** split at a nested random subset of N plants (`plant_subset_seed` fixes the draw); `null` uses all of them. This is what E3 varies. The subsets nest, so 1k ⊂ 3k ⊂ 10k and a step of the curve is never partly a change of sample composition. It selects **plants, keeping all ten views** — dropping views would change the augmentation regime that decision 6.1 fixes, which is E5's question, not E3's. `train_sorghum_4m.py` passes it to the train dataset only; a capped val/test would score the arms on different yardsticks.
- `model.model_size` ∈ {`small`, `base`, `large`, `giant`} (3M) or {`small`, `base`, `large`} (4M). This is what E4 varies: 4M builds at 25.1 M / 114.3 M / 332.2 M params.
- `model.mask_ratio` is the **total** masking fraction across all modalities; the per-modality split is sampled from `Dirichlet(α=dirichlet_alpha)` once per batch.
- `model.active_modalities` — comma-separated subset of `pc,rgb,depth,text`. Restricting it is what the E2 ablation varies; everything else stays fixed.
- `model.loss_name` ∈ {`chamfer`, `qal_loss`} with `qal_threshold` / `qal_alpha` / `qal_use_squared`. Current runs use `qal_loss`.
- `model.pc_loss_weight` scales the PC term. Chamfer values are tiny relative to MSE, so this is typically O(10) when Chamfer is selected.
- `model.spline_loss_weight` (4M only) scales the Smooth-L1 param-regression loss.
- `model.depth_norm_type` ∈ {`minmax`, `standard`} — applied to the **target** before depth MSE; the model learns to predict normalised values directly.
- `checkpointing.resume`: path to a `.pth` to resume from, or `null` for scratch.
- `distributed.world_size > 1` triggers DDP. `train_sorghum_4m.py` also accepts being launched under `torchrun`, in which case `LOCAL_RANK` is honoured and `world_size` is inferred from env.

**Global batch is `batch_size × world_size`.** `batch_size` in the YAML is per-GPU, so an ablation must adjust it to the GPU count to keep the global batch constant — a global-batch difference between arms confounds the comparison. The E2 launchers use 16×2 on the 2-GPU Blackwell nodes and 8×4 on the 4-GPU A100 nodes, both reaching 32.

**Size an ablation in optimizer steps, not epochs — and do not "fix" epoch counts that disagree.** Under `view_sampling` an epoch is one view per plant, so *epoch size is the plant count*: at global batch 32 the full train split gives 329 steps/epoch, but a 1 000-plant subset gives 32. Arms that differ in data scale therefore need very different epoch counts to receive the same number of gradient updates, and the E3 configs look wrong at a glance because of it:

| arm | plants / model | steps/ep | epochs | total steps |
|---|---|---|---|---|
| `e3_1k` | 1 000 | 32 | 6 169 | 197 408 |
| `e3_3k` | 3 000 | 94 | 2 100 | 197 400 |
| `e3_10k` | 10 000 | 313 | 631 | 197 503 |
| `e4_small` / `e4_large` | 25.1 M / 332.2 M | 329 | 600 | 197 400 |
| `e2_pcrgbdt` | 10 500, base | 329 | 600 | 197 400 |

Equalising those epoch counts would hand the 10k arm ten times the updates of the 1k arm, and the resulting curve would measure data scale and compute jointly with no way to separate them afterwards. `warmup_epochs`, `val_freq` and `save_freq` are scaled by the same per-arm factor, so every arm warms over the same fraction of its schedule and lands ~24 val points.

That shared 197 400-step budget is `e2_pcrgbdt`'s, which is why **`e2_pcrgbdt` is also E3's full-data point and E4's base-model point** rather than a separate run — and why the E3/E4 launcher's resources are deliberately byte-identical to the E2 one. `e3_10k` (10 000 plants) and `e2_pcrgbdt` (10 500) are within 5 % of each other, so that pair doubles as the only run-to-run error bar in the experiment matrix.

`persistent_workers` is deliberately **off** in the dataloaders and must stay off: persistent workers hold a copy of the dataset made at first iteration, so `train_ds.set_epoch(epoch)` would never reach them and view sampling would silently freeze at epoch 0. The cost is that workers respawn every epoch, which matters most for `e3_1k`, whose epoch is only ~1 000 items.

## Architecture (the part you'd otherwise have to read 4 files to learn)

### Forward pass shape
1. **Embed each modality** to `(B, L_m, D)`:
   - RGB / Depth → `PatchEmbed` (Conv2d patchification → 196 tokens for 224×224, patch=16).
   - Point cloud → `PointCloudEmbed`: FPS samples `num_pc_tokens=196` centres, kNN groups `group_size=32` neighbours per centre, two PointNet-style Conv1d blocks produce per-token features.
   - (4M only) Spline params → `TextLeafEmbed`: char embeddings + learned positional → mean-pool → linear → D. `n_text_tokens = 1 + max_leaves` (1 plant token + up to 24 leaf tokens).
2. **Add positional + modality embeddings** (each modality has its own learned modality bias).
3. **Dirichlet masking** in `random_masking_dirichlet`: a single Dirichlet draw decides the visible-token split across modalities for the whole batch step. Each sample's visible indices are independently random, but the per-modality count is identical batch-wide — this avoids zero-token batch elements that would corrupt encoder norms. The 4M variant also enforces `min_mask_ratio=0.25` per modality.
4. **Encoder**: visible tokens from all modalities are concatenated with a CLS token and fed through `depth` shared transformer blocks (`embed_dim=768` for base).
5. **Decoder**: project to `decoder_embed_dim=512`, restore mask tokens at original positions per modality, add decoder positional embeddings, run `decoder_depth=8` transformer blocks, then split back into modality-specific heads:
   - RGB / Depth → linear → `patch_size² × C` per token (unpatchified for visualisation).
   - PC → FoldingNet-style upsampling: each of 196 PC tokens generates `points_per_token = target_points // num_pc_tokens` 3D points by concatenating its 512-D feature with a 2D grid coordinate and passing through an MLP. `target_points` defaults to 10000, trimmed/padded to exactly that on output.
   - (4M only) Spline → 2-layer MLP → `N_PARAMS=9` floats per token, range [0, 1].

### Losses (`forward_loss`)
- **RGB**: per-patch MSE, optionally with `norm_pix_loss` (per-patch normalisation), masked mean.
- **Depth**: per-patch MSE on **normalised** target (per-image min-max or standard); the prediction is compared to the normalised target directly.
- **PC**: bidirectional Chamfer (or QAL) on the full `(B, target_points, 3)` cloud (no masking — the whole cloud is reconstructed every step), scaled by `pc_loss_weight`.
- **(4M)**: Smooth-L1 on params, masked by `text_valid` (real leaf tokens only) **and** `mask_text` (model only loses on tokens it didn't see). Scaled by `spline_loss_weight`.

`embodied_mae_4m.py` imports `PatchEmbed`, `PointCloudEmbed`, `TransformerBlock`, `chamfer_distance`, and `get_2d_sincos_pos_embed` from `embodied_mae.py` — when changing these, expect both models to be affected.

### Param normalisation (4M only)
Plant and leaf parameters are normalised to [0, 1] via fixed `_PLANT_SCALE / _PLANT_SHIFT` and `_LEAF_SCALE / _LEAF_SHIFT` arrays at the top of `embodied_mae_4m.py`. `decode_params_to_text` un-normalises them back into the human-readable strings shown in visualisations. If new fields are added to the spline YAMLs, both arrays and the `_plant_to_params` / `_leaf_to_params` builders must be updated.

## Dataset layout

`SorghumDataset` expects:
```
data_root/{train,val,test}/<sample_name>/
    rgb.png              # 224-ready RGB render
    depth.png            # big-endian packed RGBA depth
    *_nc_cam.ply         # camera-frame, normals-cleaned point cloud
    *_spline.yml         # 4M only — procedural generation params
```

Sample folders are named `Sorghum_<plant>_<view>`, so `Sorghum_0_00 … Sorghum_0_09` are ten views of one plant. **Only those four files are read by the loaders.** The `Sorghum_<n>.obj` and `Sorghum_<n>_nc.ply` that sit alongside them are source assets, are duplicated in full into every one of a plant's ten view folders, and are never opened during training — they are ~64 % of the bytes on disk.

The PC loader uniformly samples / pads to `num_points`, then centres at the centroid and scales so the max-distance point lands on the unit sphere. RGB uses ImageNet mean/std normalisation; depth is loaded as single-channel L and only `ToTensor`'d (no normalisation at load — normalisation happens inside the loss).

The active dataset on Nova is `/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K/`, an extreme-enriched 70/15/15 split (seed 42) of 15 000 plants: **105 000 train / 22 500 val / 22 500 test** view-samples. `assignment.csv` and `features.csv` at that root record the split; the tooling to regenerate or reshuffle it is in `data_split/`.

## Second dataset: Maize (separate pipeline, do not merge)

`/work/mech-ai-scratch/alloy/Maize/` holds a **maize** counterpart added
2026-09-22: 15 000 plants × 10 views, split 70/15/15 **by plant**
(train 10 500 / val 2 250 / test 2 250). It is a different generator with a
different on-disk contract, and **the decision is to keep maize and sorghum code
separate** — parallel files, not a species flag on the sorghum ones:

| sorghum | maize | built |
|---|---|---|
| `sorghum_dataset_4m.py` | `maize_dataset_4m.py` | ✅ |
| `embodied_mae_4m.py` | `embodied_mae_4m_maize.py` | ✅ |
| `train_sorghum_4m.py` | `train_maize_4m.py` | ✅ |
| `configs/config_4m.yaml` | `configs/config_maize.yaml` | ✅ |
| `slurm/scale_arm_blackwell.sbatch` | `slurm/train_maize.sbatch` | ✅ |
| `eval/linear_probe.py` | `eval/linear_probe_maize.py` | ✅ |
| `eval/latent_analysis.py` | `eval/latent_analysis_maize.py` | ✅ |

**Maize fixes decision 6.4.** Sorghum's four named targets have rank 2 — height
and leaf count are one variable (r = 0.994) and the biomass proxy is that
variable again (r = 0.996) — and neither leaf-angle column is learnable. In maize
all four are real and largely independent: height vs leaf count **r = 0.199**,
biomass (`leaf_areaProxy`, a native column not a derived proxy) vs leaf count
0.763, and `leaf_angleMean` is uncorrelated with everything (|r| < 0.034).
Condition number 197 against sorghum's 1207; effective rank 7.5 of 11. Maize is
the dataset where 6.4's metric can be reported as four independent phenotypes.
The same caveat still applies though: maize val/test are outlier-enriched
(1.5-2.6x train variance), so R² is comparable between runs on a split but is not
a portable absolute number — report MAE in native units alongside.

The maize probe zeroes the param tensor for the same reason sorghum's does, and
it matters more here: the maize plant token carries `leafCount` and
`stemInternodeSum`, i.e. **two** of the four 6.4 targets, verbatim.

**Maize model width: `N_PARAMS = 14`, `MAX_LEAVES = 28`** (sorghum: 9 and 24), and
**`num_points = 8192`** — a `pointcloud_cam.ply` holds exactly 8192 points and
asking for 8196 silently pads with duplicates. `EmbodiedMAE4MMaize` subclasses
`EmbodiedMAE4M` and rebuilds only the two modules whose shape depends on
`N_PARAMS` (`param_embed`, and the tail `nn.Linear` of `decoder_pred_params`);
everything else reads widths off the tensors. `MaizeDataset4M` is standalone —
NOT a subclass — so a change to sorghum's loader cannot silently change maize.
Its two decoders are verified copies, and its index cache has its own `maize_`
namespace so the two species can never collide.

**31 of the XML's 48 attributes are dropped**: 26 are constant across all 2,250
plants, 3 are deterministic restatements a constancy check cannot see
(`waveRAmp` is bit-identical to `waveLAmp`, `waveRFreq` to `waveLFreq`,
`waveRPhase == waveLPhase + 1.57`), `<Tassel>@seed` is the plant id (an identity
*and* split leak), and `<leaf>@id` is the token index. `leafAzimuthDeg` is stored
as `azJitterDeg = wrap180(az − 180·leaf_index)` because the raw value is
`(180·index + U(−15,15)) mod 360` — raw linear puts identical leaves a full range
apart at the 0/360 seam, and sin/cos hands a decoder R² = 0.9999 for free off
leaf parity, which is the sorghum `roll_angle` bug's twin. Validated: zero values
clipped across 27,278 leaves, round-trip error 6e-08, and all five plant-token
fields reproduce `plant_scores.csv` at r = 1.000000.

**The "Reconstructed Depth" panel borrows its silhouette from the target, in both
species.** `visualize_reconstruction_4m` computes `bg = depth_data < 0.01` from the
*ground-truth* depth and then applies it to the prediction (`pd_d[bg] = np.nan`,
`train_maize_4m.py:369`, `train_sorghum_4m.py:364`). The plant-shaped outline in that
panel is therefore free: the model supplies only the values *inside* a silhouette it
was handed, which is why a randomly-initialised model still renders a crisp plant.
**No reported number is affected** — `depth_mse` and the val/test metrics are computed
on patches, not on this render — but the figure overstates depth reconstruction, and
every depth visualisation from E1 and E2 has the same property. Do not put this panel
in the paper as evidence of depth quality without either dropping the `bg` mask on the
prediction or captioning that the silhouette is ground truth.

**The epoch-1 visualisation is a separate code path, and it crashes *after* a
successful epoch.** `train_worker` calls `visualize_reconstruction_4m` when
`epoch % viz_freq == 0 or epoch == 1`, so a fault there survives model
construction, a full training epoch and validation, then kills the run — and
recurs at every `viz_freq`. Job 16447748 died exactly this way: the maize
override of `decode_params_to_text` took `(n_tokens, N_PARAMS)` while the caller
(`train_maize_4m.py:290`) passes the whole batch `(B, n_tokens, N_PARAMS)` and
unpacks `list[list[str]]`. The wrong rank propagates silently all the way to the
f-string, which fails with `unsupported format string passed to
numpy.ndarray.__format__`. **A smoke test that only calls `forward()` does not
cover this** — exercise `visualize_reconstruction_4m` itself, on CPU with
`matplotlib.use('Agg')`, `num_samples=2` and a `Path` (not `str`) save_dir.
Any maize override of a sorghum method must keep the parent's exact contract;
this one now raises on a non-3D input rather than accepting either shape.

`slurm/train_maize.sbatch` **refuses to start on a partially transferred split**
(`exit 3`). Under `view_sampling` an epoch is one view per plant, so a half-copied
train split trains on a silent subset and no longer matches E2's step budget.
`train_maize_4m.py` refuses for the same reason when `--config` does not exist:
it deliberately has **no built-in default config**, because sorghum's entry point
does, and a copied one means a typo'd path silently builds a sorghum-width model
(24 leaves, 8196 points) at `mask_ratio` 0.15 — a two-day run comparable to
nothing.

**The Globus transfer completed 2026-09-22 18:07 and all three splits verify.**
105,000 / 22,500 / 22,500 view folders resolving to 10,500 / 2,250 / 2,250 plants,
zero folders missing any of the four files, and exact file counts (5 per folder:
the four the loader reads plus `camera_pose.json`). Each split root also holds a
`_params.json` that the `plant_*` glob ignores — it is why a raw `find | wc -l`
reads one over the expected count per split. Index caches: val 53 s, train ~20 min
(Lustre metadata-bound, and concurrent `find` sweeps over the same tree make it
much worse). First maize run is job **16447748** (`outputs/maize_4m`), started **2026-09-23 06:49** on `nova26-gpu-2` (2x RTX PRO 6000, `det_cu128`), both split guards passing 105000/105000 and 22500/22500. It runs at **4.18 it/s = ~79 s/epoch**, so 600 epochs is ~13 h (~15 h with the 24 val passes) — well inside the 2-day wall, and against sorghum's ~3.1 min/epoch on *eight* GPUs. Maize really is far less dataloader-bound: 10x smaller folders and XML params that parse ~300x faster than sorghum's YAML. **329 steps/epoch x 600 = 197,400**, matching E2/E3/E4 exactly.

**Two constructor names differ from the YAML keys**, and both silently do the
wrong thing if guessed: the model takes **`target_points`** (the trainer passes
`target_points=args.num_points`), not `num_points`, and **`pc_loss_name`**, which
the trainer reads from the YAML key `loss_name`. `active_modalities` reaches the
model as a **list**, parsed by `_parse_modalities`; handing the raw
`"pc,rgb,depth,text"` string straight to the constructor iterates it character by
character. A smoke test that builds the model by hand must mirror
`train_maize_4m.py:710-723` exactly rather than improvise the kwargs.

Both import the shared blocks (`PatchEmbed`, `PointCloudEmbed`, `TransformerBlock`,
`chamfer_distance`, `get_2d_sincos_pos_embed`) from `embodied_mae.py`.

**Four differences that break the sorghum loader outright:**
- Params are **XML**, not YAML: `maize_<id>_spline.xml`, every value an XML
  *attribute*, `<plant><Tassel/><Tiller><leaves><leaf .../></leaves></Tiller></plant>`.
- Point cloud is `pointcloud_cam.ply` — a fixed name, not `<plant>_nc_cam.ply`.
- Folders are `plant_<4-digit>_<view>` (e.g. `plant_0004_00`), so the
  `int(name.split('_')[1])` plant-id parse does **not** transfer.
- No `.obj` / `_nc.ply` duplicates, so a folder is ~450 KB against sorghum's ~5 MB.
  `rgb.png`, `depth.png`, `camera_pose.json` are the same filenames. The depth
  encoding **was verified, not assumed** (it is the one case where the wrong
  decoder yields plausible garbage rather than an error): maize is the *same*
  big-endian packed RGBA uint32, confirmed by a byte-order discriminator —
  foreground mean |horizontal gradient| 4.4e-04 big-endian against 1.6e-01 for
  every other ordering, a 350x separation. Two caveats the shared decoder
  absorbs but you should know: maize's alpha channel is a sparse 1-bit mask
  (values 0/128 on 1.4 % of pixels, never on background) contributing ~3e-08 to
  the decoded value, and decoded maize foreground spans ~0.287-0.706 against
  sorghum's ~0.03-0.064, because each renderer normalises by its own near/far.
  `depth_norm_type: minmax` is per image, so that scale gap never reaches the loss.

**Split provenance differs — do not assume it matches sorghum's.** `summary.json`
records seed **0** and scoring by Mahalanobis distance in robustly standardised
parameter space, where sorghum used seed 42 and its own "extremeness" enrichment.
Both group by plant, so no view leaks across splits.

**Target table for the probe is `plant_scores.csv`**, not `features.csv` +
`assignment.csv`: 15 000 rows keyed `plant_NNNN`, carrying `split`, `outlierScore`
and 19 features. Critically it has **real per-plant leaf angle, droop, twist and
curl** (`leaf_angleMean/Std`, `leaf_droopMean/Std`, `leaf_twistAbsMean`,
`leaf_curlMean`) — exactly what sorghum lacks, where the 6.4 leaf-angle target is
degenerate. Maize is therefore the dataset where that target is worth probing.
`plant_params.jsonl` (134 MB) carries the full generator record per plant.

`leafAzimuthDeg` is **circular** (0–360°). Encoding it linearly repeats the
`roll_angle` mistake that made sorghum's leaf angle useless; use a sin/cos pair.

## Porting to another machine

Everything machine-specific is in three places — nothing else needs touching:

1. **`data.data_root` in the YAML you run.** Every config hardcodes the Nova absolute path above.
2. **The `#SBATCH` headers in `slurm/*.sbatch`** — `--partition`, `--account`, `--gres`, `--cpus-per-task`, `--mem`. These encode Nova's partitions (`nova`, `scavenger`) and accounts (`mech-ai`, `mech-ai-scavenger`) and mean nothing elsewhere. On a non-SLURM box, ignore `slurm/` entirely and use the `torchrun` line above.
3. **The conda env.** `environment.yml` rebuilds `det`; it pins a CUDA 12.4 PyTorch, so a different GPU generation may need a different build (see the sm_120 note under Environment).

**Moving the data is the expensive part.** At ~5.0 MB per sample folder × 150 000 folders the split is **~750 GB**, but the four files training actually reads total ~1.8 MB per sample, so a filtered copy is **~265 GB** — under 40 % of the naive transfer. Copy with an include-filter rather than syncing the tree:

```bash
rsync -a --info=progress2 \
  --include='*/' \
  --include='rgb.png' --include='depth.png' \
  --include='*_nc_cam.ply' --include='*_spline.yml' \
  --exclude='*' \
  /path/to/Sorghum_15K/ user@host:/dest/Sorghum_15K/
```

Also copy `assignment.csv` and `features.csv` from the split root, and any warm-start checkpoint you need (`best_model.pth` is ~1.3 GB per run).

Note the dataset loaders build a folder index on first use and cache it — the first run on a fresh copy pays a one-off scan (~24 min for the 105 k train split); later runs print `index cache hit`. The pipeline is **dataloader-bound, not GPU-bound**: `num_workers` (≈1.6 items/s per worker) sets throughput, so give it as many CPUs as the node allows and scale `--mem` with the worker count (~2.3 GB RSS per worker plus ~10 GB per node for model and CUDA context).

## Outputs

Each run writes to `<output_dir>/`:
- `checkpoints/checkpoint_epoch_<N>.pth` every `save_freq` epochs
- `best_model.pth` when val loss improves
- `visualizations/epoch_<N>_sample_<i>_<name>.png` every `viz_freq` epochs (4-row grid for 3M, 5-row grid for 4M including text predictions); skipped automatically when the run is a reduced-modality arm
- `training_history.json` (rolling)
- `config.json` — snapshot of the effective args. **Check this to confirm what a run actually used**, especially `batch_size × world_size`, `max_plants` and `model_size`. Two fields are **not** trustworthy there: `active_modalities` and `loss_name` both serialise as `null` (the snapshot writes the raw YAML keys, while the parsed values live in `args.active_modalities` and `pc_loss_name`). This is long-standing and identical in `e2_pcrgbdt`, so it is not a maize regression — but read the arm's YAML, or infer the modality set from `Total parameters` in the log, rather than believing the `null`.

`outputs/` is gitignored apart from a small whitelist in `.gitignore`. Wandb logging is on by default (`use_wandb: true`, project `embodied-mae-sorghum`); project and run names differ between runs — check the YAML, not the script defaults.

## Things that look like dead code but aren't

- `outputs_sorghum_*/` directories at the repo root are old run outputs kept for reference; the canonical output root is `./outputs/`.
- `tools/process_depth_bg.py`, `tools/process_mask.py`, `tools/validate_sorghum_data.py`, `sweeps/vis.py`, `tools/check_structure.py` are one-off data-prep / diagnostic scripts, not part of any pipeline.
- `sweeps/vis_pc_masking.py` is a standalone tool for visualising the FPS + Dirichlet masking on a single point cloud.
- `tools/visualize_sorghum_pointclouds.py` renders multi-view PC galleries from raw `.ply` files; it doesn't touch the model.
- `eval/analyze_e2.py` reads the E2 arm output dirs and builds the modality-value-add comparison.
