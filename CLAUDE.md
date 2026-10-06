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

**Status as of 2026-09-24.** Run state goes stale fast — the table records which
experiments have been *built*, which is durable. For live progress use
`squeue -u $USER`, `outputs/<run>/training_history.json`, and
`outputs/<run>/config.json` (which records what a run actually used).

| | experiment | state | runs |
|---|---|---|---|
| E1 | headline pretrain | **done** | `4m_pretrain_15k_v2_depthfix_qal`, 1000/1000 ep, global batch 256 |
| E2 | modality value-add | **done** | all four arms 600/600; results in `reports/RESULTS_DECK_2026-09-20.md` |
| E3 | data scaling | **sorghum done; maize running** | sorghum: all three arms, `e3_1k` finished 2026-09-21 11:22 at 6169/6169. maize (Nova, jobs 16576132-4, `sbatch --job-name=maize_e3_1k slurm/scale_arm_maize.sbatch maize_e3_1k`): `maize_e3_1k` ✅ 6169/6169 (2026-09-24); `maize_e3_10k` running; `maize_e3_3k` preempted at 1017/2100 and requeued (auto-resumes). Full-data point is `maize_4m` epoch 600 |
| E4 | model scaling | **sorghum done; maize running** | sorghum: `e4_small` ✅ `e4_large` ✅ (600/600, finished 2026-09-21 21:41). maize (Nova, jobs 16576135-6, same launcher): `maize_e4_small` ✅ 600/600 (2026-09-24); `maize_e4_large` running. The base point is `maize_4m` epoch 600 ✅ (finished 2026-09-23 23:00) |
| E5 | view regime | **not built** | — |
| E6 | masking / noise | **owned here; occluded-scene masking built and run** | Yongyun since 2026-10-02. Random vs structured (`neighbour_first`) vs mixed masking on occluded scenes, both species, val + test: mixed wins, pure structured loses. Read "Yongyun's occlusion work" below |
| E7 | loss study | **owned here; partly run** | Yongyun since 2026-10-02. QAL 0.01, QAL 0.02 squared and QAL + Sinkhorn done (+ Sinkhorn wins on F1 / recall); Chamfer-only and Sinkhorn-only arms running (jobs 16735518-21). The full QAL grid is deferred. Switches: `model.loss_name` (chamfer / chamfer_cdist / qal_loss / sinkhorn), `qal_threshold`, `qal_alpha`, `qal_use_squared`, `pc_loss_weight`, `pc_sinkhorn_weight` — read "E6 and E7" below before sweeping them |
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
  **`--requeue` does NOT cover a walltime `TIMEOUT`**: Slurm requeues on preemption, node failure or
  an admin requeue only. A job that hits its `--time` ends `TIMEOUT`, nothing resubmits it, and from
  outside it looks exactly like one that is still running. Re-run the same `sbatch` line by hand —
  the launchers' auto-resume then caps the loss at `save_freq` epochs.
- **A non-GPU job can block every GPU on a node from `scavenger`.** `scavenger` is
  `PriorityTier=0` with `OverSubscribe=FORCE:1`; `nova` is `PriorityTier=100`. A scavenger job
  therefore cannot co-locate with a `nova` job on the same node, so a 2-CPU job that asks for
  a large `--mem` (or `--mem=0`, which means *all* of it) and **no GPU at all** makes all eight
  GPUs on that node unreachable from scavenger. This is why `sinfo` can show
  `gpu:rtx_pro_6000:0(IDX:N/A)` — i.e. 16 idle GPUs — while your job sits `PENDING`. The only
  two Blackwell nodes are `nova26-gpu-[1-2]` (partitions `nova,scavenger,allnodes`), so a single
  such squatter on each blocks the whole generation. Diagnose with the probe below; the fix is
  patience or the `nova` partition, never a smaller request.
- **"17 A100s idle" is usually stranded GPUs, not free ones** (checked 2026-09-28). Read
  `sinfo -N -O NodeList,StateLong,GresUsed,CPUsState,AllocMem,Memory`: a trailing `-` on the
  state (`mixed-`) means `PLANNED`, i.e. backfill has already promised those GPUs to a
  higher-priority pending job. The rest are idle GPUs on nodes whose **RAM** is gone (the
  `instruction` partition's 1-GPU/128 GB jobs left `nova22-gpu-1`/`-3` with 8 GB free) or that
  host a `nova` / `interactive` job (`OverSubscribe=NO`, which scavenger cannot share).
  `instruction` (tier 1000) *is* `FORCE:1` and does share with scavenger. `sbatch --test-only`
  on that day put even a 2-GPU/16-CPU/32 GB/6 h job only ~6 h ahead of the full
  48-CPU/160 GB/2-day one, so the wall was the queue (other scavenger users with higher
  fairshare), not the request.
- **`nova` is reachable by backfill even at the bottom of its queue — test with real jobs, not
  `--test-only`** (checked 2026-09-29). `mech-ai` had used 98 % of the cluster against a 51 % share,
  so our `nova` priority was the lowest of 153 pending jobs, and `--test-only` put even a
  1-CPU/10-minute job at Oct 3. A real one started in 25 s. `--test-only` does not model backfill;
  submit a probe that runs `hostname` (add `--deadline=now+3minutes` so a probe that cannot start
  expires on its own instead of lingering). What actually blocked a 2-GPU training job was **CPUs
  per node**: on `nova26-gpu-2` (RTX PRO 6000) 16, 24 and 32 CPUs started at once, 48 did not,
  whether at 2 days or 20 h, and 160 GB was fine. `nova` GPU jobs count against `mech-ai`'s
  shared `gres/gpu=17` cap (labmates' jobs wait on it), but unlike scavenger they are not
  preemptible. To move a *pending* scavenger job in place instead of cancelling and resubmitting:
  `scontrol update jobid=J QOS=normal Account=mech-ai Partition=nova` (all three in one call; the
  scavenger QOS is invalid on `nova`), `TresPerNode=gres/gpu:<type>:2`, and for CPUs set
  `CpusPerTask`, `MinCPUsNode` and `NumCPUs` separately — `NumCPUs` alone leaves the per-node
  minimum at the old value. A field scontrol rejects aborts that whole call, so change one per call.

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

**That arm is built: `e2_pcrgbdt_tg`** (`configs/config_e2_pcrgbdt_tg.yaml`, to run on
Delta via `slurm/delta/`). The handicap was bigger than "four ways" suggests, because
`EmbodiedMAE4M` already had a `text_mask_ratio` gate that `train_sorghum_4m.py` never
passed, so **every four-modality run so far trained with text inside the shared
Dirichlet budget**. Measured over 5,000 draws at mask 0.80: `e2_pcrgbd` sees 117.0
vision tokens; `e2_pcrgbdt` as trained sees 107.4 (-8.2 %) while text is 58 %
visible; the gated arm sees exactly 117.0 (seed-matched vision masks identical to
`e2_pcrgbd` in 5000/5000) with text at 5/25. The trainer now reads
`model.text_mask_ratio` / `--text_mask_ratio` (default `None` = every earlier run,
verified bit-identical), records it in `config.json` and in every checkpoint, and
refuses a resume whose checkpoint was trained under a different value. **Two
variables move against `e2_pcrgbdt`, not one**: vision budget 107→117 *and* text
visibility 58 %→20 %. If tg ≈ pcrgbd the arm cannot say which mattered; if tg ≈
pcrgbdt the handicap was not the cause. Compare by probe at `checkpoint_epoch_600.pth`,
not by `param_*` or total loss. `eval/linear_probe.py` still builds every model
ungated and asserts `text_mask_ratio is None` — correct for tg too, since
`forward_encoder_select` never reads it; do not "fix" the probe to forward it.

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

### Yongyun's occlusion work, 2026-10-01 to 2026-10-06 — summary for Alloy

Short version of the long section after this one. Every number below comes from
`reports/` or `outputs/<run>/training_history.json` and is one seed (seed 1).

**Task.** Input = an occluded scene (target plant + 2 neighbour plants in a row,
in RGB, depth and point cloud). Output = the clean target plant alone. The
target's procedural params are given as conditioning (all 25 tokens visible,
never reconstructed). 80 % of the 588 RGB / depth / PC tokens are masked, with
the Dirichlet split. **The scenes are synthetic and built on the fly by
`occlusion_scene.py` from the single-plant renders** — no rendered occlusion
data exists yet (see "What we need from Alloy's renders" below). Distillation is
not used anywhere in this work.

**Three masking policies** (training only — validation and test always mask uniformly):
- **random** (`mask_policy: uniform`) — hide any 80 % of tokens.
- **structured** (`neighbour_first`) — hide the tokens that show a neighbour
  first (an image patch that is >= 30 % neighbour pixels, a PC token whose FPS
  centre is a neighbour point), then fill the rest of the 80 % at random. Used on
  RGB, depth and PC alike.
- **mixed** (`neighbour_first` + `nf_prob: 0.5`) — a coin per scene: half the
  scenes structured, half random.

**Best recipe: mixed masking + QAL (0.01) + 0.5 x Sinkhorn.** Configs:
`configs/config_{sm,maize}_scene_mix_sink.yaml`. Test split, occluded input,
epoch 200 (`reports/test_scene/{sorghum,maize}.json`):

| model | sorghum F1@.01 | R@.03 | P@.03 | chamfer | maize F1@.01 | R@.03 | P@.03 | chamfer |
|---|---|---|---|---|---|---|---|---|
| random | 0.285 | 0.635 | 0.893 | 0.00224 | 0.138 | 0.674 | 0.799 | 0.00248 |
| structured | 0.186 | 0.506 | 0.816 | 0.00534 | 0.113 | 0.561 | 0.645 | 0.00929 |
| mixed | 0.296 | 0.660 | 0.895 | 0.00209 | 0.171 | 0.725 | 0.797 | 0.00235 |
| random + Sinkhorn | 0.311 | 0.669 | 0.835 | 0.00271 | 0.216 | 0.784 | 0.706 | 0.00256 |
| **mixed + Sinkhorn** | **0.324** | **0.731** | 0.839 | 0.00223 | **0.255** | **0.824** | 0.716 | 0.00241 |
| mixed + Sinkhorn, 400 ep* | running | | | | **0.401** | **0.901** | 0.817 | **0.00146** |

\* 200 + 200 epochs with a cosine warm restart; the other rows stop at 200, so
this row shows "train longer", not "better masking".

- Mixed + Sinkhorn beats random on F1@0.01 for 89 % (sorghum) / 90 % (maize) of
  test plants. The gain is at the **leaf tips**: outer-third recall@0.03 0.205 ->
  0.330 sorghum, 0.304 -> 0.598 maize.
- Mixed masking and Sinkhorn help in different ways and the gains add up:
  Sinkhorn raises recall (QAL alone clumps the predicted points), and mixed
  masking removes Sinkhorn's chamfer cost. The price is ~5-7 points of precision@0.03.
- **Pure structured masking loses** (chamfer 2.4x sorghum, 3.7x maize). Why,
  measured: it almost never lets the encoder see a neighbour (5.4 % / 9.2 % of
  visible PC tokens are neighbour, against ~60 % at test), so it never learns to
  ignore one and at test draws neighbour leaves as target. It WINS in maize
  only if the test also masks neighbours first (oracle evaluation, which needs a
  segmentation at test): chamfer 0.00152 vs 0.00236. Mixed keeps half the
  scenes random, so it learns both skills.
- **Also tried and lost to random:** leaf-weighted masking (`leaf_mask.py`).
  Edge-biased masking (`center_bias`) already lost in
  `embodiedmae4m_whole_to_whole` before this work.
- **Losses** (sorghum test, random masking, occluded F1@0.01 / R@.03 / P@.03 /
  chamfer): QAL 0.285 / 0.635 / 0.893 / 0.00224; QAL 0.02 squared 0.287 / 0.643 /
  0.905 / 0.00190; QAL + Sinkhorn 0.311 / 0.669 / 0.835 / 0.00271; QAL 0.02
  squared + Sinkhorn 0.311 / 0.678 / 0.847 / 0.00213. Chamfer-only and
  Sinkhorn-only runs are still training. Chamfer is squared and favours squared
  losses, so judge on F1 / recall / precision too.
- **Mask ratio at test** (the same 80 %-trained models, tested at 70 / 80 / 90 / 95 %):
  sorghum holds up at 90 % (mixed + Sinkhorn F1 0.324 -> 0.309), maize drops more
  (0.255 -> 0.192), and 95 % is poor for every model. Mixed + Sinkhorn has the
  best F1 and recall at every ratio. No model has been TRAINED at 90 %.
- **Long run for real data:** `outputs/long_sorghum_mixsink_s1/checkpoints/checkpoint_epoch_600.pth`
  — sorghum, mixed masking, QAL 0.02 squared + Sinkhorn, **no param stream**
  (real plants have none), 4 GPUs, 600 epochs. Test occluded F1@0.01 0.281,
  recall@0.03 0.656.

**What we need from Alloy's occluded renders, to train and test on them instead
of the synthetic scenes:**
1. A per-pixel **instance mask** (which plant each RGB / depth pixel belongs to)
   and a **per-point plant ID** in the cloud. Structured / mixed masking needs to
   know which tokens are neighbour, and the oracle evaluation needs it too.
2. The **clean target alone** (RGB, depth, cloud) from the **same camera**, as
   ground truth.
3. The **target's params only** (not the neighbours').
4. **Splits by target plant**, with neighbours drawn from the same split, so no
   plant leaks between train and val / test.
5. Ideally a **partial cloud as a depth camera would see it** (only visible
   surfaces), which is what the real data will look like.

**Where things are.** The long section below has every run, job ID and caveat.
Results: `reports/test_scene/` (test), `reports/mixed_masking/` (val, per plant),
`reports/scene_oracle.json`. Scripts: `eval/eval_test_scene.py`,
`eval/eval_scene_oracle.py`, `export/export_mixed_masking.py`. Launchers:
`slurm/sm_arm.sbatch`, `slurm/maize_scene_arm.sbatch`, `slurm/long_arm.sbatch`.
Write-ups (claude.ai pages, private until Yongyun shares them): test + loss +
400-epoch validation https://claude.ai/artifact/VopKKf1xQFbUFXFL3XEsB5, masking
explorer https://claude.ai/artifact/HHwrfv7PRxkTwEc9gJ1AK9, val arms
https://claude.ai/artifact/1W4vkx8y5NzDbudHNtmsrq. A hands-on walk-through of
the masking itself (patchify, FPS 196 x kNN 32, neighbour labels, Dirichlet
budget, the three policies, checked against the model's own masking) is in
`/work/mech-ai-scratch/yongyun/masking/` (`masking_steps.py`, README).

**Running / next (2026-10-06):** sorghum mixed + Sinkhorn 400 epochs (job
16734120, at epoch 340; its test eval runs automatically when it ends);
Chamfer-only and Sinkhorn-only arms (16735518-21). Not started: seed 2,
continuing the random baseline to 400 epochs for a fair comparison, training
at 90 % masking, and training on Alloy's renders once they exist.

### Occluded-scene training (structured-masking follow-up, built 2026-10-01)

The occlusion meeting after 2026-09-29 rejected scoring structured masking on
clean input and fixed the setup: **input** = a scene (target plant + neighbouring
plants, their leaves in its RGB, depth and cloud), **target** = the clean target
plant alone, **params** = the target's only, as conditioning (`text_mask_ratio:
0.0` -- all 25 tokens visible, none reconstructed). Alloy is rendering such
scenes properly; until they land, `occlusion_scene.py` builds them on the fly
from the single-plant renders (YAML `model.occlusion_scene`):

- Neighbours are other plants in the batch, taken to **world** coordinates via
  `camera_pose.json` (the dataset returns it with `return_pose=True`), spun about
  their stem, and stood on the target's ground along a row (`spacing` metres).
  capture.py puts every stem at the world origin (checked on 60 plants), so the
  placement is exact 3D, not a 2D billboard: the cloud and the images agree.
- Cloud: target + neighbours, cut to a `crop_radius` cylinder around the target's
  stem, resampled to `num_points`, kept in the **clean target's** normalisation
  frame. Images: neighbour points splatted with a z-buffer, hole-closed and
  colour-smoothed (raw splats speckle -- a free "neighbour" cue).
- The model scores the clean tensors (`forward(..., targets=, loss_tokens=)`);
  image patches where a neighbour shows are in the loss **even when visible**.
  `mask_policy: neighbour_first` masks neighbour tokens first -- an oracle, since
  nothing marks neighbours at test time; `uniform` is the deployable default.
- Whenever the block is present, validation runs twice: clean, and on fixed
  occluded val scenes (`val_occ_*` in `training_history.json`, `val_occ/*` in
  W&B; seeds depend on `val_seed`, rank and batch, so arms at the same GPU count
  see identical scenes). `prob: 0.0` = train clean, still score occluded -- the
  control. `best_model.pth` stays selected on clean val.
- Arms: `configs/config_sm_scene.yaml` (prob 0.5) vs `config_sm_scene_off.yaml`
  (prob 0), launched with `slurm/sm_arm.sbatch scene|scene_off <seed>`. Preview:
  `figures/plot_scene_preview.py` -> `reports/scene_preview_val.png`. At the
  defaults a val scene hides ~14 % of the target's plant pixels (4-43 %) and
  ~60 % of the input cloud is neighbour.
- `structured_mask.py` (mask-only, billboard neighbours) and
  `eval/eval_occlusion.py` are the earlier design and are untouched;
  `structured_mask` and `occlusion_scene` with prob > 0 refuse to combine.
- **Metrics and figures (added 2026-10-01).** `evaluate()` reports PC
  **F1 / precision / recall at 0.01, 0.02, 0.03** (Euclidean, unit-sphere
  cloud; precision = predicted points near the target, recall = target points
  covered) for clean and occluded val: W&B `metrics/f1@0.01`, `val_occ/f1@0.01`, …,
  history `val_pc_f1_001`, `val_occ_f1@0.01`, …. A series added mid-run is
  padded with `null` so it stays index-aligned with `val_loss` /
  `val_occ_epoch`. Chamfer is now computed chunked by `pc_scores`; the dense
  `embodied_mae.chamfer_distance` built a 12 GiB tensor at B=16 and OOMed
  `sm_scene_off_s1` mid-validation on an A100-40GB (job 16686223, epoch 120).
  `visualize_scene_4m` draws target | input | visible | reconstruction per
  modality plus the three clouds and per-sample F-scores, on fixed seeded
  scenes (`scene_visualizations` in W&B) and on clean input for arms the 5-row
  grid cannot draw (3 modalities). The `sm_scene*` configs now set `viz_freq: 20`.
  Expect low recall at 0.01 everywhere (~0.15-0.2): see "the PC decoder
  collapses" below.

**The PC decoder collapses most tokens' points, in every run (measured
2026-10-01).** The folding head (`decoder_pc_fold`) maps [512-D token feature,
2-D grid coordinate] to ABSOLUTE xyz with no per-token centre or offset. The
grid is not ignored -- its first-layer weight columns are 4-5x a feature
column's, and in `e2_pcrgbdt` the within-token output std is a third of the
across-token one -- but the spread is very uneven: most tokens emit a tight
clump (median radius 0.011) and a minority spread widely (mean 0.063). The cause
is not established. On 16 val plants, clean, mask 0.8 (share of predicted points
with another within 0.001; the target has 0.004; median radius of a token's 41
points; recall@0.01):

| run | loss | budget | dup | token radius | R@.01 |
|---|---|---|---|---|---|
| `e2_pc` | qal | 197k steps x 32 | 0.72 | 0.003 | 0.10 |
| `e2_pcrgbdt` | qal | 197k x 32 | 0.62 | 0.012 | 0.16 |
| `maize_4m` | qal | 197k x 32 | 0.68 | **0.0005** | 0.20 |
| `4m_pretrain_15k` | chamfer | ~17x the samples | 0.48 | 0.022 | 0.23 |
| E1 `..._v2_depthfix_qal` | qal | ~17x | 0.44 | 0.040 | 0.32 |

So at the shared E2/E3/E4 budget the predicted cloud is ~196 blobs (maize's is
~196 points), recall@0.01 is capped by that rather than by the loss (chamfer
collapses too), and the slow un-collapsing with more training is a large part
of why E1's chamfer is 3.5x better than `e2_pcrgbdt`'s. It also explains
"precise but low recall". Recall additionally falls with distance from the
centroid (0.31 inside r < 0.25 to 0.02 beyond r > 0.75 for `sm_scene_s1`), so
leaf tips are missed on top of the collapse. 160 of the 8196 points are exact
copies (the pad in `forward_decoder`). Any change to the head changes every PC
number, so it is a decision for new runs, not a patch to compare against E1-E4.
- **Structured masking in this setup = `mask_policy: neighbour_first`** (added
  2026-10-02): on a scene sample, patches where a neighbour shows and PC tokens
  whose FPS centre is a neighbour point are masked first (verified: 100 % of
  flagged tokens masked, the rest of the budget uniform); validation stays
  uniform for every arm. Arms: sorghum `config_sm_scene_struct.yaml` vs the
  finished `sm_scene_s1`; maize `config_maize_scene_struct.yaml` vs
  `config_maize_scene.yaml` (`slurm/maize_scene_arm.sbatch scene|scene_struct
  <seed>`). The older mask-only design (`structured_mask.py`, `sm_nbr`/`sm_off`)
  never ran here; the `paper_mask*` runs in `embodiedmae4m_whole_to_whole` used
  synthetic leaf occluders and random blob/edge masks, and all three structured
  variants lost to uniform by +71-98 % chamfer over 2 seeds.
- **Maize has occluded scenes too** (`train_maize_4m.py`, ported 2026-10-02:
  `text_mask_ratio`, `occlusion_scene`, `--seed`, F-scores, scene figures, a
  resume guard). Maize geometry was checked rather than assumed: the camera
  looks down -z, the stem is at the world origin (median 1 cm), 40 deg FOV, but
  depth near/far are **per plant** (`camera_pose.json`), so
  `MaizeDataset4M(return_pose=True)` returns `near_far` as a 9th item and
  `compose(..., near_far=)` uses it (None = sorghum's fixed planes,
  bit-identical). Maize plants are ~0.58x sorghum's footprint, so the maize
  configs use spacing [0.15, 0.35] / crop 0.45 (11.7 % of target pixels
  hidden, 65 % of the cloud neighbour; sorghum's defaults on sorghum: 16 % /
  60 %). Preview: `reports/scene_preview_maize_val.png`.
- **First results (seed 1, epoch 200, all six arms finished 2026-10-02):**
  uniform beats `neighbour_first` on occluded val in both species -- occluded
  chamfer 0.00226 vs 0.00536 sorghum, 0.00250 vs 0.00988 maize -- with lower
  precision@0.03 for the structured arm (0.82 vs 0.89, 0.65 vs 0.79), i.e. it
  reconstructs visible neighbour leaves as target: on scene samples it never
  sees a visible neighbour token, while validation masks uniformly. The
  clean-trained control is best on clean (0.00156) and 8.5x worse occluded;
  no params costs +31 % clean / +48 % occluded. Open: maize structured is
  better on CLEAN (0.00121 vs 0.00194), sorghum is not. Write-up:
  https://claude.ai/artifact/1W4vkx8y5NzDbudHNtmsrq.
- **QAL-variant pair (`*_q02sq`, submitted 2026-10-02, jobs 16702322-5):** the
  same four sorghum/maize uniform and structured scene arms with
  `qal_threshold: 0.02`, `qal_use_squared: true`, `pc_loss_weight: 15.0`
  (`configs/config_{sm,maize}_scene{,_struct}_q02sq.yaml`, launched with the
  arm names `scene_q02sq` / `scene_struct_q02sq`). Question: does structured
  masking do better under this loss? Read it two ways -- struct vs uniform
  within the new loss (does the gap close?) and struct_q02sq vs struct (did
  the loss help structured masking?). The val metrics are the same code for
  every run, so cross-loss comparison is valid, but `val_pc_chamfer` is
  squared and favours a squared loss: judge on F1 / precision / recall too.
  State 2026-10-03: `sm_scene_q02sq_s1` finished (vs `sm_scene_s1`: chamfer
  -17 % clean / -14 % occluded, F1@0.01 equal); the other three were cancelled
  by hand at 12:02 to free GPUs (last checkpoints: sorghum struct 100, maize
  140, maize struct 160) -- rerun the same `sbatch` lines to resume.
- **Leaf-weighted masking (`leaf_mask.py`, YAML `model.leaf_mask`, built
  2026-10-03).** Of the ~80 % budget, the TARGET's leaves are masked more than
  its stem: a point >= `stem_radius` (0.05 m, ~25 % of points fall inside) from
  the stem axis is leaf and gets weight `weight` (3.0); stem, background and
  neighbour tokens get 1; the masked set is a weighted sample (Gumbel top-k,
  `scores_from_weights`, reached by passing FLOAT `mask_flags`; bool flags keep
  `neighbour_first`'s path unchanged). Training only, validation stays uniform,
  no neighbour labels needed -- unlike `neighbour_first`, nothing is unavailable
  at test. Measured on real val batches: PC leaf tokens are ~92 % (sorghum) /
  78 % (maize) of all tokens, so leaves can only go from ~77 % masked to 80-84 %
  -- the main PC effect is that the stem stays visible (41 % / 52 % masked); in
  RGB/depth, leaf patches are masked 97-98 % against 79-82 % for the rest.
  Arms: `configs/config_{sm,maize}_scene_leaf.yaml` = the uniform scene arm +
  this block, old QAL 0.01 loss, so the baselines are `sm_scene_s1` /
  `maize_scene_s1`; launch arm name `scene_leaf`. **Result: worse than uniform
  in both species** -- maize epoch 200 clean chamfer 0.00285 vs 0.00194,
  occluded 0.00698 vs 0.00250; sorghum 24 % worse at epoch 100, then cancelled
  by hand (job 16707129). Likely why, untested: target leaf patches were
  masked 97-98 % while neighbour patches kept weight 1, so the visible leaves
  in training were mostly NEIGHBOURS'. Same lesson as `neighbour_first`: a
  masking policy that changes WHAT is visible relative to test teaches a
  wrong rule.
- **Why `neighbour_first` lost, measured** (`logs/nb_visibility_tmp.py`, job
  16706899, sorghum val scenes): on scene samples, visible PC tokens centred
  on a neighbour are 0.9 % under `neighbour_first` vs 62 % under uniform (=
  test); visible image patches >= 30 % neighbour 0 % vs 32.5 %. It is not
  all-or-nothing (patches < `patch_frac` neighbour and target tokens whose
  32-point group reaches a neighbour leak ~2.5 % of neighbour pixels / 4.7 %
  of group points), but the model barely practises rejecting a visible
  neighbour, and at test it draws them as target (lower precision AND recall).
- **Two follow-ups for structured masking (2026-10-03):**
  (1) `occlusion_scene.nf_prob` -- neighbour-first on only that share of the
  scene samples, uniform on the rest (1.0 = every earlier run; `to_dict` omits
  it at 1.0 so older checkpoints' resume guards still match). Arms
  `configs/config_{sm,maize}_scene_mix.yaml` (`nf_prob: 0.5`, old loss), arm
  name `scene_mix`, jobs 16710888 / 16710889 on `nova`; compare with
  `*_scene_s1` and `*_scene_struct_s1`.
  (2) Oracle evaluation, `eval/eval_scene_oracle.py` (`slurm/scene_oracle.sbatch`,
  output `reports/scene_oracle.json`): scores final checkpoints on occluded
  val under uniform AND neighbour-first test-time masking (`evaluate(...,
  oracle=True)`), i.e. assuming a target segmentation at inference -- the
  setting structured training matches. Compare structured vs uniform-trained
  arms UNDER THE SAME oracle masking, never an oracle number against a
  uniform one. **Result (job 16710985, epoch 200, seed 1; its uniform column
  reproduces the logged val_occ chamfer within ~2 %):** maize -- structured
  0.00152 vs uniform-trained 0.00236 chamfer under oracle masking (-36 %),
  F1@0.01 0.242 vs 0.145, recall@0.03 0.809 vs 0.683: structured WINS.
  Sorghum -- 0.00283 vs 0.00234 (+21 %), F1@0.01 0.247 vs 0.284: it does not.
  Uniform-trained arms barely move under oracle masking (they already ignore
  visible neighbours). The oracle ranking tracks the CLEAN ranking (maize
  structured is better on clean, sorghum structured is not), so the open
  question is still why structured training helps maize and not sorghum.
  **Mixed arms (`*_scene_mix_s1`, `nf_prob: 0.5`, finished 2026-10-04):**
  vs uniform, logged epoch 200 -- clean chamfer 0.00175 vs 0.00192 sorghum,
  0.00143 vs 0.00194 maize; occluded re-scored on identical scenes and masks
  (job 16715590) -- 0.00216 vs 0.00227 sorghum (-5 %), 0.00256 vs 0.00255
  maize (TIED), occluded F1@0.01 0.294 vs 0.285 and 0.160 vs 0.135. Under
  oracle masking mixed is the best sorghum arm (0.00210) and second in maize
  (0.00212, behind pure structured 0.00152). **Evaluation noise alone moves
  occluded chamfer up to ~5 %** (maize mixed 0.00243 logged vs 0.00256
  re-scored -- `load_pointcloud` subsamples unseeded), so compare arms on the
  paired re-score and treat differences under ~5 % as noise, before seed noise.
  One seed each; seed 2 of uniform + mixed is the next run. Write-up:
  https://claude.ai/artifact/1W4vkx8y5NzDbudHNtmsrq (version 3).
  **Per plant** (`export/export_mixed_masking.py`, job 16715835, all 2,250 val
  plants per species, same scene / cloud subsample / test mask for every arm;
  data in `reports/mixed_masking/{sorghum,maize}.json`): mixed beats uniform on
  occluded chamfer for 72 % (sorghum) / 70 % (maize) of plants, median ratio
  0.93x / 0.85x, F1@0.01 better on 65 % / 67 %; pure structured beats uniform
  on 0.3 % / 32 %. Over ALL val scenes, visible PC tokens centred on a
  neighbour are 60 % / 65 % uniform, 33 % / 37 % mixed, 5.4 % / 9.2 %
  neighbour_first (the 0.9 % above came from a 96-scene sample). Interactive
  masks + reconstructions: https://claude.ai/artifact/HHwrfv7PRxkTwEc9gJ1AK9.
- **QAL + Sinkhorn (`pc_sinkhorn.py`, YAML `model.pc_sinkhorn_weight /
  _points / _blur`, built 2026-10-04).** Adds weight x the debiased Sinkhorn
  divergence (geomloss, p=2, tensorized backend -- no KeOps/nvcc needed) on
  random point subsets to the PC loss; weight 0 = every earlier run.
  `pc_loss_weight` multiplies the WHOLE PC loss, Sinkhorn included, so the
  effective weight is the product. Measured on `*_scene_s1` (job 16716188):
  Sinkhorn GROWS from epoch 20 to 200 (sorghum 0.053 -> 0.071, maize 0.082 ->
  0.097) while QAL falls (0.058 -> 0.033) -- QAL buys nearest-point fit with a
  worse distribution (the clumping behind low recall); subsampling floor
  0.0006. At 2048 points / scaling 0.9 it nearly doubled the step time, so the
  arms use 1024 points, blur 0.02, scaling 0.8. Arms (effective weight 0.5,
  ~QAL size at epoch 200): `config_{sm,maize}_scene_sink.yaml` (QAL 0.01;
  baselines `*_scene_s1`), `config_sm_scene_q02sq_sink.yaml` (QAL 0.02 squared,
  `pc_sinkhorn_weight` 0.0333 x 15; baseline `sm_scene_q02sq_s1`). Jobs
  16716270-2 on `nova`. The logged `train_pc` / `val_pc` include the Sinkhorn
  term, so compare these arms on `val_pc_chamfer` and the F-scores only.
  **Result: Sinkhorn trades precision for a large recall gain.** Maize epoch
  200 vs `maize_scene_s1`: recall@0.01 0.250 vs 0.134 clean / 0.199 vs 0.110
  occluded, F1@0.01 0.273 vs 0.180 / 0.214 vs 0.138, recall@0.03 0.840 vs
  0.721 / 0.774 vs 0.668, precision@0.03 0.762 vs 0.837 / 0.708 vs 0.792,
  chamfer -3 % clean / +7 % occluded. Sorghum QAL 0.02 squared + Sinkhorn
  epoch 200 vs `sm_scene_q02sq_s1`: F1@0.01 0.326 vs 0.296 / 0.310 vs 0.286
  (the best sorghum F1 of any arm), recall@0.03 0.725 vs 0.681, precision@0.03
  0.851 vs 0.910, chamfer +11 %. Chamfer does not reward coverage, so judge
  these arms on F1 / recall. Mixed masking + Sinkhorn (the 2 x 2 corner):
  `config_{sm,maize}_scene_mix_sink.yaml`, jobs 16717921 (scavenger) /
  16717922 (moved to `nova`).
- **2 x 2 result (masking x Sinkhorn, QAL 0.01, epoch 200, seed 1, all
  finished 2026-10-05): the two gains STACK.** Mixed + Sinkhorn vs uniform --
  sorghum F1@0.01 0.342 vs 0.297 clean / 0.323 vs 0.284 occluded,
  recall@0.03 0.774 vs 0.664 / 0.728 vs 0.632, chamfer 0.00180 vs 0.00192 /
  0.00226 vs 0.00226, precision@0.03 0.847 vs 0.898 / 0.839 vs 0.892. Maize
  F1@0.01 0.343 vs 0.180 / 0.256 vs 0.138, recall@0.03 0.895 vs 0.721 /
  0.820 vs 0.668, chamfer 0.00142 vs 0.00194 / 0.00247 vs 0.00250,
  precision@0.03 0.794 vs 0.837 / 0.724 vs 0.792. It is the best F1 / recall
  of every arm in both species, and mixed masking removes Sinkhorn's chamfer
  cost (sorghum occluded 0.00275 uniform + Sinkhorn -> 0.00226). Cost: ~5-7
  points of precision@0.03. One seed.
- **TEST split (`eval/eval_test_scene.py`, jobs 16734136 maize / 16734246
  sorghum, `reports/test_scene/{sorghum,maize}.json`, 2026-10-05).** Last
  checkpoints, never selected on test; clean + occluded, same scenes / masks
  per arm; recall@0.03 also split into inner / middle / outer thirds of the
  plant's radius from the stem. Mixed + Sinkhorn vs uniform, occluded:
  F1@0.01 0.324 vs 0.285 sorghum, 0.255 vs 0.138 maize; recall@0.03 0.731 vs
  0.635, 0.824 vs 0.674; chamfer 0.00223 vs 0.00224, 0.00241 vs 0.00248;
  better F1@0.01 on 89 % / 90 % of plants, better recall@0.03 on 99.6 % / 94 %.
  The gain is at the EDGES: outer-third recall 0.205 -> 0.330 sorghum, 0.304
  -> 0.598 maize; inner third 0.795 -> 0.863, 0.819 -> 0.897. Test numbers
  reproduce val within noise. Write-up:
  https://claude.ai/artifact/VopKKf1xQFbUFXFL3XEsB5. Gotcha found doing it:
  geomloss turned autograd back ON inside `torch.no_grad()`; `pc_sinkhorn.py`
  now restores the caller's grad mode (training unaffected).
- **Long run for real-data testing (`configs/config_long_sorghum_mixsink.yaml`,
  `slurm/long_arm.sbatch`, job 16724098, started 2026-10-04 23:31).** Sorghum,
  mixed masking (nf_prob 0.5), QAL 0.02 squared x15 + Sinkhorn (effective
  0.5), NO parameter stream (`pc,rgb,depth` -- real plants have no procedural
  params; the planned real data is sorghum RGB-D + a ~8,096-point cloud, which
  the loader resamples to 8,196), scenes at prob 0.5, 4 GPUs x 16 = global 64,
  lr 3e-4, 600 epochs on `nova` (2-day wall, auto-resume; a TIMEOUT needs the
  same sbatch line rerun). Not comparable step-for-step with the 2 x 16 arms,
  and its occluded val scenes differ (scene seeds depend on rank). The exact
  combination had not run before; the maize mixed + Sinkhorn arm is the
  nearest check. **Finished 2026-10-05 16:49 (17 h 18 m, 600/600).** Epoch
  600, logged val: clean chamfer 0.00181, F1@0.01 0.324, recall@0.03 0.741,
  precision@0.03 0.853; occluded 0.00260 / 0.275 / 0.648 / 0.833. Without any
  parameters it matches or beats the param-conditioned 200-epoch uniform arm
  `sm_scene_s1` on clean input (0.00192 / 0.297 / 0.664) and is close on
  occluded (0.00226 / 0.284 / 0.632); against the no-param 200-epoch arm it is
  -28 % / -22 % chamfer. Budgets differ (~3x the samples) and so do the
  occluded scenes, so these comparisons are approximate until re-scored on
  the 2-rank scenes. Checkpoint for real-data tests:
  `outputs/long_sorghum_mixsink_s1/checkpoints/checkpoint_epoch_600.pth`.
- **Mixed + Sinkhorn continued to 400 epochs (2026-10-05, jobs 16734121 maize
  / 16734120 sorghum, both on `nova`).** `configs/config_{sm,maize}_scene_mix_sink400.yaml`
  = the 200-epoch arm with `epochs: 400` and its own output dir
  (`outputs/*_scene_mix_sink400_s1`), started from the 200-epoch checkpoint
  through the launchers' `INIT_CKPT` env var (used only when the run has no
  checkpoint of its own, so a preempted continuation resumes from its latest
  epoch). It is a **cosine warm restart** (LR ~0 -> ~7.9e-5 at epoch 200, back to
  0 by 400; see the maize 1000-epoch note for why): report it as "200 + 200
  epochs with a cosine restart". Validation occluded, epoch 200 -> last:
  maize (finished 2026-10-06 02:35) F1@0.01 0.256 -> 0.392, recall@0.03 0.820
  -> 0.894, precision@0.03 0.724 -> 0.814, chamfer 0.00247 -> 0.00158; sorghum
  (epoch 340 of 400 at 03:00) 0.323 -> 0.376, 0.728 -> 0.804, 0.839 -> 0.862,
  0.00226 -> 0.00170. Precision rises with recall, so the extra epochs buy
  both. Maize test (`reports/test_scene/maize_mixsink400.json`, job 16736554):
  clean chamfer 0.00079, F1@0.01 0.503, recall@0.03 0.947, precision@0.03
  0.890; occluded 0.00146 / 0.401 / 0.901 / 0.817; better than its own epoch
  200 on 85-96 % of test plants depending on the metric. **The baselines were
  not continued**, so "400-epoch mixed + Sinkhorn vs 200-epoch random" mixes
  training length with masking; continue `*_scene_s1` to 400 the same way
  before claiming a masking gain at 400.
- **Test-time mask-ratio sweep (jobs 16736233-4, `eval/eval_test_scene.py
  --mask-ratio`, `reports/test_scene/{sorghum,maize}_mr{0.7,0.9,0.95}.json`).**
  The 80 %-trained epoch-200 models, occluded test F1@0.01 at 70 / 80 / 90 /
  95 % masking -- sorghum random 0.295 / 0.285 / 0.255 / 0.205, mixed + Sinkhorn
  0.324 / 0.324 / 0.309 / 0.277; maize random 0.143 / 0.138 / 0.107 / 0.074,
  mixed + Sinkhorn 0.271 / 0.255 / 0.192 / 0.134. Mixed + Sinkhorn keeps the
  best F1 and recall at every ratio, but at 90 % in maize its precision@0.03
  falls to 0.613 and its chamfer (0.00440) is worse than random's (0.00362).
  Training at 90 % has not been tried.
- **Loss arms still missing from the E7 table (submitted 2026-10-06, jobs
  16735518-21 on `scavenger`, `--requeue`):** `config_{sm,maize}_scene_cham.yaml`
  (`loss_name: chamfer_cdist`, squared nearest-neighbour Chamfer via
  `torch.cdist`, `pc_loss_weight` 15 -- the dense `chamfer` OOMs at B=16) and
  `config_{sm,maize}_scene_sinkonly.yaml` (`loss_name: sinkhorn`: no QAL, PC
  loss = 0.5 x Sinkhorn). Random masking, so they compare against `*_scene_s1`
  (QAL) and `*_scene_sink_s1` (QAL + Sinkhorn). When they finish: `python
  eval/eval_test_scene.py --species <sp> --arms cham sinkonly --tag loss
  --no-examples`.
- **Alloy's data (checked 2026-10-06): no occluded renders exist yet.**
  `/work/mech-ai-scratch/alloy/Maize_1` (dated 2026-10-02) is a new
  single-plant maize set (one plant per view, 1024 px RGBA renders, 40 deg
  FOV), not scenes. Every occluded number in this repo therefore comes from
  `occlusion_scene.py`'s synthetic scenes, whose neighbours are splatted from
  their point clouds and look blurrier than a real render (a free cue for
  "this is a neighbour" that real occlusion will not give). Re-run the key
  arms on Alloy's renders once they land; the label list the training needs is
  in the summary section above.
- `eval/eval_scene_ckpts.py` (`slurm/scene_ckpts.sbatch`) re-scores every saved
  checkpoint of the arms on clean + occluded val with the trainer's own
  `evaluate()`, same shards and scene seeds, batches held fixed across
  checkpoints so arm differences are paired. `--wandb-runs` logs the rows
  (`backfill/*`, plot against `epoch`) and last-checkpoint figures into a
  FINISHED run's own W&B entry.

**Sorghum param schema, two incompatible layouts (2026-10-01).** Alloy's working
tree of the main checkout (`alloy/embodiedmae`, uncommitted) moved
`embodied_mae_4m.py` to the post-rewrite leaf layout: slot 4 = `width`/0.2,
slots 5-6 zero, `length`/1.25 (was waviness, waviness, `length`/1.0), and its
`load_spline_params` now raises on an original file. Every sorghum checkpoint
trained before 2026-09-30 (E1-E4, E2, the distillation runs) learned the OLD
layout, so evaluating one through that code hands it mis-scaled leaf tokens --
sorghum only, since maize has its own loader. This checkout keeps the old layout
and reads the `SorghumData` originals, which reproduces the training tensor.
**Measured, it barely matters for reconstruction**: `e2_pcrgbdt` epoch 600 on
150 test plants, paired (same masks), random masking 0.8 with params visible --
chamfer 0.002404 old layout vs 0.002415 new (+0.5 %), F1@0.01 0.262 vs 0.260. It
is a correctness trap for anything that reads the param stream (the probes zero
it; param-reconstruction metrics would not survive it), not an explanation for
reconstruction quality.

### E6 and E7 — owned here since 2026-10-02

Masking/noise (E6) and the loss study (E7) were listed as a collaborator's;
Yongyun has taken both over, so they are work for this repo and may be
scheduled from it. Their results land in the same table as everything above,
so every E6/E7 run must still match this programme on the three things that
silently break the comparison: the **same** 70/15/15 split of `Sorghum_15K`
(seed 42), the **same** global batch of 32 (16×2), and a budget stated in
**optimizer steps** rather than epochs. `outputs/<run>/config.json` records all
three; check them before a number goes into the table.

**Three traps in a QAL sweep** (`qal_loss` in `embodied_mae.py`; magnitudes
from `sm_scene_s1`, checked 2026-10-02):
- **`qal_use_squared` changes the PC loss scale, so it cannot be swept at a
  fixed `pc_loss_weight`.** At epoch 200 the PC term is 0.033 of a 0.20 total
  (~16 %). Measured on `sm_scene_s1`'s own predictions (job 16702293), QAL at
  threshold 0.02 squared is 11x smaller than threshold 0.01 Euclidean at epoch
  20 and 17x smaller at epoch 200, so a squared arm at weight 1.0 gives the PC
  head little gradient. Scale the weight so PC keeps the same share of the loss
  (the `*_q02sq` arms use 15), or the comparison measures the weight.
- **`qal_threshold` and `qal_alpha` overlap.** The weight on a near-perfect
  point is `sigmoid(-alpha * threshold)`, so cells with equal alpha x threshold
  behave alike near zero ((100, 0.02) ~ (200, 0.01)), and at alpha x threshold
  >= 6 points closer than about the threshold get essentially no gradient, which
  works against F1@0.01. A higher threshold also shrinks the PC loss, which acts
  like a smaller `pc_loss_weight`.
- **The metrics disagree on what "best" means.** `val_pc_chamfer` uses
  *squared* distances while F1 / precision / recall use Euclidean ones, so
  ranking by chamfer favours `use_squared: true`. Select on val with both
  reported, then confirm the winner on test.

That leaves **E5, E8, E10**, the rest of **E6/E7** (seed 2, equal-length
baselines, Alloy's occluded renders, the full QAL grid), the E9 sweep over the
E3/E4 arms, and the mask-ratio control arm above as the work owned here.


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

**Global batch is `batch_size × world_size`.** `batch_size` in the YAML is per-GPU, so an ablation must adjust it to the GPU count to keep the global batch constant — a global-batch difference between arms confounds the comparison. The E2 launchers use 16×2 on the 2-GPU Blackwell nodes and 8×4 on the 4-GPU A100 nodes, both reaching 32 — but **every reference run actually trained 16×2** (all E2/E3/E4 arms and `maize_4m`, per their `config.json`). Keep new arms at 16×2 too: `PointCloudEmbed`'s BatchNorm1d is not synced across ranks, so the per-GPU batch sets its statistics even when the global batch matches.

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

**The view folders' `*_spline.yml` were rewritten in place on 2026-09-30** by Alloy's
`alloy/shorgum_data/add_leaf_width.py` (per-leaf `width` added, both waviness keys
removed, all 150 000 copies). `load_spline_params` needs the waviness keys, and its
leaf filter then dropped **every** leaf silently: `Sorghum_0` loaded 1 valid token of
25. The pristine originals are untouched in `alloy/shorgum_data/SorghumData/`.
`SorghumDataset4M` now detects the rewrite at init and reads the originals
(`data.spline_root` / `$SORGHUM_SPLINE_ROOT` override; default `<data_root>/../../SorghumData`),
reproducing the tensor every earlier run trained on, and `load_spline_params` raises
on a rewritten file rather than loading an empty param stream. Copies made before
the rewrite (e.g. a Delta transfer) still load from their folders unchanged.

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
| `slurm/e2_arm_blackwell.sbatch` | `slurm/train_maize.sbatch` | ✅ |
| `slurm/scale_arm_blackwell.sbatch` | `slurm/scale_arm_maize.sbatch` | ✅ |
| `configs/config_e3_*` / `config_e4_*` | `configs/config_maize_e3_*` / `config_maize_e4_*` | ✅ |
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

**Maize runs to 1000 epochs, into a SEPARATE directory.**
`slurm/train_maize_1000.sbatch` **reads** `outputs/maize_4m` and **writes**
`outputs/maize_4m_1000ep` (submitted `--dependency=afterok:<600 job>`), so:

- `outputs/maize_4m` is **frozen at epoch 600 = 197,400 steps** — E3's full-data
  point, E4's base-model point, and the only maize artefact comparable to
  sorghum's `e2_pcrgbdt` / `e3_*` / `e4_*`. Use
  `checkpoints/checkpoint_epoch_600.pth`, not `best_model.pth`.
- `outputs/maize_4m_1000ep` is 329,000 steps **with a cosine restart in it**, and
  is comparable to none of the above.

An earlier version resumed in place and merely copied the 600-epoch artefacts to
`*_600ep.*`. That left `outputs/maize_4m/best_model.pth` and `config.json`
describing a 1000-epoch run, so plotting E4 off the directory the configs name
would have put a 329,000-step point in the middle of a curve of 197,400-step
points and inverted the model-scaling result with no error anywhere. Writing
elsewhere removes the trap rather than documenting it. **Slurm spools the batch
script at submit time**, so editing an `.sbatch` does not change an
already-queued job — cancel and resubmit (this is why job 16571996 was replaced
by 16576022).

**Why it is a warm restart.** `lr_lambda` is a cosine over `args.epochs` and
`LambdaLR.state_dict()` stores `None` for a plain-function lambda, so resuming
with `--epochs 1000` rebuilds the schedule over 1000 and restores only
`last_epoch=600`. The LR **jumps 47,101x**, 1.18e-09 at epoch 599 to 5.56e-05 at
600, then decays to 0 by 1000 — a second cosine cycle (SGDR-style), chosen
deliberately over a clean 1000-epoch run (~24 h vs ~9.6 h). Report it as
"600 epochs + 400 with a cosine restart", never as a 1000-epoch cosine.
`checkpoint_epoch_600.pth` carries `wandb_run_id` (verified: `yer9kdrv`), so the
continuation logs into the SAME W&B run — one continuous curve.

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
much worse). The first maize launch, job **16447748**, started **2026-09-23 06:49** on `nova26-gpu-2` (2x RTX PRO 6000, `det_cu128`) with both split guards passing 105000/105000 and 22500/22500, then died after 4 min in the epoch-1 viz crash described above. Its resubmission, job **16551951** (`outputs/maize_4m`), ran 08:49 to **23:00 on 2026-09-23** and finished 600/600 cleanly (14 h 11 m). It runs at **4.18 it/s = ~79 s/epoch**, so 600 epochs is ~13 h (~15 h with the 24 val passes) — well inside the 2-day wall, and against sorghum's ~3.1 min/epoch on *eight* GPUs. Maize really is far less dataloader-bound: 10x smaller folders and XML params that parse ~300x faster than sorghum's YAML. **329 steps/epoch x 600 = 197,400**, matching E2/E3/E4 exactly.

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

1. **`data.data_root` in the YAML you run.** Every config hardcodes the Nova absolute path above (maize: `/work/mech-ai-scratch/alloy/Maize`). The probes and E9 scripts (`eval/linear_probe*.py`, `eval/latent_analysis*.py`) default to the same Nova paths. Pass `--data-root` rather than editing them.
2. **`slurm/*.sbatch`: the `#SBATCH` headers *and* the body.** The headers (`--partition`, `--account`, `--gres`, `--cpus-per-task`, `--mem`) encode Nova's partitions (`nova`, `scavenger`) and accounts (`mech-ai`, `mech-ai-scavenger`) and mean nothing elsewhere. The bodies also hardcode Nova paths: `cd /work/mech-ai-scratch/alloy/embodiedmae`, `source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh`, the two conda env paths in the `nvidia-smi` switch, and, in `scale_arm_maize.sbatch` / `train_maize.sbatch`, the split-guard `ls /work/mech-ai-scratch/alloy/Maize/<split>`. If that last path is wrong, the guard counts 0 plants and the job exits 3 before training. On a non-SLURM box, ignore `slurm/` entirely and use the `torchrun` line above.
3. **The conda env.** `environment.yml` rebuilds `det`; it pins a CUDA 12.4 PyTorch, so a different GPU generation may need a different build (see the sm_120 note under Environment).

**Moving the data is the expensive part.** At ~14.3 MB per sample folder × 150 000 folders the split is **~2.1 TB**, but the four files training actually reads total ~2.6 MB per sample, so a filtered copy is **~395 GB** — under a fifth of the naive transfer (measured 2026-09-24 on sampled folders; an earlier 5.0 MB / 1.8 MB / 265 GB here was wrong). `slurm/delta/transfer_from_nova.sh` and `slurm/delta/MANIFEST.md` do this for Delta, checkpoints included. Copy with an include-filter rather than syncing the tree:

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
- `config.json` — snapshot of the effective args. **Check this to confirm what a run actually used**, especially `batch_size × world_size`, `max_plants` and `model_size`. `active_modalities: null` means **all four** streams (a reduced arm records its list, e.g. `['pc', 'rgb']`), and the PC loss is recorded as `pc_loss_name` — there is no `loss_name` key. (An earlier note here said both fields were untrustworthy nulls; checked against every E2/E3/E4 `config.json` on 2026-09-24, that was wrong.) Runs from 2026-09-24 on also record `text_mask_ratio` (`null` = text in the shared budget).

`outputs/` is gitignored apart from a small whitelist in `.gitignore`. Wandb logging is on by default (`use_wandb: true`, project `embodied-mae-sorghum`); project and run names differ between runs — check the YAML, not the script defaults.

## Things that look like dead code but aren't

- `outputs_sorghum_*/` directories at the repo root are old run outputs kept for reference; the canonical output root is `./outputs/`.
- `tools/process_depth_bg.py`, `tools/process_mask.py`, `tools/validate_sorghum_data.py`, `sweeps/vis.py`, `tools/check_structure.py` are one-off data-prep / diagnostic scripts, not part of any pipeline.
- `sweeps/vis_pc_masking.py` is a standalone tool for visualising the FPS + Dirichlet masking on a single point cloud.
- `tools/visualize_sorghum_pointclouds.py` renders multi-view PC galleries from raw `.ply` files; it doesn't touch the model.
- `eval/analyze_e2.py` reads the E2 arm output dirs and builds the modality-value-add comparison.
