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

**2026-10-02: the sorghum programme restarted on DATA VERSION 2** (rgb_nobg.png + leaf width;
see "Data version 2" at the top of the HANDOFF section). Every sorghum result in the table below
is data version 1 and is superseded as its `_d2` twin finishes. Maize is unaffected.

**Status as of 2026-09-30** (live job state: see the HANDOFF section below). Run state goes stale fast — the table records which
experiments have been *built*, which is durable. For live progress use
`squeue -u $USER`, `outputs/<run>/training_history.json`, and
`outputs/<run>/config.json` (which records what a run actually used).

| | experiment | state | runs |
|---|---|---|---|
| E1 | headline pretrain | **done** | `4m_pretrain_15k_v2_depthfix_qal`, 1000/1000 ep, global batch 256 |
| E2 | modality value-add | **done, both species** | sorghum: all four arms 600/600, results in `reports/RESULTS_DECK_2026-09-20.md`; control arms `e2_pcrgbdt_tg` / `tg40` / `tg100` done and probed (tg100 = text as target only, probe 16686383, 2026-10-01: still below PC+RGB+depth on every learnable target). maize: all four arms ✅ 600/600 and probed (2026-09-30, `reports/probe_pm_e2_600_16669530.csv`): PC+RGB(+D) beat `maize_4m` on 10/11 — see the handoff section below. Partial-cloud (1 vs 3 cameras) evaluation of the sorghum arms ✅ (2026-09-30) |
| E3 | data scaling | **done, both species** | sorghum: all three arms, `e3_1k` finished 2026-09-21 11:22 at 6169/6169. maize (Nova, jobs 16576132-4, `sbatch --job-name=maize_e3_1k slurm/scale_arm_maize.sbatch maize_e3_1k`): `maize_e3_1k` ✅ 6169/6169 (2026-09-24), `maize_e3_10k` ✅ 631/631 (2026-09-24), `maize_e3_3k` ✅ 2100/2100 (2026-09-25; last checkpoint 2024). Full-data point is `maize_4m` epoch 600. Probed + E9: `reports/probe_p{r,m}_{6168,2024,624}_*` |
| E4 | model scaling | **done, both species** | sorghum: `e4_small` ✅ `e4_large` ✅ (600/600, finished 2026-09-21 21:41). maize (Nova, jobs 16576135-6, same launcher): `maize_e4_small` ✅ 600/600 (2026-09-24), `maize_e4_large` ✅ 600/600 (2026-09-25). The base point is `maize_4m` epoch 600 ✅ (finished 2026-09-23 23:00). Probed + E9: `reports/probe_p{r,m}_600*_*` |
| E5 | view regime | **built 2026-10-15, three arms training** | `data.train_views` (SorghumDataset4M) restricts each plant's per-epoch draw to a band of the ten-view elevation ladder; arms `e5_topdown_d2` (views 0-1), `e5_horizon_d2` (4-5, the tractor-height proxy), `e5_single_d2` (view 0), reference = `e2_pcrgbdt_d2` (all ten). Same 197,400 steps. No separately rendered tractor-pass camera set exists (plan 6.2). **Maize: built 2026-10-15, not launched** (`configs/config_maize_e5_{topdown,horizon,single}_d2.yaml`, reference `maize_4m_d2`); maize view indices carry no elevation, so `MaizeDataset4M.train_views` ranks each plant's views by camera elevation (`top:2` / `horizon:2` / `[0]`) from `data_split/maize_view_elevation.csv.gz` |
| E6 | masking / noise | **owned elsewhere** | a collaborator is running this — not work for this repo |
| E7 | loss study | **owned elsewhere** | same; `model.loss_name` (chamfer / qal_loss) is the switch they need |
| E8 | baselines | **done (supervised row sorghum only)** | `eval/baseline_probe.py` + `eval/baselines/*`: EmbodiedMAE-B, MultiMAE-B, DINOv2 ViT-B/14, Point-MAE, Point-MAE + camera pose, random init, both species, `reports/probe_e8_*`. Supervised-from-scratch row (`train_supervised_4m.py`, `outputs/e8_supervised`) ✅ 600/600 (2026-09-30), scored in `reports/probe_e8_score_16676974_*`, paired deltas in `reports/e8_supervised_delta_16676974.csv` (`eval/probe_pair_bootstrap.py`). Maize camera-pose rows (2026-10-01): gravity-only (`*_gravity`, 16686506) and full pose (`*_upright`, 16684794), CIs in `reports/maize_{gravity,upright}_delta_*.csv`; arm pretrained on levelled clouds `maize_e2_pcrgbd_levelled` training (chain 16687759-64) |
| E9 | latent analysis | **built; E2/E3/E4 arms done** | `eval/latent_analysis.py` / `_maize.py`. Sorghum E2/E3/E4 and maize E3/E4 arms on train/val/test (`reports/e9_pr_*`, `reports/e9_pm_*`); sorghum control arms on val (`reports/e9_pr_tg_*`). Maize E2 arms done (`reports/e9_pm_e2_600_16669530/`); `e2_pcrgbdt_tg100` done (`reports/e9_pr_tg100_600_16686383/`) |
| E10 | real-data OOD | **partial** | `OOD_EVAL_rgb2pc.md` and the `eval_rgb2pc_*.py` scripts |

## HANDOFF — live state as of 2026-10-02 10:00 (read this first after a restart)

### MAIZE on DELTA, data version 2 — state as of 2026-10-08 12:00 (code 5ee93bd)

Maize now runs on NCSA Delta (`/work/hdd/bepk/adas14/embodiedmae`, data `/work/hdd/bepk/adas14/Maize`,
launchers in `slurm/delta/`, env notes in `slurm/delta/delta_env.sh`). **Delta pushes to GitHub
over SSH since 2026-10-14**: a write-enabled deploy key (`~/.ssh/id_ed25519_github_embodiedmae`,
`~/.ssh/config` Host github.com) and `origin` = `git@github.com:alloydas/embodiedmae.git`. There is
still no `gh` there.
Maize data version 2 (commit 4dcfe3e, 2026-10-06): `rgba.png` composited over black + the trimmed
spline XML (`param_schema: v2`, `embodied_mae_4m_maize.PLANT_FIELDS_V2 / LEAF_FIELDS_V2`); every
`_d2` config sets `data.rgb_file` / `data.param_schema`, and the trainer refuses to mix versions
in one run dir.

**Done (all on gpuA100x4, 2x A100-40GB / 32 CPU / 100G, `slurm/delta/train_maize_arm.sbatch`):**

| run | epochs | val loss | val chamfer | job | wall |
|---|---|---|---|---|---|
| `maize_4m_d2` (= E2 +params, E3 full, E4 base) | 600/600 | 0.1103 | 0.00099 | 22706793 | 34 h |
| `maize_e2_pc_d2` / `pcrgb_d2` / `pcrgbd_d2` | 600/600 | 0.0449 / 0.1226 / 0.0810 | 0.00263 / 0.00161 / 0.00121 | 22706957-9 | 21-29 h |
| `maize_e3_1k_d2` / `3k_d2` / `10k_d2` | 6169 / 2100 / 631 | 0.2505 / 0.1723 / 0.1081 | 0.00354 / 0.00146 / 0.00087 | 22706960-2 | 30-33 h |

Last checkpoints 600 / 600 / 600 / 600 / 6168 / 2024 / 624. `test_freq: 0` in every config, so no
test numbers in the histories: the held-out split is scored by the probe. Those seven cost ~454 of
the allocation's GPU-hours (gpuA100x4 bills ~2.25 h per wall-hour at this shape); **432 remain**,
so everything below went to `gpuA100x4-preempt` (half the rate; `--requeue` + the launchers'
auto-resume cover preemption, a second `afterany` chunk covers the 48 h TIMEOUT).

**Queued 2026-10-08 (user: "check from github for sorghum what experiments are there, start
them for maize"):**

| what | jobs | notes |
|---|---|---|
| probes + E9 of the seven runs | 22751615 (`pm_600`: 4m, e2_pc, e2_pcrgb, e2_pcrgbd), 22751616 (`pm_6168`), 22751618 (`pm_2024`), 22751619 (`pm_624`) | gpuA40x4, `submit.sh probe_maize`; `eval/linear_probe_maize.py` now reads each run's `rgb_file` / `param_schema` from its config.json |
| `maize_e4_small_d2` | 22751638 -> 22751639 | `configs/config_maize_e4_small_d2.yaml` |
| `maize_4m_tg_d2` / `tg40_d2` / `tg100_d2` | 22751640-1 / 22751642-3 / 22751644-5 | the maize twins of sorghum's `e2_pcrgbdt_tg*` controls: `maize_4m_d2` with `model.text_mask_ratio` 0.80 / 0.40 / 1.0. `train_maize_4m.py` gained the key (ported from the sorghum trainer; `None` is bit-identical to every earlier run; recorded in config.json and checkpoints, resume refused on a mismatch) |
| `maize_distill_all_d2` | smoke 22751636 (1 GPU, `--max_steps 40`) -> full 22751637 (`afterok`, `--kill-on-invalid-dep=yes`) | `slurm/delta/distill_maize.sbatch`, `configs/config_maize_distill_all_d2.yaml`: teacher = student init = `maize_4m_d2` epoch 600, the v1 recipe otherwise (teacher sees all four streams at `teacher_source_mask_ratio` 0.5, 100 ep, global batch 128 = 16 x 2 x accum 4). The trainer now pins `rgb_file` / `param_schema` against the teacher's config.json and a resumed checkpoint. Self-requeues 10 min before the 48 h wall (USR1 trap) |

| `maize_e8_supervised_d2` | 22752738 -> 22752739 | `train_supervised_maize_4m.py` (port of the sorghum E8 supervised trainer; `configs/config_maize_e8_supervised_d2.yaml`), `slurm/delta/e8_supervised_maize.sbatch`. After epoch 600 run it again with `SCORE=1` on 1 GPU / 16 CPU (the line in its header): `--score_head` + the cls and mean probes |

**`maize_e4_large_d2`** launched last: 22752834 -> 22752835 -> 22752836 (three 48 h chunks; ~2.5x the base
arm's 34 h). Memory test 22752773 on an A100-40GB at batch 16, mask 0.80: base 22.4 GiB alloc /
26.2 reserved (matches the known figure), large 26.9 / 31.3, so the 40 GB cards fit it.
The maize E8 supervised row is queued (table above); no maize E1 (1000-epoch, batch-256, all-views pretrain)
exists in either data version and none is planned. Then: E8 baselines on maize d2
(`eval/baseline_probe.py`, needs `rgba.png` support for maize), `#maize` page sections.


**2026-10-13 09:50 (status check).** The four probe/E9 jobs finished 2026-10-09
(`reports/probe_pm_{600,6168,2024,624}_227516*.csv`, `reports/e9_pm_*_227516*/`). On d2 the
parameter stream no longer hurts maize: `maize_4m_d2` vs `maize_e2_pcrgbd_d2` val R² height
0.635 vs 0.633, leaf angle 0.919 vs 0.926, stem radius 0.743 vs 0.760, curl 0.304 vs 0.236 (v1 had
the +params arm losing 10 of 11). PC-only is far behind (height 0.171). E3 d2 is not monotone:
height 1k / 3k / 10k / full = 0.393 / 0.524 / 0.427 / 0.635; stem radius 0.593 / 0.624 / 0.717 /
0.743 (unexplained; paired CIs not yet run). **None of the six preempt training jobs
(22751638/40/42/44, 22752738, 22752834 and their `afterany` chunks) has started in five days**:
priority 2765, 177 jobs pending on `gpuA100x4-preempt`, 0 running there, every node PLANNED or
allocated. The distillation smoke 22751636 was preempted twice, then OOMed at its epoch-20
validation (the 12 GiB chamfer block against 13.6 GiB reserved-but-unallocated), which killed
the gated full job 22751637. Fix: `run_val` in `train_maize_4m_distill.py` now calls
`torch.cuda.empty_cache()` first, and the Delta launcher sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Resubmitted: smoke **22870323** (1 GPU, 40 min,
resumes `outputs/_smoke_maize_distill_d2` at epoch 18, so it reaches the epoch-20 val in minutes;
it never exits 0 by itself because `--max_steps` only shortens epochs, so it is a test, not a
gate) and the full run **22870325** (no dependency; `outputs/maize_distill_all_d2`). Untracked
`eval/pc_fscore_*.py` + `slurm/delta/pc_fscore_maize.sbatch` (2026-10-07) were never run or committed.

**2026-10-13 10:40: MAIZE E1 launched (user: "Start the E1", "submit it").** Job **22870827**
(`gpuA100x4-preempt`, 4x A100-40GB, 64 CPU, 240G, 48 h, self-requeueing via USR1, `--requeue`):
`slurm/delta/pretrain_maize_e1.sbatch` -> `configs/config_maize_4m_pretrain_15k_d2.yaml` ->
`outputs/maize_4m_pretrain_15k_d2`. The maize twin of the sorghum E1 d2: 1000 epochs over all
105,000 views (`view_sampling: false`), global batch 256 = 16/GPU x 4 GPUs x accum 4 (the 40 GB
cards cannot take sorghum's 32/GPU, so per-rank BatchNorm sees 16), lr 6e-4, warmup 20, mask 0.80,
qal_loss, save_freq 5 pruned to the newest 3 + every 100th, 411 steps/epoch = 411,000 steps.
`train_maize_4m.py` gained `training.accum_steps` / `--accum_steps` and `checkpointing.keep_last`
/ `keep_every` (verbatim ports from the sorghum trainer; default = the old code path).
**Budget warning:** at ~100-130 items/s (loader-bound) an epoch is ~14-17 min, so ~240-280 h of
wall on 4 GPUs; the preempt partition bills ~2.25 SU per wall-hour at this shape, i.e. ~550-630
SU against a balance of 431 SU that the six other pending jobs also draw on. E1 cannot finish on
the current allocation; it will stop (job fails on a zero balance) around epoch 600-700 unless
the allocation is topped up. The checkpoints up to that point remain usable.

**2026-10-14 10:30: E1 and the distillation switched to QAL + Sinkhorn (user: "in the yongyun
branch he used new loss we have to update E1 and distill with new loss").** Ported from
`origin/yongyun` (E7 winner there: QAL 0.01 + 0.5 x debiased Sinkhorn, geomloss, 1024-point
subsets, blur 0.02): `pc_sinkhorn.py` verbatim; `embodied_mae_4m.py` gained `pc_sinkhorn_weight /
_points / _blur` (0 = every earlier run, geomloss never imported) and the `chamfer_cdist` /
`sinkhorn` loss names; `train_maize_4m.py` and `train_maize_4m_distill.py` read `model.pc_sinkhorn_*`
and pass them to the student (the teacher's weight is forced to 0; the keys are pinned on resume
and recorded in config.json). `configs/config_maize_4m_pretrain_15k_d2.yaml` and
`config_maize_distill_all_d2.yaml` set 0.5 / 1024 / 0.02 with `pc_loss_weight` 1.0 (= the branch's
effective weight). **geomloss 0.3.1 was pip-installed into the `det` env on Delta** (not in
environment.yml). The pending jobs 22870827 (E1) / 22870325 (distill) / 22870323 (distill smoke)
read the trainer and config at start, so they run the new loss without resubmission; the old
`outputs/_smoke_maize_distill_d2` was deleted so the smoke starts fresh. GPU smokes on gpuA40x4
(the regular partition, starts in minutes-hours): **22897176** (E1 path: 1 epoch, 64 plants,
`outputs/_smoke_maize_e1_d2`) and **22897183** (distill, 25 min, never exits 0 by itself).
Results page for the maize d2 programme (private artifact, user asked 2026-10-14): https://claude.ai/artifact/PqNcyvGKEcznLdm5n9z8fY ("Maize D2 Results": E1-E10 matrix with blanks for unfinished arms, curves, probe, E9, reconstructions, queue). Rebuild from a new session with the extraction + template approach (reports/probe_pm_*_2275*.csv, e9_pm_*_2275*/, outputs/*/training_history.json) and republish with `url` set. Check both logs before trusting the queued runs. **2026-10-14 11:05 (user: "move E1 to gpuA100x4"): 22870827 moved in place to `gpuA100x4` (`scontrol update ... partition=gpuA100x4`); Reason went Priority -> Resources (it needs a whole 4-GPU node). Full rate there: ~2.25 SU/wall-hour, ~1100-1260 SU for 1000 epochs against 431 SU, so it runs out around epoch 350-400 unless the allocation is topped up; the eight preempt jobs stay where they were.**
**2026-10-15 09:30: E1 22870827 RUNNING** since 10-14 13:53 on gpua078 (~15 min/epoch, epoch 79 at 19.6 h, val chamfer 0.00131; burns ~4 SU/wall-hour: 431 -> 352 SU, so ~88 h / ~350 more epochs of balance; the 48 h wall hits near epoch 190 and the USR1 trap requeues it). The eight preempt jobs have not started since 10-08. Both GPU smokes passed (22897176 E1 path exit 0 in 8 min; 22897183 distill reached epoch 10 with the Sinkhorn loss, TIMEOUT by design). Maize d2 paired bootstraps are in `reports/maize_e2_delta_d2_22751615.csv`, `maize_e2_depth_delta_d2_22751615.csv`, `maize_e3_delta_d2_2275161x.csv`, `maize_e3_3k_vs_10k_delta_d2_2275161x.csv` (paper 9205e00): depth loses tassel droop / curl / twist and wins nothing; +params vs PC+RGB+D is a draw; the E3 dip at 10k is real (3k beats 10k on height, leaf length, leaf count; loses stem radius). **Run `eval/probe_pair_bootstrap.py` on the login node only one at a time with `LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=1`** (three concurrent runs with unbounded `n_jobs=-1` finished zero targets in 25 min).

**2026-10-15 15:00: SORGHUM-ONLY CODE PORTED TO MAIZE (user: "sync with github and rewrite all the codes which are not available for maize").** Nothing launched; every piece CPU-tested on Delta.
- **Maize clouds are single-camera captures.** `tools/check_maize_cloud_views.py` (20 val plants): 2 % of a `pointcloud_cam.ply`'s points are hidden from the camera that made it, 25 % from another view's camera, and the depth image back-projects to within 1 mm of the cloud. Sorghum's `_nc_cam.ply` is the complete plant; every maize model trained on what one camera sees, and a plant's ten views are ten different clouds. The paper's "pretrained on complete clouds" is sorghum-only (noted as a \todo in `paper/discussion.tex`).
- **Maize camera poses carry no view-index structure.** `tools/maize_view_elevation.py` -> `data_split/maize_view_elevation.csv.gz` (150,000 rows: split, plant, view, sin_elev, elev_deg, az_deg, dist; elevation = camera above `plantCenter`, world +Y up). Each view index spans -85..+81 deg with a random azimuth, so sorghum's index-band E5 is meaningless here.
- **E5 maize**: `MaizeDataset4M(train_views=...)` accepts a list of indices (sorghum form) or `'top:k'` / `'horizon:k'` / `'bottom:k'` (per plant, the k views ranked by sin elevation from the table; every plant keeps exactly k views, so plants / steps / schedule are `maize_4m_d2`'s). `train_maize_4m.py`: `data.train_views` / `--train_views`, canonical form in config.json, `check_resume_train_views` refuses a run dir of another regime. Configs `config_maize_e5_topdown_d2.yaml` (`top:2`), `_horizon_d2` (`horizon:2`), `_single_d2` (`[0]`) = `config_maize_4m_d2.yaml` + the key. Realised train elevations (val split, CPU test 2026-10-15): `top:2` median +51 deg (p10 +26, p90 +73), `horizon:2` median 0 deg (p10 -12, p90 +13), view 0 median +1 deg (p10 -54, p90 +52; the probe view is a random camera in every arm). Tested: draws stay inside the chosen views over 40 epochs, ranking matches the table for 2250/2250 plants, `check_resume_train_views` refuses a mismatch. Launch: `sbatch --account=bepk-delta-gpu --partition=gpuA100x4-preempt --job-name=tr_maize_e5_topdown_d2 --cpus-per-task=32 --mem=100G --export=ALL,DELTA_ENV_FILE=$PWD/slurm/delta/delta_env.sh slurm/delta/train_maize_arm.sbatch maize_e5_topdown_d2` (x3; ~34 h each at the base shape, ~75 SU each on preempt). Then `bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_600.pth maize_e5_topdown_d2 maize_e5_horizon_d2 maize_e5_single_d2 maize_4m_d2` (submit.sh has no matched cut on record for `maize_e5_*`: it probes 600 as asked) and `eval/probe_pair_bootstrap.py --species maize --a maize_e5_*_d2 --b maize_4m_d2`.
- **Partial clouds, maize** (`eval/pc_view_eval_maize.py`, `eval/pc_view_bootstrap.py --species maize`, `slurm/delta/pc_view_maize.sbatch`): no masks; conditions are unions of the per-view clouds moved into camera 00's frame: `1view` (what the model trained on), `3view` (views 00+01+02), `full` (all ten = the reconstruction target). Transfer probe is fitted on `1view` (`ANCHOR`), deltas are `delta_vs_1view`. Coverage of the ten-camera cloud within 5 mm: 1view ~0.77, 3view ~0.97 (6-plant smoke). Launch: `sbatch --account=bepk-delta-gpu --partition=gpuA40x4 --job-name=pcview_maize_d2 --export=ALL,DELTA_ENV_FILE=$PWD/slurm/delta/delta_env.sh slurm/delta/pc_view_maize.sbatch maize_e2_pc_d2 maize_e2_pcrgb_d2 maize_e2_pcrgbd_d2 maize_4m_d2` (one data version per job; writes `reports/pcview_maize_d2_<job>_{coverage,recon,probe,bootstrap}.csv`).
- **Per-view generation, maize** (`eval/eval_views_one_plant_maize.py`): rows sorted by camera elevation (not index), chamfer per view against that view's own cloud, leaf fields follow the config's `param_schema`. Works with the pretrain checkpoint as the "before" (`--checkpoint outputs/maize_4m_d2/checkpoints/checkpoint_epoch_600.pth --label 'before distillation'`); the "after" needs `outputs/maize_distill_all_d2` (pending 22870325).
- **E8 baselines on maize d2**: `eval/baseline_probe.py --species maize --rgb-file rgba.png` (RGB adapters' caches tagged `__rgba`; PC-only adapters unchanged), paired with `probe_pair_bootstrap.py --rgb-tag rgba`. Launch on Delta with the Nova launcher's body adapted by hand (`slurm/baseline_probe.sbatch` is Nova-only) -- not yet written.
- `slurm/delta/probe_maize.sbatch` takes `FEATURE=mean` for the mean-pooled rows (as sorghum's `linear_probe.sbatch`).
- Already maize-capable, unchanged: `eval/eval_single_source_pc.py --species maize`, `eval/latent_analysis_maize.py`, `eval/probe_pair_bootstrap.py`, `train_supervised_maize_4m.py` + `slurm/delta/e8_supervised_maize.sbatch` (`SCORE=1`), `figures/make_paper_figures.py`. Sorghum-only by nature (real scans / sorghum distillation reports): `eval_rgb2pc_*`, `figures/plot_view_elevation_effect.py`, `eval/analyze_e2.py`.
Structured / mixed masking (E6) was NOT ported:
it is for occluded scenes, not the clean pretrain. The sorghum E1 (Nova) is untouched from here.

### Data version 2 — the sorghum restart (2026-10-02; code b708117)

The user changed the sorghum dataset and asked for every experiment to restart from the
beginning on it ("start the experiments from the begining"), on scavenger.
- **rgb_nobg.png** in every view folder (`shorgum_data/remove_background.py`, 2026-10-01): RGBA,
  straight alpha, background transparent, edges un-blended from the renderer's grey.
  `SorghumDataset.load_rgb` composites it over BLACK with its alpha (default
  `sorghum_dataset.RGB_FILE = 'rgb_nobg.png'`). Runs record `rgb_file` in config.json; the probe
  reads each run's own file (no key = a data-v1 run = `rgb.png`).
- **Spline files rewritten** (`shorgum_data/add_leaf_width.py`, 2026-09-30): waviness_frequency and
  waviness_period_start removed, per-leaf blade `width` (m, mesh-measured, 0.046-0.148) added. Leaf
  token layout is now `[sp, ln, ra, ba, width, 0, 0, 0, 0]` (`embodied_mae_4m.LEAF_FIELDS`), and the
  length scale is 1.25 (was 1.0, which clipped 0.54 % of leaves). **The old code would not have
  crashed on the new files: its leaf filter required the waviness keys, so it would have dropped
  EVERY leaf token.** `load_spline_params` now raises if a plant's leaves all fail the check.
- Both changes were checked in all 150,000 view folders (stat sweep), the leaf widths for all
  15,000 plants. Plant 0 has two geometry-only leaves (no params); they are not tokens or targets.
- Probe targets gained `leaf_width_mean` / `leaf_width_max` (`data_split/leaf_width_targets.csv`,
  built by `data_split/make_leaf_width_targets.py`). Between-plant spread is small (mean
  0.0985 +- 0.0023 m) and nearly uncorrelated with every other target (|r| < 0.2).
- E8 harness caches of RGB adapters are named `...__rgb_nobg.npz`; `probe_pair_bootstrap.py
  --rgb-tag rgb_nobg` reads them. Data-v1 caches keep their names and stay valid.
- The trainer refuses to resume a run dir of the other data version, frees cached GPU memory
  before val/test, and can prune checkpoints (`checkpointing.keep_last` / `keep_every`; E1 only).

**Runs (all `_d2`; on scavenger in 4 h chunks chained `afterany`, resuming from the newest
checkpoint, except three E2 arms now on nova, see below):** 2x A100 / 32 CPU / 160G / `--num_workers 14` unless noted.

| run | launcher | chain |
|---|---|---|
| `e2_pc_d2`, `e2_pcrgb_d2`, `e2_pcrgbd_d2`, `e2_pcrgbdt_d2` | `e2_arm_blackwell.sbatch <arm>_d2` | 16690567-79, 16690580-92, 16690593-605, 16690606-18 (pc, pcrgbd, pcrgbdt: nova jobs spliced in front, below) |
| `e2_pcrgbdt_tg_d2`, `_tg40_d2`, `_tg100_d2` | same | 13 chunks each, from 16695901 / 16695914 / 16695927 |
| `e3_1k_d2`, `e3_3k_d2`, `e3_10k_d2` | `scale_arm_blackwell.sbatch <slug>` | 14 chunks each, from 16695940 / 16695954 / 16695968 |
| `e4_small_d2` / `e4_large_d2` (2x **H200**) | same | 16695982-94 / 16695995-6010 |
| `e8_supervised_d2` | `E8_CONFIG=configs/config_e8_supervised_d2.yaml e8_supervised.sbatch` | 16696011-23 |
| **E1** `4m_pretrain_15k_d2` | `slurm/pretrain_15k_d2.sbatch` (self-requeuing) | **16696027**: 8x A100 80 GB, 90 CPU, 440G, `NUM_WORKERS=10` |

- First E2 chunks (2026-10-02 00:20-08:20) did 55-90 epochs per 4 h; a chunk loses up to
  save_freq epochs at its end, so ~50 effective. Extend a chain with `--dependency=afterany:<last id>`.
- **E2 moved to the Blackwell nodes through nova (2026-10-02 10:15; user: "do e2 like you are doing
  previously").** The v1 arms each ran as ONE scavenger job on 2x RTX PRO 6000 / 80 CPU / 320G /
  38 workers per rank (12-31 h per arm). Scavenger cannot get those nodes now (nova CPU jobs on both),
  so three arms run there via nova (mech-ai cap: 6 of the 17 GPUs), each splicing nova jobs IN FRONT
  of its scavenger chain, which stays as the fallback (it resumes at 600 and exits once training is done):

  | arm | nova jobs, in order | then scavenger from | fallback guard |
  |---|---|---|---|
  | `e2_pc_d2` | 16696145 (4 h, 10:15-14:15) -> **16696287** | 16690568 | 16696288 |
  | `e2_pcrgbdt_d2` | 16696146 (11 h, 10:15-21:15) -> **16696289** | 16690606 | 16696290 |
  | `e2_pcrgbd_d2` | (scavenger 16690594 until ~14:06) -> **16696291** | 16690595 | 16696292 |
  | `e2_pcrgb_d2` | none: stays on scavenger (cap full); move it when `pc` finishes | 16690581 | — |

  The bold jobs are `--time=2-00:00:00 --time-min=04:00:00 --signal=B:USR1@600 --mem=160G` with the
  launcher's new opt-in USR1 trap: backfill cuts the limit to what fits, and the job requeues itself
  10 min before the wall until epoch 600. A requeue keeps the `afterany` dependent waiting (tested,
  jobs 16696237/8). A 2-day request never got a start time (both nodes PLANNED for another user's
  12 h CPU jobs); the 4 h and 11 h ones were placed in 5 min. Each guard is `nova_fallback.sbatch`
  with GRACE 1800: if its nova job is still pending 30 min after the predecessor ends, it moves the
  nova job behind the chain's tail and lets the scavenger chain run. **A requeued nova job that pends
  for hours is NOT guarded**: check with `squeue -n e2_<arm>_d2`; to fall back by hand, hold the nova
  job, then clear its scavenger successor's dependency.
- **Why a fresh Blackwell job starts slow: the page cache, not the GPU.** On 2026-10-02 a cold NFS
  read of one item (the four files, 2.6 MB) took 1.1-1.3 s on nova26 and the login node (~2 MB/s per
  stream), while the A100 nodes that had run E2/E3 for hours returned even random items in 2 ms
  (cached). The first Blackwell epochs took 8:28, then fell every epoch to 2:22 by epoch 64, as the
  node cached the views (both arms read the same files: same `view_seed`). Steady state is set by
  the GPU: 5.7 it/s for `pc` (~1 min/epoch), 4.0 it/s for `pcrgbdt` (~1.4 min); the A100s do 2:08-2:44.
  The node's cache outlives the job, so a follow-on job on the same node starts warm.
- **E1 is the long pole.** It reads all 105,000 views per epoch and is loader-bound (~1.4-2.4
  items/s per worker on the A100 nodes, four NFS files per item). On a 96-CPU A100 node that is
  ~8-14 min/epoch, ~6-10 days of running. The user moved it from nova (8x RTX PRO 6000, 180 CPUs,
  ~3-5 days, but it needed 8 of the account's 17 GPUs) to scavenger. Both Blackwell nodes host nova
  CPU jobs, which scavenger cannot share.
- **E1 moved to 2 GPUs (2026-10-06, user: "do E1 with 2 GPUs instead of 8").** It had sat at
  epoch 21 since 10-04 (16696027, 8x A100 scavenger, pending). `train_sorghum_4m.py` gained
  `training.accum_steps` / `--accum_steps` (default 1 = the old code path; DDP `no_sync` on all but
  a group's last micro-batch; a trailing partial group still steps, so 411 steps/epoch as at 8 GPUs;
  checked equal to a full-batch step to 1e-9 under 2-process DDP). `slurm/pretrain_15k_d2.sbatch`
  now sets accum = 8 / NPROC: 2 x 32 x 4 = global batch 256, same per-GPU batch, lr, schedule, so
  the 8-GPU checkpoints resume as is. Job **16740746**: nova, 2x RTX PRO 6000, 128 CPU, 480G,
  `NUM_WORKERS=60`, `--save_freq 2`, 2-day limit with `--time-min=04:00:00` and the USR1 requeue.
  Expected ~12 min/epoch (GPU ceiling ~74 samples/s/GPU), ~8 days for the rest, if the loader
  keeps up. 16696027 (8x A100 scavenger, old spooled script, accum 1) now waits `afterany:16740746`
  as the fallback: it resumes at 1000 and exits if E1 finished.
- **E1 RESTARTED FROM SCRATCH WITH THE SINKHORN LOSS (2026-10-14, user: "in yongyun branch he
  added a new loss, update that loss and start training from the beginning for E1 sorghum").**
  Ported from `origin/yongyun` (commit 5f13ee7 there) into main: `pc_sinkhorn.py` and the
  `model.pc_sinkhorn_weight / _points / _blur` keys in `embodied_mae_4m.py` + `train_sorghum_4m.py`
  (weight 0 = every earlier run, bit-identical; also loss names `chamfer_cdist`, `sinkhorn`).
  NOT ported: the branch's structured / neighbour-occlusion masking, scene configs and evals.
  `geomloss` 0.3.1 was pip-installed into `det` and `det_cu128` (the term imports it lazily).
  New run **`4m_pretrain_15k_d2_sink`** = `configs/config_4m_pretrain15k_d2_sink.yaml`: the E1 d2
  recipe + Sinkhorn weight 0.5 / 1024 points / blur 0.02 (the branch's standard, pc_loss_weight 1,
  unsquared QAL). Job **16907077** (nova, 2x RTX PRO 6000, 128 CPU, 480G, `NUM_WORKERS=60`,
  `--save_freq 2`, USR1 requeue; launcher takes `E1_CONFIG=<yaml>`). It trains on the BRIGHTENED
  `rgb_nobg.png` throughout (the dataset was brightened in place by the user's job on 2026-10-13
  ~10:02, gain 1.85, PNG tag `brighten_gain`; the still-running `_d2` arms switched inputs
  mid-run at that moment, the finished arms never saw it). The old E1 `4m_pretrain_15k_d2`
  reached epoch 716/1000 (old loss, mixed brightness) and CONTINUES alongside the new run (user,
  2026-10-14: "keep d2 with old loss and start E1 with new loss"): job 16740746 (nova) was held
  briefly and released; the 8x A100 scavenger fallback 16696027 was CANCELLED at the user's request
  ("there is one 8gpu job in scavenger drop it"). Then (user: "try both jobs in scavenger") the two
  nova jobs 16740746 / 16907077 were cancelled and BOTH runs resubmitted on scavenger:
  **16907817** (`e1_pretrain_d2`, old loss, resumes at 716) and **16907818** (`e1_sink_d2`, Sinkhorn,
  from scratch), each 2x A100 80 GB, 40 CPU, 200G, `NUM_WORKERS=18`, 2-day limit with
  `--time-min 4h`, `--requeue` + the USR1 requeue trap, `--save_freq 2`. Expect ~25-30 min/epoch
  (36 loader workers; E1 reads all 105k views per epoch), about 2x slower than nova's 2x RTX PRO 6000.
  Then (user: "use rtx 6000 pro they are free"): sinfo showed 4 idle RTX PRO 6000 on each Blackwell
  node, but two probes (`--deadline=now+3min`, `hostname`) expired unstarted: scavenger 1-GPU
  (nova CPU + arabeh's 4-GPU jobs sit on both nodes; nova is OverSubscribe=NO) and nova 1-GPU
  (Priority: the idle GPUs are backfill-planned for others; mech-ai at 16/17). So both runs are
  QUEUED FOR THE RTX PRO 6000 ON NOVA, **16907868** (old loss) and **16907869** (Sinkhorn), 2x
  rtx_pro_6000 / 128 CPU / 480G / `NUM_WORKERS=60`, and the scavenger A100 jobs 16907817 / 16907818
  were re-pointed `afterany` behind them as fallbacks (they resume from the newest checkpoint if
  the nova job dies or times out, and exit at once if it finished). A nova job that pends for days
  blocks its fallback: to swap, hold the nova job and clear the scavenger job's dependency.
  **Old-loss E1 killed (user, 2026-10-14: "kill e1_pretrain_d2"):** 16907868 and 16907817 cancelled;
  `outputs/4m_pretrain_15k_d2` stays on disk at epoch 716 (old loss). Only the Sinkhorn run is live.
  **Sinkhorn E1 on scavenger only (user, 2026-10-14: "put it into scavenger"):** nova 16907869
  cancelled; **16907818** (scavenger, 2x A100 80 GB, 40 CPU, 200G, `NUM_WORKERS=18`, 2-day limit,
  `--time-min 4h`, requeue + USR1 trap, `--save_freq 2`) is the only E1 job, dependency cleared.
  The loss passed a 1-GPU A100 smoke (16907096: 0.48 s/step at batch 16, 22.4 GiB peak).
  **Either GPU type (user: "in scavenger rtx 6000 pro are free right why are you not requesting?"):**
  three scavenger probes for 1 RTX PRO 6000 (2-min `hostname`, `--deadline`) expired unstarted on
  2026-10-14 (nova CPU jobs + arabeh's 4-GPU nova jobs occupy both Blackwell nodes; nova is
  OverSubscribe=NO, so a tier-0 scavenger job cannot co-locate). 16907818 was therefore updated in
  place to `TresPerNode=gres/gpu:2` + `Features=(a100&NVLINK)|rtxpro6000`: it takes 2 RTX PRO 6000
  as soon as scavenger can place them, or 2 A100 80 GB (NVLINK excludes the 40 GB `a100-pcie`
  nodes, which cannot hold batch 32). Its env has `NUM_WORKERS=18` / 40 CPU either way.
  The maize trainer was not touched (its model subclasses `EmbodiedMAE4M`, default weight 0).
- **`e2_pc_d2` moved to the front of its chain (2026-10-07 09:10, user: "move e2_pc to nova").**
  Its scavenger chunks kept being preempted within minutes (stuck at 257 since 10-06). Nova job
  16696287 (2x RTX PRO 6000, 80 CPU, 2-day limit, `--time-min` 4 h, USR1 requeue) had its
  dependency cleared; the next scavenger chunk 16690568 now waits `afterany:16696287`. No guard job
  (the user does not want watchers): if 16696287 pends for long on `AssocGrpGRES`, fall back by
  hand by holding it and clearing 16690568's dependency.
- **DATA POLICY FOR THE PAPER (user, 2026-10-14: "for E1 run and dependent of E1 should be run on
  v2"):** the ablations (E2-E4, controls, E8, partial clouds, E9) are reported on sorghum v1 (the
  only probed programme) until the v2 twins are probed; E1 and EVERYTHING DERIVED FROM IT must be
  v2: the teacher = `4m_pretrain_15k_d2_sink` (Sinkhorn, job 16907818), the distillation
  (`config_4m_distill_15k_all_d2.yaml`, to create, warm start from its epoch-1000 checkpoint, with
  `pc_sinkhorn_*` set as in the E1 config), the single-sensor students and the quant generation
  numbers (`reports/quant_*`), the E1 row of the partial-cloud study (`eval/pc_view_eval.py`), and
  the paper figures (`figures/make_paper_figures.py` sorghum checkpoint). `paper/results.tex`
  marks the v1 E1-derived numbers with \todo until then.
- **Paper files (2026-10-14):** `paper/method.tex` (Sec. 3), `paper/results.tex` (Sec. 4, restructured to
  the draft PDF's outline: Setup / Regime I / Regime II / Regime III / any-subset + distillation / parameter
  stream / metrics / real-world / representation) and `paper/discussion.tex` (Secs. 5-6). Regime III and
  real-world (E10) are \todo placeholders; Regime I's leaf-count / height / F-score / 3-view / two-stage
  cells are empty. Distillation "before" = the epoch-0 warm start (`logs/evalwarm_12085946.out`), NOT
  epoch 1 of the history (that understates the gain: 19.1 % vs 5.9 %). Figures: `paper/fig_metrics.pdf`
  (`figures/make_metric_fig.py`, Chamfer vs probe over 12 sorghum arms) and `paper/fig_view_elevation.png`
  (copy of `vis_gallery/view_elevation_effect.png`). **The Overleaf project is `paper/overleaf/`** (imported
  2026-10-14 from the user's zip; `main.tex`, `sec/*.tex`, `CVPR.bib`). Its Secs. 4-6 are GENERATED from
  `paper/results.tex` / `discussion.tex` by `python paper/sync_overleaf.py` (cite-key map, label aliases): edit
  the repo drafts, re-sync, `latexmk -pdf main.tex`. Secs. 0-3 and the supplement are edited in place.
- **SORGHUM v2 EVALUATION LAUNCHED (2026-10-14 ~20:10).** All 13 `_d2` ablation arms are complete
  (E2 x4 at 600, tg/tg40/tg100 at 600, e3 at 6168/2024/624, e4 small/large at 600, e8_supervised_d2 at
  600; `config.json` records `rgb_file: rgb_nobg.png` in each). Scavenger, 1 GPU each, `--requeue`,
  caches resume: **16921549** `pr_600_d2` (probe + E9 val/test of the seven E2/control arms and the two
  E4 arms at epoch 600), **16921550/1/2** `pr_{6168,2024,624}_d2` (e3_1k/3k/10k), **16921553**
  `pr_mean600_d2` (mean-pooled rows for E8 of pc / pcrgbd / pcrgbdt), **16921554** `e8_score_d2`
  (`slurm/e8_supervised_score.sbatch` now takes `RUN=` / `E8_CONFIG=`), **16921555**
  `e8_base_sorghum_d2` (`baseline_probe.sbatch sorghum`: RGB baselines re-extract on rgb_nobg, the
  images are the BRIGHTENED ones since 10-13), **16921556** `pcview_d2` (1 vs 3 cameras, the four
  E2 d2 arms; the E1 row waits for the Sinkhorn run). Outputs: `reports/probe_<name>_<jobid>.csv`,
  `reports/e9_<name>_<jobid>/`, `reports/pcview_<jobid>_*.csv`. Then: paired CIs
  (`eval/probe_pair_bootstrap.py`, `--rgb-tag rgb_nobg` for the E8 RGB caches), swap the v1 numbers
  in `paper/results.tex` for v2, refresh the ledger artifact.
  The Sinkhorn E1 (16907818) ran 17:55-19:45 on nova22-gpu-2 (2x A100): epochs 1-2 in ~96 min
  (cold cache; checkpoint_epoch_2 saved), preempted at 19:45, requeued, pending again on Priority.
- **2026-10-15 10:30: THE FIRST v2 EVALUATION WAS CONFOUNDED AND IS BEING RE-RUN.** All eight
  jobs above completed overnight (reports moved to `reports/_stale_brightened_2026-10-14/`, caches to
  `outputs/_probe_cache{,_pcview}/_stale_brightened_2026-10-14/`), but the sorghum `rgb_nobg.png` files
  were brightened IN PLACE on 10-13 10:02 (x1.85, PNG tag `brighten_gain`) AFTER every `_d2` ablation arm
  had finished training (last one, e3_1k_d2, 10-13 02:22), and no loader undid the gain: every RGB-using
  arm, the supervised row, the RGB baselines and the partial-cloud study were probed on images 1.85x
  brighter than they trained on (PC-only rows unaffected). Fix (commit below): `SorghumDataset(rgb_gain=)`
  rescales the colour channels by `rgb_gain / <file tag>` (`read_brighten_gain`, `disk_rgb_gain`);
  both trainers record `rgb_brighten_gain` in config.json (the Sinkhorn E1's was hand-set to 1.85: it
  trains on the brightened files; `linear_probe.run_rgb_gain` reads the key, 1.0 when absent = every
  earlier run); `linear_probe.cache_stem` appends `__g<gain>` for runs on a brightenable file (v1 names
  unchanged), used by E9 and `probe_pair_bootstrap`; `pc_view_eval` pins (rgb_file, gain) per pass and
  tags its caches; `baseline_probe` serves the sorghum baselines at gain 1.0 with cache tag
  `rgb_nobg_g1` (pass `--rgb-tag rgb_nobg_g1` to the pair bootstrap). Verified on CPU against the
  figure script's undo path (max |diff| 0.016 after ImageNet normalisation, rounding + resize).
  Resubmitted (scavenger, same shapes): **16928968** `pr_600_d2`, **16928969/70/71** `pr_{6168,2024,624}_d2`,
  **16928972** `pr_mean600_d2`, **16928973** `e8_score_d2`, **16928974** `e8_base_sorghum_d2`,
  **16928975** `pcview_d2`; moved in place to nova at 10:45 (user: "mech-ai is free right now"). Numbers from the stale set, for the record (val R2 height / roll, brightened
  inputs): PC+RGB+D 0.977 / 0.778, +params 0.953 / 0.745, small 0.963 / 0.815, base 0.953, large 0.941
  / 0.699 (the E4 inversion is the kind of artefact to expect from the mismatch). Sinkhorn E1: restarted
  00:04 on nova22-gpu-3 (2x A100), epoch 23 at 09:30 = ~26 min/epoch (~17 days for 1000 if never
  preempted); val chamfer at epoch 20 is 4.94e-3 against the old-loss run's 4.82e-3 at the same epoch.
- **E5 VIEW REGIME LAUNCHED (2026-10-15 ~11:15, user: "write the code for E5 view regime").** The plan's
  two camera sets (6.2: random hemisphere vs a tractor-height pass) were never rendered; the ten views
  on disk are an exact elevation ladder (sin elev = 0.9 - 0.2 i: view 0 looks 64 deg down, 4-5 are within
  6 deg of the horizon, 9 looks 64 deg up), so E5 is built as a restriction of the TRAIN draw to a band:
  `data.train_views` (YAML list or comma string; `--train_views`), threaded to the train dataset only,
  recorded in config.json, refused without view_sampling or if a plant has none of the views. Arms =
  `configs/config_e5_{topdown,horizon,single}_d2.yaml` (views 0-1 / 4-5 / 0; everything else byte-equal
  to `config_e2_pcrgbdt_d2.yaml`, so the reference is the four-stream d2 arm and every arm runs 197,400
  steps; val/test stay on view 0). CPU-tested: draws stay inside the band over 40 epochs, plant set
  unchanged. Jobs (nova, mech-ai, 2x A100 / 32 CPU / 160G / 2 days, `scale_arm_blackwell.sbatch <slug>
  --num_workers 14`, auto-resume): see the squeue names `e5_*_d2`. Expect ~24 h each (the d2 arms did
  ~2.1-2.7 min/epoch on this shape). When done: `CKPT=checkpoints/checkpoint_epoch_600.pth
  E9_SPLITS="val test" sbatch slurm/linear_probe.sbatch e5_topdown_d2 e5_horizon_d2 e5_single_d2
  e2_pcrgbdt_d2`, the pair bootstrap against `e2_pcrgbdt_d2`, and ideally a probe per held-out VIEW
  (not built: `eval/linear_probe.py` scores view 0 only; add a `--probe-view k` to see how a regime
  generalises across elevation).
- **2026-10-15 11:40: SINKHORN E1 MOVED TO NOVA, 4x RTX PRO 6000, 4 DAYS (user: "try 4 gpu and long time
  in Nova for E1 requeueing other jobs", "Make it long for 4 days").** 16907818 (scavenger, 2x A100) was
  `scontrol requeue`d at epoch 25 (checkpoint_epoch_24 is the resume point), reshaped in place, then
  CANCELLED because a user cannot raise a job's TimeLimit (2 d) and nova allows 31 d. Replacement
  **16929801**: `E1_CONFIG=configs/config_4m_pretrain15k_d2_sink.yaml NUM_WORKERS=18 sbatch --gres=gpu:rtx_pro_6000:4
  --cpus-per-task=80 --mem=280G --time=4-00:00:00 --time-min=04:00:00 --signal=B:USR1@600 --requeue
  slurm/pretrain_15k_d2.sbatch` (nova / mech-ai / normal; accum = 8/4 = 2, global batch 256 unchanged;
  the launcher passes `--save_freq 2` and auto-resumes). It first pended on AssocGrpGRES (13 of 17 in use),
  so the E5 arms were requeued/held to make room: **16929383 / 16929384 / 16929385 are HELD (JobHeldUser)**;
  `scontrol release 16929383 16929384 16929385` once E1 is running and the cap allows (E1 4 + evals;
  E5 needs 2 each). At 11:45 E1 showed Reason=None with nova26-gpu-2 planned. Expected pace on 4x RTX PRO
  6000: ~6-8 min/epoch if the loader (72 workers) keeps up -> ~4-5 days for the remaining 975 epochs,
  i.e. it will need one USR1 self-requeue past the 4-day wall. Old-loss E1 outputs stay at epoch 716.
- **2026-10-15 11:55: EVERYTHING EXCEPT E1 CANCELLED (user: "cancel all other jobs except 16929801 I will
  ask you to requeue").** Cancelled: the gain-matched v2 evaluation 16928968 `pr_600_d2` (was 21 min in),
  16928972 `pr_mean600_d2`, 16928973 `e8_score_d2`, 16928974 `e8_base_sorghum_d2`, 16928975 `pcview_d2`
  (16928969/70/71, the E3 probes, had already COMPLETED: `reports/probe_pr_{6168,2024,624}_d2_1692896*.csv`
  + E9 dirs), and the three E5 arms 16929383/4/5 (no checkpoint yet: they restart from scratch). Partial
  feature caches (`outputs/_probe_cache/*__g1.npz`) survive and are reused. **Resubmit on request, exact
  lines (nova; add `--partition=scavenger --account=mech-ai-scavenger --requeue` instead for scavenger):**
  `CKPT=checkpoints/checkpoint_epoch_600.pth E9_SPLITS="val test" sbatch --job-name=pr_600_d2 slurm/linear_probe.sbatch e2_pc_d2 e2_pcrgb_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2 e2_pcrgbdt_tg_d2 e2_pcrgbdt_tg40_d2 e2_pcrgbdt_tg100_d2 e4_small_d2 e4_large_d2`;
  `FEATURE=mean CKPT=checkpoints/checkpoint_epoch_600.pth sbatch --job-name=pr_mean600_d2 slurm/linear_probe.sbatch e2_pc_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2`;
  `RUN=e8_supervised_d2 E8_CONFIG=configs/config_e8_supervised_d2.yaml sbatch --job-name=e8_score_d2 slurm/e8_supervised_score.sbatch`;
  `sbatch --job-name=e8_base_sorghum_d2 slurm/baseline_probe.sbatch sorghum`;
  `sbatch --job-name=pcview_d2 slurm/pc_view_eval.sbatch e2_pc_d2 e2_pcrgb_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2`;
  E5: `for s in e5_horizon_d2 e5_topdown_d2 e5_single_d2; do sbatch --job-name=$s --partition=nova --account=mech-ai --qos=normal --gres=gpu:a100:2 --cpus-per-task=32 --mem=160G --time=2-00:00:00 slurm/scale_arm_blackwell.sbatch $s --num_workers 14; done`
  (the linear_probe / e8 / pcview launchers default to nova or scavenger per their headers; the E5 line
  above is nova). Only **16929801** (Sinkhorn E1, 4x RTX PRO 6000, 4 d) is live.
  **Resubmitted 2026-10-15 10:35 (user: "requeue rest of the jobs"), all on nova / mech-ai:** 16930422
  `pr_600_d2`, 16930423 `pr_mean600_d2`, 16930424 `e8_score_d2`, 16930425 `e8_base_sorghum_d2`, 16930426
  `pcview_d2`, and the E5 arms 16930427 `e5_horizon_d2`, 16930428 `e5_topdown_d2`, 16930429 `e5_single_d2`
  (2x A100 each). The E3 probes were not resubmitted: they completed before the cancel.
- **2026-10-16 09:50: 2-GPU RESUME JOB QUEUED BEHIND E1 (user: "put a resume code with two GPUs after E1
  finish").** **16950840** `e1_sink_d2_resume`: nova / mech-ai, 2x RTX PRO 6000, 48 CPU, 180G, `NUM_WORKERS=20`,
  4-day limit, `--time-min 4h`, USR1 self-requeue, `--dependency=afterany:16929801`. The launcher resumes
  from the newest checkpoint in `outputs/4m_pretrain_15k_d2_sink/checkpoints` (accum = 8/2 = 4, global
  batch 256 unchanged) and exits at once if epoch 1000 is already there. A USR1 self-requeue of 16929801
  keeps the dependency waiting (same job id), so this only starts when 16929801 ends for good: finished,
  failed, cancelled, or a requeue that Slurm refused. E1 pace at 09:45: epoch 185, 8.7 min/epoch on 4x RTX
  PRO 6000 (val chamfer 1.88e-3 at epoch 180), ETA ~Oct 21.
- **MAIZE ROW DATA (Regime III) PIPELINE BUILT 2026-10-16 (user: "/work/mech-ai/alloydas/maize_row in this
  path I am pushing the row data write the training testing script for it").** The row renderer writes
  `<root>/<split>/plant_<id>/row.json` + `theta_<0..90>/shot_<k>/{rgba,depth,gt_rgba,gt_depth}.png`,
  `pointcloud_cam.ply` (the rover's SCENE cloud, 24,576 pts, target + two bent neighbours at d = 0.15 m),
  `gt_pointcloud_cam.ply` (the target alone, 8,192 pts, same camera), `maize_<id>_spline.xml` (schema v2
  + bendAngleDeg/bladeLength, ignored by the v2 parser), `camera_pose.json`. theta = the row's yaw vs the
  rover (0 = broadside), ten per plant. At 12:00 only `val` existed and was still uploading (811 plants);
  the loader skips plants whose row.json or shot files are incomplete, so re-index after the upload.
  Code: `maize_row_dataset.py` (`MaizeRowDataset(root, split, mode='row'|'isolated')`: inputs = row
  render or the gt_* files, targets ALWAYS the target plant's gt streams + v2 recipe; the gt cloud is
  moved into the INPUT cloud's centred/unit-sphere frame; per-item `occ_frac` (target pixels hidden by a
  neighbour, from the two depths) and `nbr_frac`; `sample_per_plant` = one (theta, shot) per plant per
  epoch), `train_maize_row_4m.py` (torchrun DDP; `EmbodiedMAE4M.forward(..., targets=(rgb, depth, pc))`
  is a new backward-compatible kwarg so the loss is taken against the target plant while the encoder
  sees the row; warm start `init.checkpoint`, v1-schema `outputs/maize_4m` ep 600 on Nova until a v2
  maize checkpoint is here; val reports Chamfer vs target AND vs input (echo detector) + v2 recipe MAE),
  `configs/config_maize_row_{ft,iso}.yaml` (row vs isolated-control arms, 100 ep, lr 1e-4, global batch
  32, QAL + Sinkhorn), `slurm/train_maize_row.sbatch [config] [extra]` (nova 2x A100, auto-resume),
  `eval/eval_maize_row.py` + `slurm/eval_maize_row.sbatch <run> [ckpt] [split] [extra]` (any maize
  checkpoint; modes row/isolated x sources all/rgb/depth/pc with the recipe never an input; per-item
  CSV + JSON summary by theta and occlusion quartile and the PAIRED row - isolated delta with a
  bootstrap CI over plants; recipe error only for v2-schema checkpoints). CPU smokes passed (2-plant
  train epoch + val; 3-plant eval of maize_4m). **To run once `train/` lands:** `sbatch
  slurm/train_maize_row.sbatch` and `sbatch --job-name=maize_row_iso slurm/train_maize_row.sbatch
  configs/config_maize_row_iso.yaml`; zero-shot baseline now: `sbatch slurm/eval_maize_row.sbatch
  maize_4m checkpoints/checkpoint_epoch_600.pth val`.
- **ROW PIPELINE MADE SPECIES-NEUTRAL (2026-10-16 ~13:00, user: "Do the same for sorghum i am preparing
  the data").** Renamed: `row_dataset.py` (`RowDataset(root, split, species='maize'|'sorghum', mode)`;
  the species table carries the recipe glob + parser (maize: `maize_*_spline.xml` schema v2; sorghum:
  `*_spline.yml`, width layout via `embodied_mae_4m.load_spline_params`), the default point count
  (8192 / 8196), max leaves (28 / 24) and the REAL parameter columns for the error metrics; plant folders
  `plant_*` or `Sorghum_*`; `rgb_gain` serves sorghum images at a run's `rgb_brighten_gain`, the renders
  being untagged = 1.0), `train_row_4m.py` (`data.species`; sorghum runs record `leaf_fields` and
  `rgb_brighten_gain` = `init.rgb_gain`), `eval/eval_row.py --species` (recipe scored only for the
  species' current layout: maize param_schema v2, sorghum leaf_fields with 'width'; sorghum images served
  at the run's recorded gain), `slurm/train_row.sbatch <config> [extra]`, `SPECIES=... slurm/eval_row.sbatch
  <run> [ckpt] [split]`, configs `config_{maize,sorghum}_row_{ft,iso}.yaml` (sorghum: root
  `/work/mech-ai/alloydas/sorghum_row` (not there yet), warm start `e2_pcrgbdt_d2` ep 600 at gain 1.0;
  switch to the Sinkhorn E1 epoch-1000 checkpoint with `init.rgb_gain: 1.85` when it exists).
  Smokes: sorghum trainer + eval on a fake 2-plant row sample built from Sorghum_15K/val (scratchpad only);
  the sorghum eval's Chamfer on that sample (1.8e-3) sits where the arm's val Chamfer does. NOTE: at
  13:05 `/work/mech-ai/alloydas/maize_row/val` had disappeared (the user is reorganising the upload);
  re-check the layout before launching anything.
- **Still to create:** the sorghum distillation on d2 (`config_4m_distill_15k_all_d2.yaml`, warm
  start from `outputs/4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth`) once E1 finishes.
  Then probes (`CKPT=checkpoints/checkpoint_epoch_600.pth slurm/linear_probe.sbatch e2_pc_d2 ...`),
  E9, the E8 baselines (RGB ones re-extract on the new images automatically), partial clouds.
- **Maize runs cancelled at the user's request (2026-10-01 17:48):** `distill_maize` 16684364 at
  epoch 59/100 (`checkpoint_epoch_58.pth` kept; resubmitting `slurm/distill_maize.sbatch` resumes it)
  and `maize_e2_pcrgbd_levelled` at epoch 9 (no checkpoint). The user wants them left cancelled.
  The video-eeg-ensembling jobs in the queue are the user's other project: never touch them.

Everything below was true at the time written; check `squeue -u $USER` and each run's
`outputs/<run>/checkpoints/` before acting. The results page is the private artifact
https://claude.ai/artifact/A6nkyfGFfnpy31dq9AdYWw ("Sorghum 4M Progress Review", v28). To update it
from a new session: `Artifact read` that URL, edit the saved copy, then publish with `url` set to it.
Its sections: sorghum headline/params/views/E2/E3·E4/6.4 probe, `#control` (token-budget control
arms), sorghum E9, `#maize` (everything for maize; `#maize-e2` is the maize E2 table), `#e8`
(baselines; `#e8-sup` is the supervised row, `#e8-pose` the maize camera-pose rows), `#pcview` (1 camera vs 3
cameras), agenda.

### Training jobs in flight — all run as chained 4-hour chunks on `scavenger`

Why chunks: on 2026-09-29 only jobs of <= 4 h could backfill onto the free scavenger GPUs (those
nodes are PLANNED for higher-priority jobs; 8-12 h requests waited 1-2 days). Each run is a chain
of `sbatch --time=04:00:00 --dependency=afterany:<previous chunk>`; every chunk auto-resumes from
the newest checkpoint that opens (save_freq 25 epochs, so a chunk end loses <= 25 epochs). The maize
and E8 launchers check that it opens; `e2_arm_blackwell.sbatch` does not, and since a268a0e
`train_sorghum_4m.py` itself walks an unreadable `--resume` down (`resolve_resume`), and every
trainer save is atomic (temp file + `os.replace`). A chunk
that starts after training already finished resumes at epoch 600, runs no epoch and exits — the
trainers do nothing after the loop but clean up — so surplus chunks are harmless.

| run | chunk shape | epochs done at 08:30 | chain job IDs still queued (last one) |
|---|---|---|---|
| `maize_e2_pc` | a100-pcie:2, 16 CPU, 96G, `--num_workers 7` | 353/600 | ...16645738, 16645739, 16669495 |
| `maize_e2_pcrgb` | a100-pcie:2, 16 CPU, 96G, `--num_workers 7` | 268/600 | ...16645744, 16645745, 16669496-98 |
| `e8_supervised` | l40s:2, 32 CPU, 160G, `--num_workers 14` | 376/600 | ...16645750, 16645751, 16669499 |
| `e2_pcrgbdt_tg100` | l40s:2, 32 CPU, 160G, `--num_workers 14` | 282/600 | ...16645756, 16645757, 16669500-02 |
| `maize_e2_pcrgbd` | nova, rtx_pro_6000:2, 32 CPU, 160G | **600/600 done** (job 16645060, 16 h 46 m) | — |

Rates were ~88 / ~67 / ~82 / ~61 epochs per 4 h chunk. If a chain runs dry before epoch 600,
extend it after its last job ID (maize needs `MAIZE_CONDA_ENV` exported at submit):

```bash
export MAIZE_CONDA_ENV=/work/mech-ai-scratch/alloy/.conda/envs/det_cu128
sbatch --job-name=maize_e2_pc --partition=scavenger --account=mech-ai-scavenger --requeue \
  --time=04:00:00 --open-mode=append --dependency=afterany:<LAST_ID> \
  --gres=gpu:a100-pcie:2 --cpus-per-task=16 --mem=96G \
  slurm/scale_arm_maize.sbatch maize_e2_pc --num_workers 7
sbatch --job-name=e8_supervised ... --gres=gpu:l40s:2 --cpus-per-task=32 --mem=160G \
  slurm/e8_supervised.sbatch --num_workers 14
sbatch --job-name=e2_pcrgbdt_tg100 ... --gres=gpu:l40s:2 --cpus-per-task=32 --mem=160G \
  slurm/e2_arm_blackwell.sbatch pcrgbdt_tg100 --num_workers 14
```

Before trusting a finished run, confirm `checkpoints/checkpoint_epoch_600.pth` exists and
`training_history.json` has 600 train epochs. Chunks that start after epoch 600 are expected to exit fast.

**UPDATE 2026-09-30 ~09:00 — the four unfinished runs move to `nova` (user: "mech-ai is free").**
Each got a `nova` job (account `mech-ai`, QOS normal, rtx_pro_6000:2, 32 CPU, 2-day limit,
`--num_workers 14`) that starts when its currently running scavenger chunk ends, and the next
scavenger chunk of each chain was re-pointed (`scontrol update Dependency=afterany:<nova job>`) so
the chain waits behind the nova job instead of racing it into the same output dir. If the nova job
finishes, the leftover scavenger chunks resume at epoch 600 and exit; if it fails, they carry on.

| run | nova job | runs after | scavenger chain now waits on it |
|---|---|---|---|
| `e2_pcrgbdt_tg100` | 16669526 | 16645755 | 16645756 -> ... |
| `e8_supervised` | 16669527 | 16645749 | 16645750 -> ... |
| `maize_e2_pc` | 16669528 | 16645737 | 16645738 -> ... |
| `maize_e2_pcrgb` | 16669529 | 16645743 | 16645744 -> ... |

CPU probes are queued behind them (scavenger, `--gres=NONE`, checkpoint_epoch_600, E9 on val):
16669530 `pm_e2_600` (maize_e2_pc/pcrgb/pcrgbd + maize_4m, after 16669528 and 16669529) and
16669531 `pr_tg100_600` (the five sorghum control-comparison arms, after 16669526). They use
`afterany`, so if a run had not reached epoch 600 its probe fails on the missing checkpoint: rerun
it by hand once the run is done. E8 supervised still needs its `--score_head` pass and a probe by hand.

**UPDATE 2026-09-30 ~12:50 (new session) — three runs on nova, `maize_e2_pc` back on scavenger.**

| run | where | epoch at 12:40 | expected 600 |
|---|---|---|---|
| `e8_supervised` | nova 16669527, nova26-gpu-2, since 10:03 (resumed 425) | 523 | ~14:45 (37 ep/h) |
| `maize_e2_pc` | scavenger chunk 16645738, since 12:23 (resumed 475) | 479 | ~16:15 (33 ep/h) |
| `maize_e2_pcrgb` | nova 16669529, nova26-gpu-1, since 11:49 (resumed 350) | 375 | ~20:30 (29 ep/h) |
| `e2_pcrgbdt_tg100` | nova 16669526, nova26-gpu-2, since 10:03 (resumed 300) | 359 | ~23:30 (22.5 ep/h) |

- **Why `maize_e2_pc` did not go to nova.** Labmates held 10 of mech-ai's 17 GPUs. The two sorghum
  nova jobs plus pcrgb's brought it to 16, so a fourth 2-GPU nova job would have pended on
  `AssocGrpGRES` (the earliest labmate release was ~17:47), with the whole chain waiting behind it.
  Rewired at 10:10: 16645737 -> 16645738 -> 16645739 -> 16669495 -> **16669528 (the nova job, now
  the chain's LAST link, `afterany:16669495,singleton`)**. It exits at once if 600 is reached and
  carries on if the chain runs dry. **Extend this chain with `afterany:16669528`, never
  afterany:16669495**: otherwise the extension and 16669528 would both start when 16669495 ends.
- `pm_e2_600` (16669530) now waits on 16669495 + 16669529.
- **`slurm/nova_fallback.sbatch`** (new): if a spliced-in nova job is still pending GRACE (15 min)
  after its predecessor ends, for a capacity reason, it holds it, re-walks the chain (same job name),
  releases the next scavenger chunk, moves the nova job behind the real last link (`,singleton`
  added) and repoints any probe. On anything unexpected it keeps the nova job HELD: a stall, never
  a double run. It was rehearsed twice on dummy chains blocked by a real AssocGrpGRES (jobs
  16675589-94, 16676307-19) and hardened after two adversarial reviews. Job 16676327 guarded
  pcrgb at 11:49: the nova job had started, so it did nothing.
- **Maize E2, first answer (probe 16675588, epoch 600, val R²):** `maize_e2_pcrgbd` beats `maize_4m`
  on 10 of 11 targets, on val and test alike: leaf angle 0.911 vs 0.865, height 0.604 vs 0.579,
  leaf width 0.781 vs 0.742, leaf count 0.908 vs 0.889, curl 0.177 vs 0.119. `maize_4m` wins only
  stem radius (0.741 vs 0.693). The run-to-run floor is ~0.02, so about half of these gaps are real.
  The sorghum finding replicates: the parameter stream makes the latent worse. The maize_4m row
  reproduces probe 16592058 exactly (same cache). E9 is in `reports/e9_pm_pcrgbd600_16675588/`.
- **E8 finish is queued:** `e8_score` 16676974 (`slurm/e8_supervised_score.sbatch`,
  afterany:16669527, reviewed clean) runs `--score_head`, then the cls and the mean probes at
  checkpoint_epoch_600.pth, writing `reports/probe_e8_score_16676974_{head,cls,mean}.csv`. It
  exits 3 if epoch 600 is missing. Then add the row to `#e8`.
- `slurm/distill_maize.sbatch`: both pre-launch fixes are ported (link-count split guard,
  `WANDB__SERVICE_WAIT=600`). The teacher decision is still open.
- **New experiment (user, 2026-09-30): 1-view vs 3-view point clouds, sorghum, evaluation only.**
  See "Point-cloud view ablation" below.

**UPDATE 2026-09-30 ~16:00 — E8 supervised row done, view ablation done, trainer saves atomic.**

| run | where | epoch at 15:51 | expected 600 |
|---|---|---|---|
| `e8_supervised` | nova 16669527 finished 14:24; leftover chunks exited in seconds | **600/600 done** | — |
| `maize_e2_pc` | scavenger 16645738 hits its 4 h limit at 16:23 (~591, resumes from 575), then 16645739 | 574 | ~1 h after 16645739 starts |
| `maize_e2_pcrgb` | nova 16669529 | 527 | ~17:20 (49 ep/h since 13:00) |
| `e2_pcrgbdt_tg100` | nova 16669526 | 462 | ~20:00 (34 ep/h since 13:00) |

- **E8 supervised row** (scored by 16676974; paired bootstrap 16683553 with the new
  `eval/probe_pair_bootstrap.py` -> `reports/e8_supervised_delta_16676974.csv`). Its head, val R²:
  height 0.974, leaf count 0.979, biomass 0.979, branch angle 0.929, roll 0.811, leaf length 0.272.
  Its CLS probe matches the head within 0.008. Frozen PC+RGB+depth minus supervised, both on CLS:
  size +0.006 (tight CIs), leaf length +0.08 to +0.12, branch angle a tie, roll a val tie and a test
  loss (−0.018). **The four-modality arm minus supervised: size −0.008 to −0.014, branch −0.027, roll
  −0.016 to −0.032, both splits.** Pretraining pays without the parameter stream, and with it the
  frozen feature is worse than no pretraining. Mean pooling flatters PC+RGB+depth (roll +0.09) but
  reads the supervised net through tokens its training never shaped. The page has it as `#e8-sup`,
  plus a row in the sorghum E8 table (mean pool).
- **View ablation done** (job 16676994, 57 min, 1 A100): results in the subsection below; page `#pcview`.
- **Trainer fix a268a0e** (found by the review workflow): both 4M trainers wrote checkpoints,
  best_model, config.json and training_history.json in place. `e2_arm_blackwell.sbatch` resumes from
  the highest-numbered checkpoint by name without opening it, so a kill mid-save would have burnt
  the whole tg100 chain at startup, and a probe starting beside a leftover chunk could read a
  config.json the chunk had just truncated. All four writes now go through temp + `os.replace`, and
  `train_sorghum_4m.py`'s `resolve_resume` falls back to the newest checkpoint that opens (it raises
  if none does). CPU-tested on scratch files and a real 1.37 GB checkpoint. Queued chunks read the
  trainer from disk, so they have it. Running processes never re-read it.
- (The planned `pm_e2_600` dependency trim was not needed: the leftover chunks got GPUs at once.)

**UPDATE 2026-09-30 ~19:50 — every training run is done; maize distillation launched.**

- **All four runs 600/600:** `maize_e2_pc` (17:12, chunk 16645739), `maize_e2_pcrgb` (17:18, nova
  16669529), `e2_pcrgbdt_tg100` (19:35, nova 16669526), `e8_supervised` (14:24). Leftover
  scavenger chunks resume at 600 and exit; let them.
- **Maize E2 is complete:** `pm_e2_600` (16669530) ran 17:18-18:22 ->
  `reports/probe_pm_e2_600_16669530.csv`, E9 in `reports/e9_pm_e2_600_16669530/`. Its pcrgbd and
  maize_4m rows match probe 16675588 to 3e-5. Val R², PC / PC+RGB / PC+RGB+D / +params: height
  0.184 / 0.654 / 0.604 / 0.579, leaf angle 0.810 / 0.916 / 0.911 / 0.865, leaf count
  0.712 / 0.894 / 0.908 / 0.889, stem radius 0.541 / 0.723 / 0.693 / 0.741. RGB is where maize
  phenotype arrives. Depth is a wash: PC+RGB leads on height, stem radius and leaf length;
  PC+RGB+D on droop, width, twist and count. The parameter stream is best on stem radius only.
  Paired CIs: `eval/probe_pair_bootstrap.py --species maize` (job 16684499 ->
  `reports/maize_e2_delta_16669530.csv`, `reports/maize_e2_depth_delta_16669530.csv`).
- **`pr_tg100_600` (16669531)** is pending on Priority on scavenger (CPU, 32 CPU / 128G). It has to
  extract tg100's features on CPU; the other four arms are cache hits. When it lands, add tg100's
  row to the page's `#control` table.
- **Maize distillation launched (decision taken, see below).** GPU smoke 16684363 (nova, 2x RTX PRO
  6000, `--max_steps 40`) passed in 2 min 21 s: epoch-0 per-source eval, 10 optimiser steps, val,
  the viz path and a checkpoint save, exit 0. PEAKMEM 25 GiB alloc / 33 GiB reserved per GPU,
  steady 2.92 it/s, so about 19 min per epoch and about 35 h for 100 epochs: one 2-day wall.
  Full run **16684364** (`afterok` on the smoke, `--kill-on-invalid-dep=yes`, nova, rtx_pro_6000:2,
  48 CPU, 160G), writing `outputs/maize_distill_all`. Its smoke output is in
  `outputs/maize_distill_smoke`; delete that whenever. **At 20:07 its limit was cut from 2 days
  to 4 h in place.** Both Blackwell nodes had all 16 GPUs idle but were PLANNED (`mixed-`), so a
  2-day job could not backfill. The launcher's USR1 trap now requeues it every ~3 h 50 m (same
  job id, same log), and each segment resumes from the newest checkpoint (save_freq 2 epochs,
  about 38 min; average loss ~20 min per segment). That makes ~10 segments for ~35 h of training.
  A user cannot raise a time limit again, so this shape stays. At 20:10 even 1-hour probes of
  that shape were pending on Priority: the run starts when the reservation clears.
- **Upright arm rows (sensitivity, maize):** new E8 adapters `maize_e2_pc_upright`,
  `maize_e2_pcrgbd_upright` and `maize_4m_upright` (`eval/baselines/_arm_upright.py`) give our
  frozen arms the same per-view camera-pose rotation as `pointmae_upright`. They are the
  like-for-like test of the "Point-MAE + pose beats maize_4m on 8/11" result, and the cheap
  first step of the gravity-aligned-clouds decision. Job **16684794**
  (`slurm/baseline_probe.sbatch maize ...`, 1 GPU) -> `reports/probe_e8_up_maize_16684794.csv`.
  A 48-plant CPU smoke completed the PC-only row end to end. Compare each row with its own
  camera-frame row and with `pointmae_upright`. A gain means the frozen encoder can use
  orientation anyway. No gain says nothing about an arm pretrained on upright clouds, which
  is the next step if this one is ambiguous.

**UPDATE 2026-10-01 ~14:30 — tg100 answered; the maize pose lead is mostly a generator artefact;
a levelled-cloud arm is training.** Page v28. Code: b30348b, 46fc650.

- **tg100 (parameters as a target only).** Probe 16669531 died at 21:12 on `OSError: [Errno 116]
  Stale file handle` (NFS), 40 min into extracting tg100's train split. All four extraction
  loops now go through `_RetryTransientIO` (`eval/linear_probe*.py`, also used by
  `baseline_probe.py` and `pc_view_eval.py`): an ESTALE/EIO item is retried in the worker with the
  RNG state restored, so point subsets match a failure-free run. Rerun 16686383 (CPU, 52 min) ->
  `reports/probe_pr_tg100_600_16686383.csv`, E9 `reports/e9_pr_tg100_600_16686383/`, paired CIs
  `reports/control_delta_16686383.csv` and `control_delta_vs_tg_16686383.csv`. Val R²: height 0.970,
  leaf count 0.973, roll 0.737, leaf length 0.259 (PC+RGB+depth 0.981 / 0.984 / 0.806 / 0.362).
  Against PC+RGB+depth it loses every learnable target on both splits: size −0.011, branch angle
  −0.028 / −0.016, roll −0.069 / −0.077, leaf length −0.102 / −0.097. So the reconstruction target
  hurts the latent too, not only the input. Against the arm as trained: size +0.005, roll −0.06.
  Against tg (20 % visible): size +0.012 to +0.014, roll −0.027 / −0.031, so the visibility trend
  is clean on size only. E9: the most size-dominated latent of the five (PC1 43 %, |r| 0.95 with
  leaf count; participation ratio 4.6). The sorghum leaf tokens carry per-leaf length and roll,
  the two targets lost most. Roll is encoded linearly although it is circular: an untested
  candidate cause.
- **Full-pose rows (16684794; CIs 16686395 -> `reports/maize_upright_delta_16684794.csv`).** Our
  frozen arms given cameraToWorld-rotated clouds. The fused arms lose 7 of 11 against their own
  camera-frame rows (CLS) and win only twist and curl. PC-only gains 6 and loses stem radius. All still lose 9 to 10 of
  11 to `pointmae_upright`. `maize_4m` against `pointmae_upright` reproduces the E8 numbers
  exactly (test height −0.115, curl −0.278, ...).
- **The full pose leaks the plant's azimuth.** In the maize renderer's world frame (y up), every
  plant's leaf plane is the same world plane: axial resultant 0.993 over 194 val plants, view 00.
  The generator places leaf i at 180 i + U(−15, 15) degrees with no per-plant yaw. Camera azimuths
  span the full circle, and elevations run −86 to +86 degrees. So `pointmae_upright` got gravity
  plus a canonical plant orientation, which no rig has. Each view's cloud is the same surface,
  resampled per view (NN 2.9e-3 against 2.1e-3 spacing).
- **Gravity-only rows (16686506; CIs 16687493 -> `reports/maize_gravity_delta_16686506.csv`).**
  `eval/baselines/_gravity.py` applies cameraToWorld and then the yaw that puts the camera back at
  azimuth 0: a level camera at its own azimuth (resultant 0.114). Rows: `pointmae_gravity`,
  `maize_{e2_pc,e2_pcrgbd,4m}_gravity`. Point-MAE val R², camera -> gravity -> full: height
  0.509 -> 0.588 -> 0.714, curl 0.199 -> 0.309 -> 0.417. Gravity helps 10 of 11; the azimuth adds a
  significant gain on all 11 on top. **Against Point-MAE + gravity, camera-frame PC+RGB+depth
  (no pose) wins 5** (test: height +0.046, biomass +0.040, twist +0.046, width +0.055, stem radius
  +0.232) **and loses 4** (leaf angle −0.021, leaf count −0.027, curl −0.181, tassel droop −0.210);
  droop and leaf length tie. `maize_4m` wins 4 and loses 5. Our frozen fused arms gain nothing
  from levelled clouds (PC+RGB+depth 0 wins and 5 losses; `maize_4m` 1 and 6). PC-only gains 5
  with no losses. Camera-frame mean-pooled features for maize_e2_pc and maize_e2_pcrgbd: 16686725
  -> `reports/probe_pm_e2_mean600_16686725.csv`. `eval/probe_pair_bootstrap.py` now reads E8
  caches by `base:<name>`. The upright caches' fingerprints were checked unchanged.
- **Decision (delegated): pretrain a levelled arm.** `maize_e2_pcrgbd_levelled` is
  `maize_e2_pcrgbd` with every cloud levelled as in the gravity rows: same seeds, views and point
  permutations (the rotation draws no RNG), 197,400 steps. `train_maize_4m_gravity.py`
  installs a dataset subclass into `train_maize_4m` rather than editing `maize_dataset_4m.py`,
  which the running distillation re-imports at every requeue. `train_maize_4m.py` only gained a
  `PC_FRAME` guard (a config's `data.pc_frame` must match the entry point). The launcher picks the
  trainer from `pc_frame`. Probe row `maize_e2_pcrgbd_levelled` asserts `config.json` records
  gravity. Chain **16687759** -> 16687760 -> ... -> **16687764** (6 x 4 h, `afterany`, 32 CPU,
  `--num_workers 14`, `MAIZE_CONDA_ENV=det_cu128`). Submitted to scavenger 2x L40S, then moved in
  place at 13:45 to **nova, 2x RTX PRO 6000, 128G** (`scontrol update` Partition/QOS/Account,
  TresPerNode, MinMemoryNode; dependencies verified intact). The L40S nodes were reserved, and
  labmates' pending jobs were waiting on Priority/Resources, not the 17-GPU cap (13 -> 15 of 17).
  `maize_e2_pcrgbd` took 16 h 46 m on this shape, so about 4.2 chunks; check that six reach 600.
  Extend with `--dependency=afterany:<last ID>`. A full-pose arm is deliberately not trained: its extra gain
  is the generator's azimuth.
- **Distillation 16684364**: epoch 47 of 100 at 11:30, ~19.5 min per epoch, requeueing cleanly
  every ~3 h 50 m (Restarts=3 at 08:00). Expected to finish early on 2 Oct. Val mean_gen has been
  flat at ~0.297 since epoch 10. Per source, epoch 45 against epoch 0:
  - src=pc: chamfer 0.052 -> 0.0021, collapse repaired; param MAE −13 %;
  - src=rgb / src=depth: chamfer −39 / −41 %, param MAE −6 %;
  - src=text: chamfer 0.033 -> 0.012.

  Train loss keeps falling (0.292 -> 0.229).

### What to do when each finishes

- **Maize E2 arms**: done, all four probed (19:50 update) and on the page as `#maize-e2`.
- **Maize distillation** (`outputs/maize_distill_all`, job 16684364): `history['val'][0]` is the
  "before". Report per source, never only the mean. The epoch-0 src=pc row is inflated by the
  warm start's full-PC collapse (smoke: src=pc pc_chamfer 0.056 vs 0.0019 from RGB). If it ends
  TIMEOUT without requeueing, re-run `sbatch --partition=nova --account=mech-ai --qos=normal --gres=gpu:rtx_pro_6000:2 --cpus-per-task=48 --mem=160G --time=2-00:00:00 slurm/distill_maize.sbatch`
  (auto-resume).
- **`e2_pcrgbdt_tg100`**: done and on the page (`#control` row, see the 2026-10-01 update): the
  reconstruction target itself hurts the latent.
- **`maize_e2_pcrgbd_levelled`** (chain 16687759-64, 6 x 4 h on nova, 2x RTX PRO 6000): before trusting
  it, confirm `checkpoint_epoch_600.pth` exists and `config.json` records `"pc_frame": "gravity"`.
  If the chain runs dry, extend it after its last ID with the same line as below. Probe it with
  `sbatch --job-name=e8_lvl_maize slurm/baseline_probe.sbatch maize maize_e2_pcrgbd_levelled`
  (NOT eval/linear_probe_maize.py, which would feed it camera-frame clouds), then pair it with
  `eval/probe_pair_bootstrap.py --species maize --feature mean --a base:maize_e2_pcrgbd_levelled
  --b base:pointmae_gravity` and `--b maize_e2_pcrgbd` (its camera-frame twin), and add it to
  the page's `#e8-pose` table.
- **`e8_supervised`**: done and on the page (`#e8-sup`, see the 16:00 update). A maize supervised
  row does not exist; it would need a maize port of `train_supervised_4m.py`.

### Point-cloud view ablation — 1 view vs 3 views (sorghum, evaluation only; added 2026-09-30)

Every model here trained on the COMPLETE plant cloud: each view's `_nc_cam.ply` is the full
`_nc.ply` moved rigidly into that camera's frame (checked point for point). This asks what a model
loses when its cloud is what a camera rig would capture: camera 00 alone, or cameras 00+01+02.
Those three sit at azimuths of about 0/+88/-138 deg, all above the plant, looking down at 64/44/30 deg.
RGB and depth stay camera 00's.

- `eval/pc_view_masks.py` + `slurm/pc_view_masks.sbatch` (done, job 16676293, 18 min, CPU):
  per plant and view, a point is visible if a ray from the camera to its closest mesh point (the
  cloud sits <= 5 mm off `<plant>.obj`) is not blocked more than 0.5 mm early, and it lies inside the
  frustum. The frustum is vertical FOV 40 deg, square, principal point at the centre, fitted from the
  silhouettes (f ~ 1406 px at 1024). Masks are in `outputs/_pcview_masks/<split>__v000102.npz`. Coverage: **1 view 48.8 % +- 7 %,
  3 views 78.3 % +- 4 %**, the same on every split. Checked against the renderer: visible points
  land on the silhouette 95 % of the time and match its depth within 5 mm 71 % of the time,
  against 3 % for occluded points. depth.png decodes to (z - 23.8 mm) / 50, a constant offset.
- `eval/pc_view_eval.py` + `slurm/pc_view_eval.sbatch`: the four E2 arms at epoch 600 and E1
  (`4m_pretrain_15k_v2_depthfix_qal`, epoch 1000) under full / 3view / 1view. It reports the 6.4
  ridge probe on the CLS token (all non-text tokens visible, as in `eval/linear_probe.py`), both
  refitted per condition ("matched") and as the full-cloud probe applied to partial clouds
  ("transfer"), and, on val, reconstruction chamfer against the complete cloud. **Reconstruction
  must run in the masked regime** (visible modalities token-masked at the run's mask_ratio 0.8).
  With every token visible the decoder is out of distribution and its cloud collapses, to chamfer
  ~0.024-0.027 against the published ~0.0017-0.005; the maize distillation teacher showed the same.
  Point subsets are seeded per plant, so the full condition is re-extracted, not read from the
  probe caches.
- **Done: job 16676994** (57 min on one A100, all 15,000 plants) ->
  `reports/pcview_16676994_{coverage,recon,probe}.csv`. Paired bootstrap over plants (job 16683300,
  `eval/pc_view_bootstrap.py`, CPU) -> `reports/pcview_16676994_bootstrap.csv`. On the page as
  `#pcview`. The harness is anchored: the full condition reproduces the published E2 val chamfer
  (0.00494 / 0.00242 / 0.00165 / 0.00243 against 0.00495 / 0.00250 / 0.00171 / 0.00248) and the
  published probe within 0.005.
  - **Matched probe (refit per condition): small costs if the arm sees RGB.** Size targets at one
    camera: −0.01 to −0.03 R², under 0.01 at three; PC-only −0.09 to −0.10 and −0.04. Roll and leaf
    length lose 0.02-0.06 at one camera. **Depth matters more on partial clouds:** on height at one
    camera (test), PC+RGB loses −0.027 [−0.030, −0.024] and PC+RGB+D −0.017 [−0.019, −0.015]. The
    params arm's deficit holds (0.012 behind PC+RGB+D at one camera, 0.015 on the full cloud).
  - **Transfer (full-cloud probe on partial clouds) collapses:** E1 height 0.983 -> 0.690 (3 cams)
    -> −0.21 (1 cam). Refitting recovers 0.966, so the information is still there, but the latent
    moves with coverage. Any readout fitted on complete clouds is biased on real scans (E10).
  - **Reconstruction: no completion.** At one camera, GT->prediction is 0.0087-0.0104 against 0.0040
    for echoing the partial input. The decoder reproduces what it is given.

### Maize distillation — launched 2026-09-30 (job 16684364) with the declared teacher departure

`train_maize_4m_distill.py`, `configs/config_maize_distill_all.yaml`, `slurm/distill_maize.sbatch`
(teacher = student init = `outputs/maize_4m/checkpoints/checkpoint_epoch_600.pth`; 100 epochs, lr 1e-4,
global batch 128 = 16 x 2 GPUs x accum 4, 821 optimizer steps/epoch like the 8-GPU sorghum run; an
in-run epoch-0 "before" eval written to `warmstart_eval.json`; `--eval_only` gives it on one GPU).
**Teacher decision (taken 2026-09-30, user: "do it yourself").** The run uses the config's
`teacher_source_mask_ratio: 0.5`: the teacher sees all four modalities, 50 % of each modality's
tokens. Sorghum's teacher saw every token (all visible, 0.0). In that regime the maize epoch-600
teacher's point cloud COLLAPSES (chamfer 0.053-0.089 against ~0.001). The model was pretrained
seeing at most 75 % of the PC tokens, so all-visible is out of distribution. The view ablation
found the same for the sorghum E2 arms. 50 % of each is the best teacher in BOTH species (maize
0.0009-0.0014; sorghum teacher_final 0.00035 against 0.00048 all-visible). Unlike `[rgb, depth,
text]`, it keeps the point cloud among the teacher's privileged inputs. The literal sorghum
recipe would have distilled every PC-token feature from a degenerate teacher. No A/B was run:
the teacher table in the config is the evidence. Report this as a declared departure.
Report per-source before/after, never only the mean — the collapse inflates the maize "before".
Runtime: the smoke measured ~19 min/epoch on 2x RTX PRO 6000 (2.92 it/s), so ~35 h. Both
pre-launch launcher fixes (link-count split guard, `WANDB__SERVICE_WAIT=600`) are in.

### E8 results (verified by an independent refit; tables in the page's `#e8` section)

Frozen probe, mean pooling for every model (the only pooling all have); Δ = paired bootstrap over
plants, 2,000 resamples; "win" = 95% CI excludes 0 on BOTH val and test.
- **Sorghum:** PC+RGB+depth beats every baseline on all 7 learnable targets; vs EmbodiedMAE-B (the
  strongest) +0.005 on size (tight CI, 0.90 floor), +0.078 roll, +0.104 leaf length. **The four-modality
  reference arm LOSES size to EmbodiedMAE-B** (height −0.006, leaf count −0.007, biomass −0.005) and
  wins roll/leaf length. **Point-MAE beats our PC-only arm on every sorghum target** (height 0.958 vs
  0.910) — our lead comes from fusing RGB with geometry, not a better PC encoder.
- **Maize:** `maize_4m` beats EmbodiedMAE-B on 9/11 and the best baseline per target on 5/11
  (E4-small on 8/11); loses leaf count and curl to Point-MAE and tassel droop to DINOv2 (0.275; every
  EmbodiedMAE-design encoder is < 0). **Camera-pose sensitivity:** Point-MAE given each view's
  camera pose (gravity-aligned clouds, which our arms never get) beats `maize_4m` on 8/11 (height −0.115,
  leaf angle −0.080, leaf count −0.056, curl −0.278 ...); we keep twist (+0.020) and stem radius
  (+0.241). Orientation alone lifts Point-MAE's height 0.509→0.714. **But that full rotation also hands
  over the generator's fixed plant azimuth** (see the 2026-10-01 update): with gravity only,
  PC+RGB+depth wins 5 and loses 4 against Point-MAE. Decision taken 2026-10-01: an arm
  pretrained on levelled clouds (`maize_e2_pcrgbd_levelled`) is training.
- Sorghum `pointmae_upright` is bitwise identical to `pointmae` (every sorghum view 00 is one camera).
- Analysis artefacts (refit script, per-plant predictions, e8_tables.json) lived in the old session's
  scratchpad and may be gone; everything needed to recompute is in `reports/probe_e8_*.csv` + the
  `outputs/_probe_cache*` feature caches.

### Infrastructure lessons from 2026-09-28/30 (all fixed in the launchers)

- **`/work/mech-ai-scratch` is NFS (novastor010) and 99% full (150/152 TB); `/work/mech-ai` 98%.**
  Under load, a readdir of the 105,000-entry maize train split hung for 45+ min in state D while
  `stat` answered in 0.2 s. The maize guards now count folders from the link count
  (`stat -c %h` − 2), never `ls`. Anything that lists the split dirs can hang a job.
- **wandb-core start timeout:** with NFS slow, all five jobs died at `wandb.init` with
  `ServicePollForTokenError` (30 s default). Launchers now `export WANDB__SERVICE_WAIT=600`
  (env `WANDB__X` maps to setting `x_...`, wandb 0.28.2).
- **40 GB A100-PCIe OOM at validation:** chamfer in `evaluate()` allocates (B,N,N,3) = 12 GiB at
  batch 16 / 8192 pts; it OOMed with 12.86 GiB reserved-but-unallocated. `train_maize_4m.py` now calls
  `torch.cuda.empty_cache()` before val and test evaluation (no effect on numbers). Batch-16 training
  itself needs ~22.4 GiB/GPU, so 40/48 GB cards are otherwise fine.
- **Scheduling:** `--test-only` does not model backfill and was wrong by days; probe with real jobs
  that run `hostname`. Blockers found in order: CPUs per node (48 did not fit, 32 did), then memory,
  then **time limit** (planned nodes only admit jobs that end before the reservation). Pending jobs
  can be reshaped in place: `scontrol update jobid=J TimeLimit=.. / CpusPerTask + MinCPUsNode + NumCPUs
  (one per call) / TresPerNode=gres/gpu:<type>:2 / MinMemoryNode=<MB>`; moving to `nova` needs
  `QOS=normal Account=mech-ai Partition=nova` in ONE call. Alternative GPU types via `Features=a|b`
  made estimates worse — name one type. `nova` jobs count against `mech-ai`'s 17-GPU cap (shared
  with labmates; the user chose to put only one run there).
- The user's standing rules: never `scancel`/kill their jobs (probes that cannot start are shrunk to
  2 CPU / 1 min so they run and exit instead); act without asking; commit and push to `origin/main`.

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

**2. E8 baselines now exist** — see "E8 results" in the handoff section below.

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
    rgb.png              # 224-ready RGB render (grey background; data version 1)
    rgb_nobg.png         # RGBA, transparent background (data version 2, the default since 2026-10-01)
    depth.png            # big-endian packed RGBA depth
    *_nc_cam.ply         # camera-frame, normals-cleaned point cloud
    *_spline.yml         # 4M only — procedural generation params (since 2026-09-30: width, no waviness)
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
| `slurm/e2_arm_blackwell.sbatch` | `slurm/train_maize.sbatch` | ✅ |
| `slurm/scale_arm_blackwell.sbatch` | `slurm/scale_arm_maize.sbatch` | ✅ |
| `configs/config_e3_*` / `config_e4_*` | `configs/config_maize_e3_*` / `config_maize_e4_*` | ✅ |
| `eval/linear_probe.py` | `eval/linear_probe_maize.py` | ✅ |
| `eval/latent_analysis.py` | `eval/latent_analysis_maize.py` | ✅ |
| `train_sorghum_4m_distill.py` | `train_maize_4m_distill.py` | ✅ built + reviewed, not yet run |
| `configs/config_4m_distill_15k_all.yaml` | `configs/config_maize_distill_all.yaml` | ✅ |
| (sorghum distill launchers) | `slurm/distill_maize.sbatch` | ✅ |

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
