# Running the remaining jobs on NCSA Delta

What moves to Delta: the **E2 control arm `e2_pcrgbdt_tg`** (the one training run
left), and the **linear probe + E9 latent analysis for every sorghum and maize run
at its matched cut**. Nova keeps its live maize runs and job 16576022; nothing here
cancels or replaces them.

Every command below goes through one script, `slurm/delta/submit.sh`. It reads
`slurm/delta/delta_env.sh`, `cd`s to `REPO_DIR`, and passes account, partition
and job name on the `sbatch` line. `DRY_RUN=1` in front of any line prints the
`sbatch` command and submits nothing.

## 1. One-time setup on Delta

```bash
ssh <you>@login.delta.ncsa.illinois.edu
accounts                                     # your allocations; GPU ones end in -delta-gpu
quota                                        # where there is room (see sizes below)
sinfo -s                                     # partition names
sinfo -p gpuA100x4 -o "%P %c %m %G %l"       # cores, MB, GPUs, time limit per node
sinfo -p gpuA40x4  -o "%P %c %m %G %l"
```

Space you need: ~390 GB for sorghum (filtered), ~70 GB for maize, ~24 GB of Nova
checkpoints, and ~35 GB for the control arm's own checkpoints. See `MANIFEST.md`.

The repo: `git clone git@github.com:alloydas/embodiedmae.git` (the trainer's
`text_mask_ratio` wiring is committed). `transfer_from_nova.sh code` (section 2)
is only needed to carry uncommitted Nova edits. If the trainer is stale,
`train_arm.sbatch` refuses the control arm and tells you why.

```bash
git clone git@github.com:alloydas/embodiedmae.git $REPO_DIR && cd $REPO_DIR
conda env create -f environment.yml          # builds `det`: torch 2.5 + CUDA 12.4, has sm_80/sm_86
conda activate det && python embodied_mae_4m.py   # CPU smoke test of the model
cp slurm/delta/delta_env.example.sh slurm/delta/delta_env.sh   # gitignored
$EDITOR slurm/delta/delta_env.sh             # every CHANGEME; submit.sh refuses until they are gone
```

If `conda` is not on your PATH, use `module avail` to find a conda/miniforge
module, or install Miniforge under `/projects`. Then set `CONDA_SH` to its
`etc/profile.d/conda.sh`. If compute nodes cannot reach `api.wandb.ai`, set
`WANDB_MODE=offline` in `delta_env.sh` and run `wandb sync` from a login node
afterwards. Otherwise run `wandb login` once.

**Hardware assumptions to check** (Delta's documentation, not measured). The
only copies are the `#SBATCH` lines in the three batch scripts. Edit them there:

| script | asks for | check |
|---|---|---|
| `train_arm.sbatch` | 2 GPUs, 64 cores, 240G, 48:00:00, `--requeue` | a node really has 64 cores / ≥240 GB, and the wall limit is 48 h. **Also check that a 2-GPU job may hold all 64 cores, and how Delta bills that.** The run is limited by data loading, so the cores matter more than the GPUs (see §4) |
| `probe.sbatch`, `probe_maize.sbatch` | 1 GPU, 16 cores, 64G, 04:00:00 | a quarter of a gpuA40x4 node |

A job that gets fewer GPUs or cores than it asked for fails at startup: its CUDA
check runs a kernel on each GPU and checks the step's CPU affinity. It does not
quietly run slower.

## 2. Transfer from Nova

Run these **on Nova**. What each job reads, the sizes, and the order are in
`MANIFEST.md`.

```bash
export DELTA_USER=<delta-login>
export DELTA_REPO=/projects/<alloc>/<you>/embodiedmae
export DELTA_SORGHUM_ROOT=/projects/<alloc>/<you>/data/Sorghum_15K     # = SORGHUM_ROOT in delta_env.sh
export DELTA_MAIZE_ROOT=/projects/<alloc>/<you>/data/Maize              # = MAIZE_ROOT
bash slurm/delta/transfer_from_nova.sh status          # what is ready; sends nothing
bash slurm/delta/transfer_from_nova.sh code            # optional: only if Nova has uncommitted edits
bash slurm/delta/transfer_from_nova.sh globus-list data-sorghum > sorghum.batch   # prints the globus command too
bash slurm/delta/transfer_from_nova.sh ckpts-sorghum   # 9 runs, 13.8 GB, while Globus runs
bash slurm/delta/transfer_from_nova.sh globus-list data-maize > maize.batch
bash slurm/delta/transfer_from_nova.sh ckpts-maize     # now, then again after each Nova run finishes
bash slurm/delta/transfer_from_nova.sh caches          # optional: old best_model features -> outputs/_probe_cache_nova
```

The data go by Globus (Nova endpoint to Delta endpoint, the batch files above).
Globus survives a dropped session and checksums every file. The rsync path
(`data-sorghum` / `data-maize`) also works. Test its filter on one folder first:
`DRY_RUN=1 ONLY_SAMPLE=val/Sorghum_10001_00 bash slurm/delta/transfer_from_nova.sh data-sorghum`.

**Submit nothing until the Globus task reports SUCCEEDED (or rsync exits 0).** A
transfer creates each folder before it fills it. The batch scripts check this in
two ways: the folder count, then the loader's own index
(`slurm/delta/check_split_index.py`). A job started too early refuses with exit 3.
It does not leave a short index cached for later jobs to reuse.

## 3. The jobs, in order

```bash
# 1. Smoke test of the control arm: 1 epoch, its own dir, no W&B (~10-40 min + a cold index scan)
bash slurm/delta/submit.sh train e2_pcrgbdt_tg --epochs 1 --no_wandb --output_dir outputs/_smoke_tg

# 2. The control arm (after the smoke test has FINISHED -- submit.sh refuses a second tr_e2_pcrgbdt_tg)
bash slurm/delta/submit.sh train e2_pcrgbdt_tg

# 3. Sorghum probe + E9, one job per matched-cut epoch. Alongside 2 is fine: the smoke
#    test already warmed the index cache, so nothing scans the tree twice.
bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_600.pth  e2_pc e2_pcrgb e2_pcrgbd e2_pcrgbdt e4_small e4_large
bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_6168.pth e3_1k
bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_2024.pth e3_3k
bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_624.pth  e3_10k

# 4. The control arm's probe, once outputs/e2_pcrgbdt_tg/checkpoints/checkpoint_epoch_600.pth exists
#    (e2_pcrgbd/e2_pcrgbdt are cache hits from step 3; they are here so all three land in one table)
bash slurm/delta/submit.sh probe checkpoints/checkpoint_epoch_600.pth  e2_pcrgbd e2_pcrgbdt e2_pcrgbdt_tg

# 5. Maize probe + E9. Ready now (transferred by ckpts-maize):
bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_600.pth  maize_4m maize_e4_small
bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_6168.pth maize_e3_1k
#    (maize_e3_1k and maize_e4_small finished on Nova 2026-09-24.)
#    After each remaining Nova run finishes AND a re-run of ckpts-maize has sent it:
bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_624.pth  maize_e3_10k                      # Nova ETA ~22:30 Thu 09-24
bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_600.pth  maize_4m maize_e4_small maize_e4_large   # ~02:30 Fri 09-25
bash slurm/delta/submit.sh probe_maize checkpoints/checkpoint_epoch_2024.pth maize_e3_3k                       # no ETA: preempted at 1017/2100, requeued
```

Each probe job takes one `--ckpt`. `submit.sh` refuses any run that is not at its
matched cut, any `best_model.pth`, `maize_4m_1000ep`, and a run of the wrong
species. The matched cuts are the last *saved* epoch, so the E3 cuts fall short
of 197,400 steps. Quote the step count next to each R²:

| run | cut | steps |
|---|---|---|
| e2_*, e2_pcrgbdt_tg, e4_*, maize_4m, maize_e4_* | 600 | 197,400 |
| e3_1k, maize_e3_1k | 6168 | 197,376 |
| e3_3k, maize_e3_3k | 2024 | 190,256 (96.4 %) |
| e3_10k, maize_e3_10k | 624 | 195,312 (98.9 %) |

When results exist, bring them back **on Nova** with
`bash slurm/delta/transfer_from_nova.sh pull-results`. It fetches the control arm's
`config.json`, `training_history.json` and epoch-600 checkpoint into
`outputs/e2_pcrgbdt_tg/`, the probe CSVs and E9 folders into `reports/delta/`, and
the logs into `logs/delta/`.

## 4. After each job

Logs are `logs/delta_<jobname>_<jobid>.{out,err}`. They are appended across
requeues, so one file holds the whole run. tqdm's s/it is in `.err`.

**Smoke test and control arm.** The `.out` must show:

```
split train: 105000/105000 ok            (and val, test 22500)
index train: 105000 samples, 10500 plants x 10 views ok     (and val, test)
Batch   : 16/GPU x 2 GPUs = 32 global
  cuda:0 NVIDIA A100-SXM4-40GB sm_80 ... kernel ok          (and cuda:1)
Multi-GPU (torchrun)  GPUs=2  batch/GPU=16  total=32
Text masking: independent at 0.8, outside the Dirichlet budget
Total parameters: 114,280,716
```

Then check the run's own record:

```bash
python -c "import json; c=json.load(open('outputs/e2_pcrgbdt_tg/config.json')); \
print(c['text_mask_ratio'], c['batch_size']*c['world_size'], c['active_modalities'], c['pc_loss_name'], c['epochs'])"
# expect: 0.8 32 None qal_loss 600
```

`active_modalities: null` means all four streams. There is no `loss_name` key:
`pc_loss_name` is the recorded field, and CLAUDE.md is wrong on both points.
Batch 16 needs ~22.4 GiB per GPU (measured on Nova, `logs/bsweep_12044382.out`)
against the A100's 40 GiB. An OOM would fail at the first step of the smoke
test, so it cannot go unnoticed.

**Probe jobs.** You get `reports/probe_pr_<epoch>_<jobid>.csv` (maize:
`probe_pm_…`) and `reports/e9_pr_<epoch>_<jobid>/`. Each log hashes the eval
scripts it ran. `E9 FAILED` with `cache missing` means the probe itself failed:
read the probe traceback above it.

**TIMEOUT vs requeue.** `sacct -j <id> -o JobID,State,Elapsed,ExitCode`:
- `PREEMPTED`/`NODE_FAIL`, then `PENDING`/`RUNNING` under the same id: Slurm
  requeued it (`--requeue`). Do nothing. It resumes from its newest readable
  checkpoint.
- `TIMEOUT`: nothing resubmits it. Re-run the same `submit.sh train e2_pcrgbdt_tg`
  line. The launcher resumes automatically and loses at most `save_freq` = 25
  epochs.
- Before re-running, check `squeue -u $USER`. `submit.sh` refuses a second
  `tr_<slug>` while one is queued or running, because two trainers in one output
  dir overwrite each other's checkpoints.

**Expected runtimes.** These are estimates, not Delta measurements:
- **Control arm, ~18 h.** The rule is CLAUDE.md's: data loading sets the speed
  at ~1.6 items/s per worker. 2 ranks × 31 workers ≈ 99 items/s gives ~106 s per
  10,500-plant epoch, so 600 epochs ≈ 17.7 h, plus 24 val passes. Nova's E2
  Blackwell run matched that rule: 76 workers predicted 122 items/s and it
  measured 3.83 it/s × 32 = 122.6. Nova's only A100 run of this model
  (`logs/e2_e2probe_13175339`: 4× A100-80GB, 8/GPU, 32 cores) took 3:50–4:16 per
  epoch, but it had half the cores, so it tells you nothing about how fast an
  A100 computes this model. **Nova has never measured A100 compute speed at
  16/GPU (fp32, no AMP).** Take the smoke test's s/it × 329 as the real epoch
  time. Above ~4.7 min/epoch, 600 epochs no longer fit in 48 h. The job then
  ends in `TIMEOUT` once and you re-run the same line. `TRAIN_GPUS=4` in front
  of the train line gives 4 GPUs (8 × 4) at the price of the shape caveat
  below. Delta's file system may also feed workers slower than Nova's Lustre.
- **Cold index scan: ~35 min** before the first job's work starts (Nova: sorghum
  train ~24 min, val and test ~5 min each; maize train ~20 min). It happens once
  per data root. Later jobs print `(cache hit)`.
- **Probe:** ~15 min for four arms cold on one GPU (CLAUDE.md), then E9 on CPU
  at ~3.4 min per run over three splits. The 4 h wall allows the 6-arm job plus
  a cold index scan.

**Reading the control-arm result** (details in the header of
`configs/config_e2_pcrgbdt_tg.yaml`):
- Compare it by probe R² against `e2_pcrgbd` and `e2_pcrgbdt`, all at 600. The
  probe is the clean comparison: the text stream is never visible and the
  params are zeroed. Val `pc_chamfer` is secondary, because the tg arm still sees
  5 real text tokens at val. Never compare `param_*` or total loss.
- **Two variables move against `e2_pcrgbdt`, not one.** The vision budget goes
  from 107.4 to 117 tokens (exactly `e2_pcrgbd`'s), and text visibility goes from
  ~58 % to 20 %. If tg ≈ pcrgbd, you cannot tell which change mattered. If tg ≈
  pcrgbdt, the token handicap was not the cause.
- **tg is the only sorghum arm trained off Nova.** The per-GPU shape is matched
  (16 × 2, so per-rank BatchNorm and Dirichlet draws match the references). The
  stack is not: A100 + torch 2.5/cu124 here, against RTX PRO 6000 + torch
  2.11/cu128 for the references. Training is not bit-reproducible even on one
  stack (FPS `randint`, unseeded point subsampling). The one run-to-run error bar
  in the matrix (`e3_10k` vs `e2_pcrgbdt`) is the yardstick for "inside the gap".
  If tg lands inside it, the clean reference is a Delta replicate of
  `e2_pcrgbdt` under a **new** slug and output dir. `train_arm.sbatch` refuses
  to "re-run" `e2_pcrgbdt` itself, because its transferred epoch-600 checkpoint
  would make that a no-op.
- `eval/linear_probe.py` builds every model ungated and asserts
  `text_mask_ratio is None`. That is correct for tg too, because
  `forward_encoder_select` never reads it. Do not "fix" the probe to forward
  `config.json`'s `text_mask_ratio`: the assert would fire on tg.

## 5. Not here, and why

- **E5 (view regime).** It needs tractor-view renders, and they do not exist
  yet.
- **E8 (baselines).** Being built in a follow-up commit (`eval/baseline_probe.py`,
  `train_supervised_4m.py`); this runbook gains its commands then.
- **E10 (real-data OOD).** It needs Maria's field data.
- **Maize 600→1000 continuation.** Job 16576022 stays queued on Nova and writes
  `outputs/maize_4m_1000ep`, which is comparable to nothing else. It is not
  probed.
- **Maize training on Delta.** No Delta launcher for `train_maize_4m.py` exists.
  The maize arms keep running on Nova. `maize_e3_3k` was preempted and is
  `PENDING` with no firm ETA.
- **E6/E7.** A collaborator owns them.
