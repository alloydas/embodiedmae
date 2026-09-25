# Delta transfer manifest

Everything NCSA Delta needs from Nova to (a) train the control arm `e2_pcrgbdt_tg`
from scratch, (b) probe + E9 every sorghum run at its matched cut, and (c) probe
+ E9 every maize run at its matched cut. `slurm/delta/transfer_from_nova.sh` moves
it, so run it on Nova.

**Measured on Nova at 2026-09-24 ~18:40 CDT.** Maize status goes stale within
hours. `bash slurm/delta/transfer_from_nova.sh status` reprints the checkpoint
table live, and it only reads.

## What each job reads (from the code, not assumed)

| job | reads | source |
|---|---|---|
| (a) train `e2_pcrgbdt_tg` | train **and val and test** splits: `rgb.png`, `depth.png`, `*_nc_cam.ply`, `*_spline.yml` per folder. No checkpoint: `resume: null` | `train_sorghum_4m.py:667-679` builds all three datasets. `sorghum_dataset.py:177-181`, `sorghum_dataset_4m.py:40` |
| (b) sorghum probe | the same 4 files, all 3 splits (extracts train, val and test, and indexes each whole split even though only view `_00` is opened). `features.csv` + `assignment.csv` at the root. `outputs/<run>/config.json` (model_size, active_modalities, num_points, max_leaves) + the checkpoint | `eval/linear_probe.py` `load_targets`, `build_model`, `run_one` |
| (b) sorghum E9 | the probe cache `outputs/_probe_cache/<run>__<ckpt stem>__<split>__…npz` + the root CSVs. It needs a checkpoint + config.json **only if the cache is cold** | `eval/latent_analysis.py:347-356` |
| (c) maize probe | `rgb.png`, `depth.png`, `pointcloud_cam.ply`, `maize_*_spline.xml`, all 3 splits. `plant_scores.csv` at the root. config.json + the checkpoint | `maize_dataset_4m.py:160-170`, `eval/linear_probe_maize.py` |
| (c) maize E9 | `outputs/_probe_cache_maize/maize_<run>__…npz` + `plant_scores.csv` | `eval/latent_analysis_maize.py:270-279` |

**Not needed, and not sent:** `best_model.pth` (anything), `outputs/maize_4m_1000ep`,
`training_history.json`, and the visualisations. Sorghum's `Sorghum_<id>.obj`,
`Sorghum_<id>_nc.ply` and `Sorghum_<id>.yml` duplicates are also left out. So are
`camera_pose.json` (no loader or probe opens it, only the E5 view scripts do, so
add it later with `WITH_CAMERA_POSE=1`), maize's per-split `_params.json` (134 MB
across the three splits), `plant_params.jsonl` (134 MB) and `params_index.json`.
`summary.json` (4.5 KB) *is* sent even though no code reads it, because it is the
only record that the maize split is seed 0 / Mahalanobis rather than sorghum's
seed 42 / extremeness.

## Items

| # | item | subcommand | files | size | status |
|---|---|---|---|---|---|
| 1 | repo working tree, with uncommitted edits | `code` | 274 | ~0.03 GB | ready |
| 2 | sorghum checkpoints + config.json | `ckpts-sorghum` | 18 | **13.83 GB** | all 9 ready |
| 3 | Sorghum_15K, filtered | `data-sorghum` | 600,000 + 2 CSVs | **382 GB ± 16** | ready |
| 4 | probe feature caches -> `outputs/_probe_cache_nova/` | `caches` | 12 | 0.17 GB | ready (optional, see note) |
| 5 | Maize, filtered | `data-maize` | 600,000 + 2 | **68.5 GB ± 1.7** | ready |
| 6 | maize checkpoints + config.json | `ckpts-maize` | 12 | **9.76 GB** (3.04 ready now) | 3 of 6 ready |
| | **total** | `all` | | **~474 GB** (data ~450, checkpoints ~23.6) | |

Data sizes come from sampling, not from a tree walk (that would hang Lustre):
random folders taken from the loaders' own index caches, 40 per split. Sorghum
averages **2.548 MB per folder** for the 4 files (SE 0.055) against 14.2 MB for
the whole folder. Maize averages 456 KB for the 4 files against 458 KB whole. The
± figures are 95 % intervals.

> **CLAUDE.md's sorghum numbers are stale.** It says ~1.8 MB/folder read and
> ~265 GB filtered, and ~5 MB/folder and ~750 GB unfiltered. Sorghum_15K as it
> is on disk now measures 2.55 MB and **~382 GB** filtered, and 14.2 MB and
> **~2.1 TB** unfiltered (8 files per folder, and the 15.9 MB `.obj` is the bulk of
> it). Size the Delta quota for ~390 GB sorghum + ~70 GB maize. The filter still
> sends only 18 % of the bytes.

### Checkpoints: the matched cut, per run

Always `outputs/<run>/checkpoints/checkpoint_epoch_<N>.pth` + `outputs/<run>/config.json`.

| run | N | GB | status |
|---|---|---|---|
| e2_pc | 600 | 1.348 | ready |
| e2_pcrgb | 600 | 1.361 | ready |
| e2_pcrgbd | 600 | 1.365 | ready |
| e2_pcrgbdt | 600 | 1.369 | ready (also E3 full-data point and E4 base point) |
| e4_small | 600 | 0.301 | ready |
| e4_large | 600 | 3.983 | ready |
| e3_1k | 6168 | 1.369 | ready |
| e3_3k | 2024 | 1.369 | ready |
| e3_10k | 624 | 1.369 | ready |
| maize_4m | 600 | 1.369 | ready. Frozen: the 1000-epoch continuation writes elsewhere |
| maize_e3_1k | 6168 | 1.369 | ready. Job COMPLETED 15:14 |
| maize_e4_small | 600 | 0.301 | ready. Job COMPLETED 17:07 |
| maize_e3_10k | 624 | ~1.369 | **pending**: at 468 (18:37), 26 epochs per ~34 min → **~22:00 CDT Thu 09-24** |
| maize_e4_large | 600 | ~3.983 | **pending**: at 325 (18:04), 25 epochs per ~44 min → **~02:15 CDT Fri 09-25** |
| maize_e3_3k | 2024 | ~1.369 | **pending, and preempted.** Last checkpoint 968 (15:19). Requeued 15:40 (`Restarts=1`), now `PENDING (Priority)`. It auto-resumes from 968 and needs ~8 h of running (12 saves × ~40 min) once it starts again. Slurm's *worst-case* StartTime is Sun 09-27 15:39. **No firm ETA.** The maize E3 curve waits on this run |

Sizes of pending checkpoints are the size of the same run's latest checkpoint;
within a run they differ by kilobytes. The ETAs assume no further preemption on
`scavenger`.

`ckpts-maize` skips anything not yet written and names the latest epoch on disk.
It also skips a checkpoint written less than `MIN_CKPT_AGE_S` (300 s) ago or one
that does not open as a complete zip. That second check is needed because
`torch.save` writes directly to the final name, so a live run's checkpoint can be
caught mid-write. Re-run the script after each ETA. rsync resends nothing that has
already arrived.

## Order to transfer in

The critical path is the sorghum data. The control arm is the longest compute
(197,400 steps), it cannot start until all three sorghum splits have landed, and
no sorghum probe can run without them either.

1. **`code`**, which takes seconds. Alternatively, commit + push and `git clone` on
   Delta, then run `code` anyway to overlay whatever is still uncommitted.
   `slurm/delta/delta_env.sh` is gitignored and never sent, so create it on Delta
   from the example.
2. **Sorghum data.** Globus is preferred for this size:
   `bash slurm/delta/transfer_from_nova.sh globus-list data-sorghum > sorghum.batch`
   prints the batch and the exact `globus transfer` command. It survives a dropped
   session and checksums every file. An rsync that loses its MFA'd ssh
   mid-tree has to rescan 150,000 folders to resume. If you use rsync
   (`data-sorghum`) instead, first prove the filter on one folder:
   `DRY_RUN=1 ONLY_SAMPLE=val/Sorghum_10001_00 bash … data-sorghum`.
3. **While it runs:** `ckpts-sorghum` (13.8 GB) and `caches` over rsync. The
   default `DELTA_SSH` multiplexes one Duo-authenticated connection for 2 h.
4. **Maize data** (68.5 GB) using `globus-list data-maize` or `data-maize`.
5. **`ckpts-maize`** now (3 runs), again after ~22:00 Thu (`maize_e3_10k`),
   after ~02:15 Fri (`maize_e4_large`), and once `maize_e3_3k` finishes.

`all` runs code → ckpts-sorghum → caches → data-sorghum → data-maize → ckpts-maize
serially over rsync. Use it only if Globus is not an option.

## Things that do not transfer, and what they cost Delta

- **The dataset index cache (`.dataset_index_cache/`) is not portable.** Its
  entries are relative folder names, so the content would travel. The key does
  not: the file name is `sha256(resolved absolute split path)` and a hit also
  requires the split directory's `st_mtime_ns` to match (`sorghum_dataset.py:41-60`,
  `maize_dataset_4m.py:74-94`). A different root path on Delta means no Nova entry
  can ever hit, so none is shipped. The first Delta job to open each split pays
  the scan once and writes it atomically to `$REPO/.dataset_index_cache/`. On Nova
  that took ~24 min for sorghum train, ~5 min for val, and ~20 min for maize train.
  Avoid starting the control arm and a probe cold at the same moment: both would
  scan (safe, just doubled metadata load). `python sorghum_dataset_4m.py
  $SORGHUM_ROOT` pre-warms train and val only, not test.
- **The probe cache is portable but mostly irrelevant, so it lands out of the
  way.** Its key is `<run>__<ckpt stem>__<split>__…` with no paths inside. Every
  file there today is `best_model`-keyed, so a matched-cut probe (`--ckpt
  checkpoints/checkpoint_epoch_N.pth`) never hits it -- but `eval/linear_probe.py`
  *defaults* `--ckpt` to `best_model.pth`, and a hand-run probe in the default
  cache dir would silently mix Nova-extracted features into a Delta table. So
  `caches` writes `outputs/_probe_cache_nova/` on Delta, which no default reads.
  To reproduce the old E2 numbers without a GPU, pass `--cache-dir
  outputs/_probe_cache_nova --ckpt best_model.pth` explicitly.
  `outputs/_probe_cache_maize/` does not exist yet.
- **Results come back with `pull-results`** (run on Nova): the control arm's
  `config.json`, `training_history.json` and epoch-600 checkpoint into
  `outputs/e2_pcrgbdt_tg/`, `reports/probe_*.csv` and `reports/e9_*/` into
  `reports/delta/`, and `logs/delta_*` into `logs/delta/`.

## After it lands: what to check on Delta

- Sorghum: 105,000 / 22,500 / 22,500 folders in train / val / test, 4 files each.
  Maize: the same counts, 4 files each. Every Delta batch script checks this
  twice before it starts: the folder count, then the loader's own index
  (`slurm/delta/check_split_index.py`), because a folder that exists before its
  files do is skipped by the index scan and the short index is then cached.
  Submit only after the Globus task reports SUCCEEDED (or rsync exits 0).
- Checkpoint byte sizes must equal the table above. For a quick integrity check,
  `python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1])" <ckpt>`.

## Open issues for whoever runs this

- **`maize_e3_3k` was preempted and has no ETA.** See the table above.
- **The E2 probe R² in CLAUDE.md were not at a matched cut.** They come from the
  `best_model` caches. From the `training_history.json` val-loss argmin, those are
  epochs 575 / 525 / 600 / 525 for `e2_pc` / `e2_pcrgb` / `e2_pcrgbd` /
  `e2_pcrgbdt`. LR is ~0 there, so the numbers should barely move, but the
  checkpoint-600 probe on Delta supersedes them.
- **Nova paths are baked in where the Delta scripts must override them.**
  `eval/linear_probe.py` and `eval/latent_analysis.py` default `--data-root` to
  the Nova sorghum root, the maize twins default to `/work/mech-ai-scratch/alloy/Maize`,
  and every YAML's `data.data_root` is Nova's. So Delta's `probe*.sbatch` must pass
  `--data-root`, and `train_arm.sbatch` must pass `--data_root` (the trainer's CLI
  flag). `config.json` also records Nova's `data_root`, which is harmless because
  neither probe reads it.
- **The Globus filter flags and `#` comments in a batch file** match the
  `globus-cli` 3.x behaviour as best known. Check `globus transfer --help` on the
  version you have. Sorghum uses an exclude-list rather than an include-list,
  so if a rule misfires it sends too much, never too little.
- **A probe-only maize copy could be 10× smaller.** The probe opens only view
  `_00` (6.9 GB instead of 68.5 GB). This is not offered: a one-view copy that
  anyone later trains on gives one view per plant, silently.
