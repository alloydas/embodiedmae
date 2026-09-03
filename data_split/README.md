# Sorghum_15K data split — extreme-enriched, reshuffle-ready

Tools + provenance for the train/val/test split of the large dataset at
`/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K`.

## Dataset
- **15,000 plants × 10 camera views = 150,000 sample folders** (8 files each:
  `rgb.png`, `depth.png`, `*_nc_cam.ply`, `*_spline.yml`, `.obj/.ply/.yml`).
- Folders named `Sorghum_<plant>_<view>` (view = 00..09).
- **Split by plant** — all 10 views of a plant stay in one split (no view leakage).

## The "trick": extreme-enriched split
Each plant is scored by **Mahalanobis distance from the population centroid** over
6 phenotype features `[n_leaves, stem_length, leaf_len_mean, leaf_len_max,
roll_mean, roll_std]`. Val/test are **probabilistically enriched** with extreme
(tail) plants so they test extrapolation, while train stays central.

- Assignment weight ∝ `rank_percentile ** GAMMA` (GAMMA=2), sampled without
  replacement into the val+test pool, then randomly halved. Seed = 42.
- Ratio 70/15/15 → **train 10,500 / val 2,250 / test 2,250 plants**
  (105k / 22.5k / 22.5k folders).
- Result: median extremeness train 2.03 vs val/test 2.76; top-decile-extreme
  share 4.7% / 21.6% / 23.0%.

## Files here
| file | what |
|---|---|
| `scan_all.py`   | parse 15k spline.yml (fast regex, skips geometry) → `features.csv` |
| `make_split.py` | features → Mahalanobis extremeness → enriched assignment → `assignment.csv` |
| `move_split.py` | initial move of flat folders into train/val/test (assumes flat source) |
| `reshuffle.py`  | **location-agnostic** re-apply of any `assignment.csv`; also `--flatten` |
| `plot_splits.py`| 8-panel per-split distribution figure |
| `assignment.csv`| current split: `plant,split,extremeness,rank` (also copied into the data dir) |
| `features.csv`  | per-plant feature matrix |

## Current on-disk state
Folders were **physically moved** (not symlinked) into
`Sorghum_15K/{train,val,test}/`. The source flat layout no longer exists;
`assignment.csv` is the record of truth.

## To reshuffle
```bash
cd data_split
# 1. change SEED / GAMMA / ratios in make_split.py, then:
python make_split.py          # rewrites assignment.csv (needs features.csv)
python reshuffle.py           # DRY RUN: shows how many folders move between splits
python reshuffle.py --go      # execute (only moves folders whose split changed)
```
`reshuffle.py` finds folders wherever they are (flat or in split dirs), so it is
safe to run repeatedly. `make_split.py` reuses `features.csv` — no need to
re-scan the 15k YAMLs unless the feature set changes (that needs `scan_all.py`,
~2 min on 16 cores).

⚠️ If a training job is reading these folders, **stop it first** — moving folders
out from under an active DataLoader will cause file-not-found errors.
