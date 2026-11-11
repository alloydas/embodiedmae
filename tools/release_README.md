# CropCode

Code for **CropCode: Inverse Procedural Modeling of Occluded Row Crop Geometry from Sparse Multimodal
Field Observations** (CVPR 2027 submission).

CropCode is one multimodal masked autoencoder over four streams of a plant: a photograph, a depth image,
a point cloud and the procedural parameters that generated the plant (the "recipe"). Any subset of the
streams can be given; the model writes the others. It is pretrained on synthetic sorghum and maize,
distilled so that a single stream reconstructs all four, and fine-tuned on crop rows where neighbours
occlude the centre plant.

The Python modules keep the `embodied_mae` prefix of the masked-autoencoder backbone they grew from; the
model in the paper is `EmbodiedMAE4M` (`embodied_mae_4m.py`) and its maize subclass
`EmbodiedMAE4MMaize` (`embodied_mae_4m_maize.py`).

## Layout

| path | contents |
|---|---|
| `embodied_mae.py`, `embodied_mae_4m.py`, `embodied_mae_4m_maize.py` | backbone, the four-stream model, the maize parameter width |
| `pc_sinkhorn.py` | debiased Sinkhorn term added to the point-cloud loss |
| `sorghum_dataset*.py`, `maize_dataset_4m.py`, `row_dataset.py` | loaders for single plants (both species) and crop rows |
| `species_conventions.py` | depth encodings of the four renderers, used for cross-species evaluation |
| `train_sorghum_4m.py`, `train_maize_4m.py` | pretraining (headline model, ablation arms) |
| `train_sorghum_4m_distill.py`, `train_maize_4m_distill.py` | cross-modal distillation (generalist and single-source students) |
| `train_row_4m.py` | row fine-tune and its isolated-plant control (Regime III) |
| `train_supervised_4m.py`, `train_supervised_maize_4m.py` | supervised-from-scratch reference of Table 10 |
| `eval/` | every evaluation behind the tables (list below); `eval/baselines/` holds the frozen-encoder adapters |
| `configs/` | one YAML per run reported in the paper |
| `data_split/` | split construction and the target tables the probes read |

## Environment

```bash
conda env create -f environment.yml
conda activate cropcode
```

Tested with Python 3.12, PyTorch 2.5 and CUDA 12.4 on A100, L40S, H200 and RTX PRO 6000 GPUs (the last
needs a CUDA 12.8 PyTorch build). `geomloss` is needed only for runs with a Sinkhorn weight above 0.

## Data

Put the datasets under `data/` (every config and script reads these relative paths):

```
data/Sorghum_15K/{train,val,test}/Sorghum_<plant>_<view>/
    rgb_nobg.png  depth.png  Sorghum_<plant>_nc_cam.ply  Sorghum_<plant>_spline.yml
data/Maize/{train,val,test}/plant_<plant>_<view>/
    rgba.png  depth.png  pointcloud_cam.ply  maize_<plant>_spline.xml  camera_pose.json
data/sorghum_row/{train,val,test}/Sorghum_<plant>/   and   data/maize_row/{train,val,test}/plant_<plant>/
    row.json  theta_<yaw>/shot_<k>/{rgba,depth,gt_rgba,gt_depth}.png
    pointcloud_cam.ply  gt_pointcloud_cam.ply  camera_pose.json  <recipe file>
```

Each species has 15,000 plants with ten views each, split 70/15/15 by plant (10,500 / 2,250 / 2,250).
`data_split/assignment.csv` is the sorghum split (`data_split/make_split.py` rebuilds it); the maize split
ships with the data (`plant_scores.csv`). Loaders build a folder index on first use and cache it.

## Training

Every config records the batch shape its run used: per-GPU batch x GPUs x accumulation steps gives the
global batch of the paper (32 for ablation arms and row fine-tunes, 256 for the headline pretraining, 128
for distillation). Launch with as many processes as the config's `distributed.world_size`:

```bash
# headline pretraining (1,000 epochs over all views)
torchrun --standalone --nproc_per_node=2 train_sorghum_4m.py --config configs/config_4m_pretrain15k_d2_sink.yaml
torchrun --standalone --nproc_per_node=4 train_maize_4m.py   --config configs/config_maize_4m_pretrain_15k_d2.yaml

# an ablation arm (197,400 optimizer steps each)
torchrun --standalone --nproc_per_node=2 train_sorghum_4m.py --config configs/config_e2_pcrgbd_d2.yaml

# distillation from the headline checkpoint (generalist, then single- and two-stream students)
torchrun --standalone --nproc_per_node=2 train_sorghum_4m_distill.py --config configs/config_4m_distill_15k_all_d2.yaml
torchrun --standalone --nproc_per_node=2 train_maize_4m_distill.py   --config configs/config_maize_distill_e1_d2.yaml

# Regime III: row fine-tune and the isolated-plant control, warm-started from the headline model
torchrun --standalone --nproc_per_node=2 train_row_4m.py --config configs/config_sorghum_row_ft.yaml
torchrun --standalone --nproc_per_node=2 train_row_4m.py --config configs/config_sorghum_row_iso.yaml
```

Runs write to `outputs/<run>/` (`checkpoints/`, `best_model.pth`, `config.json`, `training_history.json`)
and resume from `checkpointing.resume`. The ablation arms are compared at their last checkpoint
(`checkpoint_epoch_600.pth`; E3: 6168 / 2024 / 624), never at `best_model.pth`.

### Config names

| prefix | experiment |
|---|---|
| `4m_pretrain15k_d2_sink`, `maize_4m_pretrain_15k_d2` | headline model (E1) |
| `e2_*`, `maize_e2_*`, `maize_4m_d2` | which streams are present (Table 7, supplement Table 12) |
| `e2_pcrgbdt_tg*`, `maize_4m_tg*` | token-budget controls for the recipe stream (Table 8) |
| `e3_*`, `e4_*` (and `maize_`) | data scale and model scale (Table 11, supplement Table 13) |
| `e5_*` (and `maize_`) | camera-elevation regime (Regime II) |
| `e8_supervised_d2`, `maize_e8_supervised_d2` | supervised reference (Table 10) |
| `4m_distill_15k_*`, `maize_distill_*` | distillation: generalist, RGB students, two-stream students, recipe specialists |
| `*_row_ft*`, `*_row_iso*` | Regime III row fine-tune and isolated control |
| `*_xfer_*` | cross-species warm start vs scratch |

## Evaluation (paper table -> script)

| table / section | scripts |
|---|---|
| Regime I, single plant (Table 3) | `eval/regime1_eval.py`, `eval/regime1_extra.py` |
| Regime II, partial clouds and views (Table 4) | `eval/pc_view_masks.py`, `eval/pc_view_eval.py`, `eval/pc_view_eval_maize.py`, `eval/pc_view_bootstrap.py`; held-out views: `eval/linear_probe.py --probe-view k` |
| Regime III, crop rows (Table 5) | `eval/eval_row.py`, `eval/row_pair_bootstrap.py` |
| Any-subset generation and distillation (Table 6) | `eval/eval_warmstart.py`, `eval/eval_rgb2pc_quant.py`, `eval/eval_specialists.py`, `eval/pc_fscore.py`, `eval/pc_fscore_maize.py` |
| Stream ablation and controls (Tables 7, 8) | `eval/linear_probe.py`, `eval/linear_probe_maize.py`, `eval/probe_pair_bootstrap.py` |
| Frozen-encoder comparison (Table 10) | `eval/baseline_probe.py` with `eval/baselines/`; reconstruction companion `eval/recon_compare.py` |
| Representation analysis | `eval/latent_analysis.py`, `eval/latent_analysis_maize.py` |
| Cross-species | `eval/cross_species_probe.py`, `eval/eval_rgb2pc_quant.py --model_species/--data_species`, `eval/eval_row.py --model-species` |
| Real maize rows (Table 9) | `eval/real_rows.py`, `eval/eval_real_rows.py` |

Examples:

```bash
python eval/linear_probe.py --runs e2_pc_d2 e2_pcrgb_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2 --ckpt checkpoints/checkpoint_epoch_600.pth
python eval/probe_pair_bootstrap.py --a e2_pcrgbdt_d2 --b e2_pcrgbd_d2 --feature cls
python eval/regime1_eval.py --device cuda
python eval/eval_row.py --species sorghum --run sorghum_row_ft --ckpt checkpoints/checkpoint_epoch_100.pth --split test
python eval/baseline_probe.py --species sorghum --baselines pointmae multimae dinov2_vitb14
```

The probe (Table 10 and the stream ablation) is a ridge regression on frozen features: one deterministic
view per plant, the standardiser fitted on train plants only, the ridge strength chosen by 5-fold
cross-validation, scored as validation R^2. The recipe stream is never visible at probe time and its
tensor is zeroed, because the plant token carries the height target verbatim. Features are cached under
`outputs/_probe_cache/`.

The frozen baselines need their public weights: DINOv2 ViT-B/14, MultiMAE-B and Point-MAE (paths in
`eval/baselines/__init__.py`; `eval/recon_compare.py` also needs a clone of the MultiMAE repository at
`third_party/MultiMAE`). Their model code is vendored under `eval/baselines/_vendor_*` with the upstream
licences.

## Notes

- Point-cloud metrics are computed in the unit-sphere frame of each plant (`chamfer`, squared, symmetric)
  and converted to millimetres per plant with its metres-per-unit scale where the paper reports mm.
- Reconstruction is always evaluated in the masked regime the model was trained in; with every token
  visible the decoder is out of distribution.
- Sorghum and maize keep separate loaders and parameter layouts (`N_PARAMS` 9 vs 14, 24 vs 28 leaf tokens,
  8,196 vs 8,192 points).
