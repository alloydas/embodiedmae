# CropCode

Training and testing code for **CropCode: Inverse Procedural Modeling of Occluded Row Crop Geometry from
Sparse Multimodal Field Observations** (CVPR 2027 submission).

CropCode is one multimodal masked autoencoder over four streams of a plant: an image, a depth image, a
point cloud and the procedural parameters that generated the plant. Any subset of the streams can be
given; the model writes the others. It is pretrained on synthetic sorghum and maize, distilled so that a
single stream reconstructs all four, and fine-tuned on crop rows where neighbours occlude the centre plant.

The Python modules keep the `embodied_mae` prefix of the masked-autoencoder backbone they grew from; the
model in the paper is `EmbodiedMAE4M` (`embodied_mae_4m.py`) and its maize subclass
`EmbodiedMAE4MMaize` (`embodied_mae_4m_maize.py`).

## Layout

| path | contents |
|---|---|
| `embodied_mae.py`, `embodied_mae_4m.py`, `embodied_mae_4m_maize.py` | backbone, the four-stream model, the maize parameter width |
| `pc_sinkhorn.py` | debiased Sinkhorn term added to the point-cloud loss |
| `sorghum_dataset*.py`, `maize_dataset_4m.py`, `row_dataset.py`, `eval/real_rows.py` | loaders: single plants (both species), synthetic crop rows, rows of real plants |
| `species_conventions.py` | depth encodings of the renderers |
| `train_sorghum_4m.py`, `train_maize_4m.py` | pretraining (headline model and ablation arms) |
| `train_sorghum_4m_distill.py`, `train_maize_4m_distill.py` | cross-modal distillation (generalist and students) |
| `train_row_4m.py` | row fine-tuning and its isolated-plant control |
| `train_real_rows.py` | fine-tuning the row model on rows of real plants |
| `train_supervised_4m.py`, `train_supervised_maize_4m.py` | supervised-from-scratch reference |
| `eval/` | testing: the scripts that score trained models on held-out plants |
| `configs/` | one YAML per training run reported in the paper |
| `data_split/` | the sorghum split and the target tables the probes read (data only) |

## Environment

```bash
conda env create -f environment.yml
conda activate cropcode
```

Tested with Python 3.12, PyTorch 2.5 and CUDA 12.4 on A100, L40S and H200 GPUs (RTX PRO 6000 needs a
CUDA 12.8 PyTorch build). `geomloss` is needed only for runs with a Sinkhorn weight above 0.

## Data

Put the datasets under `data/` (every config and script reads these relative paths):

```
data/Sorghum_15K/{train,val,test}/Sorghum_<plant>_<view>/
    rgb_nobg.png  depth.png  Sorghum_<plant>_nc_cam.ply  Sorghum_<plant>_spline.yml
data/Maize/{train,val,test}/plant_<plant>_<view>/
    rgba.png  depth.png  pointcloud_cam.ply  maize_<plant>_spline.xml  camera_pose.json
data/sorghum_row/{train,val,test}/Sorghum_<plant>/   and   data/maize_row/{train,val,test}/plant_<plant>/
    row.json  theta_<yaw>/shot_<k>/{rgba,depth,gt_rgba,gt_depth}.png
    pointcloud_cam.ply  gt_pointcloud_cam.ply  camera_pose.json  <parameter file>
```

Each species has 15,000 plants with ten views each, split 70/15/15 by plant (10,500 / 2,250 / 2,250).
`data_split/assignment.csv` records the sorghum split; the maize split ships with the data
(`plant_scores.csv`). Loaders build a folder index on first use and cache it.

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

# row fine-tuning and the isolated-plant control, warm-started from the headline model
torchrun --standalone --nproc_per_node=2 train_row_4m.py --config configs/config_sorghum_row_ft.yaml
torchrun --standalone --nproc_per_node=2 train_row_4m.py --config configs/config_sorghum_row_iso.yaml

# fine-tuning the row model on 75 % of the real plants (one GPU)
python train_real_rows.py --species maize --init maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth --run maize_real_ft
```

Runs write to `outputs/<run>/` (`checkpoints/`, `best_model.pth`, `config.json`, `training_history.json`)
and resume from `checkpointing.resume`. Ablation arms are compared at their last checkpoint
(`checkpoint_epoch_600.pth`; data-scale arms: 6168 / 2024 / 624), never at `best_model.pth`.

### Config names

| prefix | experiment |
|---|---|
| `4m_pretrain15k_d2_sink`, `maize_4m_pretrain_15k_d2` | headline model |
| `e2_*`, `maize_e2_*`, `maize_4m_d2` | which streams are present |
| `e2_pcrgbdt_tg*`, `maize_4m_tg*` | token-budget controls for the parameter stream |
| `e3_*`, `e4_*` (and `maize_`) | data scale and model scale |
| `e5_*` (and `maize_`) | camera-elevation regime |
| `e8_supervised_d2`, `maize_e8_supervised_d2` | supervised reference |
| `4m_distill_15k_*`, `maize_distill_*` | distillation: generalist, image students, two-stream students |
| `*_row_ft*`, `*_row_iso*` | row fine-tune and isolated-plant control |
| `*_xfer_*` | cross-species warm start vs scratch |

## Testing

| result | script |
|---|---|
| single plants: parameters, height, leaf count, Chamfer (Regime I) | `eval/regime1_eval.py` |
| traits from frozen features (stream ablation, controls, scaling, supervised reference) | `eval/linear_probe.py`, `eval/linear_probe_maize.py` (held-out views: `--probe-view k`) |
| any-subset generation before and after distillation | `eval/eval_rgb2pc_quant.py` |
| point-cloud precision, recall and F1 | `eval/pc_fscore.py`, `eval/pc_fscore_maize.py` |
| crop rows (Regime III) | `eval/eval_row.py` |
| rows of real plants | `eval/eval_real_rows.py` |

Examples:

```bash
python eval/regime1_eval.py --device cuda
python eval/linear_probe.py --runs e2_pc_d2 e2_pcrgb_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2 --ckpt checkpoints/checkpoint_epoch_600.pth
python eval/eval_rgb2pc_quant.py --checkpoint outputs/4m_distill_15k_all_d2/best_model.pth --split test --source rgb \
    --out reports/quant_generalist_rgb_test.json
python eval/eval_row.py --species sorghum --run sorghum_row_ft --ckpt checkpoints/checkpoint_epoch_100.pth --split test \
    --out reports/row_sorghum_ft_test
python eval/eval_real_rows.py --species maize --out reports/real_rows_maize
```

The probe is a ridge regression on frozen features: one deterministic view per plant, the standardiser
fitted on train plants only, the ridge strength chosen by 5-fold cross-validation, scored as R^2 on held-out
plants. The parameter stream is never visible at probe time and its tensor is zeroed, because the plant
token carries the height target verbatim. Features are cached under `outputs/_probe_cache/`.

## Notes

- Point-cloud metrics are computed in the unit-sphere frame of each plant (`chamfer`, squared, symmetric)
  and converted to metres per plant with its metres-per-unit scale where the paper reports metric units.
- Reconstruction is always evaluated in the masked regime the model was trained in; with every token
  visible the decoder is out of distribution.
- Sorghum and maize keep separate loaders and parameter layouts (`N_PARAMS` 9 vs 14, 24 vs 28 leaf tokens,
  8,196 vs 8,192 points).
