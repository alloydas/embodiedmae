# RGB→PC distillation: OOD evaluation on the Sorghum_15K test split

Model: `outputs/4m_distill_rgb2pc/best_model.pth` (epoch 500, run `distill_rgb2pc`,
teacher + student init = `4m_run_v3` @ epoch 2400). Distilled on `Dataset/new_data`
(old 10k), conditioning on **RGB only** — depth/PC/text fully masked at inference.

OOD set: `Sorghum_15K/test` — the extreme-enriched split (2,250 plants × 10 views =
22,500 folders). Out-of-domain in two senses: a different render set from the 10k the
model trained on, and deliberately enriched with tail phenotypes.

## Result

| checkpoint | in-domain (old_data) | OOD (15K test) |
|---|---|---|
| ep 65  | 0.00083 | 0.0381 *(400-plant subset)* |
| ep 240 | —       | 0.04531 *(N=22,500)* |
| ep 500 (final) | **0.00027** | **0.04987** *(N=22,500, median 0.04375)* |

In-domain Chamfer fell ~10× over training while OOD Chamfer **rose** monotonically.
Final gap ≈ 185×. `best_model.pth` was selected on in-domain val loss, so it is if
anything the *worst* OOD checkpoint of the run.

## The failure is domain shift, not phenotype extrapolation

Per-plant Chamfer joined against the Mahalanobis extremeness score in
`assignment.csv` (Pearson r = **0.017**):

| extremeness quartile | n | mean Chamfer |
|---|---|---|
| Q1 (1.25–2.37, most typical) | 5,630 | 0.04932 |
| Q2 (2.37–2.76) | 5,640 | 0.04981 |
| Q3 (2.76–3.25) | 5,630 | 0.04995 |
| Q4 (3.25–6.73, most extreme) | 5,630 | 0.05038 |

The most *typical* plants in the OOD set reconstruct just as badly as the tail plants.
So the enriched split's extrapolation test is not what's failing — it is a flat,
uniform shift between the old 10k renders and Sorghum_15K. Before reading this as a
verdict on the method, check that point clouds are normalised identically across the
two datasets (centroid + unit-sphere scaling in `SorghumDataset4M`).

Retraining the distillation on the 15K train split — with the in-flight 15k pretrain
(job 11493779) as teacher — is the run whose OOD number would actually be meaningful.

## Reproduce

```bash
python plot_per_plant_chamfer.py \
    --checkpoint outputs/4m_distill_rgb2pc/best_model.pth \
    --config configs/config_4m_distill_rgb2pc.yaml \
    --data_root /work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K \
    --split test --batch_size 16 \
    --output per_plant_chamfer_15Ktest_ep500.png
```
