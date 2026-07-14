# RGB→PC distillation: OOD evaluation on the Sorghum_15K test split

Model: `outputs/4m_distill_rgb2pc/best_model.pth` (epoch 500, run `distill_rgb2pc`,
teacher + student init = `4m_run_v3` @ epoch 2400). Distilled on `Dataset/new_data`
(old 10k), conditioning on **RGB only** — depth/PC/text fully masked at inference.

OOD set: `Sorghum_15K/test` — 2,250 plants × 10 views = 22,500 folders, split by plant.

> **Correction.** The first version of this file (commit `59341ab`) attributed the OOD
> collapse to "a flat domain shift between the old 10k renders and Sorghum_15K." That is
> **wrong** and is retracted. The two datasets are statistically indistinguishable
> (see *Control* below). The real cause is a leaked validation split — see below.

## Result

| dataset | N | mean CD | median | best | worst |
|---|---|---|---|---|---|
| old10k / train — *trained on these* | 150 | 0.00029 | 0.00029 | 0.00021 | 0.00040 |
| old10k / val — *same plants, new views* | 100 | 0.00027 | 0.00027 | 0.00022 | 0.00042 |
| **15K / test — unseen plants (OOD)** | 150 | **0.04705** | 0.04280 | 0.01356 | 0.15145 |

Full OOD sweep over all 22,500 test folders: **mean 0.04987, median 0.04375**
(`per_plant_chamfer_15Ktest_ep500.json`).

Note: `SorghumDataset.load_pointcloud` subsamples points with an unseeded RNG, so these
figures move by ~0.5% between runs. The separation is four orders of magnitude larger than
that jitter.

## The old 10k validation split is leaked

`Dataset/new_data/val` contains 100 folders covering only **10 distinct plants, and all 10
also appear in `train`** — the split was made per camera view, not per plant. Every
"validation" sample is a training plant seen from a new angle. This is why train and val
scores are identical (0.00029 vs 0.00027) rather than val being worse; val is even
marginally *better*, which is the signature of a leaked split.

So the 0.0003 that this run optimised and selected against never measured generalisation.
The honest number is the 0.047–0.050 on unseen plants.

| check | result |
|---|---|
| old10k train ∩ val (plants) | 10 / 10 — fully leaked |
| Sorghum_15K train ∩ test (plants) | 0 — leak-free |
| shared plant *IDs* old10k ↔ 15K test | 137 (numbering reuse) |
| …of which identical geometry (spline md5) | 0 / 60 sampled — genuinely unseen |
| Pearson r (Chamfer, phenotype extremeness) | 0.017 — not tail extrapolation |

## Control: the two datasets are the same render domain

`compare_datasets.py` over 250 samples per split — old10k and Sorghum_15K agree on every
low-level statistic, so a render/normalisation shift cannot explain the collapse:

| | depth bits | mean fg depth | frame fill | raw PC diagonal | points/cloud | mean RGB |
|---|---|---|---|---|---|---|
| old10k / train | 8 | 0.046 | 0.143 | 2.44 | 71,208 | 0.451 |
| old10k / val | 8 | 0.046 | 0.138 | 2.56 | 81,706 | 0.453 |
| 15K / train | 8 | 0.046 | 0.132 | 2.46 | 72,455 | 0.455 |
| 15K / test | 8 | 0.046 | 0.134 | 2.46 | 73,789 | 0.454 |

## What this means

OOD Chamfer **rose** monotonically over training (0.0381 @ ep65 → 0.0453 @ ep240 →
0.0499 @ ep500) while in-domain fell ~10×. The run was memorising harder, not learning
better, and `best_model.pth` — selected on the leaked val loss — is the *worst* OOD
checkpoint it produced. Error does not track phenotype extremeness (r = 0.017), so this
is not the tail-extrapolation failure the enriched split was designed to catch: the model
simply does not transfer across plant identity, for typical and extreme plants alike.

Qualitatively (see the PDF report), on unseen plants the prediction collapses to a narrow,
generic stem-and-fan shape — plant-like, but not *that* plant.

**Next:** redo the distillation on the 15K train split (leak-free, split by plant), with the
in-flight 15k pretrain (job 11493779) as teacher, selecting on the 15K val split.

## Artefacts

- `rgb2pc_eval_report.pdf` — 6-page report: findings, distributions, per-dataset reconstructions.
- `eval_rgb2pc_recon.png`, `eval_rgb2pc_dist.png` — reconstruction grid + Chamfer distributions.
- `dataset_compare_samples.png`, `dataset_compare_stats.png` — render-domain control.
- `per_plant_chamfer_15Ktest_ep500.{json,png}` — full 22,500-folder OOD sweep.

## Reproduce

```bash
# full OOD sweep
python plot_per_plant_chamfer.py \
    --checkpoint outputs/4m_distill_rgb2pc/best_model.pth \
    --config configs/config_4m_distill_rgb2pc.yaml \
    --data_root /work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K \
    --split test --batch_size 16 \
    --output per_plant_chamfer_15Ktest_ep500.png

# per-dataset reconstructions + distributions, then the PDF
python eval_viz_rgb2pc.py --n 150
python compare_datasets.py --n_stats 250
python build_eval_pdf.py --out rgb2pc_eval_report.pdf
```
