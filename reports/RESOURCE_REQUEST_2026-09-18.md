# Resource request — GPU and storage through CVPR submission

Drafted 2026-09-18. Covers finishing the E1–E10 matrix on the current
Sorghum_15K dataset plus **two additional datasets of the same size**.

Every figure below is measured from completed jobs on this cluster, not
estimated. The derivation is in [Basis](#basis-for-the-numbers) so the request
can be defended or re-scoped without re-deriving it.

---

## Email body

> **Subject:** GPU and storage request — EmbodiedMAE sorghum project, through CVPR submission

Hi [name],

I'd like to request additional GPU allocation and storage to finish the
multi-modal MAE sorghum project, including two further datasets I plan to
generate.

**Where the project stands**

The pre-training and the first two ablations are complete on our current
15,000-plant dataset. Cross-modal generation improves 19.1 % after distillation,
and the modality ablation (E2) has finished all four arms. Data-scaling (E3) and
model-scaling (E4) are running now. To date the project has consumed **2,993
GPU-hours** across 13 training jobs on the RTX PRO 6000 and A100 nodes.

**GPU request: ~4,000 GPU-hours**

Costs below are measured from completed runs, not estimates. One ablation arm
(197,400 optimizer steps, 2 GPUs) takes 24–63 GPU-hours depending on node
contention; a full pre-training run takes 450–830 GPU-hours depending on
hardware.

| Work | GPU-hours |
|---|---|
| Finish E3/E4 on current dataset | 350 |
| Downstream probe, view-regime study, latent analysis, OOD eval | 250 |
| Baseline models (required for publication) | 400 |
| Contingency on the above (25 %) | 250 |
| **Current dataset subtotal** | **1,250** |
| Two additional datasets @ ~1,400 each (pre-train + three ablations + evals) | 2,800 |
| **Total** | **~4,050** |

**Storage request: ~2.6 TB**

This is the blocking constraint. `/work/mech-ai-scratch` is currently **100 %
full — 152 TB used, 897 GB free**. Our present dataset is 774 GB and model
outputs are 233 GB, so a single additional dataset will not fit today.

| Item | Size |
|---|---|
| Two additional datasets @ 774 GB | 1,550 GB |
| Checkpoints and outputs for those runs | 500 GB |
| Remaining runs on the current dataset | 350 GB |
| Working headroom | 200 GB |
| **Total** | **~2.6 TB** |

One efficiency worth noting: roughly 64 % of each dataset is source geometry
(`.obj` and `_nc.ply`) that the generator duplicates into all ten camera-view
folders and that training never reads. Storing those once per plant instead of
ten times would cut the two new datasets from 1,550 GB to about 530 GB, reducing
the total request to **~1.6 TB**. I'm happy to restructure the output layout
that way if storage is the tighter constraint.

**Timeline**

Our internal results freeze is 24 October, with submission 13 November. The
current-dataset work (1,250 GPU-hours) fits comfortably. The two additional
datasets at 2,800 GPU-hours would need roughly 4–5 GPUs running continuously to
complete before the freeze; if that isn't available, I'd propose treating them
as the extended evaluation for the journal version rather than delaying the
submission.

Happy to provide per-run logs or adjust scope if either figure is difficult.

Best,
Alloy

---

## Basis for the numbers

### Consumed to date

`sacct` over all 4M-project GPU jobs since 2026-07-20: **13 jobs, 2,993
GPU-hours**. This excludes unrelated zero-shot segmentation jobs on the same
account, which is why it is far below the 5,989 GPU-hours the account shows in
total.

### Measured per-run cost

| run | wall clock | GPUs | GPU-hours |
|---|---|---|---|
| `e2_pc` | 12:12:13 | 2 | 24 |
| `e2_pcrgb` | 13:22:26 | 2 | 27 |
| `e2_pcrgbd` | 1-04:51:52 | 2 | 58 |
| `e2_pcrgbdt` | 1-07:29:22 | 2 | 63 |
| `emb4m15k` (E1 pretrain) | 8-14:43:52 | 4 | 827 |
| E1 v2 pretrain, 8× RTX PRO 6000 | ~2.2 d | 8 | ~422 |

All four E2 arms run an identical 197,400 optimizer steps, so the 24 → 63
GPU-hour spread is **node contention, not the arm** — the pipeline is
dataloader-bound and co-resident jobs share the node's CPUs. Planning figures
used above: **60 GPU-h** per ablation arm, **120** for the 332M-parameter arm,
**450** for a full pre-train.

E3 arms projected from live progress (2026-09-18): `e3_10k` 498/631 at 3.54
min/epoch, `e3_3k` 273/2100 at 1.22, `e3_1k` 245/6169 at 0.39 — all three land
at 37–43 h wall clock, confirming the equal-optimizer-steps design gives equal
wall clock across data scales.

### Storage, measured

| item | size |
|---|---|
| `Sorghum_15K/train` | 105,000 folders × ~5 MB = 512 GB |
| `Sorghum_15K/val` | 22,500 × ~5 MB = 109 GB |
| `Sorghum_15K/test` | 22,500 × ~7 MB = 153 GB |
| **dataset total** | **774 GB** |
| `outputs/` (all runs to date) | 233 GB |
| one run directory (24 checkpoints) | 32 GB |
| one `best_model.pth` | 1.35 GB |
| `/work/mech-ai-scratch` | **152 TB used of 152 TB, 897 GB free** |

The 64 % figure: only `rgb.png`, `depth.png`, `*_nc_cam.ply` and `*_spline.yml`
are opened by the loaders (~1.8 MB per sample). `Sorghum_<n>.obj` and
`Sorghum_<n>_nc.ply` are byte-identical across all ten of a plant's view folders
and never read during training. See the "Dataset layout" and "Porting to another
machine" sections of `CLAUDE.md`.

### Assumptions that change the total

1. **Each new dataset gets a full pre-train plus the E2/E3/E4 ablations.** If the
   new datasets are only for OOD evaluation against existing checkpoints, the
   GPU request drops from ~4,050 to roughly **1,500** GPU-hours.
2. **Baselines (E8) are four training runs at ~100 GPU-hours each.** The
   baseline set has not been chosen yet, so this line is the softest in the
   table.
3. **25 % contingency** on current-dataset work, for preemption on the scavenger
   partition and for reruns.

### Producing a formatted copy

No LaTeX engine on Nova. Render via the pandoc module with weasyprint (available
in the `det` env):

```bash
module load pandoc
pandoc reports/RESOURCE_REQUEST_2026-09-18.md -o resource_request.pdf \
  --pdf-engine=weasyprint
```
