# RGB → {point cloud, spline params} specialist distillation

Three runs, 2026-08-24 to 2026-08-26, all stopped early. **The gain is real
(−13.8% on `mean_gen` against the undistilled teacher) but it saturates within
about 10 epochs, and every epoch after that makes the point cloud worse.**
All three runs' best checkpoints sit at epoch 1 or epoch 10.

Figure: `figures_fixed/specialist_degradation.png` (regenerate with
`python figures/plot_specialist_runs.py` — it parses the SLURM logs, because the two
cancelled 8-GPU runs never finalised their `training_history.json`).

## What this was

Take the v2 pretrain model (`outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth`,
epoch 960, val 0.1811) and fine-tune a student so that **RGB alone** — depth, PC
and spline all masked — reconstructs the point cloud *and* the growth parameters.
This is the single-source, two-target specialisation of the all-source distillation
in `configs/config_4m_distill_15k_all.yaml`.

`distill.target: [pc, text]` restricts the reconstruction term, the feature-distillation
token weighting, and the best-model metric to those two modalities
(`train_sorghum_4m_distill.py:263-287`). Teacher is frozen and sees all four modalities.

## Result

`mean_gen` is the pc+text validation loss on 100 val batches (1,600 samples).
The epoch-0 row is the teacher with **no distillation at all**, measured separately
by `eval/eval_warmstart.py` (job 12086899) on the same subset — the runs' own first
validation is already one epoch in, so without this row the gain is invisible.

| | `mean_gen` | `pc_chamfer` | `param_mae` |
|---|---|---|---|
| **epoch 0 — undistilled teacher** | 0.03923 | 0.00104 | 0.0337 |
| run A best (ep10) | 0.03544 | 0.00089 | 0.0309 |
| run B best (ep1) | 0.03387 | 0.00084 | 0.0297 |
| **run C best (ep1)** | **0.03380** | **0.00083** | **0.0298** |
| | **−13.8%** | **−20.2%** | **−11.6%** |

For scale: the in-distribution PC floor measured previously is ~0.00038 and the
pre-distillation zero-shot figure on the old 10k model was 0.0059.

## The three runs

Each run warm-starts from the previous run's best checkpoint, so this is one
chain with a decreasing learning rate, not three independent attempts.

| | job | lr | `crossmodal_prob` | global batch | GPUs | epochs done | best |
|---|---|---|---|---|---|---|---|
| A | 12086208 | 1e-4 | 1.0 | 128 | 8 | 47 of 500 | 0.03544 @ ep10 |
| B | 12087915 | 3e-5 | 1.0 | 128 | 8 | 62 of 500 | 0.03387 @ ep1 |
| C | 12099830 | 1.5e-5 | **0.8** | 64 | 4 | 36 of 500 | 0.03380 @ ep1 |

All three were cancelled by hand, none hit an error or a preemption. Run C ran
6h46m on nova26-gpu-1 at roughly 5 min/epoch.

Configs are `configs/config_4m_distill_15k_rgb2pc_spline{,_lr3e5,_mix4g}.yaml`;
sbatch files are the matching `slurm/distill_rgb2pcspline_*.sbatch`.

**Every lr drop bought an immediate one-epoch step down, then degradation resumed.**
Run B's epoch 1 (0.03387) beat run A's epoch 10 (0.03544) — which it started from —
by 4.4% after a single epoch at the lower lr. Run C's epoch 1 improved on run B's by
only 0.2%, so the chain had essentially converged by then.

**Degradation rate tracks learning rate, and only learning rate.** Lining up
`pc_chamfer` against epochs shows run C at half run B's drift, matching its halved lr:

| epoch | A (1e-4) | B (3e-5) | C (1.5e-5) |
|---|---|---|---|
| 1 | 0.00092 | 0.00084 | 0.00083 |
| 10 | 0.00089 | 0.00087 | 0.00084 |
| 20 | 0.00091 | 0.00095 | 0.00087 |
| 30 | 0.00096 | 0.00101 | 0.00088 |

## The `crossmodal_prob` hypothesis: not supported

Run C was built to test whether the degradation was overfitting to a single narrow
task — at `crossmodal_prob: 1.0` every step is the identical RGB→{PC,spline} problem.
Setting it to 0.8 makes one step in five a normal Dirichlet full-reconstruction step.

It did not stop the degradation. `mean_gen` still rose off its epoch-1 best
(0.03380 → 0.03441 by ep20, then flat through ep30) and `pc_chamfer` still crept
monotonically (+6.0% by ep30).

**This test is confounded and cannot be cleanly reported.** Run C also halved the lr
and the global batch, because no 8-GPU RTX PRO 6000 slot was free (both nodes were
held by other scavenger jobs with ~7 days left, estimated start 2026-09-02), so it ran
on 4 GPUs. Halved lr alone predicts roughly the halved drift that was observed,
leaving nothing for `crossmodal_prob` to explain. A clean test needs run C's
`crossmodal_prob: 0.8` at run B's lr 3e-5 and batch 128.

## What mixing *did* do: depth

The one unambiguous effect. Depth is not a target here, so at `crossmodal_prob: 1.0`
it receives no training signal and collapses; the full-recon steps restore it.

| | depth recon, first val → last val |
|---|---|
| A (xm 1.0) | 0.0050 → 0.0353 (**+606%**) |
| B (xm 1.0) | 0.0205 → 0.0317 (+55%) |
| C (**xm 0.8**) | 0.0031 → 0.0020 (**−36%**) |

If a specialist model ever needs to retain its other modalities, mixing in full-recon
steps is the mechanism that does it — cheaply, at one step in five.

## Reading this run's train loss: don't

`recon` is reported over both step types, and only the ~20% full-recon steps carry the
RGB term. The per-epoch crossmodal count varies ±4% (1273–1350 of 1641) because
`random.Random(step_id)` is seeded per step, not per epoch. Regressing `recon` on that
count over epochs 2–17 of run C:

```
slope = -7.9e-05 per step    r = -0.71    r² = 0.50
xm-count range 1273-1350  ->  predicted recon swing 0.0061
                              actual observed range   0.0064
```

Half the epoch-to-epoch variance, and ~95% of the observed range, is the step mix
rather than learning. Only `mean_gen` is a usable progress signal in a mixed run.

## Artifacts

- `outputs/4m_distill_15k_rgb2pc_spline_mix4g/` — run C, **25 GB**, 18 checkpoints at
  `save_freq: 2`, all superseded by `best_model.pth` (epoch 1). Safe to prune.
- `outputs/4m_distill_15k_rgb2pc_spline{,_lr3e5}/` — runs A and B, plus the
  `init_ep10_best.pth` / `init_ep1_best.pth` snapshots the chain warm-started from.
- `figures_fixed/specialist_degradation.png`, `figures/plot_specialist_runs.py`
- `eval/eval_warmstart.py` + `logs/warmr2ps_12086899.out` — the epoch-0 baseline
- Logs: `logs/r2ps500_12086208.out`, `logs/r2ps3e5_12087915.out`, `logs/r2psmix4g_12099830.out`

## If this is picked up again

1. **Use the epoch-1/epoch-10 checkpoint and stop.** The useful recipe is: warm-start,
   ~10 epochs at 1e-4, one epoch each at 3e-5 and 1.5e-5. Total −13.8%. There is no
   evidence that 500 epochs, or any epoch past ~10, helps.
2. ~~**A clean `crossmodal_prob` test** needs matched lr and batch.~~ DONE — run D
   below. The answer is that mixing does not raise the peak but does stop the drift.
3. **`val_freq: 10` is too coarse** for a curve whose entire useful range is the first
   10 epochs. Validate every epoch for the first 20.
4. The saturation is consistent with the all-source result — distillation gained −54%
   from the old 10k teacher but only −5.9% from this one. A stronger teacher leaves
   less to distill, and what remains is extracted almost immediately.

---

## Run D — the clean `crossmodal_prob` test (job 12132446, 2026-08-28)

Run B's config with **exactly one substantive change**, `crossmodal_prob 1.0 -> 0.8`
(plus `val_freq 10 -> 1`, which changes only measurement). Same `init_ep10_best.pth`,
same teacher, same lr 3e-5, same global batch 128 on 8 GPUs, same 360 CPU / 1200G
allocation, same node. Both runs' `config.json` snapshots differ in four keys only:
`crossmodal_prob`, `val_freq`, `output_dir`, `wandb_name`. Neither resumed. The val
loader is `shuffle=False` with `val_max_batches: 100`, so both score the identical
first 1600 val samples. This is a true matched pair.

Ran 3h51m, epochs 1-30 of a planned 500, then cancelled by hand. Run B validated at
epochs 1/10/20/30/40/50/60, so the arms line up at four epochs:

| epoch | mean_gen B (xm 1.0) | mean_gen D (xm 0.8) | depth B | depth D | pc_chamfer B | pc_chamfer D |
|------:|--------------------:|--------------------:|--------:|--------:|-------------:|-------------:|
|  1 | **0.03387** | 0.03413 | 0.0205 | 0.0030 | 0.00084 | 0.00084 |
| 10 | 0.03425 | 0.03429 | 0.0227 | 0.0020 | 0.00087 | 0.00085 |
| 20 | 0.03520 | 0.03446 | 0.0257 | 0.0020 | 0.00095 | 0.00087 |
| 30 | 0.03589 | 0.03476 | 0.0278 | 0.0020 | 0.00101 | 0.00091 |
| 40 | 0.03672 | — | 0.0289 | — | 0.00105 | — |
| 60 | 0.03850 | — | 0.0317 | — | 0.00115 | — |

D's own best is **0.03396 @ epoch 4**. Run C still leads all four arms at 0.03380.

**Finding 1 — mixing does not raise the peak.** 0.03396 (D) vs 0.03387 (B) is a 0.3%
difference in B's favour, inside the epoch-to-epoch jitter of D's own series (which
contains a 0.03482 spike at ep15 against a 0.03446 neighbourhood). The `crossmodal_prob`
hypothesis, stated as "mixing improves the target metric", is **not supported** — now
without run C's lr/batch confound.

**Finding 2 — mixing does cut the drift, by about two thirds.** Over ep1->ep30 B rises
+0.00202 while D rises +0.00063, i.e. D drifts at 31% of B's rate. Run C drifted
+0.00061 over the same window. Since C halved lr *and* mixed, and D mixed at B's full
lr and reached the same drift rate, the deconfounding runs the other way from what
`SPECIALIST_DISTILL` previously assumed: **C's slower degradation is attributable to
mixing, not to its halved lr.** The earlier note that "degradation rate tracks lr and
only lr" is superseded — it tracks lr *and* `crossmodal_prob`.

**Finding 3 (the largest effect here) — one full-recon step in five repairs a collapsed
depth head within a single epoch, and holds it.** Both arms start from the same
`init_ep10_best.pth`, whose depth had already collapsed to 0.0206 under run A's xm 1.0.
After one epoch: B 0.0205 (unchanged), D **0.0030** — an 85% repair in one epoch. D's
depth then settles to 0.0019-0.0020 from epoch 6 and is flat for the remaining 25
epochs, while B's climbs monotonically to 0.0317 by ep60. At the matched epoch 30 the
gap is **13.9x** (0.0278 vs 0.0020). This supersedes the older A-vs-C framing in the
"What mixing DID do" section above, which compared runs with different inits and lrs;
this is a matched pair with a measured one-epoch repair time constant.

**Finding 4 — pc_chamfer tracks the same pattern, weakly.** Identical at ep1 (0.00084),
diverging to 0.00101 vs 0.00091 by ep30. Mixing slows PC degradation too, but the
effect is ~10% where depth's is ~14x, because PC is a distillation *target* and depth
is not.

### Why this stopped at epoch 30

The follow-on job 12161772 (resume to the planned hand-cancel at ~epoch 60) sat
**PENDING for 3.7 days** and was still queued on 2026-09-01. The cause was purely a
scheduling one and is worth recording, because it cost five days for nothing:

- Only `nova26-gpu-[1-2]` carry `rtx_pro_6000` cluster-wide, and **all 16 GPUs were
  idle the whole time**. No other job on the cluster requests that GRES.
- The block was the **CPU request**. A `sbatch --test-only` sweep (submits nothing;
  vary one axis, read the projected start) found a hard cliff:
  `<=128 CPUs -> nova26-gpu-1, start Sep 2 17:08` vs `>=144 CPUs -> nova26-gpu-2,
  start Sep 7 22:05`. The job asked for 360.
- **Memory and walltime are irrelevant to placement here** — 1200G vs 800G vs 120G, and
  4h vs 12h vs 7d, all produce byte-identical projected start times. So is GPU count:
  even `--gres=...:1` waits for the same slot. Do not bother trimming those to schedule
  faster; trim CPUs.
- What actually holds the node is 452 pending `nova`-partition jobs (PriorityTier 100
  vs scavenger's 0) projected to take nova26-gpu-1's CPUs the moment the 96-CPU MPI job
  12172786 releases them.
- Dropping the request to 120 CPUs moved the estimate from Sep 7 18:16 to Sep 2 17:08.

**Epochs 31-60 would add nothing.** Depth is flat from epoch 6, mean_gen is monotone
worsening past epoch 4, and B's curve already shows where D's is heading. At the
measured 11.3 min/epoch that is ~5.7 h of 8-GPU node time for a fourth point on a
line whose slope is measured at three.

### Files

- `outputs/4m_distill_15k_rgb2pc_spline_xm08_clean/` — 15 checkpoints (ep 2..30 at
  `save_freq: 2`), `best_model.pth` = **epoch 4**, `training_history.json` with 30 train
  + 30 val entries and no gaps from the cancel. `config_scratch_12132446.json` is the
  pristine from-scratch provenance snapshot — a resume rewrites `config.json` at
  startup with `resume=<path>`, so that copy was taken before any resume could occur.
- `logs/xm08clean_12132446.{out,err}`, config
  `configs/config_4m_distill_15k_rgb2pc_spline_xm08_clean.yaml`, wandb run `ypqe8xl1`.

### Run E — the 4-GPU continuation (job 12161772)

Job 12161772 resumes run D from `checkpoint_epoch_30.pth` but on **4 GPUs at global
batch 64**, half run D's 128. Epochs 31+ are therefore **NOT comparable to run B** on
batch — this reintroduces on purpose the axis run D existed to control. What it *is*
cleanly comparable to is **run C** (xm 0.8, batch 64, lr 1.5e-5): run E holds lr at
3e-5, so E-vs-C isolates lr at fixed batch and mixing, the one contrast the four runs
did not yet cover.

Run D's clean 30-epoch record is snapshotted before the continuation touches the dir:
`training_history_runD_ep1_30_bs128.json`, `best_model_runD_ep4_bs128.pth`
(epoch 4, 0.033964), `config_scratch_12132446.json`, and the immutable
`logs/xm08clean_12132446.out`. The live `training_history.json` / `best_model.pth` will
after this point contain **mixed-regime** data — use the snapshots for the run D table
above. The output dir is deliberately left unchanged so the frozen sbatch's auto-resume
still finds the newest checkpoint on a scavenger requeue; pointing it elsewhere would
make every preemption silently restart from epoch 30.

Config at 4 GPUs: `batch_size 16` (global 64), `num_workers 2` (4 ranks x 2 = 8 workers
+ 4 mains = the 12-CPU cap exactly), lr 3e-5 unchanged, 12 CPU / 120G.
