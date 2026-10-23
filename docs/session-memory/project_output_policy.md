---
name: embodiedmae-output-policy
description: Since 2026-10-23 new run outputs go to /work/mech-ai/alloydas/embodiedmae_outputs; scratch filled to 100 percent; checkpoint retention policy
metadata:
  node_type: memory
  type: project
  originSessionId: 5c49c692-75ea-4443-a303-a6d89c1b614e
  modified: 2026-10-23T15:19:33.569Z
---

`/work/mech-ai-scratch` hit 100 % full on 2026-10-23 (five jobs died at launch, exit 0:53, no log files). Decisions made with the user that outlive the session:

**Why:** scratch is a 152 TB group share that the group keeps filling; a full disk kills jobs at the moment slurmd tries to create their stdout, and trainers at their next checkpoint save.

**How to apply:**
- New run output dirs go to `/work/mech-ai/alloydas/embodiedmae_outputs/<run>` with a symlink `outputs/<run> -> there`, so repo-relative paths keep working. Never move a RUNNING run's directory.
- Checkpoint retention (user-approved 2026-10-23, freed 923 GB): finished runs keep the final epoch, every 100th, the E3 finals (6168 / 2024 / 624) and `best_model.pth`; intermediates are deleted. Never touch running or queued runs or `4m_pretrain_15k_d2_sink`.
- Local wandb run dirs older than a week are deletable (synced); keep `latest-run` and live runs.
- Related: [[cross-species-initiative]].
