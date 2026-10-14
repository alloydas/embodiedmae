"""
Sinkhorn (entropic optimal-transport) term for the point-cloud loss.

Why. Chamfer and QAL match every point to its NEAREST neighbour, so many
predicted points can share one target point at no cost: the folding head
collapses most tokens' 41 points into a tight clump and recall rises only
slowly with training (CLAUDE.md, "the PC decoder collapses"). Optimal transport
matches the two clouds as distributions, one unit of mass each way, so a clump
that covers one spot leaves the rest of the target unmatched and pays for it --
it rewards coverage (recall) directly.

What. The debiased Sinkhorn divergence of geomloss (p = 2, so for small `blur`
it approaches half the squared 2-Wasserstein distance) between RANDOM subsets
of `n_points` predicted and `n_points` target points, the same subsets across
the batch, redrawn every call. Subsampling keeps the dense cost matrix at
B x n x n (2048^2 x 16 floats = 268 MB) on geomloss's pure-PyTorch
'tensorized' backend, which needs no KeOps / nvcc on the GPU node; over many
steps every predicted point is drawn. Sized 2026-10-04 (job 16716188) on the
uniform scene arms' own predictions: the divergence is ~0.05-0.1, about 100x
the subsampling floor (two random 2048-subsets of the SAME cloud: 0.0006), and
it GROWS from epoch 20 to 200 (sorghum 0.053 -> 0.071) while QAL falls
(0.058 -> 0.033) -- QAL training trades distribution match for nearest-point fit.
blur 0.01 and 0.02 gave the same values; 2048 points at scaling 0.9 nearly
doubled the step time, so the arms use 1024 points, blur 0.02, scaling 0.8. `blur` is in the unit-sphere frame, the
same units as the F-score thresholds (0.01 ~ 7 mm on a 1.4 m plant).

The model adds `pc_sinkhorn_weight` x this to the QAL / chamfer term
(embodied_mae_4m.EmbodiedMAE4M, YAML `model.pc_sinkhorn_*`); weight 0, the
default, never imports geomloss and leaves every earlier run unchanged.
"""

import torch

_LOSSES = {}


def sinkhorn_pc_loss(pred, target, n_points=2048, blur=0.01, scaling=0.8):
    """Mean over the batch of the debiased Sinkhorn divergence between random
    `n_points` subsets of pred (B, N, 3) and target (B, M, 3)."""
    from geomloss import SamplesLoss
    key = (float(blur), float(scaling))
    if key not in _LOSSES:
        _LOSSES[key] = SamplesLoss('sinkhorn', p=2, blur=key[0], scaling=key[1],
                                   backend='tensorized')
    N, M = pred.shape[1], target.shape[1]
    ip = torch.randperm(N, device=pred.device)[:min(int(n_points), N)]
    it = torch.randperm(M, device=target.device)[:min(int(n_points), M)]
    # geomloss switches autograd ON inside its call and does not restore it, so
    # under torch.no_grad() (validation) every later op would build a graph --
    # checked 2026-10-05: is_grad_enabled() False -> True. Restore the caller's mode.
    grad_mode = torch.is_grad_enabled()
    try:
        return _LOSSES[key](pred[:, ip].contiguous(), target[:, it].contiguous()).mean()
    finally:
        torch.set_grad_enabled(grad_mode)
