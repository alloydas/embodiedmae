"""embodiedmae -- the official EmbodiedMAE-base (Dong et al. 2025), frozen. RGB+D+PC.

This is the closest prior work, and our Dirichlet token allocation comes from it.
Weights: HF ZibinDong/embodiedmae-base, pinned to a revision and checked by
sha256. The Base model was distilled from a Giant EmbodiedMAE pretrained on
DROID-3D (paper arXiv 2505.10105, section 2.4). Table label:
"EmbodiedMAE-B (DROID-3D, frozen; mean-pooled; depth input outside its 0-2 m
pretraining range and not reconstructed)". The last clause is measured, not a
hedge: masked plant depth through the official decoder is no better than a
constant (L1 0.431 vs 0.418 sorghum, 0.334 vs 0.229 maize), 63 % of sorghum
plant pixels lie beyond 2 m, and every in-range alternative tried (2 m
backdrop; x0.5 depth; x0.4 with a 0 backdrop; plant median rescaled to 0.8 m)
reconstructs worse. So this row is effectively RGB+PC; the depth tokens are
kept because they are part of the model, not because they carry pretrained
signal here. If a reviewer asks, an RGB+PC variant (upstream's missing-
modality path, depth tokens dropped) would price them.

WHY A PURE-TORCH PORT, AND NOT THE OFFICIAL PACKAGE
The upstream code needs transformers (Dinov2Encoder), pytorch3d (FPS/kNN),
numba and einops. `det` has none of transformers, pytorch3d or numba.
`det_cu128`, which slurm/linear_probe.sbatch picks on Blackwell, does not even
have huggingface_hub or safetensors. pytorch3d also supports only
torch<=2.4.1 (upstream README). Worse, the official route has three traps, and
each gives plausible, wrong features rather than an error:
  * the HF snapshot's EmbodiedMAEModel.forward defaults add_mask=True
    (snapshot modeling_embodiedmae.py:125), which keeps a random 98 of the 588
    tokens;
  * the README's `model(...).embedding` is the PRE-encoder token embedding
    (modeling_embodiedmae.py:93). The encoder output is `.last_hidden_states`
    (:118). On real val plants the two differ by max|diff| 14-19;
  * without pytorch3d, configuration_embodiedmae.py:70-77 sets
    enable_point_cloud=False with only a warning, and the PC tokens vanish
    (modular_embodiedmae.py:565).
And the old loader is not an alternative:
/work/mech-ai-scratch/alloy/plant_point_cloud/embodiedmae_adapted.pth loaded
these weights with strict=False into THIS repo's 3M model. Its own record says
load_percentage 0.69%: 287 of 289 keys missing, and all 505 checkpoint keys
unexpected. It is a random network. The two architectures cannot share weights
at all: no CLS token or modality embeddings upstream, separate learned RGB and
depth position embeddings, a DP3 PC tokenizer (K=64 + centre MLP) instead of
our Point-MAE one (K=32), LayerScale, LayerNorm eps 1e-6, and split q/k/v.

The port below re-implements the encoder from upstream
github.com/ZibinDong/embodiedmae @ 60abd4da2aa6ccd068fc23ffc4a4c8ae2b9678ac
(MIT, per its pyproject.toml; the repo ships no LICENSE file). It uses the
UPSTREAM state-dict key names, so the checkpoint loads with strict=True and
nothing is renamed. The encoder block is transformers' Dinov2Layer (the upstream
encoder, modeling_embodiedmae.py:35). The PC tokenizer mirrors
EmbodiedMAEPointCloudEmbeddings (modular_embodiedmae.py:390-424, :552-573).
FPS and kNN mirror pytorch3d.ops.sample_farthest_points / knn_points
(fps_and_knn, modular_embodiedmae.py:290-299). No upstream source is copied
verbatim.

WEIGHT LOADING
Strict. The 242 non-decoder keys load into the port with missing=[] and
unexpected=[]: 86,996,992 parameters, the paper's "87M Base". The 263
`decoder.*` keys (the pretraining decoder, 43,960,512 values) are dropped as an
asserted set. The file is read by a ~20-line safetensors reader, so this adapter
needs neither safetensors nor huggingface_hub at run time. huggingface_hub is
only needed to download (--prefetch). The file must match CKPT_SHA256 on every
build, not only on download.

INPUT CONVERSIONS
Each conversion takes the tensor the arms see (see the contract in
baselines/__init__.py) to the upstream convention. One rule governs the choices:
a conversion must be a FIXED map, the same for every sample, so that this
baseline receives no information the arms do not. The two places this rules out
the "physically exact" choice are marked (!).

  rgb   ImageNet-normalised -> [-1, 1]: undo Normalize, then x*2-1. Upstream:
        README "rgb ... in [-1, 1]"; prepare_shuffle_idx docstring
        (modular_embodiedmae.py:215); visualize() maps back with *0.5+0.5
        (:884-887). Exact.
  depth (z-near)/(far-near), 0 = background -> metres, background 4.0 m.
        Upstream: README "depth ... in [0, 2] (meter)"; modular :217 and :805
        ("Unit is meters"); visualize() divides by 2.0 (:888-891); the paper's
        "hardware-calibrated metric depth".
        (!) Background (d == 0, no surface) becomes a 4.0 m backdrop, not 0.
          Our frames are 75-95% background, and DROID-3D's scenes are dense,
          so a zero backdrop is far outside what the model saw. Measured
          through the official decoder (6 val plants x 2 masks, masked plant
          pixels, L1 relative, sorghum / maize): background 0 m gives
          0.515 / 0.683, 2 m gives 0.428 / 0.323, 4 m gives 0.431 / 0.334.
          4 m rather than 2 m (the far end of the upstream range) because
          4 m lies behind every plant pixel: sorghum plant depth is at most
          3.60 m over 450 view-00 frames (all three splits), and maize is at
          most 3.03 m by construction (below). 2 m falls inside most plants'
          depth range (sorghum 1.8-3.2 m typical), so leaves at ~2 m would
          read as backdrop. With 4 m the fill is a bijection on the arms'
          tensor (d == 0 exactly on background) and adds no information.
          The probe barely notices the choice. At 160 plants per split, the
          mean R2 over the four 6.4 targets is: sorghum train-CV 0.853
          (0 m) vs 0.827 (4 m), val 0.818 vs 0.828; maize train-CV 0.247 vs
          0.252, val 0.374 vs 0.372. So the backdrop is picked on the
          label-free criterion above, not on R2.
        sorghum: z = 50.0*d. The sorghum renderer uses fixed near/far, and
          reprojecting *_nc_cam.ply through a fovY-40 pinhole gives
          |z| = 49.2-50.0*d on 24 views across all three splits (inlier fit),
          with signed median(|z|-50d) of +2.5 to +3.1 cm. The ~3 cm offset is
          dropped: it is 1-2% of the 1.5-3.5 m range. (checked 2026-09-24)
        (!) maize: z = 1.041 + d*(3.030-1.041), FIXED dataset constants, NOT
          the per-view near/far in camera_pose.json. The maize renderer sets
          near = 0.5114*dist and far = 1.4886*dist, where dist is the camera
          distance, and dist is framed to the plant: r = 0.82 with
          stem_internodeSum and 0.79 with leaf_lengthMean (200 val plants).
          So the per-view near/far encode plant height, while the arms' d,
          being normalised by them, is scale-free. The exact conversion would
          hand this baseline the height target. The constants are the medians
          over 480 views (80 plants x 2 views x 3 splits): near 1.0411,
          far 3.0304. So maize depth is metric up to a per-view factor
          2.036/dist (0.82-1.34), which carries the relative shape exactly.
  pc    unit sphere, OpenGL camera axes (y up, -z forward)
        -> 0.8*(x, -y, -z) + (0, 0, 0.8).
        The axis flip is physically motivated, NOT verified: upstream clouds
        come from metric depth through the camera intrinsics (paper section
        2.1), which suggests the OpenCV camera frame (y down, +z forward), but
        the DROID-3D frame is inferred from the paper, and reconstruction is
        ambivalent (PC chamfer / constant, lower is better, sorghum / maize:
        OpenCV flip 0.455 / 0.378, no flip (GL) 0.419 / 0.440, z-up 0.432 /
        0.477). One convention is kept for both species, not picked per
        species off a reconstruction score. The DROID-3D card gives xyz_range
        x,y in [-1, 1], z in [0, 1.6] m.
        (!) Scale and offset are NOT metric. The arms' PC is centred and
          unit-sphere scaled, so it carries no absolute size, and metric PC
          would add exactly that. 0.8*unit + 0.8 z places the cloud inside the
          DROID-3D box instead.
        N is as loaded (8196 sorghum / 8192 maize; upstream prefers 8192).
        FPS keeps 196 centres either way.

VERIFIED 2026-09-25 on CPU, 6 sorghum + 6 maize val plants (view 00), against
the OFFICIAL code (60abd4d; run with transformers 4.48, and with pytorch3d's
naive FPS and exact kNN in place of its C++ ops):
  * Tokens and pooled features are bitwise-equal to upstream
    EmbodiedMAEModel(add_mask=False).last_hidden_states on the converted
    inputs: max|diff| 0.0 for every modality, both species. For comparison,
    upstream `.embedding` differs by 19 and a random-init port by 10.
  * The weights are the pretrained ones. Masked-token reconstruction through
    the official pretrained decoder (98/588 tokens visible, 6 plants x 2
    Dirichlet masks), pretrained encoder vs a random-init encoder, same
    decoder and masks, sorghum / maize:
        RGB MSE, masked plant patches    0.045 vs 0.360  /  0.015 vs 0.480
        PC chamfer, relative             0.455 vs 0.967  /  0.378 vs 1.165
        depth L1, masked plant pixels    0.431 vs 0.634  /  0.334 vs 0.560
    Depth is the weak modality. It is no better than predicting the mean of
    the visible plant pixels (0.418 / 0.229): sparse plants 1.5-3.6 m away
    are outside DROID-3D.
  * The conventions. Pretrained model, one modality changed at a time
    (masked-token error, sorghum / maize):
        rgb    [-1,1] 0.022 / 0.006   vs the raw ImageNet tensor 0.073 / 0.051
        depth  metric, 4 m backdrop, plant px 0.431 / 0.334
               vs metric with 0 backdrop 0.515 / 0.683, and raw d 4.50 / 0.71
               maize per-view near/far (the leak): 0.399, no better
        pc     DROID box 0.455 / 0.378   vs the raw unit sphere (GL) 0.655 / 1.076
  * Features are deterministic: a rerun is bitwise-equal, and so are
    batch 1 vs batch 8 and --repeats 2 vs 1 (maize, 8 plants).

Known deviations, kept on purpose so that the input matches the arms':
  * Depth is the dataset's 224 px bilinear+antialias resize, so silhouette
    edges blend background (0) with foreground. Those pixels are > 0, so they
    stay plant pixels and read too near (sorghum ~0 m, maize from 1.04 m)
    rather than getting the 4 m backdrop. The arms see the same pixels.
  * Plant depths (1.5-3.6 m) and the 4 m backdrop exceed DROID-3D's 2 m
    depth range: extrapolation.
  * Depth (metric, ~2.5 m) and PC (in the 0-1.6 m box) no longer describe the
    same geometry. The encoder sees both. The reconstruction check was run
    under exactly this pairing.

TOKENS AND POOLING
588 tokens: 196 RGB + 196 depth + 196 PC, concatenated in that order with
add_mask=False semantics (prepare_shuffle_idx, modular :253-269: every token,
natural order), then 12 DINOv2 layers and a final LayerNorm. There is no CLS
token (paper section 2.2: "removing the [CLS] token"). So FEATURES=('mean',):
the mean of the 588 post-LayerNorm tokens, which is the analogue of the arms'
'mean' (latent[:, 1:].mean(1), also post-norm). Compare this row with the arms
probed at --feature mean, not with their CLS rows.

DETERMINISM
FPS starts at index 0, as pytorch3d's default random_start_point=False does
(sample_farthest_points.py:96, :167). kNN uses exact squared distances, not
cdist's matmul path. Nothing else is stochastic. So features are a
deterministic function of the loaded points, and --repeats > 1 changes nothing.
"""

import json
import os
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = 'embodiedmae'
INPUTS = ('rgb', 'depth', 'pc')
FEATURES = ('mean',)          # no CLS token upstream: 'cls' must fail, not substitute
MODEL_SIZE = 'base'
EPOCH = -1                    # external pretrain

HF_REPO = 'ZibinDong/embodiedmae-base'
HF_REVISION = 'af5399629555ef7ccf1d1fd55fc1df8c7c5bc519'
HF_FILE = 'model.safetensors'
CKPT_BYTES = 523_885_096
CKPT_SHA256 = '0b3ecb0ce4b1f15a538cb8a732a27d2d904c533c410e13e79ffd92446a2bbb09'
CODE_COMMIT = '60abd4da2aa6ccd068fc23ffc4a4c8ae2b9678ac'

N_ENCODER_KEYS = 242
N_DECODER_KEYS = 263
N_ENCODER_PARAMS = 86_996_992

SOURCE = (f'HF {HF_REPO} @ {HF_REVISION}, {HF_FILE} {CKPT_BYTES:,} B, '
          f'sha256 {CKPT_SHA256}; encoder ported to pure torch from '
          f'github.com/ZibinDong/embodiedmae @ {CODE_COMMIT[:7]} (MIT); '
          f'inputs rgb [-1,1], depth m (bg 4.0), pc 0.8*OpenCV+0.8z; '
          f'features = mean of 588 post-LN tokens')

# ── input conventions (see the module docstring for the evidence) ──
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # sorghum_dataset.py / maize_dataset_4m.py rgb_transform
IMAGENET_STD = (0.229, 0.224, 0.225)
SORGHUM_DEPTH_SCALE = 50.0              # |z| [m] ~= 50 * d, fixed near/far renderer
MAIZE_NEAR, MAIZE_FAR = 1.041, 3.030    # dataset medians; per-view values leak plant size
DEPTH_BACKGROUND_M = 4.0                # behind every plant pixel (<= 3.6 m); 0 is out of distribution
PC_SCALE, PC_Z_OFFSET = 0.8, 0.8        # unit sphere -> DROID-3D box x,y [-1,1], z [0,1.6]
GL_TO_CV = (1.0, -1.0, -1.0)            # OpenGL camera (y up, -z fwd) -> OpenCV (y down, +z fwd)

# upstream config.json at HF_REVISION
_CFG = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, mlp_ratio=4,
            layer_norm_eps=1e-6, image_size=224, patch_size=16,
            num_pc_centers=196, num_pc_knn=64)


# ─────────────────────────────── input conversion ────────────────────────────

def _check(ok, msg):
    # A changed loader convention (a new normalisation, a different depth
    # decoder) would otherwise flow through a fixed affine map into plausible
    # features. These bounds are what the conversions below assume.
    if not bool(ok):
        raise ValueError(f'embodiedmae input check failed: {msg}')


def convert_rgb(rgb):
    """ImageNet-normalised (B,3,H,W) -> upstream [-1, 1]."""
    mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x01 = rgb * std + mean
    _check((x01.min() > -1e-4) & (x01.max() < 1 + 1e-4),
           f'un-normalised rgb spans [{x01.min():.4f}, {x01.max():.4f}], expected [0, 1]: '
           f'is the loader still applying ImageNet mean/std?')
    return x01 * 2.0 - 1.0


def convert_depth(depth, species):
    """(z-near)/(far-near) in [0, 1], 0 = background -> metres, background DEPTH_BACKGROUND_M."""
    _check((depth.min() >= 0) & (depth.max() <= 1), 'depth outside [0, 1]')
    if species == 'sorghum':
        # Sorghum foreground is d ~0.03-0.07 (1.5-3.5 m). d > 0.2 would be
        # beyond 10 m: a different renderer or decoder.
        _check(depth.max() < 0.2, f'sorghum depth max {depth.max():.3f} >= 0.2 (10 m)')
        z = depth * SORGHUM_DEPTH_SCALE
    elif species == 'maize':
        z = MAIZE_NEAR + depth * (MAIZE_FAR - MAIZE_NEAR)
    else:
        raise ValueError(f'no depth convention for species {species!r}')
    # d == 0 exactly on background (both loaders decode 0 as "no surface"), so
    # the fill is a bijection on the arms' tensor, not new information.
    return torch.where(depth > 0, z, torch.full_like(z, DEPTH_BACKGROUND_M))


def convert_pc(pc):
    """Centred unit-sphere cloud, OpenGL axes -> OpenCV axes inside the DROID-3D box."""
    r = pc.norm(dim=-1).amax(dim=-1)
    _check((r <= 1 + 1e-4).all(), f'pc max norm {r.max():.4f} > 1: not the unit-sphere loader')
    scale = pc.new_tensor(GL_TO_CV) * PC_SCALE
    return pc * scale + pc.new_tensor((0.0, 0.0, PC_Z_OFFSET))


def to_upstream(batch, species):
    """The probe batch's (rgb, depth, pc) in upstream units. Params/text are never read."""
    rgb, depth, pc = batch[0], batch[1], batch[2]
    return convert_rgb(rgb), convert_depth(depth, species), convert_pc(pc)


# ─────────────────────────────── encoder port ────────────────────────────────
# Module and attribute names reproduce the upstream state-dict keys exactly.

class _SelfAttention(nn.Module):                 # Dinov2SelfAttention (eager)
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(d, d)             # qkv_bias=True
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        h = self.heads

        def split(t):
            return t.view(B, N, h, D // h).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        # Materialised softmax attention, as config attn_implementation='eager'.
        a = (q @ k.transpose(-1, -2)) / ((D // h) ** 0.5)
        return (a.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, D)


class _SelfOutput(nn.Module):                    # Dinov2SelfOutput
    def __init__(self, d):
        super().__init__()
        self.dense = nn.Linear(d, d)

    def forward(self, x):
        return self.dense(x)


class _Attention(nn.Module):                     # Dinov2Attention
    def __init__(self, d, heads):
        super().__init__()
        self.attention = _SelfAttention(d, heads)
        self.output = _SelfOutput(d)

    def forward(self, x):
        return self.output(self.attention(x))


class _LayerScale(nn.Module):                    # Dinov2LayerScale
    def __init__(self, d):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(d))   # layerscale_value 1.0

    def forward(self, x):
        return x * self.lambda1


class _MLP(nn.Module):                           # Dinov2MLP, hidden_act 'gelu' = exact erf
    def __init__(self, d, ratio):
        super().__init__()
        self.fc1 = nn.Linear(d, d * ratio)
        self.fc2 = nn.Linear(d * ratio, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Layer(nn.Module):                         # Dinov2Layer, pre-norm, drop_path 0
    def __init__(self, d, heads, ratio, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, eps=eps)
        self.attention = _Attention(d, heads)
        self.layer_scale1 = _LayerScale(d)
        self.norm2 = nn.LayerNorm(d, eps=eps)
        self.mlp = _MLP(d, ratio)
        self.layer_scale2 = _LayerScale(d)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class _Encoder(nn.Module):                       # Dinov2Encoder
    def __init__(self, d, depth, heads, ratio, eps):
        super().__init__()
        self.layer = nn.ModuleList(_Layer(d, heads, ratio, eps) for _ in range(depth))

    def forward(self, x):
        for blk in self.layer:
            x = blk(x)
        return x


class _Patchify(nn.Module):                      # Conv2dPatchify, modular :427-448
    def __init__(self, patch, d, channels):
        super().__init__()
        self.patchify = nn.Conv2d(channels, d, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.patchify(x).flatten(2).transpose(1, 2)


class _PatchEmbeddings(nn.Module):               # PatchEmbeddings, modular :451-527
    def __init__(self, image_size, patch, d, channels):
        super().__init__()
        self.embeddings = _Patchify(patch, d, channels)
        # Upstream initialises sin-cos (modular :463-466). Learned, and
        # overwritten by the strict load, so the init never reaches a feature.
        self.position_embeddings = nn.Parameter(torch.zeros(1, (image_size // patch) ** 2, d))

    def forward(self, x):
        e = self.embeddings(x)
        if e.shape[1] != self.position_embeddings.shape[1]:
            # Upstream would bicubic-interpolate (modular :469-514). The probe
            # is always 224 px, so a mismatch means a wrong input, not a feature.
            raise ValueError(f'{e.shape[1]} patches vs {self.position_embeddings.shape[1]} '
                             f'position embeddings: input is not 224x224')
        return e + self.position_embeddings


class _SharedMlp(nn.Module):                     # SharedMlp, modular :390-400
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(approximate='tanh'))

    def forward(self, x):
        return self.net(x)


class _MaxPool(nn.Module):                       # MaxPool(-2), modular :403-409
    def forward(self, x):
        return x.max(dim=-2)[0]


class _PointGroupEmbedding(nn.Module):           # PointGroupEmbedding (DP3), modular :412-424
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(_SharedMlp(3, 64), _SharedMlp(64, 128), _SharedMlp(128, 256),
                                 _MaxPool(), nn.Linear(256, d))

    def forward(self, x):
        return self.net(x)


def farthest_point_sample(x, k):
    """pytorch3d.ops.sample_farthest_points(x, K=k) indices, random_start_point=False.

    The first centre is index 0. Each next one is the argmax of the running min
    squared distance (first index on ties), as in pytorch3d's
    sample_farthest_points_naive. Deterministic given the point order.
    """
    B, N, _ = x.shape
    idx = torch.zeros(B, k, dtype=torch.long, device=x.device)
    mind = torch.full((B, N), float('inf'), dtype=x.dtype, device=x.device)
    sel = torch.zeros(B, dtype=torch.long, device=x.device)
    ar = torch.arange(B, device=x.device)
    for i in range(k):
        idx[:, i] = sel
        mind = torch.minimum(mind, ((x - x[ar, sel][:, None, :]) ** 2).sum(-1))
        sel = mind.argmax(dim=-1)
    return idx


def knn_group(centers, x, k):
    """pytorch3d.ops.knn_points(centers, x, K=k, return_nn=True).knn: (B,S,k,3).

    Exact squared distances. torch.cdist's default matmul path is off by
    rounding and could swap the k-th neighbour. Neighbour order is irrelevant:
    the group encoder max-pools.
    """
    d2 = ((centers[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1)     # (B,S,N)
    nn_idx = d2.topk(k, dim=-1, largest=False).indices
    return x[torch.arange(x.shape[0], device=x.device)[:, None, None], nn_idx]


class _PointCloudEmbeddings(nn.Module):          # EmbodiedMAEPointCloudEmbeddings, modular :552-573
    def __init__(self, d, centers, knn):
        super().__init__()
        self.num_centers, self.num_knn = centers, knn
        self.knn_embeddings = _PointGroupEmbedding(d)
        self.center_embeddings = nn.Sequential(nn.Linear(3, d), nn.GELU(approximate='tanh'),
                                               nn.Linear(d, d))

    def forward(self, pc):
        pc = pc.float()                          # fps_and_knn runs in fp32 (modular :292)
        idx = farthest_point_sample(pc, self.num_centers)
        centers = pc[torch.arange(pc.shape[0], device=pc.device)[:, None], idx]
        groups = knn_group(centers, pc, self.num_knn) - centers.unsqueeze(-2)
        return self.center_embeddings(centers) + self.knn_embeddings(groups)


class EmbodiedMAEEncoder(nn.Module):
    """EmbodiedMAEModel's encoder path at add_mask=False: embeddings, 12 layers, LayerNorm."""

    def __init__(self, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 mlp_ratio=4, layer_norm_eps=1e-6, image_size=224, patch_size=16,
                 num_pc_centers=196, num_pc_knn=64):
        super().__init__()
        d = hidden_size
        self.rgb_embeddings = _PatchEmbeddings(image_size, patch_size, d, 3)
        self.depth_embeddings = _PatchEmbeddings(image_size, patch_size, d, 1)
        self.pc_embeddings = _PointCloudEmbeddings(d, num_pc_centers, num_pc_knn)
        self.encoder = _Encoder(d, num_hidden_layers, num_attention_heads, mlp_ratio,
                                layer_norm_eps)
        self.layernorm = nn.LayerNorm(d, eps=layer_norm_eps)

    def embed(self, rgb, depth, pc):
        """(B, 588, D) pre-encoder tokens, [rgb | depth | pc], every modality required."""
        if rgb is None or depth is None or pc is None:
            # Upstream zero-fills a missing modality and masks it; this baseline's
            # row claims all three, so a missing one is a bug, not a variant.
            raise ValueError('embodiedmae needs rgb, depth and pc')
        return torch.cat([self.rgb_embeddings(rgb), self.depth_embeddings(depth),
                          self.pc_embeddings(pc)], dim=1)

    def forward(self, rgb, depth, pc):
        """(B, 588, D) post-LayerNorm tokens = upstream `.last_hidden_states`."""
        return self.layernorm(self.encoder(self.embed(rgb, depth, pc)))


# ─────────────────────────────── weights ─────────────────────────────────────

_ST_DTYPES = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
              'F64': torch.float64}


def read_safetensors(path):
    """safetensors -> {key: tensor} without the safetensors package (absent in det_cu128).

    Format: u64 little-endian header length, a JSON header of
    {key: {dtype, shape, data_offsets}}, then the raw little-endian buffer.
    """
    if sys.byteorder != 'little':
        raise RuntimeError('safetensors data is little-endian; this reader assumes a '
                           'little-endian host')
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        buf = bytearray(f.read())
    out = {}
    for key, info in header.items():
        if key == '__metadata__':
            continue
        dt = _ST_DTYPES.get(info['dtype'])
        if dt is None:
            raise ValueError(f'{key}: unsupported safetensors dtype {info["dtype"]}')
        lo, hi = info['data_offsets']
        count = (hi - lo) // torch.empty((), dtype=dt).element_size()
        t = torch.frombuffer(buf, dtype=dt, count=count, offset=lo) if count else \
            torch.empty(0, dtype=dt)
        out[key] = t.reshape(info['shape'])
    return out


def checkpoint_path(download=True):
    """The pinned model.safetensors in the shared HF cache. Downloads only on a miss."""
    from . import hf_hub_cache
    cache = hf_hub_cache()
    # HF cache layout: models--<org>--<name>/snapshots/<commit>/<file>. Resolved
    # by hand so that an offline job needs no huggingface_hub (det_cu128 has none).
    path = cache / f'models--{HF_REPO.replace("/", "--")}' / 'snapshots' / HF_REVISION / HF_FILE
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(f'{path} missing: run eval/baseline_probe.py --prefetch '
                                f'--baselines {NAME} on a node with internet')
    if os.environ.get('HF_HUB_OFFLINE', '').strip().upper() in ('1', 'ON', 'YES', 'TRUE'):
        raise FileNotFoundError(f'{path} missing and HF_HUB_OFFLINE is set: run '
                                f'eval/baseline_probe.py --prefetch --baselines {NAME} '
                                f'with the same HF_HOME / --weights-dir first')
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(f'{path} missing and huggingface_hub is not installed in this env '
                          f'(det_cu128 lacks it): prefetch from the `det` env') from e
    print(f'  downloading {HF_REPO}/{HF_FILE} @ {HF_REVISION[:7]} -> {cache}')
    return Path(hf_hub_download(HF_REPO, HF_FILE, revision=HF_REVISION, cache_dir=str(cache)))


def load_checkpoint(download=True):
    """(encoder state dict, dropped decoder keys, dropped value count), after size + sha256."""
    from . import verify_file
    path = verify_file(checkpoint_path(download), size=CKPT_BYTES, sha256=CKPT_SHA256)
    sd = read_safetensors(path)
    dropped = sorted(k for k in sd if k.startswith('decoder.'))
    enc = {k: v for k, v in sd.items() if not k.startswith('decoder.')}
    # The decoder is the only thing left out, and exactly all of it: a
    # different count means a different checkpoint or a wrong prefix filter.
    if len(dropped) != N_DECODER_KEYS or len(enc) != N_ENCODER_KEYS:
        raise RuntimeError(f'expected {N_ENCODER_KEYS} encoder + {N_DECODER_KEYS} decoder keys, '
                           f'got {len(enc)} + {len(dropped)}')
    return enc, dropped, sum(sd[k].numel() for k in dropped)


def build(device, cache_dir):
    """Frozen EmbodiedMAE-base encoder, strict-loaded. `cache_dir` (torch.hub) is unused: HF file."""
    from . import strict_load
    enc, dropped, n_dropped = load_checkpoint(download=True)
    model = EmbodiedMAEEncoder(**_CFG)
    strict_load(model, enc, f'EmbodiedMAE-base encoder ({HF_REVISION[:7]})', dropped=dropped)
    n = sum(p.numel() for p in model.parameters())
    if n != N_ENCODER_PARAMS:
        raise RuntimeError(f'encoder has {n:,} params, expected {N_ENCODER_PARAMS:,}')
    print(f'  ✓ dropped exactly the {len(dropped)} decoder.* keys ({n_dropped:,} values); '
          f'encoder {n:,} params')
    return model.eval().to(device)


def features(model, batch):
    """Mean of the 588 post-LayerNorm tokens of all-visible RGB+D+PC."""
    from . import ctx
    c = ctx(model)
    if c.feature != 'mean':
        raise ValueError(f'embodiedmae has no CLS token; feature {c.feature!r} unsupported')
    rgb, depth, pc = to_upstream(batch, c.species)
    return model(rgb, depth, pc).mean(dim=1)
