"""pointmae -- official Point-MAE ShapeNet-55 pretrain, frozen, PC only.

The standard single-modality point-cloud MAE, and the family our PC
tokenizer comes from (FPS + kNN groups -> mini-PointNet tokens). The row
asks: does a generic pretrained point encoder already give the phenotype
features that ours does, with no plant data and no RGB/depth?

WEIGHTS AND CODE
  Release checkpoint `pretrain.pth` (README.md:72, the "Pre-training / ShapeNet"
  row): 300 epochs on ShapeNet-55, 1024 points, 64 groups x 32, mask 0.6 rand
  (cfgs/pretrain.yaml). Only `ckpt['base_model']['module.MAE_encoder.*']`
  is loaded: 156 keys, strict. The other 53 keys are the pretraining decoder
  (MAE_decoder 46, decoder_pos_embed 4, increase_dim 2, mask_token 1), dropped
  as an asserted set. `ckpt['metrics'] == {'acc': 0.0}` is NOT a broken file:
  upstream's SVM validation is commented out (tools/runner_pretrain.py) and
  was never run.
  The model code is vendored as pure torch in _vendor_pointmae/ (MIT,
  7445a68). pointnet2_ops needs a CUDA build against the installed torch, and
  the knn_cuda wheel URL in upstream's README returns 404. So the upstream
  install path is dead, and importing the upstream package is not an option.

INPUT CONVERSION, this repo's batch -> upstream convention
  batch pc: (B, N, 3), N = 8196 sorghum / 8192 maize. A seeded random
  subsample of the camera-frame cloud, centred on its centroid and scaled to
  max norm 1 (sorghum_dataset.py load_pointcloud; maize_dataset_4m.py).
  1. Normalisation: none added. The loader's centre + unit-sphere scaling is
     upstream's pc_norm / pc_normalize verbatim (datasets/ShapeNet55Dataset.py
     pc_norm; datasets/ModelNetDataset.py:20-25).
  2. 1024 points by FPS, with NO re-normalisation afterwards. This is
     upstream's own downstream and frozen-feature path: ModelNetDataset
     normalises the full cloud (:135), and the runner then does
     `misc.fps(points, npoints)` with npoints=1024 (tools/runner_finetune.py:241
     test; tools/runner_pretrain.py:216,:229 validate). The FPS is a port of
     the pointnet2_ops kernel: start at index 0, never pick |p|^2 <= 1e-3.
     Grouping all N ~ 8x1024 points directly would make each 32-point
     neighbourhood cover ~1/8 of the surface it covered in pretraining, off
     the mini-PointNet's training scale.
     Pretraining itself used a random 1024 then pc_norm (ShapeNet55Dataset
     :54-65). Measured on 16 sorghum val plants: masked-group
     reconstruction/trivial 0.102 for FPS vs 0.112 for random+renorm, and
     0.102 with or without re-normalising after the FPS. So the choice
     changes nothing, and FPS is deterministic.
  3. Axes: gravity-aligned where a FIXED map does it, camera frame otherwise.
     Point-MAE was pretrained on y-up ShapeNet with no rotation augmentation
     (runner_pretrain.py:17-27, only PointcloudScaleAndTranslate). Our clouds
     are OpenGL camera frame, and at the probe's view 00:
       sorghum  ONE camera rotation for every plant: all 15,000 view-00
                camera_pose.json cameraToWorld rotations are identical to
                5.6e-16 (full scan, 2026-09-25; only the translation varies).
                World-up is (0, 0.436, 0.900) in camera axes, 64.2 deg off
                camera +y. So the rotation into the y-up world is a constant,
                like EmbodiedMAE's GL->CV flip: it carries no per-plant
                information, and it is APPLIED (SORGHUM_VIEW00_CAM_TO_WORLD,
                hard-coded; no file is read).
       maize    a random camera per plant: world-up 0-86 deg off camera +y,
                median ~30 (150 plants per split). No fixed map exists, and
                the per-plant pose is information the arms never get, so the
                maize row stays in the camera frame.
     What it costs, camera frame vs gravity-aligned. Masked reconstruction /
     trivial on 16 val plants barely moves: sorghum 0.102 vs 0.100, maize
     0.126 vs 0.111. The mini-PointNet sees centre-relative patches, so
     reconstruction is nearly orientation-blind. The FEATURES are not: the
     global pos_embed sees the tilt. Probe val R2 ('cls', first 320 train /
     160 val plants, 2026-09-25, before sorghum was aligned):
                        height  leaf_count  leaf_angle  biomass  leaf_twist
       sorghum camera   0.904     0.914       0.500      0.899      -
               upright  0.918     0.921       0.521      0.913      -
       maize   camera   0.267     0.848       0.657      0.558     0.316
               upright  0.553     0.922       0.909      0.692     0.804
     Sorghum's fixed rotation is worth ~0.01-0.02, and the headline sorghum
     row now has it. Maize's per-plant tilt costs up to 0.49, because a model
     pretrained without rotation augmentation has to untangle an unseen
     nuisance. Our arms saw random views in training, and Point-MAE did not.
     So on maize this row measures robustness to camera pose as much as it
     measures the representation, and it is published ONLY beside
     pointmae_upright, the sensitivity row that prices it: it reads each
     view's camera_pose.json (USES_CAMERA_POSE), i.e. pose the arms never
     get, and is captioned as such. On sorghum the two rows coincide.
  4. No scale conversion. Max norm 1 is exactly pretraining's pc_norm output,
     before the PointcloudScaleAndTranslate augmentation (x U(2/3, 3/2)).
     Reconstruction/trivial is 0.110 at x0.5, 0.102 at x1 and 0.109 at x2,
     so unit scale is the model's own optimum.

FEATURES, and why 'cls' is not a CLS token
  The pretrain encoder has NO CLS token: zero checkpoint keys contain 'cls'.
  PointTransformer's cls_token/cls_pos (models/Point_MAE.py:436-437) are
  created at fine-tuning and trained there. Upstream never defined a frozen
  feature: its pretrain SVM path is dead code (validate() calls a model whose
  forward returns a loss). So:
    'cls'  = concat[max-pool, mean-pool] of the 64 normed group tokens, 768-D.
             Max is the global statistic the fine-tune head uses
             (x[:, 1:].max(1), Point_MAE.py:546). Mean is the other half of
             Point-M2AE's linear-SVM feature, x_vis.mean(1) + x_vis.max(1)[0]
             (Point-M2AE models/Point_M2AE.py:258). Concatenation, not their
             sum, because a ridge on the concatenation can represent the sum,
             but not the reverse. It is also 768-D, the arms' CLS width.
    'mean' = mean-pool of the 64 tokens, 384-D. Same semantics as the arms'
             'mean' (the mean over non-CLS tokens).
  The CSV `feature` column says 'max+mean' for the first (FEATURE_LABELS), so
  a table filtered on feature == 'cls' can never set it beside real CLS
  tokens. Caption: "Point-MAE [max || mean] token pool (no CLS token)".

DETERMINISM. Nothing here draws from any RNG. FPS starts at index 0, and the
kNN is exact. So the features are a pure function of the loader's seeded
subsample, and --repeats > 1 is a no-op for this baseline.

VERIFIED 2026-09-25 on CPU (`python eval/baselines/pointmae.py --self-check
--species {sorghum,maize}`, first 16 val plants, view 00, the same four 60%
masks for every model). All 209 checkpoint keys load strictly into the full
pretraining model. Masked-group CD-L2, as a multiple of predicting every point
at its group centre:
                 pretrained   encoder random, decoder pretrained   all random
    sorghum        0.102                   0.392                     46.7
    maize          0.126                   0.727                     77.3
Pretrained beats the random-encoder hybrid on 16/16 plants in each species.
The hybrid is the control that matters: the decoder is told every masked
centre, so a pretrained decoder alone already gets part of the way. The 4-6x
gap is what the ENCODER weights add. So they are the trained ones, and the
inputs are in the convention they were trained on.
Also checked: the FPS port returns the same indices as a sequential
reference of the kernel's rules, including the skip ball. Features are
bitwise-reproducible across runs. A (B=4) batch and four (B=1) batches agree
to 1.4e-6. Capped (48/split) runs of eval/baseline_probe.py complete for both
species, and the CSV header is identical to reports/probe_probe_e2_*.csv.
"""

if __name__ == '__main__' and not __package__:
    # `python eval/baselines/pointmae.py --self-check`: re-enter as baselines.pointmae,
    # with the repo root (datasets, models) and eval/ (the probes) on sys.path.
    import sys as _sys, pathlib as _pathlib
    _h = _pathlib.Path(__file__).resolve()
    _sys.path[:0] = [str(_h.parents[2]), str(_h.parents[1])]
    import baselines.pointmae as _m
    raise SystemExit(_m._main())

import os
from pathlib import Path

import torch

from . import ctx, strict_load, verify_file
from ._vendor_pointmae import point_mae as V

NAME = 'pointmae'
INPUTS = ('pc',)
URL = 'https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth'
SIZE = 348_288_755
SHA256 = '27ded932bb0a2625d5a8eb006df199b2578598c774aee6d86b985300b6a5fd20'
SOURCE = (f'Point-MAE ShapeNet-55 pretrain (300 ep), {URL}, {SIZE:,} B, sha256 {SHA256}; '
          f'code github.com/Pang-Yatian/Point-MAE@7445a68 models/Point_MAE.py, vendored '
          f'pure-torch (eval/baselines/_vendor_pointmae); PC in camera frame, sorghum '
          f'rotated y-up by the one fixed view-00 camera rotation; FPS->1024, '
          f"64x32 groups; 'cls' = [max||mean] token pool (no CLS token)")
MODEL_SIZE = 'S-384 (21.8M)'     # ViT-S width: 384-D, 12 blocks, 6 heads
EPOCH = -1                       # external pretrain (the checkpoint says epoch 300)
FEATURES = ('cls', 'mean')
FEATURE_LABELS = {'cls': 'max+mean'}   # not a CLS token: see FEATURES in the docstring

# cameraToWorld rotation (row-major) of EVERY sorghum view-00 camera: all
# 15,000 are identical to 5.6e-16 (docstring step 3). world = R @ camera.
SORGHUM_VIEW00_CAM_TO_WORLD = (
    (0.004881331007626391, -0.8999892776095454, 0.4358847012633533),
    (0.0, 0.43588989435406733, 0.9),
    (-0.999988086232828, -0.004393197906863753, 0.0021277228572215002))

NPOINTS = 1024                   # cfgs/pretrain.yaml npoints; runner fps(points, npoints)
NUM_GROUP, GROUP_SIZE = 64, 32   # cfgs/pretrain.yaml
ENC_PREFIX = 'module.MAE_encoder.'
N_ENC_KEYS = 156
# The pretraining-only keys, by top-level module and count: anything else is fatal.
DROPPED = {'module.MAE_decoder.': 46, 'module.decoder_pos_embed.': 4,
           'module.increase_dim.': 2, 'module.mask_token': 1}


class PointMAEFeatures(torch.nn.Module):
    """Group divider + pretrained encoder. Only MAE_encoder holds weights."""

    def __init__(self):
        super().__init__()
        self.group_divider = V.Group(num_group=NUM_GROUP, group_size=GROUP_SIZE)
        self.MAE_encoder = V.MaskTransformer()

    def tokens(self, pc):
        """(B, N, 3) repo cloud -> (B, 64, 384) normed group tokens, no masking."""
        pts = to_upstream(pc)
        neighborhood, center = self.group_divider(pts)
        tok, _ = self.MAE_encoder(neighborhood, center)       # noaug path: every group visible
        return tok


def to_upstream(pc):
    """This repo's cloud -> Point-MAE's input. See the docstring, steps 1-4.

    Only the FPS is an operation. Normalisation, axes and scale are deliberate
    no-ops, and are written out here so a later edit has to face them.
    """
    if pc.ndim != 3 or pc.shape[-1] != 3:
        raise ValueError(f'pointmae expects (B, N, 3) xyz, got {tuple(pc.shape)}')
    if pc.shape[1] < NPOINTS:
        raise ValueError(f'{pc.shape[1]} points < {NPOINTS}: FPS would repeat points')
    x = pc.float()
    # A cloud that is not unit-sphere normalised means the loader changed
    # under us. Upstream's scale is load-bearing (see step 4), so fail rather
    # than feed it.
    r = x.norm(dim=-1).amax(dim=1)
    if not torch.allclose(r, torch.ones_like(r), atol=1e-4):
        raise ValueError(f'pc max norm {r.min().item():.4f}..{r.max().item():.4f}, expected 1: '
                         f'the loader no longer applies pc_norm')
    return V.fps(x, NPOINTS)          # (B, 1024, 3), no re-normalisation (ModelNet path)


def checkpoint_path(cache_dir):
    # Not torch.hub's default basename: 'pretrain.pth' in a shared checkpoints/
    # dir is a collision waiting to happen.
    return Path(cache_dir) / 'checkpoints' / 'pointmae' / 'pointmae_pretrain.pth'


def _offline():
    return os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes')


def fetch(cache_dir):
    """Download on a miss (unless offline), then verify size + sha256 on EVERY call.

    torch.hub checks the hash only when it downloads, so a truncated or
    substituted file already in the cache would otherwise load unchecked.
    """
    path = checkpoint_path(cache_dir)
    if not path.is_file():
        if _offline():
            raise FileNotFoundError(f'{path} missing and HF_HUB_OFFLINE is set: run '
                                    f'`python eval/baseline_probe.py --prefetch --baselines '
                                    f'pointmae` on a node with internet, same TORCH_HOME')
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'  downloading {URL}\n    -> {path}')
        # Downloads to a temp file in the same dir and moves it into place, so a
        # killed download never leaves a partial file under the final name.
        torch.hub.download_url_to_file(URL, str(path), hash_prefix=SHA256, progress=False)
    return verify_file(path, size=SIZE, sha256=SHA256)


def load_checkpoint(cache_dir):
    ck = torch.load(fetch(cache_dir), map_location='cpu', weights_only=True)
    if ck.get('epoch') != 300:
        raise RuntimeError(f"checkpoint epoch {ck.get('epoch')}, expected 300: not the release file")
    return ck['base_model']


def split_encoder(sd):
    """(encoder state_dict with the prefix stripped, dropped keys). Fatal on any surprise."""
    enc = {k[len(ENC_PREFIX):]: v for k, v in sd.items() if k.startswith(ENC_PREFIX)}
    dropped = [k for k in sd if not k.startswith(ENC_PREFIX)]
    counts = {p: sum(k.startswith(p) for k in dropped) for p in DROPPED}
    unexplained = [k for k in dropped if not any(k.startswith(p) for p in DROPPED)]
    if len(enc) != N_ENC_KEYS or counts != DROPPED or unexplained:
        raise RuntimeError(f'Point-MAE checkpoint layout changed: {len(enc)} encoder keys '
                           f'(expected {N_ENC_KEYS}), dropped {counts} (expected {DROPPED}), '
                           f'unexplained {unexplained[:5]}')
    if any('cls' in k for k in sd):
        raise RuntimeError("checkpoint has a 'cls' key: the pooled 'cls' definition "
                           "below no longer describes this model")
    return enc, dropped


def build(device, cache_dir):
    enc, dropped = split_encoder(load_checkpoint(cache_dir))
    model = PointMAEFeatures()
    strict_load(model.MAE_encoder, enc, 'Point-MAE MAE_encoder', dropped=dropped)
    print('    dropped: ' + ', '.join(f'{k.replace("module.", "")}* x{n}'
                                     for k, n in DROPPED.items())
          + ' (pretraining decoder)')
    # 21,825,024 parameters; the 21,826,306 values strict_load reports add the
    # two BatchNorm layers' running stats.
    n = sum(p.numel() for p in model.parameters())
    if n != 21_825_024:
        raise RuntimeError(f'encoder has {n:,} parameters, expected 21,825,024')
    # eval() is load-bearing: the mini-PointNet has BatchNorm1d, and in train
    # mode each group's token would depend on the rest of the batch.
    return model.eval().to(device)


def orient(pc, species, names=None):
    """Docstring step 3: the fixed y-up rotation where one exists (sorghum), else identity."""
    if species == 'sorghum':
        # The constant is view 00's. Another view is another camera, and this
        # rotation would then tilt the cloud instead of levelling it.
        other = [n for n in (names or ()) if not n.endswith('_00')]
        if other:
            raise ValueError(f'sorghum rotation is view 00\'s, got views {other[:3]}')
        R = torch.tensor(SORGHUM_VIEW00_CAM_TO_WORLD, dtype=torch.float64)
        return torch.einsum('ij,bnj->bni', R.to(pc.device, torch.float32), pc.float())
    if species == 'maize':
        return pc
    raise ValueError(f'no orientation convention for species {species!r}')


def pool(tok, feature):
    """(B, 64, 384) group tokens -> the pooled feature. 'cls' is NOT a CLS token."""
    if feature == 'cls':
        return torch.cat([tok.max(dim=1).values, tok.mean(dim=1)], dim=-1)
    if feature == 'mean':
        return tok.mean(dim=1)
    raise ValueError(f'unknown feature {feature!r}')


def features(model, batch):
    _rgb, _depth, pc, _params, _text_valid, names = batch
    c = ctx(model)
    return pool(model.tokens(orient(pc, c.species, names)), c.feature)   # (B, 64, 384) -> (B, D)


# ───────────────────────────── self-check ────────────────────────────────────

def self_check(species, n_plants, split, hub, seed=0, n_masks=4, device='cpu'):
    """Masked-group reconstruction on real plants: proves weights AND input convention.

    Four predictors of the 60%-masked groups, each scored with upstream's
    ChamferDistanceL2 (centre-relative):
      pretrained  all 209 checkpoint keys, strict;
      enc-random  pretrained decoder, upstream-init random encoder: what the
                  ENCODER weights add. The decoder alone already knows every
                  masked centre, so a pretrained decoder is not proof on its own;
      random      the whole model at upstream init;
      trivial     every point at its group centre.
    The same masks are used for all four, and the input goes through the
    adapter's own to_upstream(). Returns a dict of mean CD-L2.
    """
    import numpy as np
    import baseline_probe as BP

    sp = BP.get_species(species)
    ds = sp.dataset(sp.data_root, img_size=224, num_points=sp.num_points, split=split,
                    max_leaves=sp.max_leaves, view_sampling=True, deterministic_view=True)
    np.random.seed(seed)          # load_pointcloud's subsample; no worker_init in-process
    pc = torch.stack([ds[i][2] for i in range(n_plants)]).to(device)

    sd = {k[len('module.'):]: v for k, v in load_checkpoint(hub).items()}
    pre = V.PointMAE()
    strict_load(pre, sd, 'Point-MAE full pretraining model (self-check)')
    torch.manual_seed(seed)
    rnd = V.PointMAE()
    hyb = V.PointMAE()
    hyb.load_state_dict(sd, strict=True)
    hyb.MAE_encoder = V.MaskTransformer()               # upstream init, seeded
    models = {'pretrained': pre, 'enc-random': hyb, 'random': rnd}
    for m in models.values():
        m.eval().to(device)

    g = torch.Generator().manual_seed(seed)
    masks = [V.rand_mask(n_plants, NUM_GROUP, 0.6, generator=g, device=device)
             for _ in range(n_masks)]
    out = {}
    pc = orient(pc, species)       # the headline input: sorghum gravity-aligned
    with torch.no_grad():
        nb, center = pre.group_divider(to_upstream(pc))
        for tag, m in models.items():
            per = [V.chamfer_l2_per_group(*m.reconstruct(nb, center, mk)).reshape(n_plants, -1).mean(1)
                   for mk in masks]
            out[tag] = torch.stack(per).mean(0)
        triv = []
        for mk in masks:
            gt = nb[mk].reshape(-1, GROUP_SIZE, 3)
            triv.append(V.chamfer_l2_per_group(torch.zeros_like(gt), gt).reshape(n_plants, -1).mean(1))
        out['trivial'] = torch.stack(triv).mean(0)

        # Feature-level: the pooled 'cls' feature the probe will see.
        feat = PointMAEFeatures()
        feat.MAE_encoder.load_state_dict(pre.MAE_encoder.state_dict(), strict=True)
        tok = feat.eval().to(device).tokens(pc)
        f = torch.cat([tok.max(1).values, tok.mean(1)], -1)
        sd_f = f.std(0).median().item()

    z = out['trivial'].mean().item()
    print(f'\nself-check: {species} {split}, {n_plants} plants (view 00), '
          f'{n_masks} masks x 60% of {NUM_GROUP} groups, CD-L2 (upstream ChamferDistanceL2)')
    for tag in ('pretrained', 'enc-random', 'random', 'trivial'):
        v = out[tag]
        print(f'  {tag:11s} {v.mean().item():.4e}   / trivial {v.mean().item() / z:7.3f}')
    beats = (out['pretrained'] < out['enc-random']).float().mean().item()
    print(f'  pretrained beats enc-random on {beats * 100:.0f}% of plants; '
          f"'cls' feature {tuple(f.shape)}, per-dim std across plants median {sd_f:.4g}")
    ok = (out['pretrained'].mean() < 0.25 * z
          and out['pretrained'].mean() < 0.5 * out['enc-random'].mean()
          and beats == 1.0)
    print(f'SELF-CHECK {"PASS" if ok else "FAIL"}')
    return ok, {k: v.mean().item() for k, v in out.items()}


def _main():
    import argparse
    import baselines as B
    ap = argparse.ArgumentParser(description='Point-MAE adapter: masked-reconstruction self-check')
    ap.add_argument('--self-check', action='store_true', required=True)
    ap.add_argument('--species', choices=('sorghum', 'maize'), default='sorghum')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=16, help='plants (first N of the split)')
    ap.add_argument('--weights-dir', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    hub = B.configure_weight_cache(args.weights_dir)
    ok, _ = self_check(args.species, args.n, args.split, hub, device=args.device)
    return 0 if ok else 1
