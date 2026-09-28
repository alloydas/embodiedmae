"""multimae -- official MultiMAE-B (RGB + depth), frozen: the multi-modal MAE baseline.

The closest image-side prior work. MultiMAE is a masked autoencoder over
several dense modalities, uses continuous patch tokens as our RGB/depth streams
do, and splits one visible-token budget across modalities with a Dirichlet
draw, which is where our masking scheme comes from. It never saw a point
cloud, so its row answers "how far does generic multi-modal (RGB+D) MAE
pretraining get you on plant phenotypes", not "PC vs no PC".

WEIGHTS: the release checkpoint pretrained on ImageNet-1K for 1600 epochs, on
RGB with Omnidata DPT pseudo-depth and COCO pseudo-semseg
(cfgs/pretrain/multimae-b_98_rgb+-depth-semseg_1600e.yaml). The filename says
"multivit", but the file is the full pretraining model: 351 keys, 98.6M values.
Exactly the 151 encoder keys are loaded, strict. Two groups are dropped, and
both are asserted as exact sets: the 4 `input_adapters.semseg.*` keys (we have
no semseg, and an input adapter only runs for a modality that is fed) and the
196 `output_adapters.*` keys (the rgb/depth/semseg/norm_rgb decoders).
Anything else missing or extra is fatal. The sha256 is checked on EVERY load,
because torch.hub only checks it when it downloads.

CODE: vendored pure-torch encoder in _vendor_multimae/ (upstream commit
66910f5, CC BY-NC 4.0). _vendor_multimae/__init__.py says why the upstream
package cannot be imported here.

INPUT CONVERSION, from this repo's batch to upstream's pretraining convention
  RGB   unchanged. Upstream pretraining is ToTensor + Normalize(
        IMAGENET_DEFAULT_MEAN/STD) (utils/datasets.py:98-100, constants at
        utils/data_constants.py:13-14). The flag that would switch to Inception
        stats defaults to True and is store_true, so it cannot be turned off
        (run_pretraining_multimae.py:184). Our loaders apply exactly those
        constants (sorghum_dataset.py:155-159, maize_dataset_4m.py:135-139). The
        full 224 frame goes in with no crop, as it does for every arm.
        Pretraining saw RandomResizedCrop(0.2-1) + hflip (utils/datasets.py:
        80-91), so that frame is inside its augmentation range.
  DEPTH three steps. Each one prevents a specific silent failure.
    1. Direction: none needed. Upstream depth is a 16-bit Omnidata DPT
       pseudo-label / 2**16 (utils/datasets.py:95-97, SETUP.md:102), z-buffer
       style, larger = farther. Ours is (z - near)/(far - near), also larger =
       farther: maize camera_pose.json says so, and a sorghum reprojection of
       *_nc_cam.ply gave z-buffered Spearman +0.49..+0.89 on 4 plants. From
       the model's side the sign is only weakly identified (VERIFIED below:
       sorghum prefers ours, maize cannot tell).
    2. Scale to ~metres: d x 50.0 (sorghum) or d x 2.0 (maize). The
       standardisation in step 3 is invariant to any positive affine map of
       depth EXCEPT through its eps=1e-6 guard. In raw units, sorghum's
       truncated foreground variance is 6.4e-5..1.5e-4 (48 val plants), so the
       guard alone would shrink z by ~0.6% (max |dz| 0.041), by a different
       amount per image. In metres it is negligible, as it is on upstream's
       [0,1] pseudo-depth. 50.0 is the fitted |z| = 50.0 d + 0.02..0.03 m
       (sorghum reprojection, residual 6-9 mm). For maize, exact metric depth is
       near + d(far - near) with a per-view near/far in camera_pose.json. Against
       d x 2.0 (mean far - near = 1.97 over 48 val views, range 1.59-2.40) the
       standardised result differs by at most 1.8e-5, so no per-sample file is
       read.
    3. Truncated standardisation over VALID pixels only, then everything else
       set to 0.0. This is upstream's own rule for depth with invalid pixels
       (run_finetuning_depth.py:671-688): NaN-fill the invalid pixels, sort,
       keep sorted valid values [int(0.1 n), int(0.9 n)), take their mean and
       unbiased var, compute (d - mean)/sqrt(var + 1e-6), then set invalid to
       0.0. VALID IS NOT d > 0. The loaders shrink a 1024 px render to 224 px
       with an antialiased bilinear resize, whose footprint is ~+-1 output px,
       so every silhouette edge becomes a band of partial-coverage pixels
       whose value is a mix of plant depth and background 0: they read too
       near, down to ~0. Native-resolution coverage says that band is ~31 %
       (sorghum) and ~67 % (maize) of the d > 0 pixels (32 val plants each),
       far more than the 10 % the slice drops, so statistics over d > 0 are
       taken over the halo: against the native plant's own 10-90 % statistics
       the mean lands -1.1 sigma (sorghum) / -3.1 sigma (maize) off and the
       spread is 2.2x / 2.8x too wide (medians over 96 plants). depth_valid()
       therefore erodes the d > 0 silhouette by CORE_ERODE_PX = 2, keeping the
       pixels whose footprint is all plant: -0.05 / -0.20 sigma, 0.96x / 1.01x
       (1-px erosion: -0.20 / -0.72 sigma; a fixed floor d >= 0.025 / 0.12:
       -0.20 / -1.19). The halo pixels become invalid (0.0) like the
       background. Feeding them standardised instead reconstructs masked plant
       depth 6.7x (sorghum) / 1.5x (maize) worse. On thin maize the 2-px core can vanish: below
       MIN_VALID_PIXELS (32 of 15,000 view-00 plants) the image falls back to
       a 1-px erosion. The rule reads only the arms' tensor.
       The PRETRAINING rule (run_pretraining_multimae.py:488-492) takes the
       same statistics over ALL pixels. On our renders the background is 72-97%
       of each image, so that slice is mostly or entirely background: the
       variance collapses and foreground lands at 60 sigma (sorghum) to 730
       sigma (maize), measured on 48 val plants each, a plausible-looking R2
       with no error raised. depth_to_multimae() raises when the MEDIAN |z| of
       an image's valid pixels exceeds MAX_MEDIAN_ABS_Z (5); under the rule
       above it is <= 1.32 over all 30,000 view-00 frames, while the max |z|
       reaches 43 / 79 on real leaf tips, so a max-based guard would kill a
       full run.
    Kept on purpose: upstream never pretrained on a zero-filled invalid
    region, since ImageNet pseudo-depth is dense. The 0.0 fill is upstream's
    fine-tuning convention and equals the robust plant mean. A "far" fill
    looks more physical but reconstructed masked plant depth 2.4-4.2x worse
    in the first (batched, see VERIFIED) check; not re-measured since.
  PC    not an input (INPUTS), never read.

TOKENS AND POOLING
  196 RGB + 196 depth tokens (224 px, patch 16), all visible, then ONE global
  token appended LAST: 393 tokens. This is upstream's unmasked MultiViT forward
  (multimae.py:439-491). Do not use MultiMAE.forward(mask_inputs=False): it
  still draws a Dirichlet and permutes the tokens (multimae.py:320-343).
  'cls'  = the global token, tokens[:, -1]. linear_probe's latent[:, 0] would
           be the top-left RGB patch, which is background.
  'mean' = the mean of the 392 patch tokens, EXCLUDING the global token, to
           match linear_probe's latent[:, 1:].mean(1). Upstream's
           LinearOutputAdapter includes it (output_adapters.py:349-350);
           measured cos(with, without) = 0.995 on both species.
  Both are then LayerNormed WITHOUT affine, eps 1e-6: exactly what upstream's
  own linear head applies before its Linear (output_adapters.py:301, :322,
  :355). The encoder has no final norm (multimae.py:94-98, and no norm key in
  the checkpoint). That head's affine is dropped because the probe's
  StandardScaler + ridge subsume it. Our arms' features are post-LayerNorm too.
  MultiMAE's native downstream pooling is MEAN (use_mean_pooling: True,
  cfgs/finetune/cls/ft_in1k_100e_multimae-b.yaml:7; run_finetuning_cls.py:
  155-157), so --feature mean is this baseline's headline row, and its cls row
  is reported alongside. The global token is nearly input-independent: three
  massive dims sit at about -320/266/158 with std < 0.7 across plants
  (48 val plants per species).

The train/probe token-count gap (98 visible tokens over three modalities in
pretraining, 392 at probe time) is the same gap our arms have (mask 0.8 in
training, every token at probe time), so it does not confound the comparison.

VERIFIED 2026-09-25 on CPU. The check scripts were one-off and are not in the
repo. They ran upstream MultiMAE at 66910f5 in a separate process, because its
utils/ package clashes with this repo's utils.py.
  * Strict load: 151 keys, missing=[] unexpected=[]. The 200 dropped keys are
    exactly 4 semseg-input + 196 decoder. Eight corrupted checkpoints are all
    fatal: a missing, renamed, extra or wrong-shape encoder key, a renamed
    global token, a partial decoder, a perturbed pos_emb, a re-saved file.
  * On real converted batches (48 val plants per species), the tokens equal
    upstream MultiViT's (commit 66910f5, strict-loaded) to max|diff| 0.0.
    det_cu128 (torch 2.11) against det: features max|diff| 2.4e-6. A
    sample's features do not depend on the rest of its batch: 0.0.
  * Masked reconstruction through upstream's full pretraining model (351
    keys, strict), 24 val plants per species x 2 masks, 98 of 392 tokens
    visible. ONE SAMPLE PER FORWARD: upstream MultiMAE.forward keeps
    (mask_all == 0).sum() tokens summed over the whole BATCH (multimae.py:338),
    so a batched call keeps every token visible and "masked" reconstruction
    becomes autoencoding. The first version of this check was batched, and
    its numbers (and an RGB-only depth "sign check" that was really reading
    the depth tokens) are withdrawn. NMSE = SSE/SST over the masked patches,
    1.0 = predicting their mean; depth scored on masked VALID pixels.
                                          sorghum   maize
      RGB, ours (ImageNet)                  0.21     0.32   random-init 9.5 / 47
      RGB fed as [-1,1] / [0,1]        0.24 / 0.26  0.67 / 0.79
      depth, ours (eroded valid, rest 0)    0.75     0.90   random-init 2.6 / 2.3
      depth, ours sign-flipped              0.84     0.90
      depth, stats over all d > 0 (pre-fix) 3.82     2.45
      depth, all-pixel pretraining rule     7.40    17.9
    So RGB is in convention, and the depth rule is the best of those tried,
    but depth is weak: masked plant depth is only modestly better than its
    mean on sorghum and barely on maize, and the depth sign is identified
    only on sorghum. Predicting depth from RGB alone (all depth masked) gives
    r = +0.01 / +0.10 with our depth: MultiMAE's ImageNet depth prior does
    not transfer to isolated plants on a blank background (it does put the
    background farther than the plant in 100 % of images). Caption the row
    as RGB+D with that caveat.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F

NAME = 'multimae'
INPUTS = ('rgb', 'depth')
FEATURES = ('cls', 'mean')
MODEL_SIZE = 'base'
# EPOCH is left at the framework default (-1, external pretrain). Its 1600
# ImageNet epochs are not on our step axis, and a number in that column would
# read as one.

CKPT_URL = ('https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/'
            'multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth')
CKPT_NAME = CKPT_URL.rsplit('/', 1)[1]
CKPT_SIZE = 394_406_915
CKPT_SHA256 = 'afff3f8c83f4e5a22f2edda21f00b2efeb077f3303c933dc4a133becf28c8345'
UPSTREAM_COMMIT = '66910f5b5ba236f5e731883db85fe4f24ee01106'

SOURCE = (f'MultiMAE-B, ImageNet-1K 1600 ep, RGB+Omnidata depth+COCO semseg pseudo-labels '
          f'(EPFL-VILAB/MultiMAE @ {UPSTREAM_COMMIT[:7]}, CC BY-NC 4.0); weights {CKPT_URL} '
          f'({CKPT_SIZE:,} B, sha256 {CKPT_SHA256}); encoder 151 keys strict, '
          f'semseg-in + decoders dropped; depth = truncated standardisation over the '
          f'2-px-eroded silhouette (run_finetuning_depth.py rule), halo + bg 0; '
          f'features LN(global token | patch mean)')

N_ENCODER_KEYS = 151
N_DROPPED_SEMSEG_IN = 4
N_DROPPED_DECODER = 196
N_TOKENS = 196 + 196 + 1

# ~metres per depth.png unit, which only sets where upstream's eps acts (see
# the docstring, DEPTH step 2)
DEPTH_M_PER_UNIT = {'sorghum': 50.0, 'maize': 2.0}
DEPTH_STD_EPS = 1e-6         # run_finetuning_depth.py:687
HEAD_LN_EPS = 1e-6           # output_adapters.py:301
# Valid depth = the d > 0 silhouette eroded by this many pixels (docstring DEPTH
# step 3): the loader's 1024->224 antialiased resize has a footprint of about
# +-1 output px, so the partial-coverage band is up to 2 px deep.
CORE_ERODE_PX = 2
# Below this many valid pixels, a 10-90% truncated variance is noise; the
# image then falls back to a 1-px erosion. Full view-00 scan, all three splits:
# sorghum core min 865 (never falls back); maize core < 100 on 32 of 15,000
# plants, whose 1-px erosion still keeps >= 604.
MIN_VALID_PIXELS = 100
# The collapse this guards against (the all-pixel pretraining rule on a mostly-
# background frame) puts the BULK of the plant at 60-730 sigma. A max-|z| guard
# cannot tell that from a real leaf tip: under the valid-core rule max |z|
# reaches 43 (sorghum) / 79 (maize) on real plants, while the per-image median
# |z| never exceeds 1.32 over all 30,000 view-00 frames.
MAX_MEDIAN_ABS_Z = 5.0


def _offline():
    return any(os.environ.get(v, '').upper() in ('1', 'ON', 'YES', 'TRUE')
               for v in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))


def _is_encoder_key(k):
    return k == 'global_tokens' or k.startswith(
        ('input_adapters.rgb.', 'input_adapters.depth.', 'encoder.'))


def checkpoint_path(cache_dir):
    """<torch hub>/checkpoints/<release name>, downloading on a miss unless offline."""
    path = Path(cache_dir) / 'checkpoints' / CKPT_NAME
    if path.is_file():
        return path
    if _offline():
        raise FileNotFoundError(f'{path} missing and HF_HUB_OFFLINE is set: run '
                                f'eval/baseline_probe.py --prefetch on a node with internet, '
                                f'with the same TORCH_HOME / --weights-dir')
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'  downloading {CKPT_URL}\n    -> {path}')
    try:
        # download_url_to_file writes to a temp file beside `path` and moves it
        # into place, so an interrupted download never leaves a truncated
        # checkpoint under the real name. hash_prefix is the full sha256.
        torch.hub.download_url_to_file(CKPT_URL, str(path), hash_prefix=CKPT_SHA256,
                                       progress=False)
    except Exception as e:
        raise RuntimeError(f'could not download the MultiMAE checkpoint ({e}); run '
                           f'--prefetch on a node with internet') from e
    return path


def build(device, cache_dir):
    """Frozen MultiMAE-B encoder, strict-loaded from the verified release checkpoint."""
    from . import strict_load, verify_file
    from ._vendor_multimae import MultiViTEncoder, build_2d_sincos_posemb

    path = verify_file(checkpoint_path(cache_dir), size=CKPT_SIZE, sha256=CKPT_SHA256)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if set(ck) != {'model'}:
        raise RuntimeError(f'{path.name}: top-level keys {sorted(ck)}, expected ["model"]')
    sd = ck['model']

    enc = {k: v for k, v in sd.items() if _is_encoder_key(k)}
    dropped = sorted(set(sd) - set(enc))
    semseg_in = [k for k in dropped if k.startswith('input_adapters.semseg.')]
    decoder = [k for k in dropped if k.startswith('output_adapters.')]
    other = sorted(set(dropped) - set(semseg_in) - set(decoder))
    # Assert exactly what is left out. "Everything not matching a prefix" would
    # also swallow a renamed encoder key, which strict=True could then not see.
    if (len(enc) != N_ENCODER_KEYS or len(semseg_in) != N_DROPPED_SEMSEG_IN
            or len(decoder) != N_DROPPED_DECODER or other):
        raise RuntimeError(
            f'{path.name}: {len(enc)} encoder keys (want {N_ENCODER_KEYS}), '
            f'{len(semseg_in)} semseg-input (want {N_DROPPED_SEMSEG_IN}), '
            f'{len(decoder)} decoder (want {N_DROPPED_DECODER}), unexplained {other[:5]}')

    model = MultiViTEncoder()
    strict_load(model, enc, 'MultiMAE-B encoder (rgb+depth adapters, global token, 12 blocks)',
                dropped=dropped)
    print(f'    dropped: {len(semseg_in)} input_adapters.semseg.* + {len(decoder)} '
          f'output_adapters.* ({", ".join(sorted({k.split(".")[1] for k in decoder}))})')

    # The checkpoint's pos_emb must be the fixed sin-cos grid (frozen in
    # pretraining, input_adapters.py:81-82). A mismatch would mean a different
    # token layout from the one this port assumes.
    ref = build_2d_sincos_posemb(14, 14, 768)
    for m in ('rgb', 'depth'):
        dpe = float((enc[f'input_adapters.{m}.pos_emb'] - ref).abs().max())
        if dpe > 1e-5:
            raise RuntimeError(f'input_adapters.{m}.pos_emb differs from sin-cos by {dpe:.3g}')
    n = sum(p.numel() for p in model.parameters())
    fp = float(sum(p.detach().double().abs().sum() for p in model.parameters()))
    print(f'    {n:,} encoder values; pos_emb == 2D sin-cos (rgb, depth); '
          f'sum|w| fingerprint {fp:.6e}')
    return model.eval().to(device)


def _erode(mask, px):
    """Binary erosion by `px` pixels (a (2px+1)^2 min-pool). The image border
    does not erode: max_pool2d pads with -inf, which the negation ignores."""
    k = 2 * px + 1
    return (-F.max_pool2d(-mask.float(), k, stride=1, padding=px)) > 0.5


def depth_valid(depth, names=None):
    """(B,1,H,W) bool: the pixels treated as depth measurements. Docstring DEPTH step 3.

    The d > 0 silhouette eroded by CORE_ERODE_PX, i.e. the pixels whose resize
    footprint is all plant. The band around it is a coverage-weighted mix of
    plant depth and background 0, which reads too near. Per image, if fewer
    than MIN_VALID_PIXELS survive (thin maize, 0.2 % of plants), a 1-px
    erosion is used instead. Only the arms' own tensor is read.
    """
    fg = depth > 0
    core = _erode(fg, CORE_ERODE_PX) & fg
    n = core.flatten(1).sum(dim=1)
    thin = (n < MIN_VALID_PIXELS).nonzero().flatten().tolist()
    if thin:
        e1 = _erode(fg, 1) & fg
        for i in thin:
            core[i] = e1[i]
    n = core.flatten(1).sum(dim=1)
    few = (n < MIN_VALID_PIXELS).nonzero().flatten().tolist()
    if few:
        who = [names[i] if names else i for i in few[:5]]
        raise RuntimeError(f'depth: < {MIN_VALID_PIXELS} valid (eroded) pixels in {who}: '
                           f'no robust statistics to standardise with')
    return core


def depth_to_multimae(depth, species, names=None):
    """(B,1,H,W) depth.png units, 0 = background -> upstream's standardised depth.

    Docstring DEPTH steps 2-3. The per-sample loop is upstream's
    (run_finetuning_depth.py:674-688), so the slice bounds and the unbiased
    variance match it exactly; only the definition of `valid` is ours.
    """
    if species not in DEPTH_M_PER_UNIT:
        raise KeyError(f'no depth scale for species {species!r}')
    valid = depth_valid(depth, names)
    n_valid = valid.flatten(1).sum(dim=1)

    d = depth * DEPTH_M_PER_UNIT[species]
    nan_depth = d.clone()
    nan_depth[~valid] = float('nan')
    trunc = torch.sort(nan_depth.flatten(1), dim=1)[0]        # NaN (invalid) sorts last
    lo = (n_valid * 0.1).long().tolist()
    hi = (n_valid * 0.9).long().tolist()
    means = torch.stack([trunc[i, a:b].mean() for i, (a, b) in enumerate(zip(lo, hi))])
    var = torch.stack([trunc[i, a:b].var() for i, (a, b) in enumerate(zip(lo, hi))])
    z = (d - means[:, None, None, None]) / torch.sqrt(var[:, None, None, None] + DEPTH_STD_EPS)
    z[~valid] = 0.0

    zmed = torch.stack([z[i][valid[i]].abs().median() for i in range(len(z))])
    bad = (zmed > MAX_MEDIAN_ABS_Z).nonzero().flatten().tolist()
    if bad:
        who = [(names[i] if names else i, round(float(zmed[i]), 1)) for i in bad[:5]]
        raise RuntimeError(f'depth: median |z| over valid pixels > {MAX_MEDIAN_ABS_Z} in {who}: '
                           f'the statistics collapsed (all-pixel rule?), features would be garbage')
    return z


def encode(model, rgb, depth, species, names=None):
    """(B, 393, 768) upstream tokens: [196 rgb][196 depth][global]."""
    tokens = model({'rgb': rgb, 'depth': depth_to_multimae(depth, species, names)})
    if tokens.shape[1] != N_TOKENS:
        raise RuntimeError(f'{tokens.shape[1]} tokens, expected {N_TOKENS} (224 px inputs)')
    return tokens


def pool(tokens, feature):
    """Docstring TOKENS AND POOLING: global token or patch mean, then the head's LayerNorm."""
    if feature == 'cls':
        x = tokens[:, -1]                  # global token is LAST (multimae.py:464-465)
    elif feature == 'mean':
        x = tokens[:, :-1].mean(dim=1)     # patches only, like linear_probe's latent[:, 1:]
    else:
        raise ValueError(f'unknown feature {feature!r}')
    return F.layer_norm(x, (x.shape[-1],), eps=HEAD_LN_EPS)


def features(model, batch):
    from . import ctx
    c = ctx(model)
    rgb, depth, _pc, _params, _text_valid, names = batch
    return pool(encode(model, rgb, depth, c.species, names), c.feature)
