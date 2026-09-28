"""dinov2_vitb14 -- DINOv2 ViT-B/14 (LVD-142M, no registers), RGB only.

The strongest general-purpose frozen image encoder at our encoder's width
(768, 12 blocks, ~86 M params against our base's 86.7 M encoder). It asks
whether a sorghum/maize-specific multi-modal pretrain beats "just use a
foundation model on the RGB render". It sees only the RGB image the arms see:
no depth, no point cloud, and never the spline params.

WEIGHTS AND CODE
  * Weights: the official release checkpoint, WEIGHTS_URL below, the same file
    upstream's hubconf `dinov2_vitb14()` fetches (dinov2/hub/backbones.py,
    _make_dinov2_model: `<base>/dinov2_vitb14/dinov2_vitb14_pretrain.pth`).
    It is a flat backbone state dict: 175 keys, 86,580,480 values, all fp32,
    no head and no decoder, so NOTHING is dropped and the strict load covers
    the whole file. Its size and sha256 are checked on EVERY build, not only on
    download: torch.hub checks the hash only when it downloads, so a truncated
    or substituted cached file would otherwise load unchecked.
  * Code: vendored, pure torch, in _vendor_dinov2_vitb14/ (Apache-2.0, upstream
    LICENSE alongside). Not torch.hub.load, because at the pinned commit the
    hubconf imports the whole hub package (Cell-DINO, XRay-DINO, dino.txt with
    ftfy/regex), needs trust_repo plus a GitHub API call, and runs whatever code
    sits in the cache directory. The vendored copy runs unchanged in det
    (torch 2.5) and det_cu128 (torch 2.11, no timm), offline, and needs only
    the one weight file from --prefetch.
  * The vendored model equals upstream's bit for bit (2026-09-24, CPU, det):
    built under the same seed, the random-init state dicts are bitwise-equal
    (same module tree, same init order). With the released weights,
    x_norm_clstoken, x_norm_patchtokens and x_prenorm are bitwise-equal
    (max|diff| 0) to torch.hub.load(<pinned dir>, 'dinov2_vitb14',
    source='local') on 4 sorghum + 4 maize val renders.

INPUT CONVERSION: none, and here is why that is correct.
  This repo:  PIL .convert('RGB') -> Resize((224, 224)) -> ToTensor ->
              Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
              (sorghum_dataset.py:155-159, maize_dataset_4m.py:135-140).
  DINOv2:     make_classification_eval_transform (dinov2/data/transforms.py:
              77-91): Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor ->
              make_normalize_transform, whose defaults are IMAGENET_DEFAULT_MEAN
              / IMAGENET_DEFAULT_STD = the same two triples (transforms.py:42-50).
              Channel order is RGB from PIL in both.
  So the normalisation is identical and the tensor is fed as is. Two
  deviations from upstream's eval transform are DELIBERATE, and both keep the
  baseline on exactly the tensor every arm sees:
    * no CenterCrop. Upstream crops 224 from a 256 resize, discarding a
      6.25 % margin on every side. On 48 sorghum val renders (view 00) that
      margin holds foreground in 25 plants, 22 of them at the TOP edge, i.e.
      the tallest leaf tips that carry the height signal the probe scores
      (median 0.07 %, max 4.9 % of foreground pixels). Maize: 0 of 48.
    * the resize is the repo's (PIL bilinear, antialiased, 1024 -> 224), not
      bicubic. Both are antialiased downsamples of the same render.
  Measured cost of both deviations together, on 48 sorghum + 48 maize val
  renders at view 00, comparing each render's feature with the feature of the
  SAME render under upstream's exact eval transform: cosine median 0.977 /
  0.965 (cls, sorghum / maize), 0.979 / 0.986 (mean). After removing the
  across-plant mean the cosine is 0.75-0.87, and every render's feature still
  retrieves its own upstream-transform feature among the 48 (top-1 1.000 in
  all four cases). So the choice moves the features a little and preserves
  plant identity completely.
  224 px input gives a 16 x 16 grid of 14-px patches. The checkpoint's
  pos_embed is trained at 518 (37 x 37) and is bicubically interpolated to
  16 x 16 with upstream's 0.1 offset (interpolate_pos_encoding, verbatim), which
  is what DINOv2's own 224-px linear evaluation does.
  Guard: features() refuses a tensor outside the ImageNet-normalised range of
  a [0, 1] image, [-2.118, 2.640], or one with no negative value at all (an
  un-normalised [0, 1] image). Either means the loader changed underneath
  this adapter, which would otherwise give plausible, wrong features.

POOLING, i.e. what 'cls' and 'mean' mean here
  cls  = x_norm_clstoken, the final-LayerNorm CLS token (DINOv2 trains it
         directly: the DINO loss is on the CLS token). Same as model(x).
  mean = x_norm_patchtokens.mean(1), the 256 final-LayerNorm patch tokens,
         EXCLUDING CLS, as linear_probe's latent[:, 1:].mean(1) excludes ours.
  Both are post-final-norm, like the arms' latent (encoder_norm output).
  DINOv2's own linear protocol (hub/classifiers.py:60-68, layers=1) is
  concat[cls, mean] = 1536-d. The framework's modes are cls | mean, so the two
  rows together hold exactly its two halves; a concat row would need a
  'cls+mean' mode in baselines.VALID_FEATURES (linear_probe.py already has it
  for the arms). Report both rows for this baseline.

DETERMINISM
  No randomness in the forward pass and no BatchNorm, and LayerNorm is per
  token, so a row depends only on its own image: --repeats > 1 is a no-op.

VERIFIED 2026-09-24 (CPU; det unless stated)
  * Strict load: 175/175 keys, missing=[] unexpected=[], 0 dropped,
    86,580,480 params. The file's sha256 matches SOURCE, which also equals the
    fresh download (a --prefetch into an empty --weights-dir).
  * The weights are DINOv2's. DINOv2's released ImageNet linear head
    (dinov2_vitb14_linear_head.pth, Linear(1536, 1000) on concat[cls, mean],
    strict load) was applied to this adapter's two features() outputs. The
    input was the torch.hub Samoyed photo under THIS repo's RGB transform.
    Result: top-1 is class 258 'Samoyed' at p 0.600 (0.647 under upstream's
    own eval transform). The same head on the untrained network gives
    'chainlink fence' at p 0.021, and p(Samoyed) 0.0012.
  * That check does NOT test the normalisation. DINOv2 still says Samoyed on
    an un-normalised [0, 1] tensor (p 0.70) and on BGR (p 0.83): a white dog
    is nearly invariant to both mistakes. The normalisation rests on the two
    code paths using identical constants (above). The runtime guard is the
    protection: [0, 1] and [0, 255] tensors are both refused.
  * Two views of one plant. 48 val plants per species, views 00 -> 05, each
    view set centred on its own mean. Chance is top-1 0.021 and AUC 0.5,
    where AUC = P(same-plant cosine > a cross-plant cosine).
      - maize, cls: pretrained top-1 0.312, AUC 0.78, median rank 3.
        Untrained: top-1 0.083, AUC 0.66, median rank 16.
      - sorghum, mean: pretrained AUC 0.78, untrained 0.70.
      - sorghum, cls: pretrained and untrained both sit at AUC ~0.7. Sorghum
        plants come from one generator and look alike, and a random ViT
        already keys on plant extent.
  * The pretrained features differ strongly from the untrained ones. The
    untrained features are nearly constant across plants: mean cross-plant
    cosine 0.988-0.999, against 0.73-0.93 pretrained. Linear CKA between the
    two, over plants, is 0.43-0.49 on cls and 0.34-0.81 on mean.
  * Rows the probe caches are bitwise-equal to a direct forward_features on
    the same loader tensors, even though the batch sizes differ (32 vs 8).
    By the vendoring check they therefore equal upstream torch.hub's output.
  * det_cu128 (torch 2.11, no timm / einops / huggingface_hub) builds through
    this adapter and matches det to max|diff| 1.7e-5, relative 2.7e-6.
"""

import os
from pathlib import Path

import torch

NAME = 'dinov2_vitb14'
INPUTS = ('rgb',)
FEATURES = ('cls', 'mean')
MODEL_SIZE = 'base'          # ViT-B: 768 wide, 12 blocks, 12 heads
EPOCH = -1                   # external pretrain

WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
WEIGHTS_FILE = 'dinov2_vitb14_pretrain.pth'       # torch.hub's own filename for it
WEIGHTS_SIZE = 346_378_731
WEIGHTS_SHA256 = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
N_KEYS = 175
N_PARAMS = 86_580_480
CODE_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'

SOURCE = (f'DINOv2 ViT-B/14 LVD-142M backbone, {WEIGHTS_URL} '
          f'({WEIGHTS_SIZE} bytes, sha256 {WEIGHTS_SHA256}); model code vendored '
          f'from github.com/facebookresearch/dinov2@{CODE_COMMIT} (Apache-2.0)')

# ImageNet-normalised range of a [0, 1] image: (0 - mean)/std .. (1 - mean)/std
# over the three channels (transforms.py IMAGENET_DEFAULT_MEAN / _STD).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_LO = min((0.0 - m) / s for m, s in zip(_MEAN, _STD))      # -2.1179
_HI = max((1.0 - m) / s for m, s in zip(_MEAN, _STD))      # +2.6400


def _offline():
    return os.environ.get('HF_HUB_OFFLINE', '').strip().lower() in ('1', 'true', 'yes', 'on')


def weights_path(cache_dir):
    """<torch.hub dir>/checkpoints/dinov2_vitb14_pretrain.pth: where torch.hub
    itself would put it, so an existing hub download is reused as is."""
    return Path(cache_dir) / 'checkpoints' / WEIGHTS_FILE


def fetch(cache_dir):
    """The verified checkpoint path, downloading it first if it is missing.

    With HF_HUB_OFFLINE=1 a missing file is an immediate error rather than a
    download attempt: urllib has no timeout, and a compute node behind a
    firewall that drops packets would hang there instead of failing.
    """
    from . import verify_file
    path = weights_path(cache_dir)
    if not path.is_file():
        hint = ('run `python eval/baseline_probe.py --prefetch --baselines '
                f'{NAME}` on a node with internet, with the same TORCH_HOME / --weights-dir')
        if _offline():
            raise FileNotFoundError(f'{path} missing and HF_HUB_OFFLINE is set: {hint}')
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'  downloading {WEIGHTS_URL}\n           -> {path}')
        try:
            # temp file + move, and a full-length hash_prefix is a full sha256 check
            torch.hub.download_url_to_file(WEIGHTS_URL, str(path),
                                           hash_prefix=WEIGHTS_SHA256, progress=False)
        except Exception as e:
            raise FileNotFoundError(f'{path} missing and the download failed ({e}): {hint}') from e
    return verify_file(path, size=WEIGHTS_SIZE, sha256=WEIGHTS_SHA256)


def build(device, cache_dir):
    from . import strict_load
    from ._vendor_dinov2_vitb14 import vit_base_14

    path = fetch(cache_dir)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict) or len(sd) != N_KEYS:
        raise RuntimeError(f'{path.name}: expected a flat state dict of {N_KEYS} keys, '
                           f'got {type(sd).__name__} of {len(sd) if hasattr(sd, "__len__") else "?"}')

    # Construction runs upstream's random init; keep it from shifting the
    # global RNG for anything seeded later in this process.
    state = torch.random.get_rng_state()
    try:
        model = vit_base_14()
    finally:
        torch.random.set_rng_state(state)

    # Every key, encoder and all: the file IS the backbone, so nothing is dropped.
    strict_load(model, sd, label=f'{NAME} <- {path.name}')
    n = sum(p.numel() for p in model.parameters())
    if n != N_PARAMS:
        raise RuntimeError(f'{NAME}: {n:,} params, expected {N_PARAMS:,}')
    pe = tuple(model.pos_embed.shape)
    print(f'  ✓ {NAME}: {n:,} params · pos_embed {pe} (518 px grid, interpolated to 16x16 at 224) '
          f'· LayerScale |gamma| block0 {model.blocks[0].ls1.gamma.abs().mean():.3f} '
          f'-> block11 {model.blocks[11].ls1.gamma.abs().mean():.3f} (1.0 at init)')
    return model.eval().to(device)


def _check_rgb(rgb):
    if rgb is None or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f'{NAME}: expected rgb (B, 3, H, W), got '
                         f'{None if rgb is None else tuple(rgb.shape)}')
    lo, hi = float(rgb.min()), float(rgb.max())
    if lo < _LO - 1e-3 or hi > _HI + 1e-3:
        raise ValueError(f'{NAME}: rgb spans [{lo:.3f}, {hi:.3f}], outside the ImageNet-'
                         f'normalised range [{_LO:.3f}, {_HI:.3f}]: the loader changed')
    if lo >= 0.0:
        raise ValueError(f'{NAME}: rgb has no negative value (min {lo:.3f}): looks like an '
                         f'un-normalised [0, 1] image, not the ImageNet-normalised tensor')


def features(model, batch):
    """(B, 768): final-norm CLS ('cls') or mean of the 256 patch tokens ('mean')."""
    from . import ctx
    rgb = batch[0]
    _check_rgb(rgb)
    # The loader's tensor is already DINOv2's input convention: see the docstring.
    out = model.forward_features(rgb)
    feature = ctx(model).feature
    if feature == 'cls':
        return out['x_norm_clstoken']
    if feature == 'mean':
        return out['x_norm_patchtokens'].mean(dim=1)      # excludes CLS
    raise ValueError(f'unknown feature {feature!r}')
