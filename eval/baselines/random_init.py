"""random_init -- this repo's EmbodiedMAE4M base, UNTRAINED: the no-pretraining floor.

What a pretrained encoder has to beat before its pretraining counts for
anything. A random ViT over patch embeddings plus FPS/kNN point groups is not a
zero baseline: a random projection of the input already linearly encodes a lot
of plant size. Every other row is read relative to this one.

Architecture and feature path are the E2 arms' exactly:
  * embodied_mae_4m_base(active_modalities=('pc','rgb','depth')), built with
    the same kwargs as linear_probe.build_model rebuilds e2_pcrgbd from its
    config.json (img 224, 196 PC tokens, target_points 8196, qal_loss).
  * features() is linear_probe.extract_split's inner step verbatim:
    forward_encoder_select(rgb, depth, pc, zeroed params, visible=active-minus-
    text, source_mask_ratio=0.0)[0], then 'cls' = latent[:, 0] and 'mean' =
    latent[:, 1:].mean(1), which EXCLUDES CLS just as the probe does. The
    random-start FPS draws from torch's RNG, and the framework reseeds it before
    every call, exactly as the probe does.
  * One model for both species. With text inactive the maize factory builds a
    bitwise-identical network: embodied_mae_4m_maize_base(target_points=8192)
    under the same seed gives the same 289 keys, max|diff| 0.0 (checked
    2026-09-24). The widths it rebuilds (param_embed, the param head) only exist
    when text is active. target_points differs (8196 vs 8192) but only moves the
    decoder's trim, and 8196//196 == 8192//196 == 41 anyway.

Why a fixed INIT_SEED. The training entry points do not seed model
construction, so the arms' own step-0 weights are unrecoverable. A fixed seed
makes this row reproducible. If train_supervised_4m.py calls
torch.manual_seed(INIT_SEED) immediately before building the same model, this
row IS the supervised baseline's step 0, and the supervised gain is then
training alone.

Because features() is generic over any EmbodiedMAE4M, it doubles as the
framework's parity test. Feed it a model from linear_probe.build_model(...) and
eval/baseline_probe.py must reproduce linear_probe's features bit for bit.
"""

import torch

NAME = 'random_init'
INPUTS = ('rgb', 'depth', 'pc')
SOURCE = ('this repo: embodied_mae_4m.embodied_mae_4m_base, no checkpoint; '
          'torch.manual_seed(0) immediately before construction')
MODEL_SIZE = 'base'
EPOCH = 0            # step-0 weights, not "unknown"
FEATURES = ('cls', 'mean')
# The model is built from these, not from anything under eval/baselines/, so
# they join the code hash the feature cache is checked against.
CODE_DEPS = ('embodied_mae_4m.py', 'embodied_mae.py')

INIT_SEED = 0


def build(device, cache_dir):
    """Seeded untrained base model. `cache_dir` is unused: there are no weights."""
    from embodied_mae_4m import embodied_mae_4m_base

    # Save and restore the global RNG so building this baseline cannot shift
    # anything seeded later in the same process. The framework reseeds before
    # extraction anyway, but a caller composing baselines should not have to
    # know that.
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(INIT_SEED)
        model = embodied_mae_4m_base(
            active_modalities=('pc', 'rgb', 'depth'),
            img_size=224,
            num_pc_tokens=196,          # hardcoded in train_sorghum_4m.py
            target_points=8196,         # not the 10000 default
            pc_loss_weight=1.0,
            max_leaves=24,
            spline_loss_weight=5.0,
            depth_norm_type='minmax',
            pc_loss_name='qal_loss',
            qal_threshold=0.01,
            qal_alpha=100.0,
            qal_use_squared=False,
        )
    finally:
        torch.random.set_rng_state(state)

    # Nothing to load, so no strict-load report. Print a weight fingerprint
    # instead: a changed factory or init scheme then shows up as a changed
    # number rather than as a quietly different floor.
    n = sum(p.numel() for p in model.parameters())
    fp = float(sum(p.detach().double().abs().sum() for p in model.parameters()))
    print(f'  ✓ random_init: seed {INIT_SEED}, {n:,} params (untrained), '
          f'sum|w| fingerprint {fp:.6e}')

    # PointCloudEmbed holds BatchNorm1d: train mode would make features depend
    # on batch composition, and would mutate the running stats in place.
    return model.eval().to(device)


def features(model, batch):
    """linear_probe.extract_split's feature step, for any EmbodiedMAE4M."""
    from . import ctx
    rgb, depth, pc, params, _text_valid, _names = batch
    # Text never visible. An arm model with text active also needs a param
    # tensor to embed; the framework hands zeros, as the probe does.
    visible = tuple(m for m in model.active_modalities if m != 'text')
    latent = model.forward_encoder_select(
        rgb, depth, pc, params, visible=visible, source_mask_ratio=0.0)[0]
    feature = ctx(model).feature
    if feature == 'cls':
        return latent[:, 0]
    if feature == 'mean':
        return latent[:, 1:].mean(dim=1)
    raise ValueError(f'unknown feature {feature!r}')
