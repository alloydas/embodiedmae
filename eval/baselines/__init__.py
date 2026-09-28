"""E8 baselines: frozen feature extractors scored by the decision-6.4 probe.

`eval/baseline_probe.py` runs every baseline through the SAME protocol as
`eval/linear_probe.py` (sorghum) and `eval/linear_probe_maize.py` (maize): same
plants, same single deterministic view per plant, same loader seeding, same
targets, same train-only standardiser, same ridge alphas and CV, same CSV
columns. So a baseline row concatenates directly under the E2/E3/E4 arm rows.
Only feature extraction differs, and that is all an adapter supplies.

The registry imports adapters LAZILY, by name. One adapter's missing optional
dependency (e.g. a hub repo that has not been fetched) must not stop the others
from running.

ADAPTER CONTRACT -- one module per baseline, eval/baselines/<name>.py
=====================================================================
Required module attributes
--------------------------
NAME : str
    Must equal the module's REGISTRY key. It goes into the CSV `run` column and
    names the feature cache `base_<NAME>__<species>__<split>__...npz`.

INPUTS : tuple[str, ...]
    A non-empty subset of ('rgb', 'depth', 'pc'). 'text' is never allowed.
    This is ENFORCED, not merely declared: `features()` receives None in every
    modality slot not listed here. A baseline therefore cannot quietly read a
    modality its table row does not claim.

SOURCE : str
    The provenance of the weights and code: URL or HF repo plus a pinned
    revision/commit, and the expected checkpoint size in bytes and/or sha256.
    Printed with every run, and stored in the feature cache.

build(device, cache_dir) -> torch.nn.Module
    device    : str -- 'cuda', 'cuda:N' or 'cpu'.
    cache_dir : pathlib.Path, absolute. This is the torch.hub directory for the
        run, i.e. `torch.hub.get_dir()`, which the framework sets itself before
        calling build(). Torch-style checkpoints go under
        `cache_dir / 'checkpoints'`, which is torch.hub's own layout, so
        `load_state_dict_from_url(..., model_dir=cache_dir / 'checkpoints')`
        and `torch.hub.load(...)` agree. For Hugging Face files, call
        `hf_hub_cache()`: the framework exports HF_HOME before any adapter is
        imported. NEVER fall back to a library default. On Nova,
        XDG_CACHE_HOME=/tmp/.cache, so the defaults are node-local, and a
        --prefetch on a login node would never reach the compute node.
    Returns the model on `device`, in eval() mode. The framework asserts that no
    submodule is in training mode: BatchNorm in train mode makes features depend
    on batch composition. It then calls requires_grad_(False).
    Weight loading MUST be strict. A missing or unexpected ENCODER key is fatal.
    Dropping a decoder is fine, but only as an explicit, asserted set of keys
    (e.g. "exactly 263 keys, all 'decoder.*'"). Print the load report:
    `strict_load()` below prints a consistent one. A silently
    random-initialised "pretrained" baseline gives plausible, wrong numbers, and
    that is the worst failure E8 can have.
    Must work OFFLINE once `baseline_probe.py --prefetch` has run on a node with
    internet. Delta compute nodes may have none.

features(model, batch) -> torch.Tensor, shape (B, D), floating point
    model : exactly what build() returned. `model.probe_ctx` is set (see
        ProbeContext) before the first call for each split.
    batch : the 6-tuple the probe's DataLoader yields, with the framework
        having done four things to it: moved the tensors to `device`, zeroed
        the params, put None in text_valid, and put None in the non-INPUTS
        slots:
        (rgb, depth, pc, params, text_valid, names)
          rgb   (B,3,224,224) float32. Resize((224,224)) + ToTensor + ImageNet
                mean/std normalisation (0.485,0.456,0.406)/(0.229,0.224,0.225),
                identical for both species (sorghum_dataset.py rgb_transform,
                maize_dataset_4m.py rgb_transform).
          depth (B,1,224,224) float32. (z - near)/(far - near) in [0, 1] with
                0 = background, bilinear+antialias resized from 1024 px. NOT
                normalised and NOT metric. Each renderer uses its own near/far:
                sorghum foreground is ~0.03-0.064 (one fixed near/far), maize
                ~0.26-0.66 (a near/far per VIEW). Do NOT read the per-view
                values from camera_pose.json: the maize renderer frames the
                camera to the plant, so near/far correlate r = 0.82 with height
                (stem_internodeSum) over 200 val plants -- the exact metric
                conversion hands a baseline the height target. Use a fixed
                dataset constant (embodiedmae.py MAIZE_NEAR/FAR).
                The antialiased resize blends background into every silhouette
                edge: ~31 % (sorghum) and ~67 % (maize) of the d > 0 pixels are
                partial-coverage mixes of plant depth and 0, reading too near.
                An adapter that takes statistics over d > 0 is taking them over
                that halo (multimae.py depth_valid() is the fix).
          pc    (B,N,3) float32. N = 8196 (sorghum) / 8192 (maize), a seeded
                random subsample of the camera-frame cloud, centred on its
                centroid and scaled to max norm 1. Axes are the raw camera
                frame (OpenGL: y up, -z forward). Any axis or unit conversion
                is the adapter's job, and must be documented there.
          params     zeros, (B, 1+max_leaves, N_PARAMS). The spline stream is
                     never an input to any baseline. It is zeroed rather than
                     dropped only so the tuple keeps the loader's shape.
          text_valid None. As loaded it is the spline stream's validity mask,
                     and text_valid.sum() - 1 IS the leaf count (every maize
                     plant, and every sorghum plant up to the 24-leaf cap),
                     i.e. leaf_count, and on sorghum height and biomass too
                     (r > 0.99). Zeroing the params alone would not seal the
                     stream, so the framework blanks this slot as well.
          names      tuple[str] of folder names ('Sorghum_10_00',
                     'plant_0004_00'). Row i of the output belongs to names[i].
    Return the pooled feature for `model.probe_ctx.feature` ('cls' | 'mean'),
    one row per sample, in batch order. The framework runs this under
    torch.no_grad(). Immediately before EACH call it runs
    torch.manual_seed(seed + 1000003*batch_index + repeat), exactly as
    linear_probe does, so an adapter that draws from torch's RNG (e.g. a
    random-start FPS) is reproducible. When several feature modes are
    extracted in one data pass, features() is called once per mode on the same
    batch, each call after the same reseed, so every mode's rows equal a
    single-mode run's. The output must not depend on the other samples in the
    batch.

    Per-sample FILES ARE OFF LIMITS. The input is the tensor the arms see and
    nothing else. `model.probe_ctx.sample_dir(name)` raises unless the adapter
    declares USES_CAMERA_POSE = True (below). camera_pose.json holds near/far
    (a height proxy on maize, see depth above), `plantCenter`, and
    `sourceXml` (the path of the param file itself).

Optional module attributes
--------------------------
FEATURES   : tuple of the supported feature modes, default ('cls', 'mean').
             Declare it when a mode does not exist (e.g. a model with no CLS
             token). A requested mode the adapter lacks is SKIPPED for that
             adapter, with a printed line, never substituted.
FEATURE_LABELS : {mode: label} for the CSV `feature` column, default the mode
             itself. Required when a mode is not what the arms mean by it:
             Point-MAE's 'cls' is a [max || mean] token pool, so its rows say
             'max+mean', and a table filtered on feature == 'cls' cannot put
             it beside real CLS tokens.
USES_CAMERA_POSE : bool, default False. True unlocks ctx.sample_dir(), for a
             SENSITIVITY row that is given per-view camera pose the arms never
             get (pointmae_upright). Such a row must be captioned as using
             camera pose; the framework prints that at every run.
CODE_DEPS  : repo-relative paths of repo-root modules the features depend on
             (e.g. the model file random_init builds), added to the code hash
             every feature cache is checked against.
MODEL_SIZE : str for the CSV `model_size` column, default '-'.
EPOCH      : int for the CSV `epoch` column, default -1 (external pretrain).
prefetch(cache_dir) -> None
             Download and verify the weights only. The default is
             `build('cpu', cache_dir)`, which also proves the strict load
             works, so define prefetch only when build() needs something a
             login node lacks.

What an adapter does NOT do: seeding, dataset construction, view choice,
device moves, repeat averaging, caching, standardisation and the ridge fit.
Those all belong to the framework, which reuses the probe's own functions.
"""

import dataclasses
import hashlib
import importlib
import os
from pathlib import Path

# name -> module under eval/baselines/. Imported on first use only.
# The supervised baseline is NOT here: it trains (train_supervised_4m.py) and is
# scored as an ordinary run dir by eval/linear_probe.py and its own --score_head.
REGISTRY = {
    'random_init':      'random_init',       # untrained EmbodiedMAE4M base: the floor
    'embodiedmae':      'embodiedmae',       # official EmbodiedMAE-base, RGB+D+PC
    'multimae':         'multimae',          # official MultiMAE-B, RGB+D
    'dinov2_vitb14':    'dinov2_vitb14',     # DINOv2 ViT-B/14, RGB
    'pointmae':         'pointmae',          # official Point-MAE ShapeNet, PC
    'pointmae_upright': 'pointmae_upright',  # SENSITIVITY row: + per-view camera pose
}
NAMES = tuple(REGISTRY)

VALID_INPUTS = ('rgb', 'depth', 'pc')      # canonical order == MODALITIES minus text
VALID_FEATURES = ('cls', 'mean')


# ─────────────────────────────── probe context ───────────────────────────────

@dataclasses.dataclass(frozen=True)
class ProbeContext:
    """What an adapter may know about the extraction it is part of.

    Attached to the model as `model.probe_ctx` before features() is first
    called for a split. It carries the pooling mode because features() returns
    ONE (B, D) tensor, and the contract's signature has no other slot for it.
    """
    species: str          # 'sorghum' | 'maize'
    split: str            # 'train' | 'val' | 'test'
    data_root: Path
    feature: str          # 'cls' | 'mean'
    seed: int
    uses_camera_pose: bool = False     # the adapter's USES_CAMERA_POSE

    def sample_dir(self, name):
        """<data_root>/<split>/<name>, for a USES_CAMERA_POSE adapter only.

        Everything in a sample folder beyond the loader's tensors is
        information the arms never get, and some of it is the answer (maize
        near/far track height at r = 0.82; camera_pose.json names the param
        file). So a headline adapter cannot open it at all.
        """
        if not self.uses_camera_pose:
            raise PermissionError(
                'ctx.sample_dir() is only for adapters that declare USES_CAMERA_POSE = True '
                '(sensitivity rows captioned as using camera pose). A headline baseline '
                'gets exactly the tensors the arms get.')
        return Path(self.data_root) / self.split / name


def ctx(model):
    """model.probe_ctx, or a clear error if features() is called outside the framework."""
    c = getattr(model, 'probe_ctx', None)
    if c is None:
        raise RuntimeError('model.probe_ctx is unset: features() must be called '
                           'through eval/baseline_probe.py, which sets it per split')
    return c


# ─────────────────────────────── weight cache ────────────────────────────────

def configure_weight_cache(weights_dir=None):
    """Pin torch.hub and HF caches to a SHARED path; return the torch.hub dir.

    Nova exports XDG_CACHE_HOME=/tmp/.cache, and with TORCH_HOME and HF_HOME
    unset both libraries resolve to that node-local /tmp. A --prefetch on the
    login node would then land where no compute node can see it, and the
    offline job would fail, or worse, find some other copy. So:
      --weights-dir X  -> TORCH_HOME=X/torch, HF_HOME=X/huggingface (the flag wins)
      otherwise        -> honour TORCH_HOME / HF_HOME if set, else ~/.cache/{torch,huggingface}
    The values are EXPORTED, so library code that reads the env itself (the
    DINOv2 hubconf downloading its weights, hf_hub_download's default) agrees.
    Call this before anything imports huggingface_hub: it reads HF_HOME at
    import time.
    """
    import torch
    if weights_dir:
        root = Path(weights_dir).expanduser().resolve()
        os.environ['TORCH_HOME'] = str(root / 'torch')
        os.environ['HF_HOME'] = str(root / 'huggingface')
        # HF_HUB_CACHE, if already exported, would beat HF_HOME inside the library
        os.environ['HF_HUB_CACHE'] = str(root / 'huggingface' / 'hub')
    else:
        home = Path.home() / '.cache'
        os.environ.setdefault('TORCH_HOME', str(home / 'torch'))
        os.environ.setdefault('HF_HOME', str(home / 'huggingface'))
    hub = Path(os.environ['TORCH_HOME']).expanduser().resolve() / 'hub'
    torch.hub.set_dir(str(hub))
    return hub


def hf_hub_cache():
    """The HF hub cache directory, i.e. the `cache_dir=` for hf_hub_download."""
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE']).expanduser().resolve()
    if not os.environ.get('HF_HOME'):
        raise RuntimeError('HF_HOME unset: call configure_weight_cache() first '
                           '(the library default is node-local on Nova)')
    return Path(os.environ['HF_HOME']).expanduser().resolve() / 'hub'


def verify_file(path, size=None, sha256=None):
    """Check a downloaded checkpoint against SOURCE's size / sha256. Fatal on mismatch.

    Needed because torch.hub only checks the hash when it DOWNLOADS. A file
    that is already cached is loaded unchecked, so a truncated or substituted
    file would pass.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} missing: run eval/baseline_probe.py --prefetch '
                                f'on a node with internet')
    n = path.stat().st_size
    if size is not None and n != int(size):
        raise RuntimeError(f'{path.name}: {n} bytes, expected {size}')
    if sha256 is not None:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        if h.hexdigest() != sha256.lower():
            raise RuntimeError(f'{path.name}: sha256 {h.hexdigest()}, expected {sha256}')
    print(f'  ✓ {path.name}: {n:,} bytes' + (' · sha256 ok' if sha256 else ''))
    return path


def strict_load(module, state_dict, label, dropped=()):
    """load_state_dict(strict=True) plus a printed report of what was loaded.

    `dropped` lists the checkpoint keys the caller deliberately discarded
    (e.g. a pretraining decoder). They are printed as a COUNT, so the log shows
    both what was loaded and what was left out on purpose.
    """
    res = module.load_state_dict(state_dict, strict=True)   # raises on any mismatch
    assert not res.missing_keys and not res.unexpected_keys, res
    n_param = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
    print(f'  ✓ {label}: strict load, {len(state_dict)} keys / {n_param:,} values, '
          f'missing=[] unexpected=[]'
          + (f' · {len(dropped)} checkpoint keys deliberately dropped' if dropped else ''))
    return res


# ────────────────────────────────── registry ─────────────────────────────────

def load(name):
    """Import one adapter by registry name and check it honours the contract."""
    if name not in REGISTRY:
        raise KeyError(f'unknown baseline {name!r}; registered: {list(REGISTRY)}')
    try:
        mod = importlib.import_module(f'{__name__}.{REGISTRY[name]}')
    except ModuleNotFoundError as e:
        # Distinguish "adapter not written yet" from "adapter's dependency absent".
        raise ImportError(f'baseline {name!r} could not be imported: {e}') from e
    validate(mod, name)
    return mod


def validate(mod, name=None):
    """Fail on a malformed adapter before any weights or data are touched."""
    for attr in ('NAME', 'INPUTS', 'SOURCE', 'build', 'features'):
        if not hasattr(mod, attr):
            raise TypeError(f'{mod.__name__} lacks required attribute {attr!r}')
    if name is not None and mod.NAME != name:
        raise ValueError(f'{mod.__name__}.NAME={mod.NAME!r} != registry key {name!r}')
    inputs = tuple(mod.INPUTS)
    bad = [m for m in inputs if m not in VALID_INPUTS]
    if not inputs or bad or len(set(inputs)) != len(inputs):
        raise ValueError(f'{mod.NAME}: INPUTS={inputs} must be a non-empty subset of '
                         f'{VALID_INPUTS} ("text" is never an input to a baseline)')
    feats = tuple(getattr(mod, 'FEATURES', VALID_FEATURES))
    if not feats or not set(feats) <= set(VALID_FEATURES):
        raise ValueError(f'{mod.NAME}: FEATURES={feats} not a non-empty subset of {VALID_FEATURES}')
    labels = dict(getattr(mod, 'FEATURE_LABELS', {}))
    for mode, label in labels.items():
        # A label may rename a mode, never claim ANOTHER real mode's name: that
        # is exactly the mislabelling FEATURE_LABELS exists to prevent.
        if mode not in feats or (label != mode and label in VALID_FEATURES):
            raise ValueError(f'{mod.NAME}: FEATURE_LABELS {labels} must map its own FEATURES '
                             f'{feats} to their own name or a new one')
    return mod


def inputs_canonical(mod):
    """INPUTS in rgb, depth, pc order -- the order the arms' `active` column uses."""
    return tuple(m for m in VALID_INPUTS if m in mod.INPUTS)


def feature_label(mod, mode):
    """What the CSV `feature` column says for this adapter's `mode`."""
    return dict(getattr(mod, 'FEATURE_LABELS', {})).get(mode, mode)


def code_fingerprint(mod):
    """sha256 over the adapter's source and every eval/baselines file it imports.

    Stored in each feature cache. SOURCE names the weights, not the input
    conversion, so without this an edited conversion would be served its old
    features from cache. Followed: `from .x import ...` / `from . import x`
    where x is a module or package under eval/baselines/ (package __init__
    excluded: it holds no conversion), transitively. That covers
    pointmae_upright -> pointmae -> _vendor_pointmae, and imports made inside
    functions (multimae's vendored encoder).
    """
    import ast
    root = Path(__file__).resolve().parent
    seen, todo, files = set(), [Path(mod.__file__).resolve()], []
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        files.append(f)
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            base = f.parent if node.level == 1 else f.parents[node.level - 1]
            names = [node.module] if node.module else [a.name for a in node.names]
            for n in names:
                p = base.joinpath(*n.split('.'))
                cand = [p.with_suffix('.py')] if p.with_suffix('.py').is_file() else \
                       sorted(p.rglob('*.py')) if p.is_dir() else []
                todo.extend(c.resolve() for c in cand
                            if root in c.resolve().parents and c.resolve() != root / '__init__.py')
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    # Repo-root modules an adapter builds from (random_init's model), which no
    # relative import reaches. Paths are repo-relative, so a copy elsewhere
    # (Delta) hashes the same.
    repo = root.parents[1]
    for rel in sorted(getattr(mod, 'CODE_DEPS', ())):
        h.update(rel.encode())
        h.update((repo / rel).read_bytes())
    return h.hexdigest()[:16]
