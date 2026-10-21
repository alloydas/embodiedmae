"""
E8 supervised-from-scratch baseline, MAIZE: EmbodiedMAE-4M-Maize's encoder trained
END TO END to regress the decision-6.4 probe targets from PC + RGB + depth.

The maize twin of train_supervised_4m.py. A parallel file, not a species flag
(CLAUDE.md): the dataset, the target table, the plant-id parse, the model width
and the probe it must agree with all differ. Read the sorghum file's docstring
for the full rationale; every design choice there is kept, and these are the
maize-specific facts:

* THE MODEL IS maize_e2_pcrgbd's MODULE: embodied_mae_4m_maize_<size> with
  active_modalities pc,rgb,depth, built with exactly the kwargs
  train_maize_4m.py passes (including param_schema), so
  eval/linear_probe_maize.py rebuilds it from this run's config.json and
  strict-loads `model_state_dict` UNCHANGED. Text is not an active modality.
  The maize plant token carries leafCount and stemInternodeSum, TWO of the four
  6.4 targets verbatim, so the param stream can never be an input here.

* TARGETS ARE THE MAIZE PROBE'S: eval/linear_probe_maize.py's TARGETS (11:
  height, leaf_angle, leaf_count, biomass, leaf_droop, leaf_twist, leaf_curl,
  leaf_length, leaf_width, stem_radius, tassel_droop) and load_targets
  (plant_scores.csv, keyed by the STRING 'plant_NNNN'; the dataset yields the
  INT id and the lookup is keyed by it). Standardised with TRAIN-split
  statistics only (ddof=0, as fit_probe); mean/std go into every checkpoint.
  The loss is MSE averaged over the 11 targets with EQUAL weights. Unlike
  sorghum the four 6.4 targets are largely independent here (height vs leaf
  count r = 0.20), so no single axis dominates the loss. Maize val/test carry
  1.5-2.6x the train variance (Mahalanobis outlier enrichment).

* INIT IS random_init's (INIT_SEED = 0, torch.manual_seed right before
  construction). eval/baselines/random_init.py notes that with text inactive
  the maize factory builds a bitwise-identical network, so step 0 of this run
  is the E8 maize random_init row.

* THE DATASET SKIPS THE SPLINE XML PARSE. MaizeDataset4M.__getitem__ parses
  maize_*_spline.xml on every item and this run never uses it. The subclass
  keeps rgb/depth/pc and the view draw (_resolve_index) unchanged. The XML
  parse consumes no numpy RNG. If MaizeDataset4M.__getitem__ ever gains a step
  other than the parse, copy it into SupervisedMaizeDataset too.

* EVERYTHING ELSE IS THE SORGHUM DESIGN: every token visible through
  forward_encoder_select(visible=all active, source_mask_ratio=0.0); decoder
  frozen (first-step gradient assertion); linear head on the post-norm CLS
  token; val predictions all-gathered and deduplicated by plant; atomic
  checkpoint / history writes; no best_model.pth; E2-matched shape and budget
  (16 x 2 GPUs, 10,500 plants / 32 = 329 steps x 600 = 197,400 optimizer
  steps) refused otherwise; resume invariants checked before config.json is
  rewritten; a run dir of the other data version refused.

Scoring, both ways, at checkpoints/checkpoint_epoch_600.pth, IN THIS ORDER
--------------------------------------------------------------------------
  # 1. its own head, applied to the exact CLS features linear_probe_maize.py
  #    extracts (same loader, seed, cache file), so rows sit in the probe CSV.
  #    The loader shape sets which points each plant's cloud keeps: match the
  #    probe launcher's (Delta slurm/delta/probe_maize.sbatch: batch 32,
  #    workers = cpus - 1 = 15, seed 0).
  python train_supervised_maize_4m.py --config configs/config_maize_e8_supervised_d2.yaml \\
         --score_head --probe_num_workers 15
  # 2. the probe, unchanged (cache hit on the features (1) wrote):
  python eval/linear_probe_maize.py --runs maize_e8_supervised_d2 \\
         --ckpt checkpoints/checkpoint_epoch_600.pth --batch-size 32 --num-workers 15
  # On Delta both run from slurm/delta/e8_supervised_maize.sbatch with SCORE=1.

Training (2 GPUs x 16 = global batch 32, the E2 shape; anything else is refused)
--------
  torchrun --standalone --nproc_per_node=2 train_supervised_maize_4m.py \\
           --config configs/config_maize_e8_supervised_d2.yaml
  Launch with THIS entry point (Delta: slurm/delta/e8_supervised_maize.sbatch).
  slurm/delta/train_maize_arm.sbatch derives configs/config_<slug>.yaml and runs
  train_maize_4m.py; its slug check refuses maize_e8_*, and the YAML's
  entry_point key stops any trainer that reads it.

CPU smoke (one step, 8 plants, no W&B; never into the real output_dir):
  python train_supervised_maize_4m.py --config configs/config_maize_e8_supervised_d2.yaml \\
         --device cpu --world_size 1 --epochs 1 --num_workers 0 --no_wandb \\
         --smoke_plants 8 --output_dir outputs/_smoke_maize_e8_supervised
"""

import os
import sys
import json
import math
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")

from embodied_mae_4m_maize import (
    embodied_mae_4m_maize_small,
    embodied_mae_4m_maize_base,
    embodied_mae_4m_maize_large,
    MAX_LEAVES,
)
from maize_dataset_4m import MaizeDataset4M, PC_FILENAME

# Data version 1 default, as MaizeDataset4M; data v2 configs set rgba.png / v2.
RGB_FILE = 'rgb.png'

REPO = Path(__file__).resolve().parent


def _import_file(modname, path):
    """Import a module by path, reusing it if this process already has it.

    eval/ is not a package, and 'baselines' is a generic top-level name, so a
    sys.path insert could resolve to a same-named module in site-packages. A
    file path cannot. Registering under the probe's own module name means a test
    that monkeypatches `linear_probe` patches this module too.
    """
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    return mod


# The probe owns the targets and the target table. Sharing its code, rather than
# copying it, means the head and the ridge probe cannot drift onto different
# labels. random_init owns the init seed for the same reason.
LP = _import_file('linear_probe_maize', REPO / 'eval' / 'linear_probe_maize.py')
INIT_SEED = _import_file(
    '_e8_random_init', REPO / 'eval' / 'baselines' / 'random_init.py').INIT_SEED

ENTRY_POINT = 'train_supervised_maize_4m.py'

# e2_pcrgbdt's budget AND shape. E8 is matched to it, and an unmatched launch is
# refused (--allow_unmatched / --smoke_plants excepted). The shape matters, not
# just the global batch: PointCloudEmbed's BatchNorm1d is per-rank (no SyncBN),
# so the per-GPU batch sets its statistics, and every reference arm (all E2/E3/
# E4 config.json, maize_4m and every maize _d2 arm) trained 16 per GPU x 2 ranks. 8 x 4 reaches the
# same 32 and 197,400 steps and is still a different run.
E2_GLOBAL_BATCH = 32
E2_TOTAL_STEPS = 197_400
E2_PER_GPU_BATCH = 16
E2_WORLD_SIZE = 2

# What a checkpoint must agree with this launch on before it may be resumed:
# each changes what the run is mid-run. `epochs` in particular: lr_lambda is a
# plain function, so LambdaLR.state_dict() stores None for it, and a resume with
# another --epochs silently rebuilds the cosine over the new length (the maize
# 600->1000 warm restart in CLAUDE.md is this, done on purpose).
RESUME_INVARIANTS = ('epochs', 'warmup_epochs', 'batch_size', 'world_size', 'lr',
                     'weight_decay', 'view_seed', 'max_plants', 'plant_subset_seed',
                     'model_size', 'num_points', 'rgb_file', 'param_schema')

_BUILD = {'small': embodied_mae_4m_maize_small,
          'base':  embodied_mae_4m_maize_base,
          'large': embodied_mae_4m_maize_large}


# ── Config helpers ────────────────────────────────────────────────────────────

MODALITY_ORDER = ('rgb', 'depth', 'pc', 'text')


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_modalities(v):
    """None | 'rgb,pc' | ['rgb','pc'] -> tuple | None. None means all four."""
    if v is None:
        return None
    if isinstance(v, str):
        v = [t for t in (x.strip() for x in v.split(',')) if t]
    mods = tuple(str(m).strip().lower() for m in v)
    if not mods or set(mods) == set(MODALITY_ORDER):
        return None
    return mods


def merge_config_with_args(config, args):
    mapping = {
        'data': ['data_root', 'img_size', 'num_points',
                 'view_sampling', 'view_seed', 'max_plants',
                 'plant_subset_seed', 'rgb_file', 'param_schema'],
        'model': ['model_size', 'pc_loss_weight', 'depth_norm_type',
                  'spline_loss_weight', 'max_leaves', 'loss_name',
                  'qal_threshold', 'qal_alpha', 'qal_use_squared',
                  'active_modalities'],
        'training': ['batch_size', 'epochs', 'lr', 'weight_decay',
                     'warmup_epochs', 'val_freq', 'test_freq'],
        'checkpointing': ['output_dir', 'save_freq', 'resume'],
        'distributed': ['world_size', 'dist_backend', 'dist_url'],
        'system': ['num_workers', 'device'],
        'wandb': ['use_wandb', 'wandb_project', 'wandb_entity', 'wandb_name'],
    }
    for section, keys in mapping.items():
        if section not in config or config[section] is None:
            config[section] = {}
        for k in keys:
            v = getattr(args, k, None)
            if v is not None:
                config[section][k] = v
    return config


def config_to_namespace(config):
    ns = argparse.Namespace()
    ns.entry_point        = config.get('entry_point')
    ns.data_root          = config['data']['data_root']
    ns.img_size           = config['data'].get('img_size', 224)
    ns.num_points         = config['data'].get('num_points', 8192)   # a maize PLY holds exactly 8192
    ns.view_sampling      = bool(config['data'].get('view_sampling', True))
    if not ns.view_sampling:
        # Without it an epoch walks all ten views: 3,282 steps, not 329, and a
        # 600-epoch run is ten times e2_pcrgbdt's budget.
        raise SystemExit("data.view_sampling must be true: E8 is matched to "
                         "e2_pcrgbdt's one-view-per-plant epoch.")
    ns.view_seed          = int(config['data'].get('view_seed', 42))
    # Maize data version 2 (2026-10-06): rgba.png + the trimmed spline XML
    # (param_schema v2). Defaults are data version 1, as train_maize_4m.py.
    ns.rgb_file           = config['data'].get('rgb_file', RGB_FILE)
    ns.param_schema       = config['data'].get('param_schema', 'v1')
    if ns.param_schema not in ('v1', 'v2'):
        raise SystemExit(f"data.param_schema must be v1 or v2, got {ns.param_schema!r}")
    _mp                   = config['data'].get('max_plants', None)
    ns.max_plants         = None if _mp in (None, 0, 'null') else int(_mp)
    ns.plant_subset_seed  = int(config['data'].get('plant_subset_seed', 42))
    ns.model_size         = config['model'].get('model_size', 'base')
    # Construction-only. No reconstruction loss is computed here, but the probe
    # rebuilds this module from config.json with these kwargs, so they are
    # recorded exactly as passed. Defaults are random_init.build's, which are
    # the E2 arms'.
    ns.pc_loss_weight     = config['model'].get('pc_loss_weight', 1.0)
    ns.depth_norm_type    = config['model'].get('depth_norm_type', 'minmax')
    ns.spline_loss_weight = config['model'].get('spline_loss_weight', 5.0)
    ns.max_leaves         = config['model'].get('max_leaves', MAX_LEAVES)
    ns.pc_loss_name       = config['model'].get('loss_name', 'qal_loss')
    ns.qal_threshold      = config['model'].get('qal_threshold', 0.01)
    ns.qal_alpha          = config['model'].get('qal_alpha', 100.0)
    ns.qal_use_squared    = config['model'].get('qal_use_squared', False)
    mods = _parse_modalities(config['model'].get('active_modalities', 'pc,rgb,depth'))
    # None means all four, i.e. text included. The param stream is never an
    # input to a baseline. The maize plant token carries leafCount and
    # stemInternodeSum, so a run that could read text would be handed two of
    # the four 6.4 targets verbatim.
    if mods is None or 'text' in mods:
        raise SystemExit(
            f"active_modalities={config['model'].get('active_modalities')!r} includes "
            "text. E8's supervised baseline never takes the param stream as input: "
            "it carries leaf count and height verbatim. Use a subset of pc,rgb,depth.")
    ns.active_modalities  = mods
    ns.batch_size         = config['training'].get('batch_size', 16)
    ns.epochs             = config['training'].get('epochs', 600)
    ns.lr                 = config['training'].get('lr', 1.5e-4)
    ns.weight_decay       = config['training'].get('weight_decay', 0.05)
    ns.warmup_epochs      = config['training'].get('warmup_epochs', 40)
    if not 0 <= ns.warmup_epochs < ns.epochs:
        # lr_lambda's cosine divides by (epochs - warmup_epochs): equal values
        # raise ZeroDivisionError at the first scheduler.step(), AFTER an epoch.
        raise SystemExit(f"warmup_epochs ({ns.warmup_epochs}) must be in "
                         f"[0, epochs={ns.epochs})")
    ns.val_freq           = config['training'].get('val_freq', 25)
    ns.test_freq          = config['training'].get('test_freq', 0)
    ns.output_dir         = config['checkpointing'].get('output_dir', './outputs/maize_e8_supervised')
    ns.save_freq          = config['checkpointing'].get('save_freq', 25)
    ns.resume             = config['checkpointing'].get('resume', None)
    ns.world_size         = config['distributed'].get('world_size', 1)
    ns.dist_backend       = config['distributed'].get('dist_backend', 'nccl')
    ns.dist_url           = config['distributed'].get('dist_url', 'env://')
    ns.num_workers        = config['system'].get('num_workers', 8)
    ns.device             = config['system'].get('device', 'cuda')
    ns.use_wandb          = config['wandb'].get('use_wandb', True)
    ns.wandb_project      = config['wandb'].get('wandb_project', 'embodied-mae-maize')
    ns.wandb_entity       = config['wandb'].get('wandb_entity', None)
    ns.wandb_name         = config['wandb'].get('wandb_name', None)
    return ns


# ── Distributed helpers ───────────────────────────────────────────────────────

def _local_rank(rank):
    lr = int(os.environ.get('LOCAL_RANK', -1))
    return lr if lr >= 0 else rank


def _dist_timeout():
    """NCCL collective timeout. torch's default 10 min killed every maize run on
    2026-10-21 ~12:00 when the shared /work/hdd filesystem stalled (all five jobs, four
    nodes, the same minute; E1 22870827 died the same way on 10-16): one rank blocked
    on a read while the other sat in a BROADCAST/ALLREDUCE. 60 min rides that out;
    DIST_TIMEOUT_MIN overrides. No effect on numbers."""
    from datetime import timedelta
    return timedelta(minutes=int(os.environ.get('DIST_TIMEOUT_MIN', '60')))


def setup_distributed(rank, world_size, backend, url):
    if int(os.environ.get('LOCAL_RANK', -1)) >= 0:
        dist.init_process_group(backend=backend, init_method='env://',
                                world_size=world_size, rank=rank,
                                timeout=_dist_timeout())
    else:
        # A different port from train_maize_4m.py and train_supervised_4m.py
        # (12357), so the entry points can be smoke-tested side by side.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12358'
        dist.init_process_group(backend=backend, init_method='tcp://localhost:12358',
                                world_size=world_size, rank=rank,
                                timeout=_dist_timeout())
    if backend == 'nccl':
        # LOCAL rank, not the global one. The same thing on one node, but the
        # global rank is out of range on a second node.
        torch.cuda.set_device(_local_rank(rank))


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Atomic writes ─────────────────────────────────────────────────────────────

def _atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp, path)


# ── Targets ───────────────────────────────────────────────────────────────────

def _plant_of_name(name):
    """'plant_0004_07' -> 4 (MaizeDataset4M.plant_id). Ints travel through the
    collate and the all_gather; the target lookup is keyed by them."""
    return MaizeDataset4M.plant_id(name)


def _plant_of_id(pid):
    """'plant_0004' (MaizeDataset4M.plant_ids, the plant_scores.csv key) -> 4."""
    return int(pid.split('_')[-1])


def target_spec():
    names = [n for n, _, _, _ in LP.TARGETS]
    cols = [src for _, src, _, _ in LP.TARGETS]
    return names, cols


def check_plants(tgt, plants, split):
    """Every plant is in the target table, once, and on the split it claims.

    `plants` are table keys ('plant_0004'). A data_root whose plant_scores.csv
    disagrees with its folders would otherwise train on val plants, or
    standardise on test ones, and say nothing.
    """
    plants = list(plants)
    if len(set(plants)) != len(plants):
        raise RuntimeError(f'{split}: duplicate plants in the dataset index')
    missing = [p for p in plants if p not in tgt.index]
    if missing:
        raise RuntimeError(f'{split}: {len(missing)} plants missing from the target '
                           f'table, e.g. {missing[:5]}')
    wrong = [p for p, s in zip(plants, tgt.loc[plants, 'split']) if s != split]
    if wrong:
        raise RuntimeError(f'{split}: {len(wrong)} plants are assigned to another '
                           f'split in plant_scores.csv, e.g. {wrong[:5]}')


def train_statistics(tgt, train_plants, cols):
    """Mean/std over the plants this run trains on, and nothing else.

    ddof=0, which is what fit_probe's `ytr.std()` on a numpy array uses, so the
    head's standardised space is the probe's. A constant target gets std 1,
    as in fit_probe.
    """
    Y = tgt.loc[list(train_plants), cols].to_numpy(dtype=np.float64)
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def standardised_lookup(tgt, plants, cols, mean, std):
    """Keyed by the INT plant id (what the dataset yields), from table keys."""
    Y = tgt.loc[list(plants), cols].to_numpy(dtype=np.float64)
    Z = ((Y - mean) / std).astype(np.float32)
    return {_plant_of_id(p): Z[i] for i, p in enumerate(plants)}


class SupervisedMaizeDataset(MaizeDataset4M):
    """MaizeDataset4M minus the spline XML parse, plus the plant's target row.

    Yields (rgb, depth, pc, y, plant, name), where y holds the TRAIN-
    standardised targets and plant is the INT id. The view choice is
    MaizeDataset4M._resolve_index, unchanged, and rgb/depth/pc come from the
    same three loaders (load_rgb / load_depth / load_pointcloud, in the same
    order) MaizeDataset4M.__getitem__ calls, so the pixels and the point
    subsample are the MAE arms'. If MaizeDataset4M.__getitem__ ever gains a step
    other than the XML parse (an augmentation, say), it must be copied here
    too, or E8 would train on different pixels than E2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.view_sampling:
            raise ValueError('E8 needs view_sampling: an epoch is one view per plant')
        self._targets = None

    def plant_keys(self):
        """Table keys ('plant_0004'), in MaizeDataset4M.plant_ids order."""
        return list(self.plant_ids)

    def set_targets(self, lookup):
        self._targets = lookup

    def __getitem__(self, idx):
        idx = self._resolve_index(idx)
        folder = self.samples[idx]
        rgb = self.load_rgb(folder)
        depth = self.load_depth(folder / 'depth.png')
        pc = torch.from_numpy(self.load_pointcloud(folder / PC_FILENAME))
        name = folder.name
        plant = _plant_of_name(name)
        return rgb, depth, pc, torch.from_numpy(self._targets[plant]), plant, name


# ── Model ─────────────────────────────────────────────────────────────────────

def encoder_kwargs(args):
    """The kwargs linear_probe_maize.build_model reads back from config.json;
    the constructor call train_maize_4m.py makes, minus text_mask_ratio (text
    is never active here, and the probe asserts the model's value is None)."""
    return dict(
        active_modalities=args.active_modalities,
        img_size=args.img_size,
        num_pc_tokens=196,                # hardcoded in train_maize_4m.py
        target_points=args.num_points,    # 8192, not the 10000 default
        pc_loss_weight=args.pc_loss_weight,
        max_leaves=args.max_leaves,
        param_schema=args.param_schema,
        spline_loss_weight=args.spline_loss_weight,
        depth_norm_type=args.depth_norm_type,
        pc_loss_name=args.pc_loss_name,
        qal_threshold=args.qal_threshold,
        qal_alpha=args.qal_alpha,
        qal_use_squared=args.qal_use_squared,
    )


class SupervisedEncoder(nn.Module):
    """EmbodiedMAE4MMaize encoder + a linear head on the post-norm CLS token."""

    def __init__(self, mae, n_targets):
        super().__init__()
        self.mae = mae
        # Every active stream visible. Text is never active, see config_to_namespace.
        self.visible = tuple(mae.active_modalities)
        self.head = nn.Linear(mae.cls_token.shape[-1], n_targets)
        # timm's ViT head init. It is deterministic, because it draws from the
        # RNG right after the seeded encoder construction.
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def cls(self, rgb, depth, pc):
        # x_param=None: text is not active, so _embed_active never reads it.
        latent = self.mae.forward_encoder_select(
            rgb, depth, pc, None, visible=self.visible, source_mask_ratio=0.0)[0]
        return latent[:, 0]

    def forward(self, rgb, depth, pc):
        return self.head(self.cls(rgb, depth, pc))


def freeze_decoder(mae):
    """requires_grad_(False) on everything the encoder path never touches."""
    frozen = []
    for n, p in mae.named_parameters():
        if n.startswith('decoder_') or n == 'mask_token':
            p.requires_grad_(False)
            frozen.append(n)
    return frozen


def build_model(args, n_targets):
    """Seeded exactly as eval/baselines/random_init.py (INIT_SEED), so step 0 ==
    the maize random_init row: with text inactive the maize factory builds the
    same network as the sorghum one at these kwargs."""
    torch.manual_seed(INIT_SEED)
    mae = _BUILD[args.model_size](**encoder_kwargs(args))
    model = SupervisedEncoder(mae, n_targets)
    frozen = freeze_decoder(mae)
    return model, frozen


def assert_all_trainable_have_grads(base):
    """Run once, after the first backward.

    If a trainable tensor gets no gradient, DDP(find_unused_parameters=False)
    fails one step later with an error that does not name it, and a single-GPU
    run would weight-decay it for 197,400 steps. Name it here instead.
    """
    dead = [n for n, p in base.named_parameters() if p.requires_grad and p.grad is None]
    if dead:
        raise RuntimeError(
            f'{len(dead)} trainable tensors received no gradient through the encoder '
            f'path, e.g. {dead[:8]}. Freeze them in freeze_decoder().')


# ── Metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(pred, y, train_mean):
    """The four numbers linear_probe.fit_probe reports for one split, same code."""
    from sklearn.metrics import r2_score, mean_absolute_error
    return {
        'r2': r2_score(y, pred),
        'rmse': float(np.sqrt(np.mean((y - pred) ** 2))),
        'mae': mean_absolute_error(y, pred),
        'base_mae': mean_absolute_error(y, np.full_like(y, train_mean)),
    }


def head_metrics(P, Y, mean, std, names):
    """P, Y standardised (n, T) -> loss plus per-target metrics in native units."""
    out = {'loss': float(np.mean((P - Y) ** 2)), 'n': int(len(P))}
    Pn = P.astype(np.float64) * std + mean
    Yn = Y.astype(np.float64) * std + mean
    for j, name in enumerate(names):
        m = regression_metrics(Pn[:, j], Yn[:, j], mean[j])
        out[f'r2_{name}'] = m['r2']
        out[f'mae_{name}'] = m['mae']
    return out


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(model, base, dataloader, optimizer, device, epoch, check_grads):
    model.train()
    tot, n = 0.0, 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for rgb, depth, pc, y, _plant, _name in pbar:
        rgb   = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        pc    = pc.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)

        pred = model(rgb, depth, pc)
        # Mean over batch AND targets: every target weighs the same.
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if check_grads:
            assert_all_trainable_have_grads(base)
            check_grads = False
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot += loss.item()
        n += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, distributed):
    """Predictions for every plant in the split, each plant counted ONCE.

    Returns standardised (plants, P, Y). Under DDP the per-rank shards are
    all-gathered first: R^2 does not average across shards, and
    DistributedSampler pads the split to a multiple of world_size by repeating
    plants.
    """
    model.eval()
    ids, P, Y = [], [], []
    for rgb, depth, pc, y, plant, _ in tqdm(dataloader, desc='Evaluating'):
        pred = model(rgb.to(device, non_blocking=True),
                     depth.to(device, non_blocking=True),
                     pc.to(device, non_blocking=True))
        P.append(pred.float().cpu().numpy())
        Y.append(y.numpy())
        ids.append(plant.numpy())
    ids, P, Y = np.concatenate(ids), np.concatenate(P), np.concatenate(Y)

    if distributed and dist.is_available() and dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, (ids, P, Y))
        ids = np.concatenate([p[0] for p in parts])
        P = np.concatenate([p[1] for p in parts])
        Y = np.concatenate([p[2] for p in parts])

    _, first = np.unique(ids, return_index=True)
    first.sort()
    return ids[first], P[first], Y[first]


def _fmt_r2(m, names):
    return '  '.join(f"{n}: {m[f'r2_{n}']:+.4f}" for n in names)


# ── Worker ────────────────────────────────────────────────────────────────────

def _smoke_banner(n):
    bar = '!' * 78
    print(f"\n{bar}\n"
          f"!!  --smoke_plants {n}: TESTING ONLY. Every split is cut to its first {n}\n"
          f"!!  plants, so target statistics, steps/epoch and val R2 are NOT E8's.\n"
          f"!!  config.json and every checkpoint are tagged smoke_plants={n}, and a\n"
          f"!!  resume across that tag is refused.\n"
          f"{bar}\n", flush=True)


def _cap(ds, n):
    return ds if n is None else Subset(ds, range(min(int(n), len(ds))))


def train_worker(rank, world_size, args):
    distributed = world_size > 1
    if distributed:
        use_cuda = torch.cuda.is_available() and str(args.device).startswith('cuda')
        if args.dist_backend == 'nccl' and not use_cuda:
            print(f"⚠️  rank {rank}: no CUDA, using gloo instead of nccl")
            args.dist_backend = 'gloo'
        setup_distributed(rank, world_size, args.dist_backend, args.dist_url)
        device = (torch.device(f'cuda:{_local_rank(rank)}') if use_cuda
                  else torch.device('cpu'))
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    is_main = (rank == 0)
    if is_main and args.smoke_plants:
        _smoke_banner(args.smoke_plants)
    # The batch SHAPE is known now; refuse it before the minutes-long index scan.
    # The step count is checked again once the loaders exist.
    if ((args.batch_size, world_size) != (E2_PER_GPU_BATCH, E2_WORLD_SIZE)
            and not (args.smoke_plants or args.allow_unmatched)):
        raise SystemExit(f"❌ NOT E2-MATCHED: {args.batch_size} per GPU x {world_size} GPUs; "
                         f"every reference arm is {E2_PER_GPU_BATCH} x {E2_WORLD_SIZE} "
                         f"(per-rank BatchNorm). Launch torchrun --nproc_per_node=2 with "
                         f"--batch_size 16, or pass --allow_unmatched for a non-E8 run.")

    output_dir     = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)

    # ── data + targets ─────────────────────────────────────────────────────────
    if is_main: print(f"\nLoading data from: {args.data_root}")
    names, cols = target_spec()
    tgt = LP.load_targets(args.data_root)

    # view_sampling (plan §6.1): an epoch is one drawn view per PLANT. Train
    # rotates the view with (view_seed, epoch), exactly as the E2 arms do.
    # val/test pin view 00, the view linear_probe.py scores.
    # val/test pin view 00, the view linear_probe_maize.py scores. rgb_file and
    # param_schema are the config's: the XML is never parsed here, but the
    # subclass keeps the kwargs so the index and the plant set are the arms'.
    ds_kw = dict(img_size=args.img_size, num_points=args.num_points,
                 max_leaves=args.max_leaves, rgb_file=args.rgb_file,
                 param_schema=args.param_schema)
    train_base = SupervisedMaizeDataset(
        args.data_root, split='train', view_sampling=True, view_seed=args.view_seed,
        max_plants=args.max_plants, plant_subset_seed=args.plant_subset_seed, **ds_kw)
    val_base = SupervisedMaizeDataset(
        args.data_root, split='val', view_sampling=True, deterministic_view=True, **ds_kw)
    test_base = SupervisedMaizeDataset(
        args.data_root, split='test', view_sampling=True, deterministic_view=True, **ds_kw)

    n_cap = args.smoke_plants
    train_plants = train_base.plant_keys()[:n_cap] if n_cap else train_base.plant_keys()
    val_plants   = val_base.plant_keys()[:n_cap] if n_cap else val_base.plant_keys()
    test_plants  = test_base.plant_keys()[:n_cap] if n_cap else test_base.plant_keys()
    for split, plants in (('train', train_plants), ('val', val_plants), ('test', test_plants)):
        check_plants(tgt, plants, split)

    # TRAIN statistics only: the plants this run actually trains on.
    t_mean, t_std = train_statistics(tgt, train_plants, cols)
    for ds, plants in ((train_base, train_plants), (val_base, val_plants),
                       (test_base, test_plants)):
        ds.set_targets(standardised_lookup(tgt, plants, cols, t_mean, t_std))
    if is_main:
        print(f"Targets ({len(names)}, standardised on {len(train_plants)} train plants):")
        for n_, c_, m_, s_ in zip(names, cols, t_mean, t_std):
            print(f"    {n_:16s} {c_:14s} mean {m_:.6g}  std {s_:.6g}")

    # Load and validate the resume checkpoint BEFORE config.json is rewritten.
    # A refused resume must leave the run's config.json describing the run.
    resume_ckpt = None
    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_resume(resume_ckpt, args, world_size, cols, t_mean, t_std)
    elif args.resume:
        if is_main: print(f"⚠️  Checkpoint not found: {args.resume} — starting from scratch")

    train_ds = _cap(train_base, n_cap)
    val_ds   = _cap(val_base, n_cap)
    test_ds  = _cap(test_base, n_cap)

    if distributed:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)
        test_sampler  = DistributedSampler(test_ds,  world_size, rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle_train = True

    # persistent_workers stays OFF (the default): persistent workers keep the
    # dataset copy from the first iteration, so set_epoch() would never reach
    # them and view sampling would freeze at epoch 0.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=shuffle_train, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, sampler=test_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    # ── budget ─────────────────────────────────────────────────────────────────
    global_batch    = args.batch_size * world_size
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    if is_main:
        print(f"\nBudget: {len(train_ds)} plants / global batch {global_batch} "
              f"({args.batch_size} x {world_size}) = {steps_per_epoch} steps/epoch "
              f"x {args.epochs} epochs = {total_steps:,} optimizer steps")
    e2_matched = (args.batch_size == E2_PER_GPU_BATCH and world_size == E2_WORLD_SIZE
                  and total_steps == E2_TOTAL_STEPS)
    if not e2_matched:
        msg = (f"NOT E2-MATCHED: {args.batch_size} x {world_size} GPUs, {total_steps:,} steps; "
               f"e2_pcrgbdt is {E2_PER_GPU_BATCH} x {E2_WORLD_SIZE} (per-rank BatchNorm), "
               f"{E2_TOTAL_STEPS:,} steps")
        if not (args.smoke_plants or args.allow_unmatched):
            # Fatal, and before config.json / W&B: a 2-GPU torchrun that forgot
            # --batch_size runs 394,200 steps at global batch 16 for two days.
            raise SystemExit(f"❌ {msg}. Launch 2 GPUs x --batch_size 16, or pass "
                             f"--allow_unmatched for a deliberate non-E8 run.")
        if is_main:
            print(f"⚠️  {msg} (allowed: {'smoke' if args.smoke_plants else '--allow_unmatched'})")
    if resume_ckpt is not None and resume_ckpt.get('steps_per_epoch') != steps_per_epoch:
        raise SystemExit(f"--resume {args.resume} ran {resume_ckpt.get('steps_per_epoch')} "
                         f"steps/epoch, this launch {steps_per_epoch}: a different step budget.")

    # ── model ──────────────────────────────────────────────────────────────────
    if is_main: print(f"\nInitializing supervised EmbodiedMAE-4M-Maize-{args.model_size.capitalize()} "
                      f"encoder (active {','.join(args.active_modalities)}, "
                      f"init seed {INIT_SEED})...")
    model, frozen = build_model(args, len(names))
    model = model.to(device)
    base = model
    trainable = [p for p in base.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in base.parameters())
    n_train = sum(p.numel() for p in trainable)
    if is_main:
        print(f"Total parameters: {n_total:,}  trainable {n_train:,}  "
              f"(frozen: {len(frozen)} decoder-side tensors, which the encoder path never reaches)")

    if distributed:
        model = DDP(model,
                    device_ids=[device.index] if device.type == 'cuda' else None,
                    output_device=device.index if device.type == 'cuda' else None,
                    find_unused_parameters=False)

    # config.json = the EFFECTIVE run: vars(args) after YAML -> CLI -> torchrun,
    # plus the budget and targets as resolved. linear_probe.build_model rebuilds
    # the encoder from it. active_modalities is written as a list and never as
    # null, because null means all four streams to the probe.
    if is_main:
        eff = dict(vars(args))
        eff.update({
            'entry_point':      ENTRY_POINT,
            'objective':        'supervised: MSE on train-standardised linear_probe_maize.TARGETS',
            'active_modalities': list(args.active_modalities),
            'world_size':       world_size,
            'global_batch':     global_batch,
            'steps_per_epoch':  steps_per_epoch,
            'total_steps':      total_steps,
            'e2_matched':       e2_matched,
            'init_seed':        INIT_SEED,
            'head':             f'linear {base.head.in_features}->{base.head.out_features} '
                                f'on post-norm CLS',
            'n_train_plants':   len(train_plants),
            'n_val_plants':     len(val_plants),
            'n_test_plants':    len(test_plants),
            'target_names':     names,
            'target_cols':      cols,
            'target_mean':      t_mean.tolist(),
            'target_std':       t_std.tolist(),
            'total_params':     n_total,
            'trainable_params': n_train,
        })
        _atomic_json(eff, output_dir / 'config.json')

    if is_main and args.use_wandb and WANDB_AVAILABLE:
        if args.wandb_name is None:
            args.wandb_name = f"maize_e8_supervised_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run_id = resume_ckpt.get('wandb_run_id') if resume_ckpt else None
        if wandb_run_id:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       id=wandb_run_id, resume='must')
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_name, config={
                           'experiment': 'E8 supervised (maize)', 'model_size': args.model_size,
                           'active_modalities': list(args.active_modalities),
                           'num_points': args.num_points, 'batch_size': args.batch_size,
                           'global_batch': global_batch, 'epochs': args.epochs,
                           'total_steps': total_steps, 'lr': args.lr,
                           'total_params': n_total, 'trainable_params': n_train,
                           'train_plants': len(train_ds), 'val_plants': len(val_ds),
                           'targets': names, 'smoke_plants': args.smoke_plants,
                       })
        print(f"✅ W&B: {wandb.run.url}")
        wandb.watch(model, log='gradients', log_freq=500)

    # Trainable tensors only: the frozen decoder has no grad and no business in
    # the optimizer state. There are no param groups, as in train_maize_4m.py.
    optimizer = optim.AdamW(trainable, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (ep - args.warmup_epochs)
                                  / (args.epochs - args.warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    history = {'epoch': [], 'lr': [], 'train_loss': [],
               'val_epoch': [], 'val_loss': [],
               **{f'val_r2_{n}': [] for n in names},
               **{f'val_mae_{n}': [] for n in names},
               'test_epoch': [], 'test_loss': [],
               **{f'test_r2_{n}': [] for n in names},
               **{f'test_mae_{n}': [] for n in names}}

    if resume_ckpt is not None:
        if is_main: print(f"\n📂 Resuming from {args.resume}")
        ckpt, resume_ckpt = resume_ckpt, None
        sd = ckpt['model_state_dict']
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        base.mae.load_state_dict(sd, strict=True)
        base.head.load_state_dict(ckpt['head_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        for k, v in ckpt.get('history', {}).items():
            history[k] = v
        if is_main:
            print(f"✅ Resumed from epoch {ckpt['epoch']} "
                  f"(next lr {optimizer.param_groups[0]['lr']:.3e})")
        del ckpt

    if is_main:
        print(f"\nStarting training for {args.epochs} epochs...")
        print("=" * 80)

    check_grads = True
    for epoch in range(start_epoch, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
        # Advance the per-plant view draw. Workers are respawned each epoch
        # (persistent_workers is off), so they pick this up.
        train_base.set_epoch(epoch)

        lr_now = optimizer.param_groups[0]['lr']
        if is_main:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}  lr={lr_now:.6f}")
            print(f"{'='*80}")

        tr_loss = train_one_epoch(model, base, train_loader, optimizer, device,
                                  epoch, check_grads)
        check_grads = False
        history['epoch'].append(epoch)
        history['lr'].append(lr_now)
        history['train_loss'].append(tr_loss)
        if is_main:
            print(f"\nTrain — MSE (standardised): {tr_loss:.4f}")

        log = {'epoch': epoch, 'train/loss': tr_loss, 'learning_rate': lr_now}

        do_val = (epoch % args.val_freq == 0 or epoch == args.epochs or epoch == 1)
        if do_val:
            if is_main: print("\n🔍 Running validation…")
            _, P, Y = evaluate(model, val_loader, device, distributed)
            vm = head_metrics(P, Y, t_mean, t_std, names)
            history['val_epoch'].append(epoch)
            history['val_loss'].append(vm['loss'])
            for n in names:
                history[f'val_r2_{n}'].append(vm[f'r2_{n}'])
                history[f'val_mae_{n}'].append(vm[f'mae_{n}'])
            if is_main:
                print(f"Val   — MSE {vm['loss']:.4f} on {vm['n']} plants")
                print(f"Val R2 — {_fmt_r2(vm, names)}")
            log.update({'val/loss': vm['loss'],
                        **{f'val_r2/{n}': vm[f'r2_{n}'] for n in names}})

        do_test = (args.test_freq > 0
                   and (epoch % args.test_freq == 0 or epoch == args.epochs))
        if do_test:
            if is_main: print(f"\n🧪 Running TEST-set evaluation (epoch {epoch})…")
            _, P, Y = evaluate(model, test_loader, device, distributed)
            tm = head_metrics(P, Y, t_mean, t_std, names)
            history['test_epoch'].append(epoch)
            history['test_loss'].append(tm['loss'])
            for n in names:
                history[f'test_r2_{n}'].append(tm[f'r2_{n}'])
                history[f'test_mae_{n}'].append(tm[f'mae_{n}'])
            if is_main:
                print(f"Test R2 — {_fmt_r2(tm, names)}")
            log.update({'test/loss': tm['loss'],
                        **{f'test_r2/{n}': tm[f'r2_{n}'] for n in names}})

        if is_main and args.use_wandb and WANDB_AVAILABLE:
            wandb.log(log)

        scheduler.step()

        if is_main and epoch % args.save_freq == 0:
            _atomic_torch_save({
                'epoch': epoch,
                # The encoder module alone: exactly the keys
                # linear_probe_maize.build_model strict-loads. The head is kept apart.
                'model_state_dict': base.mae.state_dict(),
                'head_state_dict': base.head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'entry_point': ENTRY_POINT,
                'active_modalities': list(args.active_modalities),
                'target_names': names,
                'target_cols': cols,
                'target_mean': t_mean.tolist(),
                'target_std': t_std.tolist(),
                'init_seed': INIT_SEED,
                'smoke_plants': args.smoke_plants,
                'global_batch': global_batch,
                'steps_per_epoch': steps_per_epoch,
                **{k: getattr(args, k) for k in RESUME_INVARIANTS if k != 'world_size'},
                'world_size': world_size,
                'wandb_run_id': (wandb.run.id
                                 if args.use_wandb and WANDB_AVAILABLE
                                 and wandb.run else None),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint saved: checkpoint_epoch_{epoch}.pth")

        if is_main:
            _atomic_json(history, output_dir / 'training_history.json')

    if distributed:
        cleanup_distributed()
    if is_main and args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    if is_main:
        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"Outputs: {output_dir}")
        print(f"Score it: python {ENTRY_POINT} --config <this yaml> --score_head "
              f"--probe_num_workers <probe launcher's workers>\n"
              f"          python eval/linear_probe_maize.py --runs {output_dir.name} "
              f"--ckpt checkpoints/checkpoint_epoch_{args.epochs}.pth "
              f"--batch-size 32 --num-workers <same>")
        print(f"{'='*80}")


def check_resume_data_version(args):
    """Refuse to continue a run directory written under the other data version
    (copied from train_maize_4m.py). A config.json from before 2026-10-06 has
    neither key (= rgb.png / v1). Runs before config.json is rewritten."""
    prev = Path(args.output_dir) / 'config.json'
    if not prev.exists():
        return
    try:
        old = json.loads(prev.read_text())
    except ValueError:
        return
    was = (old.get('rgb_file', 'rgb.png'), old.get('param_schema', 'v1'))
    now = (args.rgb_file, args.param_schema)
    if was != now:
        raise SystemExit(f"{prev} records rgb_file={was[0]!r}, param_schema={was[1]!r} but "
                         f"this run uses rgb_file={now[0]!r}, param_schema={now[1]!r}: "
                         f"refusing to mix data versions in one run directory")


def check_resume(ckpt, args, world_size, cols, t_mean, t_std):
    """Refuse a resume that would change what the run is, mid-run.

    Runs before config.json is rewritten, so a refused resume leaves the run's
    config.json describing the run. steps_per_epoch is checked once the loaders
    exist, still before config.json.
    """
    if 'head_state_dict' not in ckpt:
        raise SystemExit(
            f"{args.resume} is not a {ENTRY_POINT} checkpoint (no head_state_dict). "
            "Warm-starting E8 from a pretrained MAE is a different baseline; a "
            "from-scratch run resumes only from its own checkpoints.")
    if ckpt.get('smoke_plants') != args.smoke_plants:
        raise SystemExit(
            f"--resume {args.resume} was written with smoke_plants="
            f"{ckpt.get('smoke_plants')}, this run has {args.smoke_plants}. "
            "A smoke checkpoint must never continue as the real run, or the reverse.")
    now = {**{k: getattr(args, k) for k in RESUME_INVARIANTS if k != 'world_size'},
           'world_size': world_size}
    missing = [k for k in RESUME_INVARIANTS if k not in ckpt]
    if missing:
        raise SystemExit(f"--resume {args.resume} predates the resume checks (no {missing}): "
                         f"its schedule and batch shape cannot be verified. Start over.")
    changed = {k: (ckpt[k], now[k]) for k in RESUME_INVARIANTS if ckpt[k] != now[k]}
    if changed:
        raise SystemExit(f"--resume {args.resume} would change the run mid-run "
                         f"(checkpoint, this launch): {changed}. A different --epochs "
                         f"rebuilds the cosine; a different batch shape changes the steps "
                         f"and the per-rank BatchNorm. Relaunch with the original values.")
    if list(ckpt.get('target_cols', [])) != list(cols):
        raise SystemExit(f"target columns changed since {args.resume}: "
                         f"{ckpt.get('target_cols')} -> {cols}")
    if not (np.allclose(ckpt['target_mean'], t_mean, rtol=1e-12, atol=0)
            and np.allclose(ckpt['target_std'], t_std, rtol=1e-12, atol=0)):
        raise SystemExit(
            f"train target statistics differ from {args.resume}: the target table "
            "or the train plant set changed. Resuming would silently rescale the "
            "loss mid-run.")


# ── Head scoring on the probe's own features ──────────────────────────────────

def score_head(args, cli):
    """Score the trained head on train/val/test with linear_probe_maize.py's features.

    The run is rebuilt by linear_probe_maize.build_model, the probe's own strict
    loader. Features come from linear_probe_maize.cached_features, so the loader,
    seed, view, batch/worker shape and the cache FILE are the probe's. A later
    `eval/linear_probe_maize.py --runs <run> --ckpt <same>` then gets a cache hit, and
    the head and the ridge refit are scored on byte-identical features. Rows
    use the probe CSV's columns, with run = '<slug>_head' and alpha/cv_r2 empty
    because nothing was fitted.
    """
    import pandas as pd

    run_dir = Path(args.output_dir)
    ckpt_rel = cli.score_ckpt or f'checkpoints/checkpoint_epoch_{args.epochs}.pth'
    device = args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        print('⚠️  no CUDA, falling back to CPU (slow)')
        device = 'cpu'

    model, cfg, epoch = LP.build_model(run_dir, ckpt_rel, device)
    if cfg.get('entry_point') != ENTRY_POINT:
        raise SystemExit(f"{run_dir} is not a {ENTRY_POINT} run (config.json "
                         f"entry_point={cfg.get('entry_point')!r}); it has no head.")
    if cfg.get('smoke_plants'):
        print(f"⚠️  {run_dir} is a --smoke_plants {cfg['smoke_plants']} run: TESTING ONLY")
    ckpt = torch.load(run_dir / ckpt_rel, map_location='cpu', weights_only=False)
    names, cols = target_spec()
    if list(ckpt['target_cols']) != cols:
        raise SystemExit(f"checkpoint targets {ckpt['target_cols']} != "
                         f"linear_probe_maize.TARGETS {cols}")
    W = ckpt['head_state_dict']['weight'].double().numpy()
    b = ckpt['head_state_dict']['bias'].double().numpy()
    t_mean = np.asarray(ckpt['target_mean'], dtype=np.float64)
    t_std = np.asarray(ckpt['target_std'], dtype=np.float64)
    del ckpt

    pargs = argparse.Namespace(
        cache_dir=cli.probe_cache_dir, ckpt=ckpt_rel, feature='cls',
        seed=cli.probe_seed, repeats=1, refresh=False, data_root=args.data_root,
        batch_size=cli.probe_batch_size, num_workers=cli.probe_num_workers,
        device=device)
    tgt = LP.load_targets(args.data_root)
    slug = run_dir.name
    act = ','.join(model.active_modalities)
    print(f'\n=== {slug} head · {cfg["model_size"]} · active={act} · '
          f'{ckpt_rel} @ epoch {epoch} ===')

    data = {}
    for split in ('train', 'val', 'test'):
        plants, feats = LP.cached_features(model, cfg, pargs, slug, split)
        plants = [str(p) for p in plants]        # 'plant_0004' keys, never ints
        check_plants(tgt, plants, split)
        expected = int((tgt['split'] == split).sum())
        if len(plants) != expected:
            # A stale or capped feature cache would score the head on a subset
            # and still print a plausible R2.
            msg = (f'{split}: {len(plants)} feature rows, the split has {expected} '
                   f'plants (cache {cli.probe_cache_dir})')
            if not cfg.get('smoke_plants'):
                raise SystemExit(msg + '. Delete the cache file or pass a fresh --probe_cache_dir.')
            print(f'⚠️  {msg} -- accepted only because this is a smoke run')
        data[split] = (plants, feats.astype(np.float64) @ W.T + b)
    del model

    rows = []
    for j, (name, src, prov, unit) in enumerate(LP.TARGETS):
        row = {'run': f'{slug}_head', 'model_size': cfg['model_size'], 'active': act,
               'epoch': epoch, 'feature': 'cls', 'dim': W.shape[1],
               'target': name, 'source_col': src, 'provenance': prov, 'unit': unit,
               'alpha': float('nan'), 'cv_r2': float('nan')}
        for split in ('train', 'val', 'test'):
            plants, Z = data[split]
            pred = Z[:, j] * t_std[j] + t_mean[j]
            y = tgt.loc[plants, src].to_numpy(dtype=np.float64)
            for k, v in regression_metrics(pred, y, t_mean[j]).items():
                row[f'{split}_{k}'] = v
        rows.append(row)
        print(f'  {name:16s} ({prov})  val R2 {row["val_r2"]:+.4f}   '
              f'test R2 {row["test_r2"]:+.4f}   '
              f'val MAE {row["val_mae"]:.4g} vs {row["val_base_mae"]:.4g} base [{unit}]')

    out = Path(cli.score_out) if cli.score_out else (
        run_dir / f'head_scores__{Path(ckpt_rel).stem}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\n📄 wrote {out}  ({len(rows)} rows)')
    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='E8 (maize): train EmbodiedMAE-4M-Maize supervised from scratch on the 6.4 targets')

    parser.add_argument('--config',             type=str,   default='configs/config_maize_e8_supervised_d2.yaml')
    parser.add_argument('--data_root',          type=str,   default=None)
    parser.add_argument('--img_size',           type=int,   default=None)
    parser.add_argument('--num_points',         type=int,   default=None)
    parser.add_argument('--view_seed',          type=int,   default=None)
    parser.add_argument('--rgb_file',           type=str,   default=None,
                        help=f'RGB image per view folder: rgb.png (data v1) or rgba.png (v2); default {RGB_FILE}')
    parser.add_argument('--param_schema',       type=str,   default=None, choices=['v1', 'v2'],
                        help='spline XML layout the model is built for (v2 since 2026-10-06); '
                             'construction-only here, text is never active')
    parser.add_argument('--max_plants',         type=int,   default=None,
                        help='cap the TRAIN split at N plants (nested subsets, as E3)')
    parser.add_argument('--plant_subset_seed',  type=int,   default=None)
    parser.add_argument('--model_size',         type=str,   default=None, choices=['small', 'base', 'large'])
    parser.add_argument('--active_modalities',  type=str,   default=None,
                        help="subset of 'pc,rgb,depth' (text is refused). Default pc,rgb,depth.")
    parser.add_argument('--batch_size',         type=int,   default=None)
    parser.add_argument('--epochs',             type=int,   default=None)
    parser.add_argument('--lr',                 type=float, default=None)
    parser.add_argument('--weight_decay',       type=float, default=None)
    parser.add_argument('--warmup_epochs',      type=int,   default=None)
    parser.add_argument('--val_freq',           type=int,   default=None)
    parser.add_argument('--test_freq',          type=int,   default=None)
    parser.add_argument('--output_dir',         type=str,   default=None)
    parser.add_argument('--save_freq',          type=int,   default=None)
    parser.add_argument('--resume',             type=str,   default=None)
    parser.add_argument('--world_size',         type=int,   default=None)
    parser.add_argument('--dist_backend',       type=str,   default=None)
    parser.add_argument('--dist_url',           type=str,   default=None)
    parser.add_argument('--num_workers',        type=int,   default=None)
    parser.add_argument('--device',             type=str,   default=None)
    parser.add_argument('--use_wandb',          action='store_true', default=None)
    parser.add_argument('--no_wandb',           action='store_false', dest='use_wandb')
    parser.add_argument('--wandb_project',      type=str,   default=None)
    parser.add_argument('--wandb_entity',       type=str,   default=None)
    parser.add_argument('--wandb_name',         type=str,   default=None)
    # CLI-only on purpose: a testing hook that could be set in a YAML could be
    # committed into the real config.
    parser.add_argument('--smoke_plants',       type=int,   default=None,
                        help='TESTING ONLY: cut every split to its first N plants. '
                             'Needs an explicit --output_dir.')
    parser.add_argument('--allow_unmatched',    action='store_true',
                        help='run although batch shape / steps differ from e2_pcrgbdt '
                             '(16 x 2, 197,400); the run is then NOT the E8 row')
    parser.add_argument('--score_head',         action='store_true',
                        help='score a finished run\'s head on linear_probe.py\'s '
                             'features, write CSV rows, and exit')
    parser.add_argument('--score_ckpt',         type=str,   default=None,
                        help='checkpoint inside the run dir (default '
                             'checkpoints/checkpoint_epoch_<epochs>.pth)')
    parser.add_argument('--score_out',          type=str,   default=None)
    # Defaults are slurm/delta/probe_maize.sbatch's (batch 32, seed 0, workers =
    # cpus - 1 = 15 at its 16-CPU header). Which points a plant's cloud keeps
    # depends on (seed, batch size, worker count), so these must match the arm
    # caches' shape for the rows to share point subsets.
    parser.add_argument('--probe_cache_dir',    type=str,
                        default=str(REPO / 'outputs' / '_probe_cache_maize'))
    parser.add_argument('--probe_seed',         type=int,   default=0)
    parser.add_argument('--probe_batch_size',   type=int,   default=32)
    parser.add_argument('--probe_num_workers',  type=int,   default=15)

    cli = parser.parse_args()

    if not os.path.exists(cli.config):
        # No silent fallback to a built-in config, for the same reason as
        # train_maize_4m.py: a typo'd path must not become a run that
        # compares to nothing.
        raise SystemExit(f"Config not found: {cli.config}\n"
                         f"{ENTRY_POINT} has no default config on purpose. Pass "
                         f"--config configs/config_maize_e8_supervised_d2.yaml.")
    print(f"📋 Loading config: {cli.config}")
    raw = load_config(cli.config)
    if raw.get('entry_point') != ENTRY_POINT:
        # An E2 config would otherwise train a supervised model into that E2
        # arm's output_dir, overwriting its checkpoint_epoch_N.pth files.
        raise SystemExit(
            f"{cli.config} is not an E8 supervised config (entry_point="
            f"{raw.get('entry_point')!r}, expected {ENTRY_POINT!r}). Refusing, "
            f"so that this trainer never writes into an MAE run's output_dir.")
    args = config_to_namespace(merge_config_with_args(raw, cli))
    args.smoke_plants = cli.smoke_plants
    args.allow_unmatched = cli.allow_unmatched

    if cli.score_head:
        score_head(args, cli)
        return

    if args.smoke_plants and cli.output_dir is None:
        raise SystemExit("--smoke_plants needs an explicit --output_dir, so a test "
                         f"can never write into {args.output_dir}.")
    prior = Path(args.output_dir) / 'config.json'
    if prior.exists():
        try:
            prior_ep = json.loads(prior.read_text()).get('entry_point')
        except ValueError:
            prior_ep = None
        if prior_ep != ENTRY_POINT:
            raise SystemExit(
                f"{args.output_dir} already holds a run from another entry point "
                f"(config.json entry_point={prior_ep!r}). Refusing to write "
                f"supervised checkpoints over it.")
    check_resume_data_version(args)

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = world_size
        if rank == 0:
            print(f"\n🚀 Multi-GPU (torchrun)  GPUs={world_size}  "
                  f"batch/GPU={args.batch_size}  total={args.batch_size*world_size}")
        train_worker(rank, world_size, args)
    elif args.world_size > 1:
        print(f"\n🚀 Multi-GPU (mp.spawn)  GPUs={args.world_size}  "
              f"batch/GPU={args.batch_size}  total={args.batch_size*args.world_size}")
        mp.spawn(train_worker, args=(args.world_size, args),
                 nprocs=args.world_size, join=True)
    else:
        args.world_size = 1
        train_worker(0, 1, args)


if __name__ == '__main__':
    main()
