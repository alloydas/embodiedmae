"""Plant-level subsetting and per-epoch view sampling.

The Sorghum folders are named ``Sorghum_<plant>_<view>``: every plant appears
once per camera view (10 views in Sorghum_15K), so the 105,000 train folders are
10,500 plants x 10 views.  ``torch.utils.data`` sees only the flat list, which
makes two things awkward for a scaling study:

* a random subset of *folders* mixes "more plants" with "more views of the same
  plant", so a 1k/3k/10k data-scaling axis built on it measures neither;
* an epoch over all views shows the model the same plant 10 times in a row of
  gradient steps, which is 10x the compute for far less than 10x the signal.

This module fixes both.  ``fixed_plant_subset`` picks whole plants (all their
views stay available), and ``ViewSampler`` draws a fresh view per plant every
epoch, so an epoch is "one look at each plant" while the model still sees every
view over the course of training.

Both are deterministic given a seed, and ``ViewSampler`` shards across DDP ranks
the way ``DistributedSampler`` does, so every rank runs the same number of steps.
"""

import math
import re

import torch
from torch.utils.data import Subset
from torch.utils.data import Sampler

__all__ = ['plant_id', 'sample_names', 'plant_groups', 'fixed_plant_subset',
           'ViewSampler']

_VIEW_SUFFIX = re.compile(r'^(?P<plant>.+)_(?P<view>\d+)$')


def plant_id(name):
    """'Sorghum_0_07' -> 'Sorghum_0'.  Names without a numeric view suffix are
    returned unchanged, i.e. treated as a plant with a single view."""
    m = _VIEW_SUFFIX.match(name)
    return m.group('plant') if m else name


def sample_names(dataset):
    """Folder names of a SorghumDataset, seen through any stack of Subsets."""
    if isinstance(dataset, Subset):
        parent = sample_names(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    return [folder.name for folder in dataset.samples]


def plant_groups(dataset):
    """[(plant_id, [dataset indices])] in first-appearance order."""
    groups = {}
    for idx, name in enumerate(sample_names(dataset)):
        groups.setdefault(plant_id(name), []).append(idx)
    return list(groups.items())


def fixed_plant_subset(dataset, num_plants, seed):
    """Deterministic subset of whole plants, keeping all views of each.

    None/0/>= the plant count is a no-op, so this is safe to call from
    config-driven code.  The chosen plants are nested across sizes: with the
    same seed, the 1k set is a subset of the 3k set, which is a subset of the
    10k set, because the plants are drawn from one fixed permutation.
    """
    groups = plant_groups(dataset)
    if not num_plants or num_plants >= len(groups):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(groups), generator=generator)[:num_plants].tolist()
    indices = [i for slot in sorted(order) for i in groups[slot][1]]
    return Subset(dataset, indices)


class ViewSampler(Sampler):
    """One (or `views_per_epoch`) randomly chosen view per plant, per epoch.

    An epoch is therefore ``num_plants * views_per_epoch`` samples rather than
    the full folder count.  Which view each plant contributes is redrawn every
    epoch from ``seed + epoch``, so over many epochs every view is used, while a
    single epoch never repeats a plant.

    DDP: like ``DistributedSampler``, the plant order is padded to a multiple of
    ``num_replicas`` so all ranks step in lockstep; the padding wraps around to
    the front of the same epoch's order.
    """

    def __init__(self, dataset, views_per_epoch=1, seed=42,
                 num_replicas=1, rank=0, shuffle=True):
        groups = plant_groups(dataset)
        if not groups:
            raise ValueError('ViewSampler: dataset has no samples')
        self.views_per_epoch = int(views_per_epoch)
        if self.views_per_epoch < 1:
            raise ValueError('views_per_epoch must be >= 1')
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        counts = [len(idx) for _, idx in groups]
        self.num_plants = len(groups)
        self.max_views = max(counts)
        if self.views_per_epoch > min(counts):
            raise ValueError(
                f'views_per_epoch={self.views_per_epoch} exceeds the {min(counts)} '
                f'view(s) of the sparsest plant')

        # (P, V) padded index table + validity mask, so a whole epoch's view
        # draw is one vectorised op rather than a Python loop over plants.
        self._table = torch.zeros(self.num_plants, self.max_views, dtype=torch.long)
        self._valid = torch.zeros(self.num_plants, self.max_views, dtype=torch.bool)
        for row, (_, idx) in enumerate(groups):
            self._table[row, :len(idx)] = torch.as_tensor(idx, dtype=torch.long)
            self._valid[row, :len(idx)] = True

        per_epoch = self.num_plants * self.views_per_epoch
        self.num_samples = math.ceil(per_epoch / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)

        # Smallest `views_per_epoch` random keys per plant = that many distinct
        # views, drawn uniformly; invalid slots are pushed out of reach.
        keys = torch.rand(self.num_plants, self.max_views, generator=g)
        keys = keys.masked_fill(~self._valid, float('inf'))
        picked = keys.topk(self.views_per_epoch, dim=1, largest=False).indices
        indices = self._table.gather(1, picked).reshape(-1)

        if self.shuffle:
            indices = indices[torch.randperm(indices.numel(), generator=g)]

        pad = self.total_size - indices.numel()
        if pad > 0:
            indices = torch.cat([indices, indices[:pad]])
        return iter(indices[self.rank:self.total_size:self.num_replicas].tolist())
