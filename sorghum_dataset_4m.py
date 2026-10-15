"""
SorghumDataset4M — extends SorghumDataset with parametric spline loading.

Each sample returns:
    rgb         : (3, H, W)                      float32
    depth       : (1, H, W)                      float32
    pc          : (num_points, 3)                float32
    param_floats: (1 + max_leaves, N_PARAMS)     float32  — encoder input + target
    text_valid  : (1 + max_leaves,)              float32  — 1=real, 0=padding
    name        : str
"""

from pathlib import Path
import random

import torch
from sorghum_dataset import (
    RGB_FILE, SorghumDataset, _read_index_cache, _write_index_cache,
)
from embodied_mae_4m import load_spline_params


class SorghumDataset4M(SorghumDataset):

    def __init__(self, data_root, img_size=224, num_points=8196, split=None,
                 max_leaves=24, view_sampling=False, view_seed=0,
                 deterministic_view=False, max_plants=None,
                 plant_subset_seed=42, rgb_file=RGB_FILE, rgb_gain=None,
                 train_views=None):
        super().__init__(data_root, img_size=img_size,
                         num_points=num_points, split=split, rgb_file=rgb_file,
                         rgb_gain=rgb_gain)
        self.max_leaves = max_leaves

        # Same caching treatment as the base index: one glob per folder over 105k
        # folders is minutes of shared-filesystem traffic at every job start, and
        # the resolved name also removes the per-__getitem__ glob below.
        entries = _read_index_cache(self.load_dir, 'spline')
        if entries is None:
            entries = []
            for folder in self.samples:
                ymls = list(folder.glob('*_spline.yml'))
                if ymls:
                    entries.append([folder.name, ymls[0].name])
                else:
                    print(f"⚠️  Skipping {folder.name}: no *_spline.yml")
            _write_index_cache(self.load_dir, 'spline', entries)
        else:
            print(f"⚡ spline index cache hit ({len(entries)} samples)")

        self._spline_names = {name: yml for name, yml in entries}
        self.samples = [self.load_dir / name for name, _ in entries]
        print(f"✅ {len(self.samples)} samples have spline data")

        # ── Data scaling (CVPR plan §5, experiment E3) ────────────────────
        # max_plants restricts the split to a NESTED random subset of plants:
        # the full plant list is shuffled once with plant_subset_seed and the
        # first N kept, so 1k ⊂ 3k ⊂ 10k. Three independent draws would make
        # every step of the scaling curve part sample-composition, and no
        # amount of averaging afterwards separates the two effects.
        #
        # The subset is by PLANT, never by view. Dropping views would hold the
        # plant count fixed and shrink the augmentation pool instead, which is
        # E5's question (view regime), not E3's.
        #
        # Only ever pass this for split='train'. val/test must stay identical
        # at every scale or the arms are scored on different yardsticks; the
        # training entry point enforces that by passing it to train_ds alone.
        if max_plants is not None:
            groups = {}
            for i, folder in enumerate(self.samples):
                groups.setdefault(folder.name.rsplit('_', 1)[0], []).append(i)
            all_plants = sorted(groups)
            if int(max_plants) < len(all_plants):
                shuffled = list(all_plants)
                random.Random(int(plant_subset_seed)).shuffle(shuffled)
                keep = set(shuffled[:int(max_plants)])
                self.samples = [f for f in self.samples
                                if f.name.rsplit('_', 1)[0] in keep]
                self._spline_names = {
                    k: v for k, v in self._spline_names.items()
                    if k.rsplit('_', 1)[0] in keep}
                print(f"📉 data scaling: {len(keep)} of {len(all_plants)} "
                      f"plants (subset seed {plant_subset_seed}) -> "
                      f"{len(self.samples)} samples")
            else:
                print(f"📉 data scaling: max_plants={max_plants} >= "
                      f"{len(all_plants)} available; using every plant")

        # ── View sampling (CVPR plan §6.1) ────────────────────────────────
        # Folders are <plant>_<view>, ten consecutive views of each plant. Left
        # alone, one epoch walks all ten views of every plant, so an epoch costs
        # 10x what the plan budgets for and the ablation queue does not close.
        # With view_sampling the dataset is indexed BY PLANT and one view is
        # drawn per plant per epoch: the same view diversity across a run, at a
        # tenth of the per-epoch cost.
        #
        # The draw is a pure function of (view_seed, epoch, plant index), so it
        # needs no shared RNG state -- it is identical in every dataloader worker
        # and on every DDP rank, and a run is reproducible from its seed. Call
        # set_epoch() once per epoch or every epoch draws the same views.
        self.view_sampling      = bool(view_sampling)
        self.deterministic_view = bool(deterministic_view)
        self._view_seed         = int(view_seed)
        self._epoch             = 0
        self.plant_views        = None
        self.plant_ids          = None
        # ── View regime (CVPR plan E5) ───────────────────────────────────
        # train_views restricts the per-epoch draw to these view indices (the
        # <view> suffix of each folder). The ten renders are an elevation ladder,
        # sin(elev) = 0.9 - 0.2 x index (view 0 looks 64 deg down, 4-5 sit within
        # 6 deg of the horizon, 9 looks 64 deg up), so a regime is a contiguous
        # band of it: a tractor-height pass is views 4-5, a top-down rig 0-1, the
        # full hemisphere None. Plants and steps per epoch are unchanged, so an
        # E5 arm runs the same 197,400 steps as the E2/E3/E4 arms; only the
        # augmentation pool shrinks. Pass it for the TRAIN split only: val/test
        # stay on view 0 (deterministic_view) so every regime is scored alike.
        self.train_views = None if train_views is None else sorted({int(v) for v in train_views})
        if self.train_views is not None and not self.train_views:
            raise ValueError('train_views must name at least one view')
        if self.train_views is not None and not self.view_sampling:
            raise ValueError('train_views needs view_sampling=True (one drawn view per plant)')
        if self.view_sampling:
            groups = {}
            keep = None if self.train_views is None else set(self.train_views)
            for i, folder in enumerate(self.samples):
                # Sorghum_10001_07 -> Sorghum_10001, view 7
                plant, view = folder.name.rsplit('_', 1)
                if keep is not None and int(view) not in keep:
                    continue
                groups.setdefault(plant, []).append(i)
            if keep is not None:
                n_plants = len({f.name.rsplit('_', 1)[0] for f in self.samples})
                if len(groups) != n_plants:
                    raise ValueError(f'train_views={self.train_views}: {n_plants - len(groups)} '
                                     f'of {n_plants} plants have none of these views')
            self.plant_ids   = sorted(groups)
            self.plant_views = [groups[pid] for pid in self.plant_ids]
            n_v = {len(v) for v in self.plant_views}
            print(f"🎥 view sampling: {len(self.plant_views)} plants, "
                  f"{sorted(n_v)} views each -> epoch is "
                  f"{len(self.plant_views)} items, not {len(self.samples)}"
                  + ("  [deterministic: view 0]" if self.deterministic_view else "")
                  + (f"  [E5 view regime: views {self.train_views}]" if keep is not None else ""))

    def set_epoch(self, epoch):
        """Advance the view draw. No-op unless view_sampling is on."""
        self._epoch = int(epoch)

    def __len__(self):
        if self.view_sampling:
            return len(self.plant_views)
        return super().__len__()

    def _resolve_index(self, idx):
        """Plant index -> sample index, when view sampling is on."""
        if not self.view_sampling:
            return idx
        views = self.plant_views[idx]
        if self.deterministic_view:
            # val/test: hold the view fixed so the metric moves only because the
            # model moved. A rotating val view would add view variance to every
            # comparison between epochs and between E2 arms.
            return views[0]
        # Explicit integer mix rather than hash() of a tuple, so the draw does
        # not depend on any interpreter hashing detail.
        rng = random.Random(
            (self._view_seed * 1_000_003 + self._epoch) * 1_000_033 + idx)
        return views[rng.randrange(len(views))]

    def __getitem__(self, idx):
        idx = self._resolve_index(idx)
        rgb, depth, pc, name = super().__getitem__(idx)

        folder = self.samples[idx]
        cached = self._spline_names.get(folder.name)
        yml    = (folder / cached) if cached else list(folder.glob('*_spline.yml'))[0]
        text_valid, param_floats = load_spline_params(yml, self.max_leaves)

        return rgb, depth, pc, param_floats, text_valid, name


if __name__ == '__main__':
    import argparse
    from embodied_mae_4m import N_PARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    args = parser.parse_args()

    for split in ('train', 'val'):
        ds = SorghumDataset4M(args.data_root, split=split)
        print(f"{split}: {len(ds)} samples")
        if ds:
            rgb, depth, pc, param_floats, text_valid, name = ds[0]
            print(f"  name={name}")
            print(f"  rgb={tuple(rgb.shape)}  depth={tuple(depth.shape)}  pc={tuple(pc.shape)}")
            print(f"  param_floats={tuple(param_floats.shape)}  text_valid={tuple(text_valid.shape)}")
            print(f"  real_tokens={int(text_valid.sum())}")
            print(f"  param_floats[0] (plant): {param_floats[0].tolist()}")
            print(f"  param_floats[1] (leaf1): {param_floats[1].tolist()}")
