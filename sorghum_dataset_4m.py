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
    SorghumDataset, _read_index_cache, _write_index_cache,
)
from embodied_mae_4m import load_spline_params


class SorghumDataset4M(SorghumDataset):

    def __init__(self, data_root, img_size=224, num_points=8196, split=None,
                 max_leaves=24, view_sampling=False, view_seed=0,
                 deterministic_view=False):
        super().__init__(data_root, img_size=img_size,
                         num_points=num_points, split=split)
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
        if self.view_sampling:
            groups = {}
            for i, folder in enumerate(self.samples):
                # Sorghum_10001_07 -> Sorghum_10001
                plant = folder.name.rsplit('_', 1)[0]
                groups.setdefault(plant, []).append(i)
            self.plant_ids   = sorted(groups)
            self.plant_views = [groups[pid] for pid in self.plant_ids]
            n_v = {len(v) for v in self.plant_views}
            print(f"🎥 view sampling: {len(self.plant_views)} plants, "
                  f"{sorted(n_v)} views each -> epoch is "
                  f"{len(self.plant_views)} items, not {len(self.samples)}"
                  + ("  [deterministic: view 0]" if self.deterministic_view else ""))

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
