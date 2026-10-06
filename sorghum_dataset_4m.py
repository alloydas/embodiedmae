"""
SorghumDataset4M — extends SorghumDataset with parametric spline loading.

Each sample returns:
    rgb         : (3, H, W)                      float32
    depth       : (1, H, W)                      float32
    pc          : (num_points, 3)                float32
    param_floats: (1 + max_leaves, N_PARAMS)     float32  — encoder input + target
    text_valid  : (1 + max_leaves,)              float32  — 1=real, 0=padding
    name        : str
    pc_norm     : (4,)                           float32  — only with
                  return_pc_norm=True: the cloud's centroid (xyz, metres, camera
                  frame) and scale, so pc * scale + centroid is the camera-frame
                  cloud. Structured masking needs it to project points.
    cam2world   : (4, 4)                         float32  — only with
                  return_pose=True (which implies return_pc_norm): the view's
                  camera_pose.json cameraToWorld. Occluded-scene training needs
                  it to stand neighbouring plants on the target's ground.

Spline params: on 2026-09-30 Alloy's alloy/shorgum_data/add_leaf_width.py
rewrote every *_spline.yml copy in Sorghum_15K in place (per-leaf `width` added,
both waviness keys dropped). load_spline_params needs the waviness keys, and its
leaf filter used to drop every leaf of a rewritten file without a word -- the
text stream became the plant token alone. See _resolve_spline_root.
"""

from pathlib import Path
import json
import os
import random
import re

import torch
from sorghum_dataset import (
    SorghumDataset, _read_index_cache, _write_index_cache,
)
from embodied_mae_4m import load_spline_params


def is_rewritten_spline(text):
    """True for a *_spline.yml that add_leaf_width.py rewrote (width, no waviness)."""
    return ('waviness_' not in text
            and re.search(r'^\s+width:', text, re.MULTILINE) is not None)


class SorghumDataset4M(SorghumDataset):

    def __init__(self, data_root, img_size=224, num_points=8196, split=None,
                 max_leaves=24, view_sampling=False, view_seed=0,
                 deterministic_view=False, max_plants=None,
                 plant_subset_seed=42, return_pc_norm=False,
                 return_pose=False, spline_root=None):
        super().__init__(data_root, img_size=img_size,
                         num_points=num_points, split=split)
        self.max_leaves = max_leaves
        self.return_pose = bool(return_pose)
        self.return_pc_norm = bool(return_pc_norm) or self.return_pose

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
        self.spline_root = self._resolve_spline_root(spline_root)

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

    def _resolve_spline_root(self, spline_root):
        """Directory to read <plant>_spline.yml from, or None for the view folders.

        The SorghumData originals the per-view copies were made from are
        untouched by the 2026-09-30 rewrite (add_leaf_width.py only rewrites a
        copy that is byte-identical to its original), so reading them gives the
        exact tensor every run before the rewrite trained on. Order: the
        argument, then $SORGHUM_SPLINE_ROOT, then -- only when the folder copies
        turn out to be rewritten -- Nova's layout, <data_root>/../../SorghumData.
        A copy left original keeps loading from its folder, unchanged.
        """
        explicit = spline_root or os.environ.get('SORGHUM_SPLINE_ROOT')
        if explicit:
            root = Path(explicit)
            if not root.is_dir():
                raise ValueError(f"spline_root {root} is not a directory")
            print(f"📄 spline params from {root} (set explicitly)")
            return root
        if not self.samples:
            return None
        folder = self.samples[0]
        if not is_rewritten_spline(
                (folder / self._spline_names[folder.name]).read_text()):
            return None
        root = self.data_root.parent.parent / 'SorghumData'
        probe = root / f"{folder.name.rsplit('_', 1)[0]}_spline.yml"
        if not probe.exists() or is_rewritten_spline(probe.read_text()):
            raise ValueError(
                f"{folder.name}'s *_spline.yml was rewritten by add_leaf_width.py "
                f"(no waviness keys), and no original was found at {probe}. Set "
                f"data.spline_root (or $SORGHUM_SPLINE_ROOT) to the SorghumData "
                f"folder; without it the text stream loads no leaves.")
        print(f"📄 spline copies under {self.load_dir.name}/ were rewritten on "
              f"2026-09-30 (no waviness); reading the originals in {root}")
        return root

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
        if self.spline_root is not None:
            yml = self.spline_root / f"{folder.name.rsplit('_', 1)[0]}_spline.yml"
        else:
            cached = self._spline_names.get(folder.name)
            yml    = (folder / cached) if cached else list(folder.glob('*_spline.yml'))[0]
        text_valid, param_floats = load_spline_params(yml, self.max_leaves)

        if self.return_pc_norm:
            pc_norm = torch.from_numpy(self._last_pc_norm.copy())
            if self.return_pose:
                with open(folder / 'camera_pose.json') as fh:
                    c2w = json.load(fh)['cameraToWorld']
                cam2world = torch.tensor(c2w, dtype=torch.float32).view(4, 4)
                return (rgb, depth, pc, param_floats, text_valid, name, pc_norm,
                        cam2world)
            return rgb, depth, pc, param_floats, text_valid, name, pc_norm
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
