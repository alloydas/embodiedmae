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
import torch
from sorghum_dataset import SorghumDataset
from embodied_mae_4m import load_spline_params


class SorghumDataset4M(SorghumDataset):

    def __init__(self, data_root, img_size=224, num_points=8196, split=None,
                 max_leaves=24):
        super().__init__(data_root, img_size=img_size,
                         num_points=num_points, split=split)
        self.max_leaves = max_leaves

        valid = []
        for folder in self.samples:
            if list(folder.glob('*_spline.yml')):
                valid.append(folder)
            else:
                print(f"⚠️  Skipping {folder.name}: no *_spline.yml")
        self.samples = valid
        print(f"✅ {len(self.samples)} samples have spline data")

    def __getitem__(self, idx):
        rgb, depth, pc, name = super().__getitem__(idx)

        folder = self.samples[idx]
        yml    = list(folder.glob('*_spline.yml'))[0]
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
