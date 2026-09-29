"""
SorghumDataset4M — extends SorghumDataset with parametric spline loading.

Each sample returns:
    rgb         : (3, H, W)                      float32
    depth       : (1, H, W)                      float32
    pc          : (num_points, 3)                float32
    param_floats: (1 + max_leaves, N_PARAMS)     float32  — encoder input + target
    text_valid  : (1 + max_leaves,)              float32  — 1=real, 0=padding
    name        : str
    cond        : (N_COND,)                      float32  — only with geometry_cond=True
                  (camera rotation + point-cloud normalisation radius; see
                  embodied_mae_4m.geometry_condition). Off by default so every
                  existing 6-tuple consumer keeps working.
"""

from pathlib import Path
import torch
from sorghum_dataset import SorghumDataset
from embodied_mae_4m import geometry_condition, load_spline_params


class SorghumDataset4M(SorghumDataset):

    def __init__(self, data_root, img_size=224, num_points=8196, split=None,
                 max_leaves=24, param_encoding='v1', geometry_cond=False):
        super().__init__(data_root, img_size=img_size,
                         num_points=num_points, split=split)
        self.max_leaves     = max_leaves
        self.param_encoding = param_encoding
        self.geometry_cond  = bool(geometry_cond)

        def _with_spline():
            keep = []
            for folder in self.samples:
                if list(folder.glob('*_spline.yml')):
                    keep.append(folder)
                else:
                    print(f"⚠️  Skipping {folder.name}: no *_spline.yml")
            return keep

        # Same reason as the base class: this is another glob per directory.
        self.samples = self.cached_index('4m', _with_spline)
        print(f"✅ {len(self.samples)} samples have spline data", flush=True)

    def __getitem__(self, idx):
        item   = self.load_item(idx)
        folder = item['dir']
        yml    = list(folder.glob('*_spline.yml'))[0]
        text_valid, param_floats = load_spline_params(
            yml, self.max_leaves, encoding=self.param_encoding)

        out = (item['rgb'], item['depth'], item['pc'], param_floats, text_valid,
               item['name'])
        if self.geometry_cond:
            cond = geometry_condition(folder / 'camera_pose.json', item['pc_radius'])
            out = out + (torch.from_numpy(cond),)
        return out


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
