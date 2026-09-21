"""Export the point clouds behind "The same plant, progressively hidden".

The static plate is `sweeps/rgb_mask_sweep.py`'s qualitative half: ONE validation
plant (Sorghum_10001_00) pushed through the encoder five times with 0, 20, 50,
80 and 95% of its 196 RGB patch tokens dropped, everything else masked, and the
point cloud read off the decoder each time. This re-runs exactly that on CPU and
packs the clouds for the interactive viewer.

Provenance notes that matter, and their limits:

  * checkpoint. sweeps/rgb_mask_sweep.py hard-codes ./outputs/4m_distill_15k_all/
    best_model.pth -- the ALL-SOURCE distilled generalist, epoch 95, val 0.2779
    (logs/rgbmask_12086400.out). Not the rgb2pc specialists used by the other
    two exported figures.

  * the masking mechanism is copied verbatim from forward_partial_rgb: one
    torch.rand + argsort shuffle over the 196 RGB tokens, keep the first
    round(196*(1-f)), depth/pc/text contribute zero tokens, decoder restores
    everything. Same len_keep at every level as the plate (196/157/98/39/10).

  * the plate ran on CUDA, so its torch.Generator(device='cuda') draw cannot be
    reproduced on CPU even at the same seed=0 -- the *number* of surviving
    patches matches, the particular subset does not.

  * SorghumDataset4M draws a fresh random 8196-point subsample of the .ply on
    every read; the plate read through a DataLoader worker whose numpy seed came
    from an unseeded torch base_seed, so its exact target cloud is not
    recoverable either. np.random.seed(0) here makes THIS export reproducible.

Both of those move single-plant Chamfer by O(1e-4), which is why the page's own
note tells the reader to trust the 208-plant table, not these frames.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse, json, pathlib
import numpy as np
import torch

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both
from export_pc_web import pack

CKPT = 'outputs/4m_distill_15k_all/best_model.pth'      # sweeps/rgb_mask_sweep.py:20
LEVELS = [0.0, 0.20, 0.50, 0.80, 0.95]                  # sweeps/rgb_mask_sweep.py:22
PLANT = 'Sorghum_10001_00'                              # sweeps/rgb_mask_sweep.py:23
# population means over 208 val plants, job 12086400 (logs/rgbmask_12086400.out)
POP = {0.0: 0.00107, 0.20: 0.00112, 0.50: 0.00153, 0.80: 0.00359, 0.95: 0.00680}


@torch.no_grad()
def forward_partial_rgb(m, rgb, depth, pc, params, mask_frac, seed=0):
    """rgb_mask_sweep.forward_partial_rgb, verbatim but device-agnostic."""
    x_rgb   = m.rgb_embed(rgb)      + m.pos_embed_2d   + m.modality_embed_rgb
    x_depth = m.depth_embed(depth)  + m.pos_embed_2d   + m.modality_embed_depth
    x_pc    = m.pc_embed(pc)        + m.pos_embed_pc   + m.modality_embed_pc
    x_text  = m.param_embed(params) + m.pos_embed_text + m.modality_embed_text
    B, L, D = x_rgb.shape
    dev = rgb.device
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.rand(B, L, device=dev, generator=g)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    len_keep = max(1, int(round(L * (1.0 - mask_frac))))
    ids_keep = ids_shuffle[:, :len_keep]
    xr_v = torch.gather(x_rgb, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    ar = lambda n: torch.arange(n, device=dev).unsqueeze(0).expand(B, n)
    x = torch.cat([xr_v, x_depth[:, :0], x_pc[:, :0], x_text[:, :0]], dim=1)
    x = torch.cat([m.cls_token.expand(B, -1, -1), x], dim=1)
    for blk in m.encoder_blocks:
        x = blk(x)
    x = m.encoder_norm(x)

    out = m.forward_decoder(x, ids_restore, ar(x_depth.shape[1]),
                            ar(x_pc.shape[1]), ar(x_text.shape[1]),
                            len_keep, 0, 0, 0)
    return out[2], len_keep                                   # pred_pc, len_keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_points', type=int, default=2600)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--data_root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--plant', default=PLANT)
    ap.add_argument('--load_seed', type=int, default=0, help='numpy seed for the .ply resample')
    ap.add_argument('--mask_seed', type=int, default=0, help='seed of the RGB token shuffle (plate used 0)')
    ap.add_argument('--out', default='vis_gallery/pc_masksweep.json')
    args = ap.parse_args()

    ds = SorghumDataset4M(args.data_root, split='val', num_points=args.num_points)
    by_name = {f.name: i for i, f in enumerate(ds.samples)}
    assert args.plant in by_name, f'{args.plant} not in the val split'

    model = embodied_mae_4m_base(target_points=args.num_points).eval()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, 'checkpoint does not match the model'
    epoch = int(ck.get('epoch', -1))
    print(f'checkpoint {CKPT}  epoch {epoch}  val {ck.get("val_loss", ck.get("best_val_loss"))}')

    # one read of the plant: the target cloud is the SAME for all five rows
    np.random.seed(args.load_seed)
    rgb, depth, pc, par, tv, nm = ds[by_name[args.plant]]
    b = lambda t: t.unsqueeze(0)
    gt = pc.numpy()

    rng = np.random.default_rng(0)
    gt_b64 = pack(gt, args.n_points, rng)

    rows, base = [], None
    for lv in LEVELS:
        pred, keep = forward_partial_rgb(model, b(rgb), b(depth), b(pc), b(par),
                                         lv, seed=args.mask_seed)
        f, r = chamfer_both(pred, pc.unsqueeze(0))
        cd = float(f + r)
        if base is None:
            base = cd
        rows.append(dict(
            label=f'{lv*100:.0f}% hidden', name=nm,
            chamfer=round(cd, 8),
            clouds={'gt': gt_b64, 'gen': pack(pred[0].numpy(), args.n_points, rng)},
            meta=[['RGB hidden', f'{lv*100:.0f}%'],
                  ['visible patches', f'{keep} / {model.num_patches}'],
                  ['this plant, vs 0%', f'{cd/base:.2f}x'],
                  ['208-plant mean', f'{POP[lv]:.5f}']]))
        print(f'  {lv*100:>3.0f}% hidden  {keep:>3}/196 patches  chamfer {cd:.5f}  '
              f'({cd/base:.2f}x)  population {POP[lv]:.5f}')

    out = {
        'id': 'mask_sweep',
        'source_figure': 'The same plant, progressively hidden',
        'checkpoint': CKPT, 'epoch': epoch, 'split': 'val',
        'n_points': args.n_points,
        'panels': [{'key': 'gt',  'label': 'ground truth', 'color': '#9aa7ad'},
                   {'key': 'gen', 'label': 'generated from RGB alone', 'color': '#1baf7a'}],
        'rows': rows,
        'note': (f'sweeps/rgb_mask_sweep.py on {nm}, val split, all-source distilled generalist '
                 f'{CKPT} (epoch {epoch}) — the checkpoint the static plate hard-codes. Each row '
                 f'keeps a random round(196x(1-f)) of the RGB patch tokens and masks depth, point '
                 f'cloud and spline params entirely; the ground-truth cloud is one single dataset '
                 f'read, identical in every row, so only the generated cloud degrades. Loader '
                 f'resample seeded np.random.seed({args.load_seed}) and the token shuffle seeded '
                 f'{args.mask_seed} (the plate used seed 0 on a CUDA generator, which draws a '
                 f'different subset of the same size).'),
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {p}  ({p.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
