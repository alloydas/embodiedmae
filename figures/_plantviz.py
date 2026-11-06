"""Shared helpers for the paper's dataset and visual-result figures (figures/make_dataset_fig.py,
make_dataset_stats_fig.py, make_teaser_fig.py, make_any2any_fig.py, make_views_fig.py).

- image loading for display (RGBA over white, packed-RGBA depth decoded and colour-mapped),
- clouds (open3d .ply, consistent 3-D framing, rasterised scatter),
- the species' models (headline = E1 epoch 1000, generalist = distilled best_model, row = row fine-tune)
  and one-call generation from a chosen source set, with the RGB / depth predictions unpatchified,
- a "recipe glyph": the procedural parameters drawn as a stem-and-leaves diagram (insertion height,
  leaf length, leaf width, side of the stem, and for maize the droop), the same drawing for the ground
  truth and for a predicted recipe. It is a schematic, not the generator's mesh: the generator is not
  part of this repository.
"""
import json, pathlib, re, sys
import numpy as np, torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import embodied_mae_4m as S
import embodied_mae_4m_maize as M
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M
from train_sorghum_4m import unpatchify, _unnorm_pix

ROOTS = {'sorghum': pathlib.Path('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'),
         'maize': pathlib.Path('/work/mech-ai-scratch/alloy/Maize_1')}
ROW_ROOTS = {'sorghum': pathlib.Path('/work/mech-ai/alloydas/sorghum_row'), 'maize': pathlib.Path('/work/mech-ai/alloydas/maize_row')}
O = REPO / 'outputs'
CKPT = {'sorghum': {'headline': O / '4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth',
                    'generalist': O / '4m_distill_15k_all_d2/best_model.pth',
                    'row': O / 'sorghum_row_ft/checkpoints/checkpoint_epoch_100.pth'},
        'maize': {'headline': O / 'maize_4m_pretrain_15k_d2/checkpoints/checkpoint_epoch_1000.pth',
                  'generalist': O / 'maize_distill_e1_d2/best_model.pth',
                  'row': O / 'maize_row_ft_e1/checkpoints/checkpoint_epoch_100.pth'}}
PTS = {'sorghum': 8196, 'maize': 8192}
RGB_FILE = {'sorghum': 'rgb_nobg.png', 'maize': 'rgba.png'}
# camera elevation ABOVE the plant for the sorghum view ladder (view 0 looks 64 deg down, view 9 64 deg up)
SORGHUM_ELEV = np.degrees(np.arcsin(0.9 - 0.2 * np.arange(10)))
IMNET_MEAN, IMNET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
# palette shared by all figures (white pages, print-safe)
C = dict(gt='#2f6b2a', pred='#1f6f8b', before='#9a6a00', neighbor='#c0392b', hidden='#e67e22', stem='#4a3b2a', muted='#777')


# ---------------------------------------------------------------- images
def rgba_on_white(path, size=None):
    im = Image.open(path).convert('RGBA')
    if size: im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255
    return a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4])


def alpha_mask(path, size=None):
    im = Image.open(path).convert('RGBA')
    if size: im = im.resize((size, size), Image.NEAREST)
    return np.asarray(im)[..., 3] > 0


def decode_depth_png(path, size=None):
    """Packed big-endian RGBA depth -> float in [0, 1] (background 0), native resolution."""
    im = Image.open(path)
    if size: im = im.resize((size, size), Image.NEAREST)
    a = np.asarray(im)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a.astype(np.uint32)
        v = (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]
        return v.astype(np.float64) / 4294967295.0
    a = a.astype(np.float64)
    return a / (65535.0 if a.max() > 255 else 255.0)


def depth_rgb(d, cmap='viridis', lo=None, hi=None):
    """Colour-map the foreground of a depth array, white background."""
    import matplotlib.cm as cm
    fg = d > 0
    if not fg.any(): return np.ones(d.shape + (3,))
    lo = np.percentile(d[fg], 1) if lo is None else lo; hi = np.percentile(d[fg], 99) if hi is None else hi
    t = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = cm.get_cmap(cmap)(t)[..., :3]
    rgb[~fg] = 1.0
    return rgb


def tensor_rgb(t):
    """(3,H,W) ImageNet-normalised tensor -> display RGB (what the model saw)."""
    x = t.permute(1, 2, 0).cpu().numpy() * IMNET_STD + IMNET_MEAN
    return np.clip(x, 0, 1)


# ---------------------------------------------------------------- clouds
def read_cloud(path):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)


def normalise_cloud(P):
    c = P.mean(0); X = P - c; s = float(np.linalg.norm(X, axis=1).max()) or 1.0
    return X / s, c, s


def frame_of(P, pad=1.05):
    lo, hi = P.min(0), P.max(0)
    return (lo + hi) / 2, (hi - lo).max() / 2 * pad


def scatter_cloud(ax, P, c, frame=None, s=0.6, alpha=0.85, elev=15, azim=-40, cmap=None, vmin=None, vmax=None, title=None, fs=7, zoom=1.45):
    """Camera-frame cloud (x right, y down, z away) drawn as (x, z, y) so the plant stands up."""
    P = np.asarray(P)
    ctr, h = frame if frame is not None else frame_of(P)
    sc = ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=s, c=c, alpha=alpha, linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(ctr[0] - h, ctr[0] + h); ax.set_ylim(ctr[2] - h, ctr[2] + h); ax.set_zlim(ctr[1] - h, ctr[1] + h)
    ax.set_box_aspect((1, 1, 1), zoom=zoom); ax.set_axis_off()
    if title: ax.set_title(title, fontsize=fs, pad=2)
    return sc


# ---------------------------------------------------------------- datasets and models
def run_cfg(ck):
    for d in (ck.parent, ck.parent.parent):
        if (d / 'config.json').exists(): return json.load(open(d / 'config.json'))
    return {}


def dataset(species, split='val'):
    """The loader as the species' final-data models read it (same files, gain and schema)."""
    if species == 'sorghum':
        cfg = run_cfg(CKPT['sorghum']['generalist'])
        return SorghumDataset4M(ROOTS['sorghum'], split=split, num_points=PTS['sorghum'], view_sampling=False,
                                rgb_file=cfg.get('rgb_file', 'rgb_nobg.png'), rgb_gain=cfg.get('rgb_brighten_gain'))
    return MaizeDataset4M(ROOTS['maize'], split=split, num_points=PTS['maize'], view_sampling=False, rgb_file='rgba.png', param_schema='v2')


def item_by_name(ds, name, seed=0):
    idx = next(i for i, f in enumerate(ds.samples) if f.name == name)
    np.random.seed(seed)                      # the loader subsamples the cloud with np.random
    return ds[idx]


def folder(species, split, plant, view):
    return ROOTS[species] / split / f'{plant}_{view:02d}'


def plant_name(species, folder_name):
    m = re.match(r'(Sorghum_\d+|plant_\d+)_(\d+)$', folder_name)
    return m.group(1), int(m.group(2))


def load_model(species, which, device='cpu'):
    ck = CKPT[species][which]
    build = S.embodied_mae_4m_base if species == 'sorghum' else M.embodied_mae_4m_maize_base
    cfg = run_cfg(ck)
    m = build(target_points=int(cfg.get('gt_points') or cfg.get('num_points') or PTS[species])).to(device).eval()
    st = torch.load(ck, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in st.get('model_state_dict', st).items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (ck, len(miss), len(unexp))
    return m, cfg, int(st.get('epoch', -1))


@torch.no_grad()
def generate(model, item, visible, device='cpu', seed=0, source_mask_ratio=0.0, zero_params=True):
    """One item (rgb, depth, pc, par, tv, name) -> dict(rgb, depth, pc, par) generated with the given
    source streams visible. Recipe input zeroed unless it is a source. RGB/depth returned as display arrays."""
    rgb, depth, pc, par, tv = (item[k][None].to(device) for k in range(5))
    par_in = par if ('text' in visible or not zero_params) else torch.zeros_like(par)
    torch.manual_seed(seed)
    _, _, (pr, pd, ppc, ppar), _ = model(rgb, depth, pc, par_in, tv, visible=list(visible), source_mask_ratio=source_mask_ratio)
    pr = _unnorm_pix(model, pr, rgb)
    pr_img = unpatchify(pr, model.patch_size, 3, model.img_size)[0]
    pd_img = unpatchify(pd, model.patch_size, 1, model.img_size)[0, 0]
    return dict(rgb=tensor_rgb(pr_img), depth=pd_img.float().cpu().numpy(), pc=ppc[0].float().cpu().numpy(),
                par=ppar[0].clamp(0, 1).float().cpu(), tv=item[4])


def chamfer(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    d = torch.cdist(a[None], b[None])[0]
    return float(d.min(1).values.pow(2).mean() + d.min(0).values.pow(2).mean())


# ---------------------------------------------------------------- recipes
def decode_recipe(species, par, tv=None, predicted=False):
    """Normalised (1+K, N_PARAMS) -> dict(height, leaves=[{...raw fields...}]) in the generator's units.
    Ground truth: the real leaf slots (tv). Predicted: the leaf count read from the recipe itself (the
    rule of the paper's Regime I table): maize from the plant token, sorghum from the stem length."""
    par = np.asarray(par, dtype=np.float64)
    if species == 'sorghum':
        plant = par[0] * S._PLANT_SCALE - S._PLANT_SHIFT
        H = float(plant[0])
        n = int(np.clip(np.rint((H + 0.001) / 0.05), 0, 24)) if predicted else int(np.asarray(tv)[1:].sum())
        raw = par[1:1 + n, :len(S.LEAF_FIELDS)] * S._LEAF_SCALE[:len(S.LEAF_FIELDS)] - S._LEAF_SHIFT[:len(S.LEAF_FIELDS)]
        leaves = [dict(zip(S.LEAF_FIELDS, r)) for r in raw]
        return dict(height=H, leaves=leaves, n=n)
    plant = par[0] * M._PLANT_SCALE_V2 - M._PLANT_SHIFT_V2
    n = int(np.clip(np.rint(plant[0]), 0, 28)) if predicted else int(np.asarray(tv)[1:].sum())
    nf = len(M.LEAF_FIELDS_V2)
    raw = par[1:1 + n, :nf] * M._LEAF_SCALE_V2[:nf] - M._LEAF_SHIFT_V2[:nf]
    leaves = [dict(zip(M.LEAF_FIELDS_V2, r)) for r in raw]
    return dict(height=float(plant[2]), leaves=leaves, n=n)


def draw_recipe_glyph(ax, species, rec, color, title=None, fs=7, ref=None, lw_scale=35.0):
    """Stem-and-leaves diagram. Stem: vertical line of the recipe's height. Leaf i: a segment (sorghum) or a
    drooping curve (maize) from its insertion height, length and width from the recipe, on the side its
    azimuth points to. `ref` (another recipe dict) sets the frame so GT and prediction share the scale."""
    H = rec['height']; L = rec['leaves']
    ax.plot([0, 0], [0, H], color=C['stem'], lw=2.2, solid_capstyle='round', zorder=2)
    xs = [0.05]
    if species == 'sorghum':
        # The sorghum recipe has no droop field: every leaf arcs by the same fixed 75 deg in this
        # schematic (a constant of the drawing), so only the fields it does have vary between leaves.
        for l in L:
            y = float(l['starting_point']) * H; ln = float(l['length']); b = np.radians(float(l['branching_angle']))
            side = 1.0 if np.cos(np.radians(float(l['roll_angle']))) >= 0 else -1.0
            phi0, phi1 = b, b + np.radians(75.0)
            p0 = np.array([0.0, y]); p1 = p0 + 0.5 * ln * np.array([side * np.sin(phi0), np.cos(phi0)])
            p2 = p1 + 0.5 * ln * np.array([side * np.sin(phi1), np.cos(phi1)])
            t = np.linspace(0, 1, 24)[:, None]
            q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            ax.plot(q[:, 0], q[:, 1], color=color, lw=0.6 + 0.6 * lw_scale * float(l['width']), solid_capstyle='round', alpha=0.9, zorder=3)
            xs.append(abs(q[:, 0]).max())
    else:
        y = 0.0
        for i, l in enumerate(L):
            y += float(l['distance']); ln = float(l['leafLength'])
            side = 1.0 if np.cos(np.radians(180.0 * i + float(l['azJitterDeg']))) >= 0 else -1.0
            phi0 = np.radians(35.0 + float(l['leafAngle']))                 # emergence, from the stem axis
            phi1 = phi0 - np.radians(float(l['droopiness']))                # the blade droops by `droopiness`
            p0 = np.array([0.0, y]); p1 = p0 + 0.5 * ln * np.array([side * np.sin(phi0), np.cos(phi0)])
            p2 = p1 + 0.5 * ln * np.array([side * np.sin(phi1), np.cos(phi1)])
            t = np.linspace(0, 1, 24)[:, None]
            q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            ax.plot(q[:, 0], q[:, 1], color=color, lw=0.6 + lw_scale * float(l['leafWidth']), solid_capstyle='round', alpha=0.9, zorder=3)
            xs.append(abs(q[:, 0]).max())
    top = max(H, max(xs)) if ref is None else max(ref['height'], H, max(xs))
    w = max(max(xs), 0.3) if ref is None else max(max(xs), 0.3, 0.6 * ref['height'])
    ax.set_xlim(-w * 1.1, w * 1.1); ax.set_ylim(-0.02 * top, top * 1.08)
    ax.set_aspect('equal'); ax.set_axis_off()
    if title: ax.set_title(title, fontsize=fs, pad=2)
    return dict(n=len(L), height=H)


def equalise(axes):
    xl = [ax.get_xlim() for ax in axes]; yl = [ax.get_ylim() for ax in axes]
    x = (min(l[0] for l in xl), max(l[1] for l in xl)); y = (min(l[0] for l in yl), max(l[1] for l in yl))
    for ax in axes: ax.set_xlim(*x); ax.set_ylim(*y)


# ---------------------------------------------------------------- rows
def row_shot_dir(species, split, plant, theta):
    d = ROW_ROOTS[species] / split / plant / f'theta_{theta:03d}'
    if (d / 'rgba.png').exists(): return d
    shots = sorted(d.glob('shot_*'))
    return shots[0]


ROW_NEARFAR = {'sorghum': (0.05, 4.5), 'maize': (0.05, 3.0)}


def row_masks(species, shot_dir, size=None):
    """(target pixels, neighbour pixels, target pixels hidden behind a neighbour) as in RowDataset.occlusion."""
    near, far = ROW_NEARFAR[species]
    dec = lambda c: c * (far - near) + near
    scene_a, gt_a = alpha_mask(shot_dir / 'rgba.png', size), alpha_mask(shot_dir / 'gt_rgba.png', size)
    sd, gd = dec(decode_depth_png(shot_dir / 'depth.png', size)), dec(decode_depth_png(shot_dir / 'gt_depth.png', size))
    nearer = (sd > near) & (sd < gd - 2e-3)
    return gt_a, scene_a & ~gt_a, gt_a & nearer


def row_overlay(species, shot_dir, size=None, nb=(0.86, 0.22, 0.2), hid=(0.95, 0.55, 0.1)):
    """The row photograph with neighbour pixels tinted red and hidden target pixels orange."""
    img = rgba_on_white(shot_dir / 'rgba.png', size)
    tgt, nbr, hidden = row_masks(species, shot_dir, size)
    out = img.copy()
    out[nbr] = 0.45 * img[nbr] + 0.55 * np.array(nb)
    out[hidden] = 0.35 * img[hidden] + 0.65 * np.array(hid)
    return out, dict(occ=float(hidden.sum() / max(1, tgt.sum())), nbr=float(nbr.sum() / max(1, (nbr | tgt).sum())))
