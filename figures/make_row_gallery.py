"""Qualitative Regime III figure: what the row-trained and isolated-trained models reconstruct from a row.

For each species, three held-out test plants at yaw 0 (broadside) and 90 (end-on). Columns:
  row photograph (the input RGB, neighbors included) | target plant's true cloud |
  row-trained model's prediction from the row (all three streams, recipe never an input) |
  isolated-trained control's prediction from the same row.
Clouds are drawn in the input camera frame (x right, y up); the prediction is colored over the true
cloud in light gray, and each panel is labeled with its Chamfer against the target (1e-4 m^2).
Uses the same model loading, depth convention, masking and seeds as eval/eval_row.py.

  python figures/make_row_gallery.py            # writes paper/figs/fig_rows_examples.{pdf,png}
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'eval'))
import os, numpy as np, torch
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import eval_row as ER
from row_dataset import RowDataset, IN_FILES

REPO = Path(__file__).resolve().parents[1]; OUT = REPO / 'paper' / 'figs'
SPEC = {'sorghum': ('/work/mech-ai/alloydas/sorghum_row', 'sorghum_row_ft', 'sorghum_row_iso'),
        'maize': ('/work/mech-ai/alloydas/maize_row', 'maize_row_ft', 'maize_row_iso')}
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#c9c8c4'
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def plants_for(root, split, k=3):
    names = sorted((e.name for e in os.scandir(Path(root) / split) if e.is_dir()), key=lambda s: int(s.split('_')[-1]))
    return [names[int(i)] for i in np.linspace(0, len(names) - 1, k + 2)[1:-1]]


def run_species(sp, thetas=(0, 90), seed=0):
    root, ft_run, iso_run = SPEC[sp]
    dev = 'cpu'
    ft, cfg, _ = ER.build_model(REPO / 'outputs' / ft_run, 'checkpoints/checkpoint_epoch_100.pth', dev, sp)
    iso, _, _ = ER.build_model(REPO / 'outputs' / iso_run, 'checkpoints/checkpoint_epoch_100.pth', dev, sp)
    info_sp = ER.species_info(sp)
    gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
    ds = RowDataset(root, 'test', sp, 'row', cfg.get('img_size', 224), cfg.get('num_points', info_sp['num_points']),
                    info_sp['max_leaves'], thetas=list(thetas), shots=[0], plants=plants_for(root, 'test'), rgb_gain=gain,
                    gt_points=cfg.get('gt_points', cfg.get('num_points', info_sp['num_points'])), occlusion_stats=False)
    mr = float(cfg.get('mask_ratio', 0.8)); rows = []
    for i in range(len(ds)):
        np.random.seed(seed + i)
        rgb, depth, pc, params, valid, _, _, gt_pc, inf = ds[i]
        b = lambda x: x[None].to(dev)
        out = {}
        for tag, m in (('ft', ft), ('iso', iso)):
            pred, _ = ER.predict(m, 'all', b(rgb), b(depth), b(pc), b(params), mr, seed + 1000003 * i)
            scale = torch.tensor([float(inf['scale'])])
            _, cg_m2, _, _ = ER.score_cloud(pred, b(gt_pc), scale)
            out[tag] = (pred[0].numpy(), float(cg_m2[0]) * 1e4)
        plant, theta, shot, folder = ds.items[ds._resolve(i)]
        im = Image.open(folder / IN_FILES['row'][0]).convert('RGBA')
        im = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 255)), im).convert('RGB').resize((320, 320), Image.LANCZOS)
        rows.append((plant, theta, np.asarray(im), gt_pc.numpy(), out))
        print(sp, plant, theta, {k: round(v[1], 1) for k, v in out.items()}, flush=True)
    return rows


def main():
    data = {sp: run_species(sp) for sp in ('sorghum', 'maize')}
    n = sum(len(v) for v in data.values())
    fig, ax = plt.subplots(n, 4, figsize=(6.9, 1.62 * n), gridspec_kw={'wspace': 0.05, 'hspace': 0.28})
    r = 0
    for sp, rows in data.items():
        for plant, theta, im, gt, out in rows:
            a = ax[r]
            a[0].imshow(im); a[0].set_title(f'{sp} · {plant.split("_")[-1]} · yaw {theta}°', fontsize=6.5, loc='left')
            lim = np.abs(gt[:, :2]).max() * 1.15
            a[1].scatter(gt[:, 0], gt[:, 1], s=0.25, c='#3c3b38', lw=0); a[1].set_title('target plant (truth)', fontsize=6.5)
            for k, (tag, col, lab) in enumerate((('ft', BLUE, 'row-trained'), ('iso', ORANGE, 'isolated-trained'))):
                p, cd = out[tag]; ap = a[2 + k]
                ap.scatter(gt[:, 0], gt[:, 1], s=0.25, c=GRAY, lw=0); ap.scatter(p[:, 0], p[:, 1], s=0.25, c=col, lw=0)
                ap.set_title(f'{lab}: {cd:.1f}', fontsize=6.5, color=col)
            for j in range(1, 4):
                a[j].set_xlim(-lim, lim); a[j].set_ylim(-lim, lim); a[j].set_aspect('equal')
            for x in a: x.set_xticks([]); x.set_yticks([]); [s.set_visible(False) for s in x.spines.values()]
            r += 1
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_rows_examples.{ext}', dpi=220, bbox_inches='tight')
    print('wrote', OUT / 'fig_rows_examples.png')


if __name__ == '__main__':
    main()
