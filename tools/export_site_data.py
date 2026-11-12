"""Data for the project page's interactive point-cloud viewers (site/).

Every item is one JSON file (its clouds as one base64 binary field `b64`) and an image or two:
  binary layout, per cloud: int16 x, y, z (n x 3, quantised by the manifest's `q`), then, for clouds with
  `err`, uint8 per-point error (n; millimetres = byte / 255 * err_max_mm), padded to an even length.
  Coordinates are each item's own frame: camera frame of the input (x right, y up, z toward the camera),
  centred and scaled as the model read it; `scale_m` converts a unit to metres.
Per-point error of a prediction = distance to the nearest true point, in millimetres.

Tasks (paper sections; both species each):
  rows      Regime III: the row scene (center plant marked), the true center plant, the zero-shot headline model,
            the isolated-trained control and the row fine-tune from all three streams, plus the row fine-tune
            from each single stream; yaw 0 and 90. Plants: the paper's Fig. 9 plants and the ten of the
            supplement gallery (figures/make_row_gallery_v2.py).
  dataset   the data the paper uses: isolated plants (ten views, the point cloud), synthetic crop rows
            (scene with the center plant marked, two yaws) and rows of real plants.

usage: python tools/export_site_data.py --tasks rows dataset [--out site/data] [--species sorghum maize]
CPU (about 15 minutes for both tasks and species); run it as a Slurm job.
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import argparse, csv, io, json, pickle
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', default=['rows', 'dataset'])
ap.add_argument('--species', nargs='+', default=['sorghum', 'maize'])
ap.add_argument('--out', default=str(R / 'site/data'))
ap.add_argument('--points', type=int, default=4096, help='points kept per object cloud (scene clouds keep 2x)')
a = ap.parse_args()
OUT = pathlib.Path(a.out); torch.set_num_threads(8)
ERR_MAX = {'maize': 40.0, 'sorghum': 100.0}      # colour-scale cap of the per-point error (mm), per species
ERR_MAX_MM = 40.0
GALLERY = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/row_gallery_dump.pkl')
FIG9 = {'sorghum': ['Sorghum_10756', 'Sorghum_14816', 'Sorghum_4815'], 'maize': ['plant_0320', 'plant_6665', 'plant_4604']}
ROWRUN = {'sorghum': dict(ft='sorghum_row_ft', iso='sorghum_row_iso', head='4m_pretrain_15k_d2_sink'),
          'maize': dict(ft='maize_row_ft_e1', iso='maize_row_iso_e1', head='maize_4m_pretrain_15k_d2')}
RS = np.random.default_rng(0)


def sub(P, n, seed):
    P = np.asarray(P, np.float32)
    if len(P) <= n: return P
    return P[np.random.default_rng(seed).choice(len(P), n, replace=False)]


def nn_dist(P, Q):
    """distance from every point of P to its nearest point of Q (torch, chunked)"""
    P, Q = torch.as_tensor(P, dtype=torch.float32), torch.as_tensor(Q, dtype=torch.float32)
    return torch.cat([torch.cdist(P[i:i + 4096], Q).min(1).values for i in range(0, len(P), 4096)]).numpy()


def chamfer(P, Q):
    return float((nn_dist(P, Q) ** 2).mean() + (nn_dist(Q, P) ** 2).mean())


class Item:
    """collects clouds for one manifest + binary"""
    def __init__(self, scale_m, err_max=ERR_MAX_MM):
        self.scale_m, self.clouds, self.err_max = float(scale_m), [], float(err_max)

    def add(self, name, P, err_mm=None, mark=None):
        self.clouds.append((name, np.asarray(P, np.float32), None if err_mm is None else np.asarray(err_mm, np.float32),
                            None if mark is None else np.asarray(mark, bool)))

    def write(self, stem, meta):
        q = 32767.0 / max(1e-6, max(float(np.abs(P).max()) for _, P, _, _ in self.clouds)) * 0.999
        buf, entries = io.BytesIO(), {}
        for name, P, e, m in self.clouds:
            off = buf.tell(); buf.write(np.round(P * q).astype('<i2').tobytes())
            ent = dict(off=off, n=int(len(P)))
            if e is not None:
                buf.write(np.clip(np.round(e / self.err_max * 255), 0, 255).astype(np.uint8).tobytes()); ent['err'] = True
            if m is not None:
                buf.write(m.astype(np.uint8).tobytes()); ent['mark'] = True
            if buf.tell() % 2: buf.write(b'\0')
            entries[name] = ent
        stem.parent.mkdir(parents=True, exist_ok=True)
        import base64                                       # binary inside the JSON (static hosts serve .json everywhere)
        meta = dict(meta, b64=base64.b64encode(buf.getvalue()).decode(), q=q, scale_m=self.scale_m, err_max_mm=self.err_max, clouds=entries)
        stem.with_suffix('.json').write_text(json.dumps(meta, separators=(',', ':')))
        return meta, buf.tell()


def save_jpg(arr, path, side=360, quality=88):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)) if arr.dtype != np.uint8 else Image.fromarray(arr)
    s = side / max(im.size); im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True); im.convert('RGB').save(path, quality=quality); return path.name


def rgba_over_black(path, gain=None):
    a_ = np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255
    return np.clip(a_[..., :3] * (gain or 1.0), 0, 1) * a_[..., 3:]


# ====================================================================== task: crop rows (Regime III)
def export_rows(sp):
    import _plantviz as V
    from row_dataset import RowDataset
    from eval_row import build_model, predict, DEFAULT_ROOT
    G = pickle.load(open(GALLERY, 'rb'))[sp]
    plants = [(p, 'Fig. 9') for p in FIG9[sp]] + [(p, f'{q:.0f}th percentile') for p, q in G['more']]
    runs = ROWRUN[sp]
    M = {k: build_model(R / 'outputs' / runs[k], 'checkpoints/checkpoint_epoch_1000.pth' if k == 'head' else 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
         for k in ('ft', 'iso', 'head')}
    cfg = M['ft'][1]; gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
    ds = RowDataset(DEFAULT_ROOT[sp], 'test', sp, 'row', cfg.get('img_size', 224), cfg['num_points'], cfg['max_leaves'], thetas=[0, 90], shots=[0],
                    plants=[p for p, _ in plants], rgb_gain=gain, gt_points=cfg['gt_points'], occlusion_stats=True)
    tag = dict(plants)
    index, total = [], 0
    for i in range(len(ds)):
        plant, theta, shot, folder = ds.items[ds._resolve(i)]
        seed = int(plant.split('_')[-1]) * 7 + theta
        np.random.seed(seed)
        rgb, depth, pc, params, valid, _, _, gt_pc, inf = ds[i]
        scale = float(inf['scale']); gt = gt_pc.numpy(); scene = pc.numpy()
        it = Item(scale, ERR_MAX[sp])
        sc = sub(scene, 2 * a.points, seed); it.add('scene', sc, mark=nn_dist(sc, gt) < 0.012)
        gts = sub(gt, a.points, seed + 1); it.add('truth', gts)
        rec_t = V.decode_recipe(sp, params.numpy(), valid.numpy())
        metrics = {}
        for key, srcs in (('head', ('all',)), ('iso', ('all',)), ('ft', ('all', 'rgb', 'depth', 'pc'))):
            m, mcfg, _ = M[key]
            for src in srcs:
                with torch.no_grad():
                    ppc, ppar = predict(m, src, rgb[None], depth[None], pc[None], params[None], float(mcfg.get('mask_ratio', 0.8)), seed)
                P = ppc[0].float().numpy(); Ps = sub(P, a.points, seed + 2)
                it.add(f'{key}:{src}', Ps, err_mm=nn_dist(Ps, gt) * scale * 1000)
                n_par = m.param_embed.proj.in_features if hasattr(m, 'param_embed') else None
                rec = V.decode_recipe(sp, ppar[0].clamp(0, 1).float().numpy(), valid.numpy(), predicted=True) if n_par == params.shape[-1] else None
                metrics[f'{key}:{src}'] = dict(chamfer=round(chamfer(P, gt) * scale ** 2 * 1e4, 2),
                                               height=None if rec is None else round(float(rec['height']), 3),
                                               leaves=None if rec is None else int(rec['n']))
        img = save_jpg(rgba_over_black(folder / 'rgba.png', gain), OUT / 'rows' / sp / f'{plant}_y{theta}.jpg')
        meta, nb = it.write(OUT / 'rows' / sp / f'{plant}_y{theta}', dict(
            task='rows', species=sp, plant=plant, yaw=theta, tag=tag[plant], img=img, occ=round(float(inf['occ_frac']), 3),
            truth=dict(height=round(float(rec_t['height']), 3), leaves=int(rec_t['n'])), metrics=metrics))
        total += nb
        index.append(dict(id=f'{plant}_y{theta}', plant=plant, yaw=theta, tag=tag[plant], json=f'{plant}_y{theta}.json',
                          ft=metrics['ft:all']['chamfer']))
        print(sp, plant, theta, {k: v['chamfer'] for k, v in metrics.items()}, f'{nb / 1e3:.0f} kB', flush=True)
    (OUT / 'rows' / sp / 'index.json').write_text(json.dumps(dict(task='rows', species=sp, items=index), indent=0))
    print(sp, 'rows', len(index), 'items', f'{total / 1e6:.1f} MB', flush=True)


# ====================================================================== task: the datasets
def export_dataset(sp):
    import _plantviz as V
    from row_dataset import read_ply_xyz
    out = OUT / 'dataset' / sp; index = []
    # (1) isolated plants: ten views and the point cloud (sorghum: the complete plant; maize: the cloud of view 00's camera)
    picks = {'sorghum': ['Sorghum_10036', 'Sorghum_10113', 'Sorghum_7712'], 'maize': ['plant_0106', 'plant_1875', 'plant_14506']}[sp]
    for plant in picks:
        views = []
        for v in range(10):
            f = V.folder(sp, 'test', plant, v)
            if not f.exists(): continue
            img = V.rgba_on_white(f / ('rgb_nobg.png' if sp == 'sorghum' else 'rgba.png'))
            views.append(save_jpg(img, out / 'isolated' / f'{plant}_v{v}.jpg', side=300))
        f0 = V.folder(sp, 'test', plant, 0)
        ply = next(f0.glob('*_nc_cam.ply')) if sp == 'sorghum' else f0 / 'pointcloud_cam.ply'
        P = V.read_cloud(ply); P, c, s = V.normalise_cloud(P)
        it = Item(s); it.add('plant', sub(P, 2 * a.points, 1))
        meta, nb = it.write(out / 'isolated' / plant, dict(task='dataset', kind='isolated', species=sp, plant=plant, views=views,
                                                         note='complete plant' if sp == 'sorghum' else 'what camera 00 sees'))
        index.append(dict(kind='isolated', id=plant, json=f'isolated/{plant}.json'))
        print(sp, 'isolated', plant, len(views), 'views', f'{nb / 1e3:.0f} kB', flush=True)
    # (2) synthetic crop rows: scene with the center plant marked, two yaws
    root = V.ROW_ROOTS[sp] / 'test'
    for plant in FIG9[sp][:2]:
        for theta in (0, 90):
            shot = V.row_shot_dir(sp, 'test', plant, theta)
            scene = read_ply_xyz(shot / 'pointcloud_cam.ply'); gt = read_ply_xyz(shot / 'gt_pointcloud_cam.ply')
            c = scene.mean(0); s = float(np.linalg.norm(scene - c, axis=1).max())
            scene, gt = (scene - c) / s, (gt - c) / s
            sc = sub(scene, 3 * a.points, theta)
            it = Item(s); it.add('scene', sc, mark=nn_dist(sc, gt) < 0.012)
            img = save_jpg(rgba_over_black(shot / 'rgba.png', 1.85 if sp == 'sorghum' else None), out / 'rows' / f'{plant}_y{theta}.jpg')
            meta, nb = it.write(out / 'rows' / f'{plant}_y{theta}', dict(task='dataset', kind='row', species=sp, plant=plant, yaw=theta, img=img))
            index.append(dict(kind='row', id=f'{plant}_y{theta}', json=f'rows/{plant}_y{theta}.json'))
            print(sp, 'row', plant, theta, f'{nb / 1e3:.0f} kB', flush=True)
    # (3) rows of real plants: one stop per layout, the scene with the center plant marked
    from real_rows import RealRowDataset, SPECIES as RSP
    stops = {'maize': [('parallel', '0148', 14), ('perpendicular', '0137', 4)], 'sorghum': None}[sp]
    if stops is None:
        ds_ = RealRowDataset(layouts=('parallel', 'perpendicular'), species='sorghum', rgb_gain=None)
        seen, stops = set(), []
        for lay, d, rid, stop in ds_.items:
            if lay not in seen and stop == 9: stops.append((lay, rid, stop)); seen.add(lay)
    for lay, rid, stop in stops:
        ds_ = RealRowDataset(layouts=(lay,), rows=[rid], stops={stop}, seed=0, species=sp)
        rgb, dep, pin, par, valid, pgt, info = ds_[0]
        P, G_ = pin.numpy(), pgt.numpy()
        sc = sub(P, 3 * a.points, 0)
        it = Item(info['scale']); it.add('scene', sc, mark=nn_dist(sc, G_) < 0.012)
        d = next(pathlib.Path(x) for x in RSP[sp]['layouts'][lay].glob(f'row_{rid}_seed1'))
        fr = list(csv.DictReader(open(d / f'row_{rid}_seed1_frames.csv')))[stop - 1]
        img = save_jpg(np.asarray(Image.open(d / fr['photo']).convert('RGB')), out / 'real' / f'{lay}_{rid}_{stop}.jpg', side=420)
        meta, nb = it.write(out / 'real' / f'{lay}_{rid}_{stop}', dict(task='dataset', kind='real', species=sp, layout=lay, row=rid, stop=stop,
                                                                      img=img, height=round(float(info['height']), 3)))
        index.append(dict(kind='real', id=f'{lay}_{rid}_{stop}', json=f'real/{lay}_{rid}_{stop}.json'))
        print(sp, 'real', lay, rid, stop, f'{nb / 1e3:.0f} kB', flush=True)
    (out / 'index.json').write_text(json.dumps(dict(task='dataset', species=sp, items=index), indent=0))


if __name__ == '__main__':
    for sp in a.species:
        if 'dataset' in a.tasks: export_dataset(sp)
        if 'rows' in a.tasks: export_rows(sp)
    print('done', OUT, flush=True)
