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
  single    Stage 1 masked pretraining (input with 80 % of tokens hidden -> reconstruction) and Stage 2 any input ->
            every stream (each single stream and the three pairs, headline model before and generalist after
            distillation); the paper's figure plants plus a 5th..95th percentile spread of the test set.
  partial   Regime II: 1 camera, 3 cameras, full cloud -> reconstruction by the headline model; 8 evenly spaced test plants.
  real      E10: rows of real plants (held-out only): headline, synthetic-row fine-tune, real-row fine-tune.
  dataset   the data the paper uses: isolated plants (ten views, the point cloud), synthetic crop rows
            (scene with the center plant marked, two yaws) and rows of real plants.

usage: python tools/export_site_data.py --tasks rows dataset single partial real [--out site/data] [--species sorghum maize]
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
        metrics = {}; recipes = {'truth': full_recipe(rec_t)}
        for key, srcs in (('head', ('all',)), ('iso', ('all',)), ('ft', ('all', 'rgb', 'depth', 'pc'))):
            m, mcfg, _ = M[key]
            for src in srcs:
                with torch.no_grad():
                    ppc, ppar = predict(m, src, rgb[None], depth[None], pc[None], params[None], float(mcfg.get('mask_ratio', 0.8)), seed)
                P = ppc[0].float().numpy(); Ps = sub(P, a.points, seed + 2)
                it.add(f'{key}:{src}', Ps, err_mm=nn_dist(Ps, gt) * scale * 1000)
                n_par = m.param_embed.proj.in_features if hasattr(m, 'param_embed') else None
                rec = V.decode_recipe(sp, ppar[0].clamp(0, 1).float().numpy(), valid.numpy(), predicted=True) if n_par == params.shape[-1] else None
                if rec is not None: recipes[f'{key}:{src}'] = full_recipe(rec)
                metrics[f'{key}:{src}'] = dict(chamfer=round(chamfer(P, gt) * scale ** 2 * 1e4, 2),
                                               height=None if rec is None else round(float(rec['height']), 3),
                                               leaves=None if rec is None else int(rec['n']))
        img = save_jpg(rgba_over_black(folder / 'rgba.png', gain), OUT / 'rows' / sp / f'{plant}_y{theta}.jpg')
        meta, nb = it.write(OUT / 'rows' / sp / f'{plant}_y{theta}', dict(
            task='rows', species=sp, plant=plant, yaw=theta, tag=tag[plant], img=img, occ=round(float(inf['occ_frac']), 3),
            truth=dict(height=round(float(rec_t['height']), 3), leaves=int(rec_t['n'])), metrics=metrics, recipes=recipes))
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


# ====================================================================== shared helpers for single plants
def full_recipe(rec):
    """the decoded recipe for the page's procedural model: height, leaf count, one row of the species' leaf fields per leaf"""
    if rec is None: return None
    f = list(rec['leaves'][0].keys()) if rec['leaves'] else []
    return dict(h=round(float(rec['height']), 4), n=int(rec['n']), f=f, L=[[round(float(l[k]), 4) for k in f] for l in rec['leaves']])


def recipe_of(V, sp, par, tv, predicted):
    r = V.decode_recipe(sp, np.asarray(par), np.asarray(tv), predicted=predicted)
    return dict(height=round(float(r['height']), 3), leaves=int(r['n']))


def glyph_png(sp, rec, path, side=2.2):
    import _fig12_kit as K
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(K.tile_glyph(sp, rec, side=side, px=180, lw_scale=0.9)); return path.name


# ====================================================================== tasks: masked pretraining + any input -> every stream
SOURCES = [('rgb',), ('depth',), ('pc',), ('text',), ('rgb', 'depth'), ('rgb', 'text'), ('depth', 'text')]


def single_plants(sp):
    """the paper's figure plants plus a spread over the test set's image-to-cloud Chamfer (5th..95th percentile)"""
    rep = json.load(open(R / {'sorghum': 'reports/quant_generalist_rgb_d2_test.json', 'maize': 'reports/quant_maize_generalist_rgb_d2_test.json'}[sp]))
    names, ch = rep['plants'], np.array(rep['per_sample']['model']); order = np.argsort(ch)
    fig = {'sorghum': [('Sorghum_7712_05', 'Fig. 12'), ('Sorghum_10036_00', 'Fig. 2')], 'maize': [('plant_1875_09', 'Fig. 11')]}[sp]
    out, seen = list(fig), {n for n, _ in fig}
    for q in (5, 20, 35, 50, 65, 80, 95):
        k = int(round(q / 100 * (len(order) - 1)))
        while names[order[k]] in seen: k += 1
        out.append((names[order[k]], f'{q}th percentile')); seen.add(names[order[k]])
    return out


def export_single(sp):
    import _plantviz as V
    from train_sorghum_4m import unpatchify, _unnorm_pix
    import _fig12_kit as K
    ds = V.dataset(sp, 'test')
    head, hcfg, hep = V.load_model(sp, 'headline'); gen, gcfg, gep = V.load_model(sp, 'generalist')
    out = OUT / 'single' / sp; index = []
    for name, tag in single_plants(sp):
        item = V.item_by_name(ds, name, seed=0)
        rgb, depth, pc, par, tv = (item[k][None] for k in range(5))
        plant, view = name.rsplit('_', 1)
        f0 = V.folder(sp, 'test', plant, int(view))
        ply = next(f0.glob('*_nc_cam.ply')) if sp == 'sorghum' else f0 / 'pointcloud_cam.ply'
        s_m = float(V.normalise_cloud(V.read_cloud(ply))[2])          # metres per unit of the normalised cloud
        truth = pc[0].numpy(); it = Item(s_m, ERR_MAX[sp] / 2)
        it.add('truth', sub(truth, a.points, 1))
        # --- masked pretraining (Stage 1): the headline model at its training mask ratio
        rec = {}
        orig = head.pc_embed.fps
        def fps(xyz, n, _o=orig, _r=rec):
            idx = _o(xyz, n); _r['idx'] = idx.detach().clone(); return idx
        head.pc_embed.fps = fps; torch.manual_seed(3)
        with torch.no_grad():
            _, _, (pr, pd_, ppc, ppar), (mr, md, mp, mt) = head(rgb, depth, pc, par, tv, mask_ratio=0.8)
        head.pc_embed.fps = orig
        cent = truth[rec['idx'][0].numpy()]
        grp = np.linalg.norm(truth[:, None] - cent[None], axis=2).argmin(1)
        vis = (mp[0].numpy() < 0.5)[grp]
        it.add('masked', sub(truth, a.points, 1), mark=sub(np.c_[truth, vis], a.points, 1)[:, 3] > 0.5)
        P = ppc[0].float().numpy(); Ps = sub(P, a.points, 2)
        it.add('recon', Ps, err_mm=nn_dist(Ps, truth) * s_m * 1000)
        visible = {k: int((1 - m[0]).sum()) for k, m in (('rgb', mr), ('depth', md), ('pc', mp), ('text', mt))}
        img_rgb = V.tensor_rgb(rgb[0]); d0 = depth[0, 0].numpy()
        rec_rgb = V.tensor_rgb(unpatchify(_unnorm_pix(head, pr, rgb), head.patch_size, 3, head.img_size)[0])
        pn = unpatchify(pd_, head.patch_size, 1, head.img_size)[0, 0].float().numpy()
        zd = pn * (d0.max() - d0.min()) + d0.min(); zd[pn < 0.05] = 0
        files = dict(rgb=save_jpg(img_rgb, out / f'{name}_rgb.jpg', 300), depth=save_jpg(V.depth_rgb(d0), out / f'{name}_depth.jpg', 300),
                     rgb_masked=save_jpg(K.masked(img_rgb, mr[0].numpy()), out / f'{name}_rgb_masked.jpg', 300),
                     depth_masked=save_jpg(K.masked(V.depth_rgb(d0), md[0].numpy()), out / f'{name}_depth_masked.jpg', 300),
                     rgb_rec=save_jpg(rec_rgb, out / f'{name}_rgb_rec.jpg', 300), depth_rec=save_jpg(V.depth_rgb(zd), out / f'{name}_depth_rec.jpg', 300),
                     params=glyph_png(sp, V.decode_recipe(sp, par[0].numpy(), tv[0].numpy()), out / f'{name}_params.png'))
        metrics = {'recon': dict(chamfer=round(V.chamfer(P, truth) * 1e3, 3))}
        recipes = {'truth': full_recipe(V.decode_recipe(sp, par[0].numpy(), tv[0].numpy())),
                   'recon': full_recipe(V.decode_recipe(sp, ppar[0].clamp(0, 1).float().numpy(), tv[0].numpy(), predicted=True))}
        # --- any input -> every stream (Stage 2): the headline model (before) and the distilled generalist (after)
        for src in SOURCES:
            key = '+'.join(src)
            for tag2, m in (('before', head), ('after', gen)):
                g = V.generate(m, item, list(src), seed=0)
                Pg = np.asarray(g['pc'], np.float32); Pgs = sub(Pg, a.points, 3)
                it.add(f'{key}:{tag2}', Pgs, err_mm=nn_dist(Pgs, truth) * s_m * 1000)
                r_ = recipe_of(V, sp, g['par'], tv[0].numpy(), True) if 'text' not in src else None
                if 'text' not in src: recipes[f'{key}:{tag2}'] = full_recipe(V.decode_recipe(sp, np.asarray(g['par']), tv[0].numpy(), predicted=True))
                metrics[f'{key}:{tag2}'] = dict(chamfer=round(V.chamfer(Pg, truth) * 1e3, 3), **(r_ or {}))
        meta, nb = it.write(out / name, dict(task='single', species=sp, plant=name, tag=tag, files=files, visible=visible,
                                             truth=recipe_of(V, sp, par[0].numpy(), tv[0].numpy(), False), metrics=metrics, recipes=recipes))
        index.append(dict(id=name, tag=tag, json=f'{name}.json'))
        print(sp, 'single', name, tag, {k: v['chamfer'] for k, v in metrics.items()}, f'{nb / 1e3:.0f} kB', flush=True)
    (out / 'index.json').write_text(json.dumps(dict(task='single', species=sp, items=index), indent=0))


# ====================================================================== task: partial clouds (Regime II)
def export_partial(sp, n_plants=8):
    """the headline model given 1 camera, 3 cameras or the full cloud (sorghum: visibility masks of cameras 00-02 on the
    complete plant; maize: the per-view clouds fused in camera 00's frame, 'full' = all ten), view 00's image and depth,
    reconstruction in the masked regime as eval/pc_view_eval*.py. Clouds are shown in the full cloud's frame."""
    import importlib.util, open3d as o3d
    import _plantviz as V
    from scipy.spatial import cKDTree
    def imp(name):
        spec = importlib.util.spec_from_file_location(name, R / 'eval' / f'{name}.py'); m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m; spec.loader.exec_module(m); return m
    model, cfg, ep = V.load_model(sp, 'headline')
    n_text = model.pos_embed_text.shape[1]; n_par = model.param_embed.proj.in_features
    visible = tuple(m for m in model.active_modalities if m != 'text'); mr_ = float(cfg.get('mask_ratio', 0.8))
    N = int(cfg.get('num_points') or V.PTS[sp])
    if sp == 'sorghum':
        from sorghum_dataset_4m import SorghumDataset4M
        PM = imp('pc_view_masks'); masks = PM.Masks('test', (0, 1, 2), R / 'outputs' / '_pcview_masks')
        base = SorghumDataset4M(V.ROOTS[sp], img_size=224, num_points=N, split='test', max_leaves=24, view_sampling=True, deterministic_view=True,
                                rgb_file=cfg.get('rgb_file', 'rgb_nobg.png'), rgb_gain=cfg.get('rgb_brighten_gain'))
        conds = {'1view': (0,), '3view': (0, 1, 2), 'full': None}
    else:
        from maize_dataset_4m import MaizeDataset4M
        PVM = imp('pc_view_eval_maize')
        base = MaizeDataset4M(V.ROOTS[sp], img_size=224, num_points=N, split='test', max_leaves=28, view_sampling=True, deterministic_view=True,
                              rgb_file='rgba.png', param_schema='v2')
        conds = {'1view': (0,), '3view': (0, 1, 2), 'full': tuple(range(10))}
    folders = [base.samples[v[0]] for v in base.plant_views]
    pick = [folders[int(i)] for i in np.linspace(0, len(folders) - 1, n_plants + 2)[1:-1]]      # evenly spaced, not selected
    out = OUT / 'partial' / sp; index = []
    def take(rng, n): return rng.choice(n, N, replace=False) if n >= N else np.concatenate([np.arange(n), rng.choice(n, N - n, replace=True)])
    for folder in pick:
        rgb = base.load_rgb(folder)[None]; depth = base.load_depth(folder / 'depth.png')[None]
        if sp == 'sorghum':
            pid = int(folder.name.split('_')[1])
            pts = np.asarray(o3d.io.read_point_cloud(str(base.find_pointcloud_file(folder))).points, dtype=np.float64)
            sel = {c: pts if v is None else pts[masks.get(pid, v)] for c, v in conds.items()}
            gt_raw = pts[take(np.random.default_rng([0, pid, 99]), len(pts))]
        else:
            plant = MaizeDataset4M.plant_of(folder.name); pid = int(plant.split('_')[-1])
            _, W2C0 = PVM._pose(folder); clouds = []
            for v in range(10):
                f = folder.with_name(f'{plant}_{v:02d}')
                P = np.asarray(o3d.io.read_point_cloud(str(f / PVM.PC_FILENAME)).points, dtype=np.float64)
                if v: P = PVM._apply(W2C0, PVM._apply(PVM._pose(f)[0], P))
                clouds.append(P)
            sel = {c: np.concatenate([clouds[v] for v in vs]) for c, vs in conds.items()}
            gt_raw = sel['full'][take(np.random.default_rng([0, pid, 99]), len(sel['full']))]
        x_full = sel['full'][take(np.random.default_rng([0, pid, 2]), len(sel['full']))]
        cf = x_full.mean(0); sf = float(np.linalg.norm(x_full - cf, axis=1).max())
        it = Item(sf, ERR_MAX[sp] / 2); truth = (gt_raw - cf) / sf
        it.add('truth', sub(truth, a.points, 1))
        metrics, tree = {}, cKDTree(gt_raw)
        params = torch.zeros(1, n_text, n_par)
        it_true = base[folders.index(folder)]                          # the plant's item (view sampling indexes plants); its recipe only
        tv = it_true[4].numpy(); recipes = {'truth': full_recipe(V.decode_recipe(sp, it_true[3].numpy(), tv))}
        for ci, (c, _) in enumerate(conds.items()):
            x = sel[c][take(np.random.default_rng([0, pid, ci]), len(sel[c]))]
            cc = x.mean(0); sc = float(np.linalg.norm(x - cc, axis=1).max())
            pc_in = torch.from_numpy(((x - cc) / sc).astype(np.float32))[None]
            torch.manual_seed(1)
            with torch.no_grad():
                enc = model.forward_encoder_select(rgb, depth, pc_in, params, visible=visible, source_mask_ratio=mr_)
                dec = model.forward_decoder(enc[0], *enc[5:9], *enc[9:13]); pred = dec[2][0].float().numpy()
            recipes[f'{c}:recon'] = full_recipe(V.decode_recipe(sp, dec[3][0].clamp(0, 1).float().numpy(), tv, predicted=True))
            pred_full = (pred * sc + cc - cf) / sf
            it.add(f'{c}:input', sub((x - cf) / sf, a.points, 4))
            Ps = sub(pred_full, a.points, 5); it.add(f'{c}:recon', Ps, err_mm=nn_dist(Ps, truth) * sf * 1000)
            gt_c = (gt_raw - cc) / sc
            cov = float((cKDTree(sel[c]).query(gt_raw)[0] <= 0.005).mean())
            metrics[c] = dict(chamfer=round(chamfer(pred, gt_c) * 1e3, 3), coverage=round(cov, 3))
        img = save_jpg(V.tensor_rgb(rgb[0]), out / f'{folder.name}.jpg', 300)
        meta, nb = it.write(out / folder.name, dict(task='partial', species=sp, plant=folder.name, img=img, metrics=metrics, recipes=recipes))
        index.append(dict(id=folder.name, json=f'{folder.name}.json'))
        print(sp, 'partial', folder.name, metrics, f'{nb / 1e3:.0f} kB', flush=True)
    (out / 'index.json').write_text(json.dumps(dict(task='partial', species=sp, items=index), indent=0))


# ====================================================================== task: real plants (E10)
E10 = {'maize': dict(report='reports/e10_real_ft_test_17383192.csv', real='maize_real_ft', row='maize_row_ft_e1', head='maize_4m_pretrain_15k_d2',
                     fig=[('parallel', '0019', 17), ('parallel', '0206', 8), ('perpendicular', '0137', 4), ('perpendicular', '0215', 10)]),
       'sorghum': dict(report='reports/e10_real_ft_sorghum_test_17384357.csv', real='sorghum_real_ft', row='sorghum_row_ft', head='4m_pretrain_15k_d2_sink', fig=[])}


def export_real(sp):
    from eval_row import build_model, predict
    from real_rows import RealRowDataset, SPECIES as RSP
    from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2
    c = E10[sp]
    rows = [r for r in csv.DictReader(open(R / c['report'])) if r['source'] == 'all' and r['model'] == c['real']]
    picks = [(l, r_, s_, 'Fig. E10') for l, r_, s_ in c['fig']]
    if sp == 'maize':                                            # 6 stops per layout on distinct plants, 10th..98th percentile
        for lay in ('parallel', 'perpendicular'):
            g = sorted((r for r in rows if r['layout'] == lay), key=lambda r: float(r['chamfer_m2']))
            used = {p for l, p, _, _ in picks if l == lay}
            for q in (10, 30, 50, 70, 90, 98):
                k = int(round(q / 100 * (len(g) - 1)))
                while g[k]['row'] in used: k += 1
                picks.append((lay, g[k]['row'], int(g[k]['stop']), f'{q}th percentile')); used.add(g[k]['row'])
    else:                                                        # every held-out plant, its middle stop, layouts alternating
        for i, pl in enumerate(sorted({r['row'] for r in rows})):
            picks.append(('parallel' if i % 2 == 0 else 'perpendicular', pl, 9, 'held-out plant'))
    M = {}
    for k in ('head', 'row', 'real'):
        run = c[k]; ck = 'best_model.pth' if k == 'real' else ('checkpoints/checkpoint_epoch_1000.pth' if k == 'head' else 'checkpoints/checkpoint_epoch_100.pth')
        M[k] = build_model(R / 'outputs' / run, ck, 'cpu', sp)
    gain = float(M['real'][1].get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
    H_IDX = PLANT_FIELDS_V2.index('height')
    out = OUT / 'real' / sp; index = []
    for i, (lay, rid, stop, tag) in enumerate(picks):
        ds = RealRowDataset(layouts=(lay,), rows=[rid], stops={stop}, seed=0, species=sp, rgb_gain=gain)
        rgb, dep, pin, par, valid, pgt, info = ds[0]
        scale = float(info['scale']); it = Item(scale, ERR_MAX[sp] * (1 if sp == 'maize' else 3))
        sc = sub(pin.numpy(), 2 * a.points, i); gt = pgt.numpy()
        it.add('scene', sc, mark=nn_dist(sc, gt) < 0.012); it.add('truth', sub(gt, a.points, i + 1))
        metrics = {}; recipes = {}
        import _plantviz as V
        for k, (m, mcfg, _) in M.items():
            with torch.no_grad():
                pp, ppar = predict(m, 'all', rgb[None], dep[None], pin[None], par[None], float(mcfg.get('mask_ratio', 0.8)), 1000003 * i)
            P = pp[0].float().numpy(); Ps = sub(P, a.points, i + 2)
            it.add(k, Ps, err_mm=nn_dist(Ps, gt) * scale * 1000)
            h = None
            if sp == 'maize':
                p0 = ppar[0, 0].clamp(0, 1).float().numpy(); h = round(float(p0[H_IDX] * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX]), 3)
            metrics[k] = dict(chamfer=round(chamfer(P, gt) * scale ** 2 * 1e4, 2), height=h)
            if m.param_embed.proj.in_features == par.shape[-1]:
                recipes[k] = full_recipe(V.decode_recipe(sp, ppar[0].clamp(0, 1).float().numpy(), valid.numpy(), predicted=True))
        d = next(pathlib.Path(x) for x in RSP[sp]['layouts'][lay].glob(f'row_{rid}_seed1'))
        fr = list(csv.DictReader(open(d / f'row_{rid}_seed1_frames.csv')))[stop - 1]
        img = save_jpg(np.asarray(Image.open(d / fr['photo']).convert('RGB')), out / f'{lay}_{rid}_{stop}.jpg', side=420)
        sid = f'{lay}_{rid}_{stop}'
        meta, nb = it.write(out / sid, dict(task='real', species=sp, layout=lay, row=rid, stop=stop, tag=tag, img=img,
                                            truth=dict(height=round(float(info['height']), 3)), occ=round(float(info['occ_frac']), 3), metrics=metrics, recipes=recipes))
        index.append(dict(id=sid, tag=tag, layout=lay, json=f'{sid}.json'))
        print(sp, 'real', sid, tag, {k: v['chamfer'] for k, v in metrics.items()}, f'{nb / 1e3:.0f} kB', flush=True)
    (out / 'index.json').write_text(json.dumps(dict(task='real', species=sp, items=index), indent=0))


if __name__ == '__main__':
    for sp in a.species:
        if 'dataset' in a.tasks: export_dataset(sp)
        if 'rows' in a.tasks: export_rows(sp)
        if 'single' in a.tasks: export_single(sp)
        if 'partial' in a.tasks: export_partial(sp)
        if 'real' in a.tasks: export_real(sp)
    print('done', OUT, flush=True)
