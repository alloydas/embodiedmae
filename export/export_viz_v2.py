"""Final-data (sorghum v2) data files for the ledger's interactive viewers.

Writes, under --out-dir:
  clouds_v2.js   window.CLOUD_DATA_V2 = {samples:[{name, clouds:[{label, xyz, nn}, {label, xyz, nn}]}]}
                 Eight held-out plants (view 00). Both panes show the plant's GROUND-TRUTH cloud;
                 nn = per-GT-point distance to the nearest point the model GENERATED FROM THE
                 PHOTOGRAPH ALONE (visible=['rgb']; every other stream fully masked, recipe zeroed).
                 Pane 0 = the Sinkhorn E1 at epoch 1,000 (no distillation), pane 1 = the distilled
                 generalist student at --student-ckpt.
  views_v2.js    window.VIEW_DATA_V2 = {plant, views:[{view, img, chamfer, pred, gt, nn}]}
                 One plant, all ten views, the student generating from each photograph alone;
                 img = view_v2_<vv>.jpg, the rgb_nobg.png the loader opens composited over black.
  params_v2.js   window.PARAM_DATA_V2 = {leaf_fields, leaf_units, plant_fields,
                 plants:[{name, n, plant:{gt,pc,rgb}, leaves:[{i, gt, pc, rgb}]}]}
                 The student's recipe head from the point cloud alone and from RGB alone, decoded to
                 the generator's units (final-data leaf layout: start, length, roll, branch, width).
  viz_v2_meta.json  checkpoints, epochs, per-plant Chamfer and miss fractions.

The schemas are the ones the September review's viewers consume (clouds.js / views.js /
params.js), so the same viewer code renders them.
  python export/export_viz_v2.py --out-dir /path/to/dir
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from embodied_mae_4m import embodied_mae_4m_base, LEAF_FIELDS, _LEAF_SCALE, _LEAF_SHIFT, _PLANT_SCALE, _PLANT_SHIFT
from sorghum_dataset_4m import SorghumDataset4M

ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
E1 = 'outputs/4m_pretrain_15k_d2_sink/checkpoints/checkpoint_epoch_1000.pth'


def load(path, dev):
    m = embodied_mae_4m_base(target_points=8196).to(dev).eval()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck['model_state_dict'].items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (path, len(miss), len(unexp))
    return m, int(ck.get('epoch', -1))


@torch.no_grad()
def generate(model, item, visible, dev, seed):
    rgb, depth, pc, par, tv, _ = item
    torch.manual_seed(seed)
    out = model(rgb[None].to(dev), depth[None].to(dev), pc[None].to(dev), torch.zeros_like(par)[None].to(dev),
                tv[None].to(dev), visible=list(visible))
    _, _, (_, _, pred_pc, pred_par), _ = out
    return pred_pc[0].float().cpu(), pred_par[0].float().cpu()


def nn_and_chamfer(gt, pred):
    d = torch.cdist(gt, pred)
    dg = d.min(1).values; dp = d.min(0).values
    return dg.numpy(), float((dg ** 2).mean() + (dp ** 2).mean())


def r4(a): return [round(float(x), 4) for x in np.asarray(a).ravel()]


def decode_leaf(v): return (np.clip(v[:5], 0, 1) * _LEAF_SCALE[:5] - _LEAF_SHIFT[:5]).tolist()
def decode_plant(v): return (np.clip(v[:4], 0, 1) * _PLANT_SCALE[:4] - _PLANT_SHIFT[:4]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student-ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-plants', type=int, default=8)
    ap.add_argument('--view-plant', default='Sorghum_10001')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    ds = SorghumDataset4M(ROOT, split='val', num_points=8196, view_sampling=False,
                          rgb_file='rgb_nobg.png', rgb_gain=1.85)
    idx = {f.name: i for i, f in enumerate(ds.samples)}
    plants = sorted({n.rsplit('_', 1)[0] for n in idx if n.endswith('_00')},
                    key=lambda s: int(s.split('_')[1]))[:a.n_plants]
    e1, e1_ep = load(E1, dev); st, st_ep = load(a.student_ckpt, dev)
    lab0, lab1 = f'E1, no distillation (ep {e1_ep})', f'distilled student (ep {st_ep})'
    meta = {'e1': E1, 'e1_epoch': e1_ep, 'student': a.student_ckpt, 'student_epoch': st_ep, 'plants': {}}

    def item(name, seed):
        np.random.seed(seed)
        return ds[idx[name]]

    # ---- 1. coverage viewer
    samples = []
    for k, p in enumerate(plants):
        it = item(f'{p}_00', 1000 + k); gt = it[2].float()
        clouds = []
        for lab, m in ((lab0, e1), (lab1, st)):
            pred, _ = generate(m, it, ['rgb'], dev, 7 + k)
            nn, cd = nn_and_chamfer(gt, pred)
            clouds.append({'label': lab, 'xyz': r4(gt.numpy()), 'nn': r4(nn)})
            meta['plants'].setdefault(p, {})[lab] = {'chamfer': cd, 'miss_0p01': float((nn > 0.01).mean()), 'miss_0p02': float((nn > 0.02).mean()), 'mean_nn': float(nn.mean())}
        samples.append({'name': f'{p}_00', 'clouds': clouds})
        print(p, {l: round(v['chamfer'] * 1e3, 3) for l, v in meta['plants'][p].items()}, flush=True)
    (out / 'clouds_v2.js').write_text('window.CLOUD_DATA_V2=' + json.dumps({'samples': samples}, separators=(',', ':')) + ';')

    # ---- 2. ten-view gallery (student, RGB alone)
    views = []
    for v in range(10):
        name = f'{a.view_plant}_{v:02d}'
        if name not in idx: continue
        it = item(name, 2000 + v); gt = it[2].float()
        pred, _ = generate(st, it, ['rgb'], dev, 11 + v)
        nn, cd = nn_and_chamfer(gt, pred)
        im = Image.open(ds.samples[idx[name]] / 'rgb_nobg.png').convert('RGBA')
        bg = Image.new('RGBA', im.size, (0, 0, 0, 255)); im = Image.alpha_composite(bg, im).convert('RGB').resize((384, 384), Image.LANCZOS)
        fn = f'view_v2_{v:02d}.jpg'; im.save(out / fn, quality=85)
        views.append({'view': f'{v:02d}', 'img': fn, 'chamfer': round(cd, 6), 'pred': r4(pred.numpy()), 'gt': r4(gt.numpy()), 'nn': r4(nn)})
        print('view', v, round(cd * 1e3, 3), flush=True)
    (out / 'views_v2.js').write_text('window.VIEW_DATA_V2=' + json.dumps({'plant': a.view_plant, 'views': views}, separators=(',', ':')) + ';')
    meta['views'] = {v['view']: v['chamfer'] for v in views}

    # ---- 3. parameter browser (student, PC alone / RGB alone)
    P = []
    for k, p in enumerate(plants):
        it = item(f'{p}_00', 3000 + k); par, tv = it[3].numpy(), it[4].numpy().astype(bool)
        _, pp = generate(st, it, ['pc'], dev, 21 + k); _, pr = generate(st, it, ['rgb'], dev, 31 + k)
        pp, pr = pp.numpy(), pr.numpy()
        leaves = [{'i': int(j), 'gt': r4(decode_leaf(par[j])), 'pc': r4(decode_leaf(pp[j])), 'rgb': r4(decode_leaf(pr[j]))}
                  for j in range(1, par.shape[0]) if tv[j]]
        P.append({'name': f'{p}_00', 'n': len(leaves), 'plant': {'gt': r4(decode_plant(par[0])), 'pc': r4(decode_plant(pp[0])), 'rgb': r4(decode_plant(pr[0]))}, 'leaves': leaves})
    (out / 'params_v2.js').write_text('window.PARAM_DATA_V2=' + json.dumps({
        'leaf_fields': list(LEAF_FIELDS), 'leaf_units': ['', '', '°', '°', ' m'],
        'plant_fields': ['stem_length', 'stem_dir_x', 'stem_dir_y', 'stem_dir_z'], 'plants': P}, separators=(',', ':')) + ';')
    (out / 'viz_v2_meta.json').write_text(json.dumps(meta, indent=1))
    print('wrote', sorted(x.name for x in out.iterdir()))


if __name__ == '__main__':
    main()
