#!/usr/bin/env python3
"""One MAIZE plant, all ten camera views, generated from a single modality each time.

The maize twin of eval/eval_views_one_plant.py, with two differences that change
what the rows mean:

  * Sorghum's ten views are an elevation ladder (camera height falls with the view
    index). Maize's cameras are random (tools/maize_view_elevation.py: every index
    spans -85..+81 deg of elevation with a random azimuth), so the rows are sorted
    by the camera's actual elevation above the plant centre, read from
    camera_pose.json ("position" - "plantCenter", world +Y up), not by index.
  * Sorghum's `_nc_cam.ply` is the same complete cloud in every view folder, so
    the point-cloud target is equally hard from every view. A maize
    `pointcloud_cam.ply` is what ITS OWN camera sees (tools/check_maize_cloud_views.py),
    so the chamfer column scores each view's generation against that view's
    visible surface: a per-view target, comparable across views only as a trend.

What does not move: the spline parameters are a property of the PLANT, and the
XML is byte-identical across a plant's ten folders. The ten parameter predictions
are ten estimates of one number, so their SPREAD across views is a direct measure
of how view-invariant the parameter head is, needing no extra labels.

    python eval/eval_views_one_plant_maize.py --plant plant_0004 --source rgb \\
        --config configs/config_maize_distill_all_d2.yaml \\
        --checkpoint outputs/maize_distill_all_d2/best_model.pth
    # "before": the teacher / student init, strict-loaded into the same model
    python eval/eval_views_one_plant_maize.py --plant plant_0004 --source rgb \\
        --checkpoint outputs/maize_4m_d2/checkpoints/checkpoint_epoch_600.pth \\
        --label 'before distillation'

The config fixes the data version (data.rgb_file / data.param_schema) and the
leaf-field layout follows it (LEAF_FIELDS / LEAF_FIELDS_V2).
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, maize_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import train_maize_4m_distill as DIS
import embodied_mae_4m_maize as M4
from maize_dataset_4m import MaizeDataset4M


def leaf_layout(param_schema):
    """(field names, scale, shift) of the real leaf slots under this schema."""
    if param_schema == 'v2':
        return list(M4.LEAF_FIELDS_V2), np.asarray(M4._LEAF_SCALE_V2), np.asarray(M4._LEAF_SHIFT_V2)
    return list(M4.LEAF_FIELDS), np.asarray(M4._LEAF_SCALE), np.asarray(M4._LEAF_SHIFT)


def camera_elevation(folder):
    """(sin(elevation), elevation in deg, camera height) from camera_pose.json."""
    f = Path(folder) / 'camera_pose.json'
    if not f.exists():
        return None, None, None
    m = json.loads(f.read_text())
    p, c = m.get('position'), m.get('plantCenter')
    if not p or not c:
        return None, None, None
    rel = [a - b for a, b in zip(p, c)]
    n = math.sqrt(sum(x * x for x in rel))
    s = rel[1] / n if n else None
    return s, (math.degrees(math.asin(max(-1.0, min(1.0, s)))) if s is not None else None), p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_maize_distill_all_d2.yaml')
    ap.add_argument('--checkpoint', default='outputs/maize_distill_all_d2/best_model.pth')
    ap.add_argument('--label', default='after distillation')
    ap.add_argument('--plant', default='plant_0004')
    ap.add_argument('--split', default='val')
    ap.add_argument('--source', default='rgb', choices=['rgb', 'depth', 'pc', 'text'])
    ap.add_argument('--data_root', default=None, help='default: the config\'s data.data_root')
    ap.add_argument('--out', default='vis_views_maize')
    ap.add_argument('--export_gallery', default=None,
                    help='directory to write the input renders and the generated '
                         'clouds into, for the interactive gallery')
    ap.add_argument('--gallery_px', type=int, default=448,
                    help='long edge of the exported render (source is 1024)')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.data_root:
        cfg['data']['data_root'] = a.data_root
    args = DIS.config_to_namespace(cfg)
    device = torch.device(a.device)
    root = Path(args.data_root) / a.split

    ds = MaizeDataset4M(args.data_root, img_size=args.img_size, num_points=args.num_points,
                        split=a.split, max_leaves=args.max_leaves,
                        rgb_file=args.rgb_file, param_schema=args.param_schema)
    names = [p.name for p in ds.samples]
    idx = [i for i, n in enumerate(names) if MaizeDataset4M.plant_of(n) == a.plant]
    if not idx:
        raise SystemExit(f"no folders for plant {a.plant} in {root}")
    idx.sort(key=lambda i: names[i])
    print(f"{a.plant}: {len(idx)} views -> {[names[i] for i in idx]}  "
          f"(rgb {args.rgb_file}, params {args.param_schema})")

    dl = DataLoader(Subset(ds, idx), batch_size=len(idx), shuffle=False, num_workers=4)
    rgb, depth, pc, params, tv, nm = next(iter(dl))
    rgb, depth, pc = rgb.to(device), depth.to(device), pc.to(device)
    params, tv = params.to(device), tv.to(device)

    model = DIS.build_model(args, device)
    DIS.load_weights_into(model, a.checkpoint, device, 'views')
    model.eval()
    m = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        _, _, (_, _, ppc, pparam), _ = m(rgb, depth, pc, params, tv, visible={a.source})

    # Per-leaf params in native units. The GT is identical for every view (the XML is
    # one file copied into ten folders), so these are N estimates of one number. Only
    # the first len(fields) leaf slots are real; the rest are zero padding in both GT
    # and prediction and would put a 0/0 into every ratio below.
    fields, scale, shift = leaf_layout(args.param_schema)
    NF = len(fields)
    valid = tv[0, 1:].bool().cpu().numpy()
    scale_t = torch.tensor(scale, device=device, dtype=torch.float32)
    shift_t = torch.tensor(shift, device=device, dtype=torch.float32)
    gt_leaf = (params[0, 1:][valid][:, :len(scale)] * scale_t - shift_t).cpu().numpy()[:, :NF]
    pred_leaf = np.stack([
        (pparam[v, 1:][valid][:, :len(scale)] * scale_t - shift_t).cpu().numpy()[:, :NF]
        for v in range(len(idx))])                      # (views, leaves, NF)

    rows = []
    for v, i in enumerate(idx):
        se, ed, h = camera_elevation(root / names[i])
        cd = float(((ppc[v:v+1].unsqueeze(2) - pc[v:v+1].unsqueeze(1))
                    ** 2).sum(-1).min(2).values.mean()
                   + ((ppc[v:v+1].unsqueeze(2) - pc[v:v+1].unsqueeze(1))
                      ** 2).sum(-1).min(1).values.mean())
        err = np.abs(pred_leaf[v] - gt_leaf)            # (leaves, NF)
        rows.append({'view': names[i][-2:], 'name': names[i],
                     'sin_elev': se, 'elev_deg': ed, 'cam_height': h, 'chamfer': cd,
                     'field_mae': err.mean(0).tolist(),
                     'leaf_mae_norm': float((err / (gt_leaf.std(0) + 1e-9)).mean())})
    order = sorted(range(len(rows)), key=lambda k: -(rows[k]['sin_elev'] if rows[k]['sin_elev'] is not None else -9))

    print(f"\n{a.label}  ·  source = {a.source}  ·  {gt_leaf.shape[0]} real leaves  "
          f"(rows sorted by camera elevation, looking down first)\n")
    print(f"{'view':>5}{'elev deg':>10}{'sin(elev)':>11}{'chamfer':>11}"
          + ''.join(f"{f[:9]:>10}" for f in fields))
    for k in order:
        r = rows[k]
        ed = f"{r['elev_deg']:>10.1f}" if r['elev_deg'] is not None else f"{'?':>10}"
        se = f"{r['sin_elev']:>11.3f}" if r['sin_elev'] is not None else f"{'?':>11}"
        print(f"{r['view']:>5}{ed}{se}{r['chamfer']:>11.6f}"
              + ''.join(f"{x:>10.3f}" for x in r['field_mae']))

    across = pred_leaf.std(0).mean(0)                   # (NF,)
    gtsd = gt_leaf.std(0)
    print(f"\n{'':>35}" + ''.join(f"{f[:9]:>10}" for f in fields))
    print(f"{'sd of prediction ACROSS views':>35}" + ''.join(f"{x:>10.3f}" for x in across))
    print(f"{'sd of the field across leaves':>35}" + ''.join(f"{x:>10.3f}" for x in gtsd))
    print(f"{'view sensitivity (ratio)':>35}"
          + ''.join(f"{x:>10.2f}" for x in across / (gtsd + 1e-9)))

    ch = [r['chamfer'] for r in rows]
    print(f"\nchamfer over views (each against ITS OWN camera's cloud): best {min(ch):.6f} "
          f"({rows[int(np.argmin(ch))]['view']}) worst {max(ch):.6f} "
          f"({rows[int(np.argmax(ch))]['view']}) -> {max(ch)/min(ch):.2f}x spread")

    if a.export_gallery:
        from PIL import Image
        g = Path(a.export_gallery); g.mkdir(parents=True, exist_ok=True)
        gal = {'plant': a.plant, 'source': a.source, 'label': a.label, 'views': []}
        for v, i in enumerate(idx):
            nmv = names[i]
            # The file the loader opens (rgba.png composited over black for data v2),
            # bar the resize to img_size and ImageNet normalisation.
            with Image.open(root / nmv / args.rgb_file) as im:
                if im.mode == 'RGBA':
                    im = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 255)), im)
                im = im.convert('RGB')
                im.thumbnail((a.gallery_px, a.gallery_px), Image.LANCZOS)
                im.save(g / f'view_{nmv[-2:]}.jpg', quality=88, optimize=True)
            gal['views'].append({
                'view': nmv[-2:], 'name': nmv,
                'sin_elev': rows[v]['sin_elev'], 'elev_deg': rows[v]['elev_deg'],
                'cam_height': rows[v]['cam_height'], 'chamfer': rows[v]['chamfer'],
                'img': f'view_{nmv[-2:]}.jpg',
                'pred': [round(float(x), 3) for x in ppc[v].cpu().numpy().ravel()],
                'gt':   [round(float(x), 3) for x in pc[v].cpu().numpy().ravel()],
                'nn':   [round(float(x), 4) for x in
                         torch.cdist(ppc[v:v+1], pc[v:v+1], p=2)[0]
                              .min(dim=0).values.cpu().numpy()],
            })
        (g / 'views.json').write_text(json.dumps(gal, separators=(',', ':')))
        print(f"gallery -> {g}  ({(g/'views.json').stat().st_size/1e6:.1f} MB json, "
              f"{len(idx)} renders)")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f'views_{a.plant}_{a.source}_{a.label.replace(" ", "-")}.json'
    p.write_text(json.dumps(
        {'plant': a.plant, 'source': a.source, 'label': a.label,
         'checkpoint': a.checkpoint, 'param_schema': args.param_schema,
         'rgb_file': args.rgb_file, 'leaf_fields': fields,
         'n_leaves': int(gt_leaf.shape[0]), 'rows': rows,
         'pred_sd_across_views': across.tolist(),
         'gt_sd_across_leaves': gtsd.tolist()}, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()
