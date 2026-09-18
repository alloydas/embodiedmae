#!/usr/bin/env python3
"""One plant, all ten camera views, generated from a single modality each time.

This is a natural experiment the dataset hands you for free. The ten folders
`<plant>_00 .. _09` are not arbitrary: camera height descends linearly with the
view index (y = y0 - 0.5 * index on every plant checked) while azimuth is
randomised, so the index is an ELEVATION LADDER from looking down at the plant
to looking up at it.

What makes it clean is that the answer does not move:

  * `<plant>_nc.ply` is the same cloud in all ten folders; `_nc_cam.ply` is that
    cloud rotated into the view frame, and the loader centres and unit-scales
    before Chamfer, so the point-cloud target is equally hard from every view.
  * The spline parameters are a property of the PLANT. All ten views share one
    ground-truth vector, so parameter predictions from ten different images are
    ten estimates of the same number -- their SPREAD is a direct measure of how
    view-invariant the parameter head is, needing no extra labels.

So anything that changes across the rows below is caused by the input image and
nothing else.

    python eval_views_one_plant.py --plant Sorghum_10001 --source rgb
    python eval_views_one_plant.py --plant Sorghum_10001 --source rgb \\
        --checkpoint outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth \\
        --label "before distillation"
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import train_sorghum_4m_distill as DIS
from embodied_mae_4m import _LEAF_SCALE, _LEAF_SHIFT
from sorghum_dataset_4m import SorghumDataset4M

LEAF_FIELDS = ['starting_point', 'length', 'roll_angle', 'branching_angle',
               'waviness_frequency', 'waviness_period_start_0',
               'waviness_period_start_1']


def camera_elevation(folder):
    """sin(elevation) of the camera, and its raw height, from camera_pose.json."""
    f = Path(folder) / 'camera_pose.json'
    if not f.exists():
        return None, None
    p = json.loads(f.read_text()).get('position')
    if not p:
        return None, None
    n = math.sqrt(sum(x * x for x in p))
    return (p[1] / n if n else None), p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_4m_distill_15k_all.yaml')
    ap.add_argument('--checkpoint', default='outputs/4m_distill_15k_all/best_model.pth')
    ap.add_argument('--label', default='after distillation')
    ap.add_argument('--plant', default='Sorghum_10001')
    ap.add_argument('--split', default='val')
    ap.add_argument('--source', default='rgb',
                    choices=['rgb', 'depth', 'pc', 'text'])
    ap.add_argument('--out', default='vis_views')
    ap.add_argument('--export_gallery', default=None,
                    help='directory to write the input renders and the generated '
                         'clouds into, for the interactive gallery')
    ap.add_argument('--gallery_px', type=int, default=448,
                    help='long edge of the exported render (source is 1024)')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    args = DIS.config_to_namespace(cfg)
    device = torch.device(a.device)
    root = Path(cfg['data']['data_root']) / a.split

    ds = SorghumDataset4M(str(root), img_size=args.img_size,
                          num_points=args.num_points, max_leaves=args.max_leaves)
    names = [p.name for p in ds.samples]
    idx = [i for i, n in enumerate(names) if n.rsplit('_', 1)[0] == a.plant]
    if not idx:
        raise SystemExit(f"no folders for plant {a.plant} in {root}")
    idx.sort(key=lambda i: names[i])
    print(f"{a.plant}: {len(idx)} views -> {[names[i] for i in idx]}")

    dl = DataLoader(Subset(ds, idx), batch_size=len(idx), shuffle=False, num_workers=4)
    rgb, depth, pc, params, tv, nm = next(iter(dl))
    rgb, depth, pc = rgb.to(device), depth.to(device), pc.to(device)
    params, tv = params.to(device), tv.to(device)

    model = DIS.build_model(args, device)
    DIS.load_weights_into(model, a.checkpoint, device, 'views')
    model.eval()
    m = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        _, _, (_, _, ppc, pparam), _ = m(rgb, depth, pc, params, tv,
                                         visible={a.source})

    # Per-leaf params in native units; the GT is identical for every view, so
    # these are N estimates of one number.
    # The leaf vector is N_PARAMS=9 wide but only the first 7 slots are real
    # fields; the last two are padding that is identically zero in both GT and
    # prediction. Left in, they contribute a 0/0 to every ratio below and print
    # view-sensitivity figures in the hundreds of thousands. Slice them off.
    NF = len(LEAF_FIELDS)
    valid = tv[0, 1:].bool().cpu().numpy()
    scale = torch.tensor(_LEAF_SCALE, device=device)
    shift = torch.tensor(_LEAF_SHIFT, device=device)
    gt_leaf = (params[0, 1:][valid] * scale + shift).cpu().numpy()[:, :NF]
    pred_leaf = np.stack([
        (pparam[v, 1:][valid] * scale + shift).cpu().numpy()[:, :NF]
        for v in range(len(idx))])                      # (views, leaves, 7)

    rows = []
    for v, i in enumerate(idx):
        se, h = camera_elevation(root / names[i])
        cd = float(((ppc[v:v+1].unsqueeze(2) - pc[v:v+1].unsqueeze(1))
                    ** 2).sum(-1).min(2).values.mean()
                   + ((ppc[v:v+1].unsqueeze(2) - pc[v:v+1].unsqueeze(1))
                      ** 2).sum(-1).min(1).values.mean())
        err = np.abs(pred_leaf[v] - gt_leaf)            # (leaves, 7)
        rows.append({'view': names[i][-2:], 'name': names[i],
                     'sin_elev': se, 'cam_height': h, 'chamfer': cd,
                     'field_mae': err.mean(0).tolist(),
                     'leaf_mae_norm': float((err / (gt_leaf.std(0) + 1e-9)).mean())})

    print(f"\n{a.label}  ·  source = {a.source}  ·  {gt_leaf.shape[0]} real leaves\n")
    print(f"{'view':>5}{'cam h':>8}{'sin(elev)':>11}{'chamfer':>11}"
          + ''.join(f"{f[:9]:>10}" for f in LEAF_FIELDS))
    for r in rows:
        print(f"{r['view']:>5}{r['cam_height']:>8.2f}{r['sin_elev']:>11.3f}"
              f"{r['chamfer']:>11.6f}"
              + ''.join(f"{x:>10.3f}" for x in r['field_mae']))

    # Spread ACROSS views of the prediction for each field: the GT is one
    # number, so this is pure view-sensitivity with no label needed.
    across = pred_leaf.std(0).mean(0)                   # (7,)
    gtsd = gt_leaf.std(0)
    print(f"\n{'':>35}" + ''.join(f"{f[:9]:>10}" for f in LEAF_FIELDS))
    print(f"{'sd of prediction ACROSS views':>35}" + ''.join(f"{x:>10.3f}" for x in across))
    print(f"{'sd of the field across leaves':>35}" + ''.join(f"{x:>10.3f}" for x in gtsd))
    print(f"{'view sensitivity (ratio)':>35}"
          + ''.join(f"{x:>10.2f}" for x in across / (gtsd + 1e-9)))

    ch = [r['chamfer'] for r in rows]
    print(f"\nchamfer over views: best {min(ch):.6f} ({rows[int(np.argmin(ch))]['view']}) "
          f"worst {max(ch):.6f} ({rows[int(np.argmax(ch))]['view']}) "
          f"-> {max(ch)/min(ch):.2f}x spread")

    if a.export_gallery:
        from PIL import Image
        g = Path(a.export_gallery); g.mkdir(parents=True, exist_ok=True)
        gal = {'plant': a.plant, 'source': a.source, 'label': a.label, 'views': []}
        for v, i in enumerate(idx):
            nmv = names[i]
            # The render, not a de-normalised tensor: this is the actual file the
            # loader opens, so what you see is what the model was given (bar the
            # resize to img_size and ImageNet normalisation).
            im = Image.open(root / nmv / 'rgb.png').convert('RGB')
            im.thumbnail((a.gallery_px, a.gallery_px), Image.LANCZOS)
            im.save(g / f'view_{nmv[-2:]}.jpg', quality=88, optimize=True)
            gal['views'].append({
                'view': nmv[-2:], 'name': nmv,
                'sin_elev': rows[v]['sin_elev'], 'cam_height': rows[v]['cam_height'],
                'chamfer': rows[v]['chamfer'],
                'img': f'view_{nmv[-2:]}.jpg',
                'pred': [round(float(x), 3) for x in ppc[v].cpu().numpy().ravel()],
                'gt':   [round(float(x), 3) for x in pc[v].cpu().numpy().ravel()],
                # Per GT point, the distance to the NEAREST predicted point, so
                # the ground-truth cloud can be coloured by whether the model
                # covered it. Shipped as a distance rather than a boolean: the
                # threshold is the thing a reader wants to move.
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
         'checkpoint': a.checkpoint, 'leaf_fields': LEAF_FIELDS,
         'n_leaves': int(gt_leaf.shape[0]), 'rows': rows,
         'pred_sd_across_views': across.tolist(),
         'gt_sd_across_leaves': gtsd.tolist()}, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()
