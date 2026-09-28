"""Export the MAIZE "ten images of one plant" gallery: one held-out val plant,
all ten camera views, and the point cloud the base model (`maize_4m`,
checkpoints/checkpoint_epoch_600.pth) generates from each photograph ALONE.

Produces, under --out-dir:
  maize_view_00.jpg .. maize_view_09.jpg
        the rgb.png the loader opens for <plant>_<vv>, resized to --img-px wide
        (LANCZOS), JPEG quality --jpeg-q.
  views_maize.js
        window.VIEW_DATA = {plant, views: [{view, img, elev, chamfer,
                            pred: [x,y,z,...], gt: [x,y,z,...], nn: [...]}, ...]};
        pred = generated cloud, gt = the loader's cloud, nn = per-GT-point
        Euclidean (NOT squared) distance to the nearest predicted point.
        Arrays are rounded to 4 decimals; the per-view scalars to 6.
        Extra keys: per view elev_deg, azim_deg, frac_nn_gt_0p01 and
        frac_nn_gt_0p02 -- both fractions computed from the ROUNDED nn array
        the page actually ships (nn > thr), so they equal the page's live
        readout at that slider value; top level elev_order = the view ids
        sorted by elev (looking down first), because maize has no ladder.
  views_maize_report.json
        every number printed below (camera geometry, frame evidence, per-view
        Chamfer, miss anatomy, the parameter view-invariance table).

CAMERA GEOMETRY -- measured from camera_pose.json, not assumed.
  * `cameraToWorld` is an OpenGL-style camera: `plantCenter` maps to
    (0, 0, -d) in camera coordinates in every view, so the camera looks down
    its own -z. The JSON field `viewDirection` is the camera's +z column, i.e.
    it points from the plant BACK to the camera; the viewing direction is
    forward = -cameraToWorld[:3, 2] = -viewDirection (the same formula the
    sorghum exporters use). `elev` = forward_y = sin(elevation) in the Y-up
    world; positive = camera below the plant looking up.
  * Whether the ten views form an elevation ladder (sorghum's did:
    sin elev = -0.9 + 0.2 * index) is tested, not assumed; the linear fit and
    its residual are printed.

FRAME -- decided by carrying world-up through worldToCamera, not silhouettes.
  * World up is +Y: the renderer's camera x axis has an exactly-zero world-Y
    component in every view (a lookAt with up = +Y), and world +Y maps to a
    POSITIVE camera y in every view. So camera +y is up.
  * Forward is camera -z (plantCenter at negative z; every PLY point has
    -far <= z <= -near).
  * Horizontal handedness is then checked against the depth map by VALUE:
    each PLY point is projected with u = cx + sx*fx*x/(-z), v = cy - sy*fy*y/(-z)
    and its (-z - near)/(far - near) compared with the decoded depth.png at
    that pixel, for all four (sx, sy) sign pairs. Only the pair matching the
    world-up result is accepted.
  * The loader only centres and unit-scales (no rotation), so the loader's
    frame is the PLY's frame. If that frame is x right / y up / z toward the
    viewer (a three.js camera at +Z looking down -Z), the export is the
    identity; otherwise the diagonal sign flip that makes it so is applied and
    printed (a det = -1 flip is flagged as a mirror).

NO LEAKS. RGB is the only source: `forward_encoder_select(visible={'rgb'},
source_mask_ratio=0.0)` hands depth, PC and text ZERO encoder tokens (asserted
from the returned n_vis and masks), and the param tensor passed in is all zeros
(the maize plant token carries leafCount verbatim). This is then checked, not
assumed: every view is re-run with the REAL params and with depth and PC
replaced by random noise, and the predicted cloud and params must match the
main pass bit for bit (max|diff| = 0.0).

DETERMINISM. numpy and torch are seeded (seed = --seed + view) before each
dataset read (load_pointcloud permutes with np.random.choice) and again before
each forward (FPS draws its first centroid with torch.randint).

PARAMETERS. The same forward pass also yields the params predicted from RGB
alone. For every LEAF field, decoded to native units with the maize inverse
(raw = clip(norm, 0, 1) * _LEAF_SCALE - _LEAF_SHIFT; waveLPhase by atan2 of its
sin/cos pair, with circular differences and circular std), the report gives
MAE vs ground truth over views x real leaves, and view spread = mean over
leaves of the std across the ten views / that field's std across the plant's
real leaves (both ddof = 0). The GT is a property of the plant, identical in
all ten folders, so the across-view std is pure view sensitivity.

Chamfer is the repo's own `embodied_mae.chamfer_distance` (squared NN
distances, mean both ways) so it sits on the same scale as val_pc_chamfer.

MISS ANATOMY -- what a red point in the coverage pane actually is. A "miss" is
a GT point more than --miss-thr (0.05, loader-normalised units) from every
generated point. Both clouds are taken back to the PLY's camera frame in world
units (the loader's centroid and scale are recomputed from the PLY, and the
loader cloud is asserted to be a permutation of it), then projected through the
view's own intrinsics (1024 px). For each miss the report gives the distance in
PIXELS to the nearest generated point in the image plane, and, for that
image-plane neighbour, the offset along the line of sight
(dz = depth_gt - depth_pred, + = the generated point sits nearer the camera).
A miss with an image-plane neighbour within --px-thr pixels is a point the
model drew where the photograph shows it, at the wrong depth -- NOT a missing
leaf; the coverage pane opens at the camera's own viewpoint, where such a
depth error is invisible until the cloud is rotated. The same numbers are given
for all GT points as a control.

Runs on CPU (no CUDA-only ops), from the repo root:
    python export/export_maize_views.py --out-dir /path/to/review/maize
    python export/export_maize_views.py --plant plant_11052 --split val \\
        --run maize_4m --ckpt checkpoints/checkpoint_epoch_600.pth
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, maize_dataset_4m, eval/*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from embodied_mae import chamfer_distance
from embodied_mae_4m_maize import (LEAF_FIELDS, PLANT_FIELDS, _LEAF_SCALE,
                                   _LEAF_SHIFT, _PLANT_SCALE, _PLANT_SHIFT)
from maize_dataset_4m import MaizeDataset4M, PC_FILENAME
from eval.linear_probe_maize import build_model

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = '/work/mech-ai-scratch/alloy/Maize'

# 12 linear leaf slots in model order, then the (sin, cos) phase pair merged.
LEAF_OUT = list(LEAF_FIELDS[:12]) + ['waveLPhase']
LEAF_UNITS = {'leafAngle': 'deg', 'droopiness': 'deg', 'stemInclinationDeg': 'deg',
              'leafTwist': 'deg', 'azJitterDeg': 'deg', 'waveLPhase': 'rad'}


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def circ_std(a, axis):
    """Circular std sqrt(-2 ln R) of angles in radians."""
    R = np.hypot(np.cos(a).mean(axis), np.sin(a).mean(axis))
    return np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))


def decode_leaves(pf):
    """(1+max_leaves, 14) normalised -> (max_leaves, 13) native units."""
    p = np.clip(np.asarray(pf, np.float64), 0.0, 1.0)
    raw = p[1:] * _LEAF_SCALE.astype(np.float64) - _LEAF_SHIFT.astype(np.float64)
    phase = np.arctan2(raw[:, 12], raw[:, 13])
    return np.concatenate([raw[:, :12], phase[:, None]], axis=1)


def decode_plant(pf):
    p = np.clip(np.asarray(pf, np.float64), 0.0, 1.0)
    raw = p[0] * _PLANT_SCALE.astype(np.float64) - _PLANT_SHIFT.astype(np.float64)
    return raw[:len(PLANT_FIELDS)]


# ── camera geometry and frame evidence (from camera_pose.json) ───────────────

def load_depth_full(path):
    """Decode depth.png at full resolution (big-endian packed RGBA uint32)."""
    a = np.asarray(Image.open(path)).astype(np.float64)
    return (a[..., 0] * 256.0 ** 3 + a[..., 1] * 256.0 ** 2
            + a[..., 2] * 256.0 + a[..., 3]) / float(256 ** 4 - 1)


def camera_and_frame(folder):
    """Measured camera elevation/azimuth and the PLY frame evidence for one view."""
    import open3d as o3d
    d = json.loads((folder / 'camera_pose.json').read_text())
    C = np.array(d['cameraToWorld'], np.float64).reshape(4, 4)
    W = np.array(d['worldToCamera'], np.float64).reshape(4, 4)
    K = d['intrinsics']
    centre = np.array(d['plantCenter'], np.float64)
    pos = np.array(d['position'], np.float64)

    forward = -C[:3, 2]                         # camera looks down its -z
    to_centre = (centre - pos) / np.linalg.norm(centre - pos)
    centre_cam = (W @ np.r_[centre, 1.0])[:3]
    up_cam = W[:3, :3] @ np.array([0.0, 1.0, 0.0])   # world +Y in camera coords

    pts = np.asarray(o3d.io.read_point_cloud(str(folder / PC_FILENAME)).points)
    zf = -pts[:, 2]
    dep = load_depth_full(folder / 'depth.png')
    target = (zf - d['near']) / (d['far'] - d['near'])
    reproj = {}
    for sx in (1, -1):
        for sy in (1, -1):
            u = np.floor(K['cx'] + sx * K['fx'] * pts[:, 0] / zf).astype(int)
            v = np.floor(K['cy'] - sy * K['fy'] * pts[:, 1] / zf).astype(int)
            ok = (u >= 0) & (u < dep.shape[1]) & (v >= 0) & (v < dep.shape[0])
            got = np.full(len(pts), np.nan)
            got[ok] = dep[v[ok], u[ok]]
            reproj[f'{sx:+d},{sy:+d}'] = {
                'match_1e-3': float(np.mean(np.abs(got - target) < 1e-3)),
                'on_foreground': float(np.mean(got > 0))}

    return {
        'sin_elev': float(forward[1]),
        'elev_deg': float(np.degrees(np.arcsin(np.clip(forward[1], -1, 1)))),
        'azim_deg': float(np.degrees(np.arctan2(pos[0] - centre[0], pos[2] - centre[2]))),
        'cam_dist': float(np.linalg.norm(pos - centre)),
        'viewDirection_y': float(d['viewDirection'][1]),
        'cos_forward_to_centre': float(forward @ to_centre),
        'centre_cam': centre_cam.tolist(),
        'up_cam': up_cam.tolist(),
        'camx_world_y': float(C[1, 0]),
        'det_R': float(np.linalg.det(C[:3, :3])),
        'ply_z_range': [float(pts[:, 2].min()), float(pts[:, 2].max())],
        'near_far': [float(d['near']), float(d['far'])],
        'reproj': reproj,
    }


def decide_frame(geo):
    """Diagonal sign flip taking the PLY/loader frame to x right, y up, z back."""
    sy_up = 1 if all(g['up_cam'][1] > 0 for g in geo) else (
        -1 if all(g['up_cam'][1] < 0 for g in geo) else 0)
    assert sy_up != 0, 'world-up maps to inconsistent camera-y signs across views'
    assert all(abs(g['camx_world_y']) < 1e-9 for g in geo), \
        'camera x axis is not horizontal -- not a Y-up lookAt'
    fwd_negz = all(g['centre_cam'][2] < 0 and g['ply_z_range'][1] < 0 for g in geo)
    fwd_posz = all(g['centre_cam'][2] > 0 and g['ply_z_range'][0] > 0 for g in geo)
    assert fwd_negz or fwd_posz, 'forward axis inconsistent across views'
    if fwd_posz:
        raise NotImplementedError('forward = +z: reprojection test below assumes -z')
    # Pick the reprojection sign pair with the best depth-value agreement,
    # summed over views, and require it to agree with the world-up result.
    keys = geo[0]['reproj'].keys()
    score = {k: np.mean([g['reproj'][k]['match_1e-3'] for g in geo]) for k in keys}
    best = max(score, key=score.get)
    sx, sy = (int(s) for s in best.split(','))
    assert sy == sy_up, f'depth reprojection ({best}) disagrees with world-up ({sy_up})'
    assert score[best] > 0.9, f'best reprojection match only {score[best]:.3f}'
    D = np.diag([float(sx), float(sy), 1.0])      # z: forward is -z already
    return D, score, best


def ply_norm(folder):
    """The raw PLY (camera frame, world units) and the loader's centroid / scale."""
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(str(folder / PC_FILENAME)).points, np.float64)
    c = pts.mean(0)
    return pts, c, float(np.linalg.norm(pts - c, axis=1).max())


def project(X, K):
    """Camera-frame points (forward = -z) -> continuous pixel coords, depth."""
    zf = -X[:, 2]
    assert (zf > 0).all(), 'a point is behind the camera'
    return np.stack([K['cx'] + K['fx'] * X[:, 0] / zf,
                     K['cy'] - K['fy'] * X[:, 1] / zf], 1), zf


def nn2d(a, b, chunk=2048):
    """For each row of a: (distance, index) of the nearest row of b (2-D)."""
    at, bt = torch.from_numpy(a), torch.from_numpy(b)
    d, j = [], []
    for i in range(0, len(at), chunk):
        m = torch.cdist(at[i:i + chunk], bt).min(dim=1)
        d.append(m.values)
        j.append(m.indices)
    return torch.cat(d).numpy(), torch.cat(j).numpy()


def miss_anatomy(gt_n, pred_n, nn_gt, c, s, K, miss_thr, px_thr, y_low):
    """Where the GT points no generated point comes near actually sit."""
    G, P = gt_n * s + c, pred_n * s + c          # camera frame, world units
    uvG, zG = project(G, K)
    uvP, zP = project(P, K)
    px_all, j_all = nn2d(uvG, uvP)
    dz_all = zG - zP[j_all]                      # + = generated point nearer camera
    miss = nn_gt > miss_thr
    out = {'miss_thr': miss_thr, 'px_thr': px_thr, 'y_low': y_low,
           'frac_miss': float(miss.mean()), 'n_miss': int(miss.sum()),
           'all_frac_px_within': float((px_all <= px_thr).mean()),
           'all_median_px': float(np.median(px_all)),
           'all_median_abs_dz_world': float(np.median(np.abs(dz_all)))}
    if miss.any():
        pm, dzm = px_all[miss], dz_all[miss]
        out.update({
            'miss_frac_y_below': float((gt_n[miss, 1] < y_low).mean()),
            'miss_frac_px_within': float((pm <= px_thr).mean()),
            'miss_median_px': float(np.median(pm)),
            'miss_median_abs_dz_world': float(np.median(np.abs(dzm))),
            'miss_median_abs_dz_norm': float(np.median(np.abs(dzm)) / s),
            'miss_median_dz_world_signed': float(np.median(dzm)),
            'miss_frac_pred_nearer': float((dzm > 0).mean()),
            'miss_median_nn3d_world': float(np.median(nn_gt[miss]) * s),
        })
    ok = ~miss
    out['hit_median_abs_dz_world'] = float(np.median(np.abs(dz_all[ok]))) if ok.any() else None
    out['scale_world_per_norm'] = s
    return out


# ── the RGB-only forward ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_rgb_only(model, rgb, depth, pc, params, seed):
    """RGB fully visible; depth, PC and text fully masked (0 encoder tokens)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = model.forward_encoder_select(rgb, depth, pc, params, visible={'rgb'},
                                       source_mask_ratio=0.0)
    (latent, mr, md, mp, mt, rr, rd, rp, rt, lr_, ld_, lp_, lt_) = out
    assert (ld_, lp_, lt_) == (0, 0, 0), (ld_, lp_, lt_)
    ps = model.patch_size
    assert lr_ == (rgb.shape[-2] // ps) * (rgb.shape[-1] // ps), lr_
    for m in (md, mp, mt):
        assert bool((m == 1).all()), 'a non-source modality is not fully masked'
    assert bool((mr == 0).all())
    _, _, pred_pc, pred_params = model.forward_decoder(
        latent, rr, rd, rp, rt, lr_, ld_, lp_, lt_)
    return pred_pc, pred_params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='maize_4m')
    ap.add_argument('--ckpt', default='checkpoints/checkpoint_epoch_600.pth')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split', default='val')
    ap.add_argument('--plant', default='plant_11052')
    ap.add_argument('--n-views', type=int, default=10)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--img-px', type=int, default=448, help='output JPEG width')
    ap.add_argument('--jpeg-q', type=int, default=85)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--nn-thresh', type=float, default=0.01)
    ap.add_argument('--miss-thr', type=float, default=0.05,
                    help='GT point counts as a miss beyond this normalised distance')
    ap.add_argument('--px-thr', type=float, default=5.0,
                    help='image-plane radius (px of 1024) for "drawn but misplaced"')
    ap.add_argument('--y-low', type=float, default=-0.4,
                    help='normalised camera-y below which a miss is "bottom of plant"')
    args = ap.parse_args()

    if 'best_model' in args.ckpt:
        raise SystemExit('best_model.pth is selected on total val loss at a '
                         'run-dependent epoch; pass checkpoints/checkpoint_epoch_N.pth')
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.data_root) / args.split
    views = [f'{args.plant}_{v:02d}' for v in range(args.n_views)]
    t0 = time.time()

    # ── camera geometry + frame, straight from the files ────────────────────
    geo = [camera_and_frame(split_dir / n) for n in views]
    D, score, best = decide_frame(geo)
    print(f'[{time.time()-t0:.0f}s] camera geometry for {len(views)} views')
    print(f"{'view':>5}{'sin elev':>10}{'elev°':>8}{'azim°':>8}{'dist':>8}"
          f"{'cos(fwd,ctr)':>13}{'ctr_cam z':>10}{'up_cam y':>9}{'z range':>18}")
    for n, g in zip(views, geo):
        print(f"{n[-2:]:>5}{g['sin_elev']:>+10.4f}{g['elev_deg']:>+8.2f}"
              f"{g['azim_deg']:>+8.1f}{g['cam_dist']:>8.4f}"
              f"{g['cos_forward_to_centre']:>13.6f}{g['centre_cam'][2]:>10.4f}"
              f"{g['up_cam'][1]:>9.4f}  [{g['ply_z_range'][0]:.3f},{g['ply_z_range'][1]:.3f}]")
    idx = np.arange(len(views), dtype=np.float64)
    se = np.array([g['sin_elev'] for g in geo])
    slope, icpt = np.polyfit(idx, se, 1)
    resid = se - (slope * idx + icpt)
    r = float(np.corrcoef(idx, se)[0, 1])
    ladder = {'slope': float(slope), 'intercept': float(icpt),
              'max_abs_resid': float(np.abs(resid).max()), 'pearson_r_index': r}
    print(f'ladder test: sin elev ~ {slope:+.4f} * index {icpt:+.4f}, '
          f'max |resid| {np.abs(resid).max():.4f}, r(index, sin elev) = {r:+.3f}')
    print('depth reprojection, fraction of PLY points within 1e-3 of depth.png '
          '(mean over views), by (sx, sy):')
    for k, s in score.items():
        print(f'   {k}: {s:.4f}' + ('   <- chosen' if k == best else ''))
    print(f'frame conversion loader -> (x right, y up, z toward viewer): '
          f'diag({", ".join(f"{x:+.0f}" for x in np.diag(D))})  det {np.linalg.det(D):+.0f}'
          + ('  (identity)' if np.allclose(D, np.eye(3)) else '')
          + ('  ** MIRROR **' if np.linalg.det(D) < 0 else ''))

    # ── model + data ─────────────────────────────────────────────────────────
    run_dir = REPO / 'outputs' / args.run
    model, cfg, epoch = build_model(run_dir, args.ckpt, 'cpu')
    assert 'rgb' in model.active_modalities and 'text' in model.active_modalities
    print(f'[{time.time()-t0:.0f}s] {args.run} {args.ckpt} (epoch field {epoch}), '
          f'{cfg["model_size"]}, active={model.active_modalities}')
    ds = MaizeDataset4M(args.data_root, img_size=cfg.get('img_size', 224),
                        num_points=cfg.get('num_points', 8192), split=args.split,
                        max_leaves=model.max_leaves)
    where = {p.name: i for i, p in enumerate(ds.samples)}
    missing = [n for n in views if n not in where]
    assert not missing, f'not in the {args.split} index: {missing}'

    Dt = torch.tensor(D, dtype=torch.float32)
    rows, js_views = [], []
    pred_leaf, pred_plant = [], []
    gt_leaf = gt_plant = n_leaves = None
    leak_pc = leak_par = 0.0
    for vi, name in enumerate(views):
        seed = args.seed + vi
        np.random.seed(seed)
        torch.manual_seed(seed)
        rgb, depth, pc, params, valid, nm = ds[where[name]]
        assert nm == name
        b = lambda t: t.unsqueeze(0)
        zeros = torch.zeros_like(params)
        assert float(zeros.abs().max()) == 0.0
        pred_pc, pred_par = predict_rgb_only(model, b(rgb), b(depth), b(pc), b(zeros), seed)

        # Leak check: real params, noise depth and noise PC -> must be identical.
        g = torch.Generator().manual_seed(seed + 7)
        pc_noise = torch.randn(pc.shape, generator=g)
        pc_noise = pc_noise / pc_noise.norm(dim=1).max()
        d_noise = torch.rand(depth.shape, generator=g)
        pc2, par2 = predict_rgb_only(model, b(rgb), b(d_noise), b(pc_noise), b(params), seed)
        leak_pc = max(leak_pc, float((pc2 - pred_pc).abs().max()))
        leak_par = max(leak_par, float((par2 - pred_par).abs().max()))

        gt = b(pc)
        cd = float(chamfer_distance(pred_pc, gt))
        dmat = torch.cdist(gt, pred_pc)[0]                 # (M_gt, N_pred), Euclidean
        nn_gt = dmat.min(dim=1).values                     # per GT point
        nn_pr = dmat.min(dim=0).values                     # per pred point
        cd_check = float((nn_pr ** 2).mean() + (nn_gt ** 2).mean())
        assert abs(cd_check - cd) <= 1e-6 + 1e-3 * cd, (cd, cd_check)
        frac = float((nn_gt > args.nn_thresh).float().mean())
        # What the page shows: its slider compares the ROUNDED shipped nn, so
        # the fractions are taken from exactly the list written to the JS.
        nn_ship = [round(float(x), 4) for x in nn_gt.numpy()]
        nn_r = np.array(nn_ship)
        frac_page = float((nn_r > args.nn_thresh).mean())
        frac_page_02 = float((nn_r > 0.02).mean())

        # Miss anatomy in the photograph's own image plane (loader frame = PLY
        # frame; checked here by recovering the PLY from the loader cloud).
        pts, c_ply, s_ply = ply_norm(split_dir / name)
        back = pc.numpy().astype(np.float64) * s_ply + c_ply
        perm_err = float(nn2d(back, pts)[0].max())
        assert perm_err < 1e-5, f'loader cloud is not a permutation of the PLY ({perm_err})'
        K = json.loads((split_dir / name / 'camera_pose.json').read_text())['intrinsics']
        anat = miss_anatomy(pc.numpy().astype(np.float64),
                            pred_pc[0].numpy().astype(np.float64),
                            nn_gt.numpy().astype(np.float64), c_ply, s_ply, K,
                            args.miss_thr, args.px_thr, args.y_low)
        anat['ply_permutation_max_err'] = perm_err
        rows.append({'view': name[-2:], 'name': name, **{k: geo[vi][k] for k in
                     ('sin_elev', 'elev_deg', 'azim_deg')},
                     'chamfer': cd, 'acc_pred_to_gt': float((nn_pr ** 2).mean()),
                     'comp_gt_to_pred': float((nn_gt ** 2).mean()),
                     'frac_gt_nn_gt_thresh': frac,
                     'frac_gt_nn_gt_thresh_page': frac_page,
                     'frac_gt_nn_gt_0p02_page': frac_page_02,
                     'nn_gt_median': float(nn_gt.median()),
                     'miss_anatomy': anat,
                     'seed': seed})

        # image: the file the loader opens
        im = Image.open(split_dir / name / 'rgb.png').convert('RGB')
        h = round(im.height * args.img_px / im.width)
        im = im.resize((args.img_px, h), Image.LANCZOS)
        img_name = f'maize_view_{name[-2:]}.jpg'
        im.save(out / img_name, quality=args.jpeg_q, optimize=True)

        P = (pred_pc[0] @ Dt.T).numpy()
        G = (pc @ Dt.T).numpy()
        rnd = lambda a: [round(float(x), 4) for x in np.asarray(a).ravel()]
        js_views.append({'view': name[-2:], 'img': img_name,
                         'elev': round(geo[vi]['sin_elev'], 6),
                         'elev_deg': round(geo[vi]['elev_deg'], 3),
                         'azim_deg': round(geo[vi]['azim_deg'], 3),
                         'chamfer': round(cd, 6),
                         'frac_nn_gt_0p01': round(frac_page, 6),
                         'frac_nn_gt_0p02': round(frac_page_02, 6),
                         'pred': rnd(P), 'gt': rnd(G), 'nn': nn_ship})

        # params
        vmask = valid.numpy().astype(bool)
        n_real = int(vmask[1:].sum())
        gl = decode_leaves(params.numpy())[:n_real]
        if gt_leaf is None:
            gt_leaf, gt_plant, n_leaves = gl, decode_plant(params.numpy()), n_real
        else:     # GT is a property of the plant: identical in every view folder
            assert n_real == n_leaves and np.array_equal(gl, gt_leaf)
        pred_leaf.append(decode_leaves(pred_par[0].numpy())[:n_real])
        pred_plant.append(decode_plant(pred_par[0].numpy()))
        print(f'[{time.time()-t0:.0f}s] {name}  sin elev {geo[vi]["sin_elev"]:+.4f}  '
              f'chamfer {cd:.6f}  frac nn>{args.nn_thresh} {frac:.4f} '
              f'(page, rounded nn: {frac_page:.4f}; >0.02 {frac_page_02:.4f})  '
              f'leafCount pred {pred_plant[-1][0]:.2f} (gt {gt_plant[0]:.0f})')
        a = anat
        if a['n_miss']:
            print(f"      misses (nn>{args.miss_thr}) {a['frac_miss']:.2%}: "
                  f"{a['miss_frac_y_below']:.1%} at y<{args.y_low}; "
                  f"{a['miss_frac_px_within']:.1%} have a generated point within "
                  f"{args.px_thr:g} px (median {a['miss_median_px']:.2f} px); "
                  f"line-of-sight |dz| median {a['miss_median_abs_dz_world']:.4f} world "
                  f"= {a['miss_median_abs_dz_norm']:.4f} norm "
                  f"(signed {a['miss_median_dz_world_signed']:+.4f}, "
                  f"{a['miss_frac_pred_nearer']:.0%} pred nearer camera)")
        print(f"      all GT: {a['all_frac_px_within']:.1%} within {args.px_thr:g} px "
              f"(median {a['all_median_px']:.2f} px), |dz| median "
              f"{a['all_median_abs_dz_world']:.4f} world; perm err {perm_err:.1e}")

    print(f'leak check over {len(views)} views: max|diff| pred cloud {leak_pc}, '
          f'pred params {leak_par}  (real params + noise depth/PC vs zeros + real)')
    assert leak_pc == 0.0 and leak_par == 0.0, 'a masked input reached the output'

    ch = np.array([r['chamfer'] for r in rows])
    ib, iw = int(ch.argmin()), int(ch.argmax())
    summ = {'mean_chamfer': float(ch.mean()), 'median_chamfer': float(np.median(ch)),
            'best': [rows[ib]['view'], float(ch[ib])],
            'worst': [rows[iw]['view'], float(ch[iw])],
            'worst_over_best': float(ch[iw] / ch[ib]),
            'pearson_sin_elev_chamfer': float(np.corrcoef(se, ch)[0, 1]),
            'mean_frac_nn_gt_0p01': float(np.mean([r['frac_gt_nn_gt_thresh'] for r in rows])),
            'mean_frac_nn_gt_0p01_page': float(np.mean([r['frac_gt_nn_gt_thresh_page'] for r in rows])),
            'mean_frac_nn_gt_0p02_page': float(np.mean([r['frac_gt_nn_gt_0p02_page'] for r in rows]))}
    print(f"\nchamfer mean {summ['mean_chamfer']:.6f}  median {summ['median_chamfer']:.6f}  "
          f"best {summ['best'][0]} {summ['best'][1]:.6f}  worst {summ['worst'][0]} "
          f"{summ['worst'][1]:.6f}  worst/best {summ['worst_over_best']:.3f}  "
          f"r(sin elev, chamfer) {summ['pearson_sin_elev_chamfer']:+.3f}")

    # ── parameter view-invariance table ─────────────────────────────────────
    PL = np.stack(pred_leaf)                     # (views, leaves, 13)
    table = []
    for j, f in enumerate(LEAF_OUT):
        p, g = PL[:, :, j], gt_leaf[:, j]
        if f == 'waveLPhase':
            err = np.abs(wrap_pi(p - g[None]))
            sd_views = circ_std(p, axis=0)           # per leaf
            sd_leaves = float(circ_std(g, axis=0))
            base = np.abs(wrap_pi(g - np.angle(np.exp(1j * g).mean())))
        else:
            err = np.abs(p - g[None])
            sd_views = p.std(0)
            sd_leaves = float(g.std())
            base = np.abs(g - g.mean())
        table.append({'field': f, 'unit': LEAF_UNITS.get(f, 'gen. units'),
                      'mae': float(err.mean()),
                      'mae_own_leaf_mean': float(base.mean()),
                      'sd_across_views_mean': float(sd_views.mean()),
                      'sd_across_leaves_gt': sd_leaves,
                      'view_spread': float(sd_views.mean() / sd_leaves) if sd_leaves > 0 else None})
    print(f'\nleaf fields, {n_leaves} real leaves x {len(views)} views (RGB alone):')
    print(f"{'field':>20}{'unit':>11}{'MAE':>11}{'MAE(own mean)':>15}"
          f"{'sd views':>11}{'sd leaves':>11}{'spread':>9}")
    for t in table:
        vs = 'n/a' if t['view_spread'] is None else f"{t['view_spread']:.3f}"
        print(f"{t['field']:>20}{t['unit']:>11}{t['mae']:>11.4g}{t['mae_own_leaf_mean']:>15.4g}"
              f"{t['sd_across_views_mean']:>11.4g}{t['sd_across_leaves_gt']:>11.4g}{vs:>9}")
    PP = np.stack(pred_plant)
    plant_rows = [{'field': f, 'gt': float(gt_plant[k]), 'pred_mean': float(PP[:, k].mean()),
                   'pred_sd_views': float(PP[:, k].std()),
                   'mae': float(np.abs(PP[:, k] - gt_plant[k]).mean()),
                   'per_view': PP[:, k].tolist()} for k, f in enumerate(PLANT_FIELDS)]
    print('\nplant token (RGB alone):')
    for t in plant_rows:
        print(f"{t['field']:>28}  gt {t['gt']:.5g}  pred {t['pred_mean']:.5g} "
              f"± {t['pred_sd_views']:.3g} (sd over views)  MAE {t['mae']:.4g}")

    # ── write ───────────────────────────────────────────────────────────────
    blob = {'plant': args.plant, 'source': 'rgb', 'run': args.run, 'ckpt': args.ckpt,
            'epoch': epoch, 'split': args.split,
            'frame': 'x right, y up, z toward the viewer (OpenGL camera; forward = -z)',
            'elev_order': [v['view'] for v in sorted(js_views, key=lambda v: v['elev'])],
            'views': js_views}
    js = out / 'views_maize.js'
    js.write_text('window.VIEW_DATA = ' + json.dumps(blob, separators=(',', ':')) + ';')
    rep = {'plant': args.plant, 'run': args.run, 'ckpt': args.ckpt, 'epoch': epoch,
           'split': args.split, 'seed_scheme': f'{args.seed} + view index',
           'camera': [{'view': n[-2:], **g} for n, g in zip(views, geo)],
           'ladder_fit': ladder, 'reproj_score_mean': score, 'reproj_chosen': best,
           'frame_conversion_diag': np.diag(D).tolist(),
           'rows': rows, 'summary': summ,
           'leak_check_max_abs_diff': {'pred_pc': leak_pc, 'pred_params': leak_par},
           'n_real_leaves': n_leaves, 'leaf_table': table, 'plant_table': plant_rows}
    (out / 'views_maize_report.json').write_text(json.dumps(rep, indent=1))
    print(f'\nwrote {js} ({js.stat().st_size / 1e6:.2f} MB), '
          f'{len(views)} JPEGs, views_maize_report.json  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
