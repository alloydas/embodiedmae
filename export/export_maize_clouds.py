"""Export the MAIZE "which points the model cannot predict" viewer data (E3 data scaling).

Produces, under --out-dir:

  clouds_maize.js         window.CLOUD_DATA = {samples: [{name, clouds: [
                              {label: '1,000 plants',  xyz: [...], nn: [...]},
                              {label: '10,500 plants', xyz: [...], nn: [...]}]}, ...]};
                          -- the schema the results page's coverage viewer already
                          consumes for sorghum, unchanged.
  clouds_maize_meta.json  everything the .js deliberately does not carry: per-sample
                          and averaged Chamfer / mean NN / fraction missed, the seeds,
                          the per-sample up-rotation, the visible-token draw and the
                          proof that both arms saw the identical draw, and a larger
                          random val subset as a sanity check against val_pc_chamfer.

WHAT IS COMPARED.  The 1,000-plant arm (`maize_e3_1k`, checkpoint_epoch_6168.pth,
its final cut) against the 10,500-plant reference (`maize_4m`,
checkpoint_epoch_600.pth).  Both are base-size models on the same 197,400-step
budget.  Never best_model.pth: it is selected on total val loss and lands at
different fractions of the schedule in different arms.

WHAT THE PREDICTION IS.  The ordinary masked-autoencoder forward pass under the
run's own training-time masking -- `model(rgb, depth, pc, params, text_valid,
mask_ratio=cfg['mask_ratio'])`, exactly the call `train_maize_4m.evaluate` makes
for val_pc_chamfer: all four streams in one Dirichlet(alpha=1.0) budget at
mask_ratio 0.80, min_mask_ratio 0.25, real params (as validation feeds them).
This is NOT a single-source cross-modal generation, so the "zero the params"
rule for single-source probes does not apply; the meta file records per sample
how many text tokens were visible and whether the plant token was among them.

SAME DRAW FOR BOTH ARMS.  Inputs are loaded ONCE per plant (numpy seeded before
the read -- the loader permutes the 8192 PLY points with np.random.choice) and
the identical tensors go to both models.  torch and numpy are re-seeded with the
same per-plant seed immediately before each forward, so FPS's first centroid
(torch.randint), the Dirichlet split and the per-token shuffle are identical.
The script does not trust that: it hooks `pc_embed.fps` and
`random_masking_dirichlet` and asserts the FPS indices and every modality's
ids_keep are bit-identical between the two arms.

METRICS.  `chamfer` is the repo's `embodied_mae.chamfer_distance(pred, gt)` in
float32 on the (1, 8192, 3) clouds -- the function and precision val_pc_chamfer
uses (squared distances, both directions summed).  `nn[k]` is the plain
Euclidean (NOT squared) distance from GT point k to its nearest predicted point,
computed exactly in float64 in chunks (torch.cdist's matmul path is not used: it
loses precision at the 1e-3 scale that matters here).  `frac_missed` is the
fraction of GT points with nn > 0.01, the qal_threshold default the page's
slider starts at.

FRAME.  `pointcloud_cam.ply` is in an OpenGL-style camera frame (points at
z < 0, camera looks down -z); the loader only centres and scales it, so the
output is still camera-oriented.  camera_pose.json's cameraToWorld maps it to a
Y-up world (plant base on y = 0, verified per sample).  World up expressed in
camera coordinates is u = R_c2w^T e_y = R_c2w[1, :].  Each sample is rotated by
Rot = [x'; u; x' x u] with x' = e_x orthogonalised against u -- the camera frame
with its pitch taken out, so +y is plant-up and the photo's azimuth is kept.
u has zero x-component for every view (the renderer has no roll), so Rot is a
pure rotation about the camera x axis by the view's elevation.  The camera
elevation differs per plant even at view 00 (-26.5 to +76.2 degrees for the
default eight; some of those cameras sit BELOW the ground plane looking up), so
the rotation is per sample, and
the SAME rotation is applied to GT and both predictions.  Distances are computed
before rotating and re-checked after (a rotation cannot change them).

Run from the repo root (CPU is fine, ~1-2 s per forward at 12 threads):
    source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
    conda activate /work/mech-ai/alloy/.conda/envs/det
    python export/export_maize_clouds.py --out-dir <dir>
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, maize_dataset_4m, eval/*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embodied_mae import chamfer_distance
from maize_dataset_4m import MaizeDataset4M
from eval.linear_probe_maize import build_model

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = '/work/mech-ai-scratch/alloy/Maize'
PLANTS = ['plant_8968', 'plant_5483', 'plant_0633', 'plant_11052',
          'plant_1196', 'plant_13733', 'plant_7102', 'plant_6078']
# (run dir, checkpoint, label on the page) -- order == order of the two panes.
ARMS = [('maize_e3_1k', 'checkpoints/checkpoint_epoch_6168.pth', '1,000 plants'),
        ('maize_4m',    'checkpoints/checkpoint_epoch_600.pth',  '10,500 plants')]
MODS = ('rgb', 'depth', 'pc', 'text')


# ───────────────────────────── geometry ─────────────────────────────────────

def up_rotation(folder):
    """Rotation taking the loader's camera-frame cloud to a +y = plant-up frame."""
    cp = json.loads((Path(folder) / 'camera_pose.json').read_text())
    C = np.asarray(cp['cameraToWorld'], np.float64).reshape(4, 4)
    W = np.asarray(cp['worldToCamera'], np.float64).reshape(4, 4)
    assert np.abs(C @ W - np.eye(4)).max() < 1e-9, 'pose matrices disagree'
    R = C[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, 'cameraToWorld is not a rotation'
    u = R.T @ np.array([0.0, 1.0, 0.0])          # world up in camera coordinates
    ex = np.array([1.0, 0.0, 0.0])
    x = ex - (ex @ u) * u
    x /= np.linalg.norm(x)
    Rot = np.stack([x, u, np.cross(x, u)])        # rows; Rot @ u == e_y
    assert np.allclose(Rot @ Rot.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(Rot) - 1.0) < 1e-12
    assert np.allclose(Rot @ u, [0, 1, 0], atol=1e-12)
    elev = float(np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1))))   # <0 looks down
    return Rot, u, C, elev


def world_check(folder, pc_loader, u, C):
    """Tie the loader's cloud to the world frame: returns (world y range, corr).

    The loader output is (ply[perm] - mean) / s, so ply[perm] = pc*s + mean with
    mean and s recomputed from the raw PLY (both permutation-invariant).  Mapping
    that through cameraToWorld gives world y; it must equal (pc @ u)*s + const.
    """
    import open3d as o3d
    raw = np.asarray(o3d.io.read_point_cloud(str(Path(folder) / 'pointcloud_cam.ply')).points)
    mean = raw.mean(0)
    s = np.linalg.norm(raw - mean, axis=1).max()
    cam = pc_loader.astype(np.float64) * s + mean
    world = cam @ C[:3, :3].T + C[:3, 3]
    y_rot = pc_loader.astype(np.float64) @ u
    r = float(np.corrcoef(world[:, 1], y_rot)[0, 1])
    resid = float(np.abs(world[:, 1] - (y_rot * s + (mean @ u + C[1, 3]))).max())
    return float(world[:, 1].min()), float(world[:, 1].max()), r, resid


@torch.no_grad()
def nn_euclid(src, dst, chunk=1024):
    """Exact float64 Euclidean distance from each src point to its nearest dst point."""
    src = torch.as_tensor(src, dtype=torch.float64)
    dst = torch.as_tensor(dst, dtype=torch.float64)
    out = torch.empty(src.shape[0], dtype=torch.float64)
    for i in range(0, src.shape[0], chunk):
        d2 = ((src[i:i + chunk, None, :] - dst[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d2.min(1).values
    return out.sqrt().numpy()


@torch.no_grad()
def nn_self(pts, chunk=1024):
    """Distance from each point to its nearest OTHER point of the same cloud --
    the GT sampling floor: a threshold near it counts even a perfect surface as
    'missed' wherever the prediction samples it at different spots."""
    x = torch.as_tensor(pts, dtype=torch.float64)
    out = torch.empty(x.shape[0], dtype=torch.float64)
    for i in range(0, x.shape[0], chunk):
        d2 = ((x[i:i + chunk, None, :] - x[None, :, :]) ** 2).sum(-1)
        d2[torch.arange(d2.shape[0]), torch.arange(i, i + d2.shape[0])] = float('inf')
        out[i:i + chunk] = d2.min(1).values
    return out.sqrt().numpy()


SWEEP = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)


class _Seeded(torch.utils.data.Dataset):
    """ds[idx] with numpy seeded per item, so worker processes reproduce the
    same PLY permutation the single-process read would."""
    def __init__(self, ds, idxs, seeds):
        self.ds, self.idxs, self.seeds = ds, idxs, seeds

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, j):
        return load_item(self.ds, self.idxs[j], self.seeds[j])[:5]


# ───────────────────────────── model hooks ──────────────────────────────────

def attach_recorder(model):
    """Record FPS indices and the visible-token draw of the next forward pass."""
    rec = {}
    fps0 = model.pc_embed.fps
    mask0 = model.random_masking_dirichlet

    def fps(xyz, npoint):
        idx = fps0(xyz, npoint)
        rec['fps'] = idx.clone()
        return idx

    def masking(embeds, mask_ratio_total=0.75, min_mask_ratio=0.25):
        out = mask0(embeds, mask_ratio_total, min_mask_ratio)
        rec['mask_ratio_total'] = mask_ratio_total
        rec['min_mask_ratio'] = min_mask_ratio
        rec['keep'] = {n: torch.argsort(v[2], dim=1)[:, :v[0].shape[1]].clone()
                       for n, v in out.items()}
        rec['n_vis'] = {n: int(v[0].shape[1]) for n, v in out.items()}
        rec['L'] = {n: int(v[1].shape[1]) for n, v in out.items()}
        return out

    model.pc_embed.fps = fps
    model.random_masking_dirichlet = masking
    return rec


@torch.no_grad()
def forward_pc(model, cfg, batch, seed):
    """Val-path forward (train_maize_4m.evaluate); returns pred_pc and masks."""
    rgb, depth, pc, params, valid = batch
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    _, _, (_, _, pred_pc, _), (mr, md, mp, mt) = model(
        rgb, depth, pc, params, valid, mask_ratio=cfg['mask_ratio'])
    return pred_pc, mt


def load_item(ds, idx, seed):
    np.random.seed(seed % (2 ** 32))          # the loader permutes the PLY points
    rgb, depth, pc, params, valid, name = ds[idx]
    return rgb, depth, pc, params, valid, name


def r4(a):
    return np.round(np.asarray(a, dtype=np.float64), 4).ravel().tolist()


# ─────────────────────────────── main ───────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split', default='val')
    ap.add_argument('--plants', nargs='+', default=PLANTS)
    ap.add_argument('--view', default='00')
    ap.add_argument('--seed', type=int, default=0,
                    help='per-plant seed = seed*100003 + integer plant id')
    ap.add_argument('--threshold', type=float, default=0.01,
                    help='nn above this counts as missed (qal_threshold default)')
    ap.add_argument('--sanity-n', type=int, default=256,
                    help='extra random val plants (view 00, batches of 16 like '
                         'evaluate) to compare the mean Chamfer with val_pc_chamfer; '
                         '0 skips it')
    ap.add_argument('--sanity-batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8, help='loader workers, sanity set only')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = MaizeDataset4M(args.data_root, split=args.split, num_points=8192,
                        view_sampling=False)
    by_name = {p.name: i for i, p in enumerate(ds.samples)}
    scores = pd.read_csv(Path(args.data_root) / 'plant_scores.csv').set_index('plant')

    # ---- showcase inputs: loaded once, shared by both arms -------------------
    items = []
    for plant in args.plants:
        name = f'{plant}_{args.view}'
        assert name in by_name, f'{name} not in the {args.split} split'
        seed = args.seed * 100003 + MaizeDataset4M.plant_id(name)
        rgb, depth, pc, params, valid, nm = load_item(ds, by_name[name], seed)
        assert nm == name and pc.shape == (8192, 3)
        folder = ds.samples[by_name[name]]
        Rot, u, C, elev = up_rotation(folder)
        assert abs(u[0]) < 1e-9, f'{name}: camera has roll, Rot is not a pure pitch'
        ymin, ymax, r, resid = world_check(folder, pc.numpy(), u, C)
        assert r > 0.999999 and resid < 1e-6, (name, r, resid)
        items.append(dict(name=name, plant=plant, seed=seed, folder=folder,
                          batch=tuple(t.unsqueeze(0) for t in (rgb, depth, pc, params, valid)),
                          gt=pc.numpy(), Rot=Rot, u=u, elev=elev, C=C,
                          world_y=(ymin, ymax), world_corr=r, world_resid=resid,
                          leaf_count=int(scores.loc[plant, 'leaf_count']),
                          n_leaf_tokens=int(valid.sum().item()) - 1,
                          self_nn=nn_self(pc.numpy()), arms={}))
    print(f'loaded {len(items)} showcase plants ({time.time()-t0:.0f}s)', flush=True)

    # sanity subset: random val plants at view 00, fixed draw
    sanity = []
    if args.sanity_n:
        plants_all = sorted({MaizeDataset4M.plant_of(p.name) for p in ds.samples})
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(len(plants_all), args.sanity_n, replace=False).tolist())
        names = [f'{plants_all[j]}_{args.view}' for j in pick]
        seeds = [args.seed * 100003 + MaizeDataset4M.plant_id(n) for n in names]
        dl = torch.utils.data.DataLoader(
            _Seeded(ds, [by_name[n] for n in names], seeds), batch_size=None,
            shuffle=False, num_workers=args.workers, persistent_workers=False)
        sanity = list(dl)
        print(f'loaded {len(sanity)} sanity plants ({time.time()-t0:.0f}s)', flush=True)

    # ---- run each arm once ----------------------------------------------------
    arm_meta, sanity_cd = [], {}
    for run, ckpt, label in ARMS:
        model, cfg, epoch = build_model(REPO / 'outputs' / run, ckpt, 'cpu')
        assert cfg.get('text_mask_ratio') is None and 'dirichlet_alpha' not in cfg
        assert model.dirichlet_alpha == 1.0 and model.training is False
        assert tuple(model.active_modalities) == MODS, model.active_modalities
        rec = attach_recorder(model)
        arm_meta.append(dict(run=run, ckpt=ckpt, label=label, epoch=epoch,
                             model_size=cfg['model_size'], mask_ratio=cfg['mask_ratio'],
                             max_plants=cfg.get('max_plants'),
                             dirichlet_alpha=model.dirichlet_alpha))
        print(f'\n=== {run} ({label})  {cfg["model_size"]}  {ckpt} @ epoch {epoch}  '
              f'mask_ratio {cfg["mask_ratio"]}  ({time.time()-t0:.0f}s)', flush=True)

        for it in items:
            pred, mt = forward_pc(model, cfg, it['batch'], it['seed'])
            gt_t = it['batch'][2]
            cd = float(chamfer_distance(pred, gt_t).item())
            pred_np = pred[0].numpy()
            nn = nn_euclid(it['gt'], pred_np)                 # GT -> pred
            nn_rev = nn_euclid(pred_np, it['gt'])             # pred -> GT
            cd64 = float((nn ** 2).mean() + (nn_rev ** 2).mean())
            # distances survive the rotation (checked, not assumed)
            nn_rot = nn_euclid(it['gt'] @ it['Rot'].T, pred_np @ it['Rot'].T)
            assert np.abs(nn_rot - nn).max() < 1e-9
            nn4 = np.round(nn, 4)
            it['arms'][run] = dict(
                pred_rot=pred_np @ it['Rot'].T, nn=nn,
                chamfer=cd, chamfer_f64=cd64,
                gt_to_pred=float((nn ** 2).mean()), pred_to_gt=float((nn_rev ** 2).mean()),
                mean_nn=float(nn.mean()), median_nn=float(np.median(nn)),
                p95_nn=float(np.percentile(nn, 95)), max_nn=float(nn.max()),
                frac_missed=float((nn > args.threshold).mean()),
                frac_missed_rounded=float((nn4 > args.threshold).mean()),
                frac_missed_sweep={str(t): float((nn > t).mean()) for t in SWEEP},
                fps=rec['fps'].clone(), keep={k: v.clone() for k, v in rec['keep'].items()},
                n_vis=dict(rec['n_vis']), L=dict(rec['L']),
                mask_ratio_total=rec['mask_ratio_total'],
                min_mask_ratio=rec['min_mask_ratio'],
                plant_token_visible=bool(mt[0, 0].item() == 0),
                leaf_tokens_visible=int(((mt[0, 1:] == 0) & (it['batch'][4][0, 1:] > 0)).sum()))
            a = it['arms'][run]
            print(f'  {it["name"]:<15} chamfer {cd:.6f}  meanNN {a["mean_nn"]:.5f}  '
                  f'missed@{args.threshold} {100*a["frac_missed"]:.2f}%  '
                  f'vis {a["n_vis"]}', flush=True)

        if sanity:
            cds = []
            for b0 in range(0, len(sanity), args.sanity_batch):
                chunk = sanity[b0:b0 + args.sanity_batch]
                batch = tuple(torch.stack([c[k] for c in chunk]) for k in range(5))
                pred, _ = forward_pc(model, cfg, batch, 7_000_003 + b0)
                for i in range(len(chunk)):
                    cds.append(float(chamfer_distance(pred[i:i + 1], batch[2][i:i + 1]).item()))
            sanity_cd[run] = np.asarray(cds)
            print(f'  sanity: {len(cds)} random val plants  mean chamfer '
                  f'{np.mean(cds):.6f}  (sem {np.std(cds, ddof=1)/np.sqrt(len(cds)):.6f})  '
                  f'({time.time()-t0:.0f}s)', flush=True)
        del model

    # ---- identical-draw proof -------------------------------------------------
    r0, r1 = ARMS[0][0], ARMS[1][0]
    for it in items:
        a, b = it['arms'][r0], it['arms'][r1]
        assert torch.equal(a['fps'], b['fps']), f'{it["name"]}: FPS differs'
        for m in MODS:
            assert torch.equal(a['keep'][m], b['keep'][m]), f'{it["name"]}: {m} ids_keep differ'
        assert a['n_vis'] == b['n_vis'] and a['plant_token_visible'] == b['plant_token_visible']
    print('\nFPS indices and ids_keep identical between arms for every sample, every modality')

    # ---- CLOUD_DATA -------------------------------------------------------------
    samples = []
    for it in items:
        gt_rot = it['gt'].astype(np.float64) @ it['Rot'].T
        xyz = r4(gt_rot)
        samples.append({'name': it['name'], 'clouds': [
            {'label': label, 'xyz': xyz, 'nn': r4(it['arms'][run]['nn'])}
            for run, _, label in ARMS]})
    js = out / 'clouds_maize.js'
    js.write_text('window.CLOUD_DATA = ' + json.dumps({'samples': samples},
                                                      separators=(',', ':')) + ';')

    # ---- meta -----------------------------------------------------------------
    def arm_stats(run):
        keys = ('chamfer', 'mean_nn', 'frac_missed', 'gt_to_pred', 'pred_to_gt')
        d = {k: float(np.mean([it['arms'][run][k] for it in items])) for k in keys}
        d['frac_missed_sweep'] = {str(t): float(np.mean(
            [it['arms'][run]['frac_missed_sweep'][str(t)] for it in items])) for t in SWEEP}
        return d

    hist = {}
    for run, _, _ in ARMS:
        h = json.loads((REPO / 'outputs' / run / 'training_history.json').read_text())
        c = json.loads((REPO / 'outputs' / run / 'config.json').read_text())
        v = h['val_pc_chamfer']
        # val runs at epoch 1, every val_freq, and at the final epoch
        ep = [1] + [c['val_freq'] * k for k in range(1, len(v))]
        ep[-1] = min(ep[-1], c['epochs'])
        hist[run] = dict(zip(ep, v))

    meta = dict(
        page_schema='window.CLOUD_DATA = {samples:[{name, clouds:[{label, xyz, nn}]}]}',
        split=args.split, view=args.view, threshold=args.threshold,
        seed_rule='per plant: torch.manual_seed & np.random.seed(seed*100003 + plant_id) '
                  'before the data read and again before every forward',
        arms=arm_meta,
        camera_elevation_note=('degrees of the camera look direction above horizontal: '
                               '<0 looks down on the plant, >0 looks up at it from '
                               'below (camera_world_y < 0 is under the ground plane)'),
        frame=('xyz = loader cloud (centred, unit-sphere) @ Rot.T, Rot = rows '
               '[x_cam orthogonalised against u; u; x x u], u = cameraToWorld[:3,:3]^T e_y. '
               'Pure pitch about camera x (u_x == 0 for every view). Same Rot for GT '
               'and both predictions; distances computed before rotation and re-checked after.'),
        per_sample=[dict(
            name=it['name'], leaf_count=it['leaf_count'], n_leaf_tokens=it['n_leaf_tokens'],
            seed=it['seed'], camera_elevation_deg=round(it['elev'], 3),
            camera_world_y=round(float(it['C'][1, 3]), 4),
            up_in_camera=[round(float(x), 6) for x in it['u']],
            rotation=[[round(float(x), 6) for x in row] for row in it['Rot']],
            world_y_range=[round(it['world_y'][0], 4), round(it['world_y'][1], 4)],
            world_y_corr=it['world_corr'], world_y_resid=it['world_resid'],
            gt_y_range_rotated=[round(float(v), 4) for v in
                                ((it['gt'] @ it['Rot'].T)[:, 1].min(),
                                 (it['gt'] @ it['Rot'].T)[:, 1].max())],
            gt_self_nn=dict(mean=float(it['self_nn'].mean()),
                            median=float(np.median(it['self_nn'])),
                            p95=float(np.percentile(it['self_nn'], 95)),
                            min=float(it['self_nn'].min())),
            visible_tokens=it['arms'][r0]['n_vis'], tokens=it['arms'][r0]['L'],
            plant_token_visible=it['arms'][r0]['plant_token_visible'],
            leaf_tokens_visible=it['arms'][r0]['leaf_tokens_visible'],
            masks_identical_across_arms=True,
            arms={run: {k: it['arms'][run][k] for k in
                        ('chamfer', 'chamfer_f64', 'gt_to_pred', 'pred_to_gt', 'mean_nn',
                         'median_nn', 'p95_nn', 'max_nn', 'frac_missed',
                         'frac_missed_rounded', 'frac_missed_sweep')}
                  for run, _, _ in ARMS})
            for it in items],
        averages={run: arm_stats(run) for run, _, _ in ARMS},
        gt_self_nn_mean_over_samples=float(np.mean([it['self_nn'].mean() for it in items])),
        gt_self_nn_median_over_samples=float(np.mean([np.median(it['self_nn']) for it in items])),
        val_history_pc_chamfer={run: {str(k): v for k, v in hist[run].items()}
                                for run in hist},
        sanity=({run: dict(n=int(len(v)), mean=float(v.mean()),
                           sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                           median=float(np.median(v)))
                 for run, v in sanity_cd.items()} if sanity_cd else None),
    )
    mj = out / 'clouds_maize_meta.json'
    mj.write_text(json.dumps(meta, indent=1))

    # ---- summary --------------------------------------------------------------
    print(f'\n{"sample":<15} {"leaves":>6} ' + ' '.join(
        f'{"cd " + lab:>16} {"meanNN":>8} {"miss%":>6}' for _, _, lab in ARMS))
    for it in items:
        print(f'{it["name"]:<15} {it["leaf_count"]:>6} ' + ' '.join(
            f'{it["arms"][r]["chamfer"]:>16.6f} {it["arms"][r]["mean_nn"]:>8.5f} '
            f'{100*it["arms"][r]["frac_missed"]:>6.2f}' for r, _, _ in ARMS))
    for run, _, lab in ARMS:
        s = meta['averages'][run]
        print(f'mean {lab:<14} chamfer {s["chamfer"]:.6f}  meanNN {s["mean_nn"]:.5f}  '
              f'missed {100*s["frac_missed"]:.2f}%')
    print(f'\nwrote {js} ({js.stat().st_size/1e6:.2f} MB)\nwrote {mj}  '
          f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
