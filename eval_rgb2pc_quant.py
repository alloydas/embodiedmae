"""Quantitative RGB -> point-cloud generation eval over many DISTINCT val plants.

figures/gen_rgb2pc_gallery.py makes the picture; this makes the numbers behind it. The
two differences that matter:

  * plants, not folders.  The split is Sorghum_<plant>_<view> and sorts so that
    ten consecutive folders are ten views of one plant.  Striding by a number
    coprime with 10 (as the gallery does) is enough for 8 rows but starts
    repeating plants at N in the hundreds, so this samples plant ids directly
    and takes one seeded view of each.

  * controls.  A Chamfer of 0.0008 means nothing on its own -- the clouds are
    unit-sphere normalised, so every sorghum already overlaps every other one.
    Two baselines bracket it:
      shuffled  : the model's own prediction scored against a DIFFERENT plant
      gt_vs_gt  : one plant's ground truth used as a prediction for another
    `gt_vs_gt` is the honest "predict the average sorghum" floor; a model that
    has learnt nothing plant-specific lands there, not at `shuffled`.

Both Chamfer directions are reported separately.  They fail differently:
  pred->gt  (accuracy)     rises when the model puts points where no plant is
  gt->pred  (completeness) rises when the model misses parts of the plant
A cloud that collapses to a dense blob has good accuracy and bad completeness,
and the symmetric sum hides that.

CPU by default -- the GPUs are training.  ~5 s/plant at 8196 points.
"""
import os
import argparse, json, pathlib, re, collections
import numpy as np
import torch

from embodied_mae_4m import embodied_mae_4m_base
from sorghum_dataset_4m import SorghumDataset4M
from maize_dataset_4m import MaizeDataset4M
from embodied_mae_4m_maize import embodied_mae_4m_maize_base
import species_conventions as SC
from sorghum_dataset import RGB_FILE

_NAME = re.compile(r'^(?P<stem>.+)_(?P<plant>\d+)_(?P<view>\d+)$')


def chamfer_both(pred, target):
    """(B,N,3),(B,M,3) -> (pred->target, target->pred) squared-distance means."""
    d = torch.cdist(pred, target) ** 2                 # (B,N,M)
    return d.min(dim=2).values.mean(dim=1), d.min(dim=1).values.mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--model_species', choices=('sorghum', 'maize'), default='sorghum',
                    help='which species the CHECKPOINT is (picks the model class)')
    ap.add_argument('--data_species', choices=('sorghum', 'maize'), default=None,
                    help='which species the INPUT data is; default = model_species. When they differ '
                         '(PLAN_cross_species.md Z2) depth is re-encoded into the model convention and '
                         'the recipe stream is zeroed at the model width.')
    ap.add_argument('--target_points', type=int, default=None,
                    help='model output size; default 8196 (sorghum model) / 8192 (maize model)')
    ap.add_argument('--label', default=None, help='name for this checkpoint in the report')
    ap.add_argument('--data_root', default='/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_plants', type=int, default=120)
    ap.add_argument('--num_points', type=int, default=8196)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--source', default='rgb', help="visible modality, e.g. rgb / depth / rgb,depth")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True, help='path to write the JSON report')
    ap.add_argument('--rgb_file', default=None, help="RGB file name; default: the run's config.json (rgb_file), else the dataset default")
    ap.add_argument('--rgb_gain', type=float, default=None, help="serve RGB at this brighten gain; default: the run's config.json (rgb_brighten_gain), else the files as on disk")
    args = ap.parse_args()

    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    data_species = args.data_species or args.model_species
    cross = data_species != args.model_species
    _SORGHUM_ROOT = '/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K'
    if data_species == 'maize' and args.data_root == _SORGHUM_ROOT:
        args.data_root = '/work/mech-ai-scratch/alloy/Maize'
    if args.num_points == 8196 and data_species == 'maize':
        args.num_points = 8192                       # a maize cloud holds exactly 8192 points
    target_points = args.target_points or (8196 if args.model_species == 'sorghum' else 8192)

    # Serve the images the checkpoint trained on: the run dir's config.json records rgb_file
    # (data v2 = rgb_nobg.png) and rgb_brighten_gain (1.85 for runs trained after 2026-10-13).
    import json as _json
    _run_cfg = {}
    for _d in (os.path.dirname(args.checkpoint), os.path.dirname(os.path.dirname(args.checkpoint))):
        _cj = os.path.join(_d, 'config.json')
        if os.path.exists(_cj):
            _run_cfg = _json.load(open(_cj)); break
    if data_species == 'sorghum':
        # same-species: serve the images the checkpoint trained on; cross: the data's native files
        rgb_file = args.rgb_file or (_run_cfg.get('rgb_file') if not cross else 'rgb_nobg.png') or RGB_FILE
        rgb_gain = args.rgb_gain if args.rgb_gain is not None else (_run_cfg.get('rgb_brighten_gain') if not cross else None)
        print(f'RGB: {rgb_file} at gain {rgb_gain if rgb_gain is not None else "as on disk"}')
        ds = SorghumDataset4M(args.data_root, split=args.split, num_points=args.num_points, rgb_file=rgb_file, rgb_gain=rgb_gain)
    else:
        rgb_file = args.rgb_file or (_run_cfg.get('rgb_file') if not cross else None) or 'rgb.png'
        print(f'RGB: {rgb_file} (maize data)')
        ds = MaizeDataset4M(args.data_root, split=args.split, num_points=args.num_points, rgb_file=rgb_file,
                            param_schema='v2' if rgb_file == 'rgba.png' else 'v1')
    if cross:
        _enc = SC.depth_encoder(SC.model_depth_convention(args.model_species, _run_cfg))
        print(f'CROSS: {data_species} data -> {args.model_species} model; depth re-encoded to '
              f'{SC.model_depth_convention(args.model_species, _run_cfg)}; recipe zeroed at model width')

    # one seeded view per plant, plants themselves sampled without replacement
    by_plant = collections.defaultdict(list)
    for i, folder in enumerate(ds.samples):
        m = _NAME.match(folder.name)
        by_plant[m.group('plant') if m else folder.name].append(i)
    plants = sorted(by_plant)
    take = rng.choice(len(plants), size=min(args.n_plants, len(plants)), replace=False)
    idx = [int(rng.choice(by_plant[plants[p]])) for p in sorted(take)]
    print(f"{len(ds)} folders / {len(plants)} plants -> {len(idx)} sampled, one view each")

    build = embodied_mae_4m_base if args.model_species == 'sorghum' else embodied_mae_4m_maize_base
    model = build(target_points=target_points).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    epoch = int(ck.get('epoch', -1))
    print(f"loaded {args.checkpoint}  epoch {epoch}")

    visible = [s.strip() for s in args.source.split(',') if s.strip()]
    names, preds, gts = [], [], []
    for s in range(0, len(idx), args.batch_size):
        chunk = idx[s:s + args.batch_size]
        items = [ds[i] for i in chunk]
        rgb   = torch.stack([it[0] for it in items]).to(dev)
        if cross:
            fixed = []
            for k, it in zip(chunk, items):
                dec = (SC.depth_decoder('sorghum_std') if data_species == 'sorghum'
                       else SC.depth_decoder('maize_std', ds.samples[k] if hasattr(ds, 'samples') else None))
                fixed.append(SC.reencode_depth(it[1], dec, _enc))
            depth = torch.stack(fixed).to(dev)
            B = rgb.shape[0]
            n_text = 1 + (24 if args.model_species == 'sorghum' else 28)
            n_par = 9 if args.model_species == 'sorghum' else 14
            par = torch.zeros(B, n_text, n_par, device=dev)
            tv = torch.zeros(B, n_text, dtype=torch.bool, device=dev)
        else:
            depth = torch.stack([it[1] for it in items]).to(dev)
            par   = torch.stack([it[3] for it in items]).to(dev)
            tv    = torch.stack([it[4] for it in items]).to(dev)
        pc    = torch.stack([it[2] for it in items]).to(dev)
        with torch.no_grad():
            _, _, (_, _, pred_pc, _), _ = model(rgb, depth, pc, par, tv, visible=visible)
        names += [it[5] for it in items]
        preds.append(pred_pc.cpu()); gts.append(pc.cpu())
        print(f"  {min(s + args.batch_size, len(idx))}/{len(idx)}", flush=True)

    pred = torch.cat(preds); gt = torch.cat(gts)
    # derangement so no sample is ever compared against itself
    roll = torch.roll(torch.arange(len(pred)), 1)

    per = {}
    for key, a, b in (('model',    pred,       gt),
                      ('shuffled', pred,       gt[roll]),
                      ('gt_vs_gt', gt[roll],   gt)):
        f, r = [], []
        for s in range(0, len(a), args.batch_size):
            fw, bw = chamfer_both(a[s:s + args.batch_size], b[s:s + args.batch_size])
            f += fw.tolist(); r += bw.tolist()
        per[key] = dict(pred_to_gt=f, gt_to_pred=r,
                        symmetric=[x + y for x, y in zip(f, r)])

    def stats(v):
        v = np.asarray(v)
        return dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)),
                    p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                    min=float(v.min()), max=float(v.max()))

    report = dict(
        label=args.label or pathlib.Path(args.checkpoint).parent.name,
        model_species=args.model_species, data_species=data_species,
        checkpoint=args.checkpoint, epoch=epoch, split=args.split, source=visible,
        n_plants=len(idx), num_points=args.num_points, seed=args.seed,
        plants=names,
        summary={k: {d: stats(v[d]) for d in v} for k, v in per.items()},
        per_sample={k: v['symmetric'] for k, v in per.items()},
    )
    m = report['summary']['model']['symmetric']['mean']
    g = report['summary']['gt_vs_gt']['symmetric']['mean']
    report['skill_vs_mean_plant'] = float(1 - m / g)   # 1.0 perfect, 0.0 no better than another plant

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    for k in ('model', 'shuffled', 'gt_vs_gt'):
        s = report['summary'][k]
        print(f"{k:9s} sym {s['symmetric']['mean']:.5f}  "
              f"pred->gt {s['pred_to_gt']['mean']:.5f}  gt->pred {s['gt_to_pred']['mean']:.5f}")
    print(f"skill vs another-plant baseline: {report['skill_vs_mean_plant']*100:.1f}%")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
