"""Fine-tune the maize row model on the real maize rows (E10 data, eval/real_rows.py) with 75 % of the plants.

Split: the 520 real plants (each row of the real data is one plant model repeated 20 times; both layouts of a plant
go to the same side) are shuffled with --split-seed; 25 % are held out for test, --val-plants of the rest select the
checkpoint, the remainder trains. Written to data_split/e10_real_split.csv and to the run directory.

What the real files supervise, and nothing else:
  cloud    the centre plant alone as the camera sees it (exact; the model's PC loss, QAL + Sinkhorn as the row run)
  depth    the centre plant alone, splatted from all its points in the window, the ones its neighbours hide included
           (exact; the model's masked-patch depth loss)
  recipe   the plant token's height and leaf count only (Smooth-L1 in normalised units; height targets above the
           head's 2.0 m range clip to 2.0 m). The files carry no per-leaf recipe, so leaf tokens get no loss.
  image    none: the photograph does not show the parts of the centre plant its neighbours hide.
Inputs follow the E10 protocol: image + depth + cloud visible at the run's source mask ratio, the recipe never given.

usage: python train_real_rows.py [--init run:ckpt] [--run maize_real_ft] [--epochs 15] [--lr 2e-5]
Then: python eval/eval_real_rows.py --rows <test plants> --models maize_real_ft:best_model.pth maize_row_ft_e1:...
"""
import sys, pathlib; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), str(pathlib.Path(__file__).resolve().parent / 'eval')]
import argparse, json, math, os, shutil, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from eval_row import build_model, predict, score_cloud
from real_rows import RealRowDataset, collate, SPECIES, row_id
from embodied_mae_4m_maize import _PLANT_SCALE_V2, _PLANT_SHIFT_V2, PLANT_FIELDS_V2

# Sorghum (--species sorghum): the 30 scanned plants. Their truth is the plant height, which the sorghum recipe does
# not hold (its plant token carries the stem length) and which is 2.2-5.3 m against 0.2-1.7 m in training, so the
# recipe gets no loss by default (--recipe-weight 0): cloud and depth only. Defaults: 3 validation plants.
ap = argparse.ArgumentParser()
ap.add_argument('--species', default='maize', choices=('maize', 'sorghum'))
ap.add_argument('--source', default='rows3000', choices=('rows3000', 'rows2026'),
                help='rows3000: the 3000-row upload of 2026-11-13 (split by scan: maize 520 plants, sorghum 52 scans '
                     'with their four tilted copies on one side); rows2026: the first upload (split by row = plant)')
ap.add_argument('--init', default=None, help='run:ckpt (default: the species row fine-tune at epoch 100)')
ap.add_argument('--run', default=None, help='default <species>_real_ft (rows2026) / <species>_real_ft_r3000 (rows3000)')
ap.add_argument('--out-root', default='/work/mech-ai/alloydas/embodiedmae_outputs')
ap.add_argument('--test-frac', type=float, default=0.25); ap.add_argument('--val-plants', type=int, default=None)
ap.add_argument('--split-seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--warmup-epochs', type=int, default=1)
ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--weight-decay', type=float, default=0.05)
ap.add_argument('--batch-size', type=int, default=16); ap.add_argument('--num-workers', type=int, default=14)
ap.add_argument('--recipe-weight', type=float, default=None, help='maize 5.0 (= the row run spline_loss_weight); sorghum 0')
ap.add_argument('--val-every', type=int, default=1)
ap.add_argument('--smoke', type=int, default=0, help='>0: that many train/val plants, 2 epochs, CPU-friendly')
ap.add_argument('--split-only', action='store_true', help='write the split (run dir + data_split/) and exit')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
SORG = a.species == 'sorghum'
a.init = a.init or ('sorghum_row_ft:checkpoints/checkpoint_epoch_100.pth' if SORG else 'maize_row_ft_e1:checkpoints/checkpoint_epoch_100.pth')
R3 = a.source == 'rows3000'
a.run = a.run or f"{a.species}_real_ft{'_r3000' if R3 else ''}"     # never the first upload's run dirs
a.val_plants = a.val_plants if a.val_plants is not None else ((5 if R3 else 3) if SORG else 39)
a.recipe_weight = a.recipe_weight if a.recipe_weight is not None else (0.0 if SORG else 5.0)
LAY = SPECIES[a.species]['layouts']
dev = torch.device(a.device)
H_IDX, N_IDX = PLANT_FIELDS_V2.index('height'), PLANT_FIELDS_V2.index('leafCount')

# ---- split by plant --------------------------------------------------------------------------------
if R3:                                         # scans; a scan's rows may also hold other splits' plants as neighbours
    from real_rows import load_index_3000, scan_id
    plants = sorted({scan_id(m['plant_id']) for m in load_index_3000(a.species)})
else:
    plants = sorted({row_id(d) for d in LAY['parallel'].glob('row_*_seed1')} &
                    {row_id(d) for d in LAY['perpendicular'].glob('row_*_seed1')})
rng = np.random.default_rng(a.split_seed); perm = list(rng.permutation(plants))
n_test = int(round(a.test_frac * len(plants)))
test, rest = sorted(perm[:n_test]), perm[n_test:]
val, train = sorted(rest[:a.val_plants]), sorted(rest[a.val_plants:])
if a.smoke:
    train, val, a.epochs = train[:a.smoke], val[:max(1, a.smoke // 2)], 2
split = pd.DataFrame([(p, 'train') for p in train] + [(p, 'val') for p in val] + [(p, 'test') for p in test], columns=['plant', 'split'])
print(f'{len(plants)} plants: train {len(train)}, val {len(val)}, test {len(test)} (seed {a.split_seed})', flush=True)

# ---- run directory (outputs on /work/mech-ai, symlinked under outputs/) -----------------------------
run_dir = pathlib.Path('outputs') / a.run
if not run_dir.exists():
    real = pathlib.Path(a.out_root) / a.run if not a.smoke else pathlib.Path(a.out_root) / '_smoke' / a.run
    real.mkdir(parents=True, exist_ok=True)
    if a.smoke: run_dir = real
    else: run_dir.symlink_to(real)
(run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
split.to_csv(run_dir / 'split.csv', index=False)
if not a.smoke:
    split.to_csv(f"data_split/e10_real_split{'_sorghum' if SORG else ''}{'_rows3000' if R3 else ''}.csv", index=False)

if a.split_only:
    print('split written to', run_dir / 'split.csv'); sys.exit(0)

init_run, init_ckpt = a.init.split(':', 1)
model, cfg, init_ep = build_model(pathlib.Path('outputs') / init_run, init_ckpt, dev, a.species)
gain = float(cfg.get('rgb_brighten_gain', 1.0)) if SORG else None
model.pc_sinkhorn_weight = float(cfg.get('pc_sinkhorn_weight', 0.0))      # build_model turns it off for eval
model.pc_sinkhorn_points = int(cfg.get('pc_sinkhorn_points', 1024)); model.pc_sinkhorn_blur = float(cfg.get('pc_sinkhorn_blur', 0.02))
mask = float(cfg.get('mask_ratio', 0.8)); pc_w = float(cfg.get('pc_loss_weight', 1.0))
run_cfg = dict(cfg, real_finetune=dict(init=a.init, init_epoch=init_ep, split_seed=a.split_seed, n_train=len(train), n_val=len(val),
                                       n_test=len(test), source=a.source, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, recipe_weight=a.recipe_weight,
                                       supervised=['pc', 'depth'] + (['height', 'leafCount'] if a.recipe_weight > 0 else []), source_mask_ratio=mask))
(run_dir / 'config.json').write_text(json.dumps(run_cfg, indent=1, default=str))

sel = (lambda ps: dict(source='rows3000', plants=ps)) if R3 else (lambda ps: dict(rows=ps))
train_ds = RealRowDataset(with_targets=True, seed=1, species=a.species, rgb_gain=gain, **sel(train))
val_ds = RealRowDataset(seed=0, species=a.species, rgb_gain=gain, **sel(val))
mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh, num_workers=a.num_workers, collate_fn=collate,
                               persistent_workers=False, drop_last=sh)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
steps_per_ep = max(1, len(train_ds) // a.batch_size); total = a.epochs * steps_per_ep; warm = a.warmup_epochs * steps_per_ep
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def recipe_target(info):
    h = torch.tensor([min(i['height'], _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX]) for i in info])
    n = torch.tensor([float(i['leaves']) for i in info])
    return torch.stack([(n + _PLANT_SHIFT_V2[N_IDX]) / _PLANT_SCALE_V2[N_IDX], (h + _PLANT_SHIFT_V2[H_IDX]) / _PLANT_SCALE_V2[H_IDX]], 1)


@torch.no_grad()
def validate():
    model.eval(); ch, hs, ns = [], [], []
    for bi, (rgb, dep, pin, par, valid, pgt, info) in enumerate(mk(val_ds, False)):
        rgb, dep, pin, par, pgt = (x.to(dev) for x in (rgb, dep, pin, par, pgt))
        pp, ppar = predict(model, 'all', rgb, dep, pin, par, mask, 1000003 * bi)
        scale = torch.tensor([i['scale'] for i in info], device=dev)
        ch += score_cloud(pp.float(), pgt, scale)[1].tolist()
        p0 = ppar[:, 0].clamp(0, 1).float().cpu().numpy()
        if SORG:                                    # stem length (plant token field 0) against the plant height
            import embodied_mae_4m as S4
            h = p0[:, 0] * S4._PLANT_SCALE[0] - S4._PLANT_SHIFT[0]; n = np.full(len(info), -99)
        else:
            h = p0[:, H_IDX] * _PLANT_SCALE_V2[H_IDX] - _PLANT_SHIFT_V2[H_IDX]
            n = np.rint(p0[:, N_IDX] * _PLANT_SCALE_V2[N_IDX] - _PLANT_SHIFT_V2[N_IDX])
        hs += list(np.abs(h - np.array([i['height'] for i in info])) / np.array([i['height'] for i in info]))
        ns += list(np.abs(n - np.array([i['leaves'] for i in info])) <= 1)
    model.train()
    return dict(val_chamfer_m2=float(np.mean(ch)), val_height_rel_err=float(np.median(hs)), val_leaves_pm1=float(np.mean(ns)))


hist = dict(epoch=[], train_loss=[], train_pc=[], train_depth=[], train_recipe=[], val=[])
hist['val'].append(dict(epoch=0, **validate())); print('epoch 0 (init)', hist['val'][-1], flush=True)
best = hist['val'][-1]['val_chamfer_m2']
model.train(); step = 0
for ep in range(1, a.epochs + 1):
    train_ds.seed = 1 + ep                                                  # a fresh point subsample per epoch
    t0, sums, nb = time.time(), np.zeros(4), 0
    for rgb, dep, pin, par, valid, pgt, g_dep, info in mk(train_ds, True):
        rgb, dep, pin, par, valid, pgt, g_dep = (x.to(dev) for x in (rgb, dep, pin, par, valid, pgt, g_dep))
        _, (_, ld, lp, _), (_, _, _, ppar), _ = model(rgb, dep, pin, torch.zeros_like(par), valid, visible={'rgb', 'depth', 'pc'},
                                                      source_mask_ratio=mask, targets=(rgb, g_dep, pgt))
        lr_ = (F.smooth_l1_loss(ppar[:, 0, [N_IDX, H_IDX]], recipe_target(info).to(dev), beta=0.1) if a.recipe_weight > 0
               else ppar.new_zeros(()))
        loss = pc_w * lp + ld + a.recipe_weight * lr_
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        sums += [loss.item(), lp.item(), ld.item(), lr_.item()]; nb += 1
        if nb % 100 == 0:
            print(f'  ep {ep} step {nb}/{steps_per_ep} loss {sums[0] / nb:.4f} pc {sums[1] / nb:.4f} depth {sums[2] / nb:.4f} '
                  f'recipe {sums[3] / nb:.4f} ({time.time() - t0:.0f} s)', flush=True)
    for k, v in zip(('train_loss', 'train_pc', 'train_depth', 'train_recipe'), sums / max(1, nb)):
        hist[k].append(float(v))
    hist['epoch'].append(ep)
    ck = dict(model_state_dict=model.state_dict(), epoch=ep)
    if ep % a.val_every == 0 or ep == a.epochs:
        hist['val'].append(dict(epoch=ep, **validate()))
        print(f'epoch {ep}: train {sums[0] / max(1, nb):.4f}  {hist["val"][-1]}  ({time.time() - t0:.0f} s)', flush=True)
        if hist['val'][-1]['val_chamfer_m2'] < best:
            best = hist['val'][-1]['val_chamfer_m2']; torch.save(ck, run_dir / 'best_model.pth.tmp'); os.replace(run_dir / 'best_model.pth.tmp', run_dir / 'best_model.pth')
    torch.save(ck, run_dir / 'checkpoints' / 'last.pth.tmp'); os.replace(run_dir / 'checkpoints' / 'last.pth.tmp', run_dir / 'checkpoints' / f'checkpoint_epoch_{a.epochs}.pth' if ep == a.epochs else run_dir / 'checkpoints' / 'last.pth')
    (run_dir / 'training_history.json').write_text(json.dumps(hist, indent=1))
if not (run_dir / 'best_model.pth').exists():                               # no epoch beat the warm start
    shutil.copy(run_dir / 'checkpoints' / f'checkpoint_epoch_{a.epochs}.pth', run_dir / 'best_model.pth')
print('test plants:', ' '.join(test))
print('done', run_dir)
