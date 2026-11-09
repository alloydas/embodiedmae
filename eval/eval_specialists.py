"""Final-data parameter specialist against the distilled generalist at recipe regression from the point cloud
(the claim in the intro, the discussion and the supplement). Scores checkpoints with the distillation trainer's own
per-source evaluator (the source fully visible, every other stream masked and generated; param MAE over the real
leaf and plant tokens) on the FULL validation split (22,500 views; the training histories use the first 1,600).
usage: python eval/eval_specialists.py --ckpt generalist=outputs/4m_distill_15k_all_d2/best_model.pth \
         spec=outputs/4m_distill_15k_pc2text_d2/best_model.pth [--config configs/config_4m_distill_15k_pc2text_d2.yaml]
         [--sources pc] [--max-batches 0] [--device cuda] [--out reports/specialists_<job>.csv]
Missing checkpoints are skipped with a note. CPU smoke: --max-batches 2 --device cpu --num-workers 2.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, os, time
import pandas as pd, torch, yaml
from torch.utils.data import DataLoader
from sorghum_dataset_4m import SorghumDataset4M
import train_sorghum_4m_distill as DIS

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/config_4m_distill_15k_pc2text_d2.yaml')
ap.add_argument('--ckpt', nargs='+', required=True, help='label=path')
ap.add_argument('--sources', nargs='+', default=['pc'])
ap.add_argument('--split', default='val'); ap.add_argument('--max-batches', type=int, default=0)
ap.add_argument('--device', default='cuda'); ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--num-workers', type=int, default=8); ap.add_argument('--out', default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); args = DIS.config_to_namespace(cfg); args.batch_size = a.batch_size
ds = SorghumDataset4M(args.data_root, img_size=args.img_size, num_points=args.num_points, split=a.split,
                      max_leaves=args.max_leaves, rgb_file=args.rgb_file, rgb_gain=args.rgb_gain)
dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=(a.device == 'cuda'))
print(f'config {a.config}  split {a.split}  {len(ds)} views  sources {a.sources}  max_batches {a.max_batches}', flush=True)
rows = []
for spec in a.ckpt:
    label, path = spec.split('=', 1)
    if not os.path.exists(path):
        print(f'SKIP {label}: {path} missing', flush=True); rows.append(dict(label=label, ckpt=path, status='missing')); continue
    m = DIS.build_model(args, a.device)
    DIS.load_weights_into(m, path, a.device, label, None)
    st = torch.load(path, map_location='cpu', weights_only=False); ep = st.get('epoch') if isinstance(st, dict) else None; del st
    t0 = time.time()
    per, _ = DIS.evaluate_crossmodal(m, dl, a.device, a.sources, target=None, max_batches=a.max_batches)
    for s in a.sources:
        p = per[s]
        rows.append(dict(label=label, ckpt=path, epoch=ep, source=s, status='ok', n_batches=(a.max_batches or len(dl)), **{k: float(v) for k, v in p.items()}))
        print(f'{label:18s} ep {ep} src={s}: param_mae {p["param_mae_masked"]:.4f}  pc_chamfer {p["pc_chamfer"]:.5f}  '
              f'text_loss {p["text_loss"]:.5f}  total {p["total"]:.4f}  ({time.time() - t0:.0f} s)', flush=True)
    del m
    if a.device == 'cuda': torch.cuda.empty_cache()
out = a.out or f"reports/specialists_{os.environ.get('SLURM_JOB_ID', 'manual')}.csv"
pd.DataFrame(rows).to_csv(out, index=False); print('wrote', out)
