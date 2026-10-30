"""Measure the true epoch-0 cross-modal baseline (the warm start, before any
distillation). The run's first val point is at epoch 1, so the published -5.9%
excludes whatever the first epoch bought.

usage: python eval/eval_warmstart.py [config] [sources]
  config   a distillation config; its data block (rgb_file, rgb_gain, max_leaves) and
           its distill.student_init are used, so a final-data config scores the
           final-data warm start (default: the first-release generalist config).
  sources  optional comma list overriding distill.sources, e.g. rgb+depth,rgb+text."""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import torch, yaml
from torch.utils.data import DataLoader
from sorghum_dataset_4m import SorghumDataset4M
import train_sorghum_4m_distill as DIS

DEV='cuda'
import sys
CFG_PATH=sys.argv[1] if len(sys.argv)>1 else 'configs/config_4m_distill_15k_all.yaml'
CFG=yaml.safe_load(open(CFG_PATH))
print(f"config: {CFG_PATH}")
args=DIS.config_to_namespace(CFG)
if len(sys.argv)>2: args.sources=[s for s in sys.argv[2].split(',') if s]
ds=SorghumDataset4M(args.data_root, img_size=args.img_size, num_points=args.num_points, split='val',
                    max_leaves=args.max_leaves, rgb_file=args.rgb_file, rgb_gain=args.rgb_gain)
dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=True)

m=DIS.build_model(args,DEV)
# the first-release config predates distill.student_init pointing at a checkpoint on disk
INIT=args.student_init if args.student_init and __import__('os').path.exists(args.student_init) \
     else './outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth'
print(f"warm start: {INIT}  sources: {args.sources}  target: {args.targets}")
DIS.load_weights_into(m,INIT,DEV,'warmstart',getattr(args,'expect_init_epoch',None))
per,mg=DIS.evaluate_crossmodal(m,dl,DEV,args.sources,target=args.targets,
                               max_batches=args.val_max_batches)
print("\n=== EPOCH 0 (warm start, no distillation) ===")
for s in args.sources:
    p=per[s]
    print(f"  src={s:<5} total {p['total']:.4f}  pc_loss {p['pc_loss']:.5f}  "
          f"text_loss {p['text_loss']:.5f}  pc_chamfer {p['pc_chamfer']:.5f}  "
          f"param_mae {p['param_mae_masked']:.4f}")
print(f"  MEAN generation metric: {mg:.5f}")
print("EVAL_DONE")
