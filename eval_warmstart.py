"""Measure the true epoch-0 cross-modal baseline (the warm start, before any
distillation). The run's first val point is at epoch 1, so the published -5.9%
excludes whatever the first epoch bought."""
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
root=CFG['data']['data_root']
ds=SorghumDataset4M(f"{root}/val", img_size=args.img_size, num_points=args.num_points)
dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=True)

m=DIS.build_model(args,DEV)
DIS.load_weights_into(m,'./outputs/4m_pretrain_15k_v2_depthfix_qal/teacher_final.pth',DEV,'warmstart')
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
