"""Regenerate the report figures with the RGB de-normalisation fix.

Pins the exact plants used in the published figures via a Subset, so the
before/after distillation frames show the same plant and are directly
comparable. Writes into ./figures_fixed/.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import sys, torch, yaml
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader, Subset

from sorghum_dataset_4m import SorghumDataset4M
import train_sorghum_4m_distill as DIS
import train_sorghum_4m as PRE

DEV = 'cuda'
OUT = Path('figures_fixed'); OUT.mkdir(exist_ok=True)
CFG = yaml.safe_load(open('configs/config_4m_distill_15k_all.yaml'))
PRE_DIR = Path('./outputs/4m_pretrain_15k_v2_depthfix_qal')

args = DIS.config_to_namespace(CFG)

root = CFG['data']['data_root']
ds = SorghumDataset4M(f"{root}/val", img_size=args.img_size, num_points=args.num_points)
# ds.samples is a list of folder Paths - match on the folder name, no decoding
names = [p.name for p in ds.samples]
print(f"val set: {len(ds)} samples")

def find(target):
    for i, n in enumerate(names):
        if target in str(n):
            return i
    raise SystemExit(f"could not locate {target} in the val split")

def loader_for(idxs, bs):
    return DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=False,
                      num_workers=4, pin_memory=True)

# ── 1. distillation: same plant, before vs after ──────────────────────────
i_d = find('Sorghum_10001_00')
dl = loader_for([i_d], 1)
model = DIS.build_model(args, DEV)

for tag, ckpt, ep in [('warmstart', f'{PRE_DIR}/teacher_final.pth', 0),
                      ('distilled', './outputs/4m_distill_15k_all/best_model.pth', 95)]:
    DIS.load_weights_into(model, ckpt, DEV, tag)
    d = OUT / tag; d.mkdir(exist_ok=True)
    torch.manual_seed(0)
    DIS.visualize_crossmodal(model, dl, DEV, ep, d, 'pc', num_samples=1)
    print(f"  -> {tag}: {[p.name for p in d.glob('*.png')]}")

del model; torch.cuda.empty_cache()

# ── 2. pretrain: the 5-row masked reconstruction ──────────────────────────
i_p = find('Sorghum_1001_02')
pl = loader_for([i_p], 1)
pm = DIS.build_model(args, DEV)
DIS.load_weights_into(pm, f'{PRE_DIR}/best_model.pth', DEV, 'pretrain')
d = OUT / 'pretrain'; d.mkdir(exist_ok=True)
torch.manual_seed(0)
PRE.visualize_reconstruction_4m(pm, pl, DEV, 960, d, num_samples=1,
                                mask_ratio=args.mask_ratio)
print(f"  -> pretrain: {[p.name for p in d.glob('*.png')]}")
print("REGEN_DONE")
