#!/usr/bin/env python
"""Re-apply a train/val/test split to the Sorghum_15K dataset, location-agnostic.

The dataset folders may currently sit either flat under BASE/ or inside
BASE/{train,val,test}/. This script finds every Sorghum_<plant>_<view> folder
wherever it is, reads a target assignment.csv (plant,split,...), and moves only
the folders whose split needs to change. All 10 views of a plant follow its
plant-level split (no view leakage).

Typical reshuffle workflow:
    1. python make_split.py            # regenerate assignment.csv (edit SEED/GAMMA/ratios there)
    2. python reshuffle.py             # DRY RUN — shows the move plan
    3. python reshuffle.py --go        # execute

To flatten back to a single directory (no splits):  python reshuffle.py --flatten --go
"""
import csv, os, sys, shutil, re
from collections import Counter

BASE   = "/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K"
ASSIGN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assignment.csv")
SPLITS = ("train", "val", "test")
_PAT   = re.compile(r"^Sorghum_(\d+)_(\d+)$")

def current_locations():
    """folder_name -> current split dir ('' means flat at BASE root)."""
    loc = {}
    for sub in ("",) + SPLITS:
        d = os.path.join(BASE, sub) if sub else BASE
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if _PAT.match(name):
                loc[name] = sub
    return loc

def main():
    go       = "--go" in sys.argv
    flatten  = "--flatten" in sys.argv
    loc      = current_locations()
    assign   = {int(r["plant"]): r["split"] for r in csv.DictReader(open(ASSIGN))}

    plan = []  # (folder, src_sub, dst_sub)
    for name, src in loc.items():
        m = _PAT.match(name); plant = int(m.group(1))
        dst = "" if flatten else assign.get(plant)
        if dst is None:
            print(f"⚠️  {name}: plant {plant} not in assignment.csv — leaving in {src or 'root'}")
            continue
        if dst != src:
            plan.append((name, src, dst))

    moves = Counter(f"{s or 'root'}->{d or 'root'}" for _, s, d in plan)
    print(f"folders found: {len(loc)}   moves needed: {len(plan)}")
    for k, v in sorted(moves.items()):
        print(f"  {v:6d}  {k}")
    if not plan:
        print("nothing to do — already in target layout"); return
    if not go:
        print("\nDRY RUN — pass --go to execute"); return

    for sub in SPLITS:
        if not flatten:
            os.makedirs(os.path.join(BASE, sub), exist_ok=True)
    done = 0
    for name, src, dst in plan:
        s = os.path.join(BASE, src, name) if src else os.path.join(BASE, name)
        d = os.path.join(BASE, dst, name) if dst else os.path.join(BASE, name)
        shutil.move(s, d); done += 1
        if done % 20000 == 0:
            print(f"  moved {done}/{len(plan)}")
    print(f"DONE moved {done} folders")

if __name__ == "__main__":
    main()
