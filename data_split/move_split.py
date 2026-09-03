import csv, os, shutil, sys
BASE="/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K"
SP="/tmp/claude-490224/-work-mech-ai-scratch-alloy-embodiedmae/36e07a10-6288-4299-a7ff-576ed7a36a14/scratchpad"
assign={int(r['plant']):r['split'] for r in csv.DictReader(open(f"{SP}/assignment.csv"))}
DRY = '--go' not in sys.argv

# verify views present, plan moves
existing=set(os.listdir(BASE))
splits={'train','val','test'}
missing=[]; plan=[]  # (src_folder, split)
for n,sp in assign.items():
    for v in range(10):
        name=f"Sorghum_{n}_{v:02d}"
        if name in existing: plan.append((name, sp))
        else: missing.append(name)
# also detect any folders on disk not in assignment
known=set(f"Sorghum_{n}_{v:02d}" for n in assign for v in range(10))
extra=[e for e in existing if e not in known and e not in splits and not e.startswith('.')]
from collections import Counter
c=Counter(sp for _,sp in plan)
print(f"folders to move: {len(plan)}  ({dict(c)})")
print(f"missing (in assignment, not on disk): {len(missing)}  e.g. {missing[:3]}")
print(f"extra   (on disk, not in assignment): {len(extra)}  e.g. {extra[:3]}")
if DRY:
    print("\nDRY RUN — pass --go to execute"); sys.exit(0)
for s in splits: os.makedirs(os.path.join(BASE,s), exist_ok=True)
done=0
for name,sp in plan:
    shutil.move(os.path.join(BASE,name), os.path.join(BASE,sp,name))
    done+=1
    if done%20000==0: print(f"  moved {done}/{len(plan)}")
print(f"DONE moved {done} folders")
