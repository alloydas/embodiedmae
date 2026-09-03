import os, re, numpy as np, csv
from multiprocessing import Pool
BASE="/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K"
OUT="/tmp/claude-490224/-work-mech-ai-scratch-alloy-embodiedmae/36e07a10-6288-4299-a7ff-576ed7a36a14/scratchpad/features.csv"
scal=re.compile(r'^\s*(stem_length|panicle_seed_amount|starting_point|length|roll_angle|branching_angle|waviness_frequency):\s*([-\d.eE]+)')
KEYS=['n_leaves','stem_length','leaf_len_mean','leaf_len_max','roll_mean','roll_std','branch_mean','wav_mean']
def feats(n):
    y=f"{BASE}/Sorghum_{n}_00/Sorghum_{n}_spline.yml"
    if not os.path.exists(y): return None
    d={'length':[],'roll_angle':[],'branching_angle':[],'waviness_frequency':[]}; plant={}
    with open(y) as f:
        for line in f:
            m=scal.match(line)
            if m:
                k,v=m.group(1),float(m.group(2))
                if k in d: d[k].append(v)
                elif k not in plant: plant[k]=v
    ln=np.array(d['length'] or [0.]); ra=np.array(d['roll_angle'] or [0.]); ba=np.array(d['branching_angle'] or [0.]); wf=np.array(d['waviness_frequency'] or [0.])
    return [n, len(d['length']), plant.get('stem_length',0.), ln.mean(), ln.max(), ra.mean(), ra.std(), ba.mean(), wf.mean()]
if __name__=='__main__':
    with Pool(16) as p:
        rows=[r for r in p.map(feats, range(15000), chunksize=64) if r]
    rows.sort()
    with open(OUT,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['plant']+KEYS); w.writerows(rows)
    print(f"wrote {len(rows)} plants -> {OUT}")
