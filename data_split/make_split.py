import numpy as np, csv, sys, json
SP="/tmp/claude-490224/-work-mech-ai-scratch-alloy-embodiedmae/36e07a10-6288-4299-a7ff-576ed7a36a14/scratchpad"
rows=list(csv.DictReader(open(f"{SP}/features.csv")))
plants=np.array([int(r['plant']) for r in rows])
FEATS=['n_leaves','stem_length','leaf_len_mean','leaf_len_max','roll_mean','roll_std']
X=np.array([[float(r[f]) for f in FEATS] for r in rows])

# standardize + Mahalanobis distance from centroid
mu=X.mean(0); Xs=(X-mu)/X.std(0)
cov=np.cov(Xs.T); inv=np.linalg.pinv(cov)
d2=np.einsum('ij,jk,ik->i', Xs, inv, Xs)   # squared Mahalanobis
extremeness=np.sqrt(d2)
rank=extremeness.argsort().argsort()/(len(extremeness)-1)  # percentile 0..1

# enriched probabilistic assignment ---------------------------------------
rng=np.random.default_rng(42)
N=len(plants); n_test=int(round(N*0.15)); n_val=int(round(N*0.15)); n_pool=n_test+n_val
GAMMA=2.0                          # enrichment strength; weight ~ percentile^GAMMA
w=(rank+0.02)**GAMMA; w/=w.sum()
pool=rng.choice(N, size=n_pool, replace=False, p=w)   # extreme-enriched pool
rng.shuffle(pool)
test_idx=set(pool[:n_test].tolist()); val_idx=set(pool[n_test:].tolist())
split=np.array(['train']*N, dtype=object)
for i in val_idx:  split[i]='val'
for i in test_idx: split[i]='test'

# report enrichment
def summ(name):
    m=split==name
    return (f"{name:5s} n={m.sum():5d}  extremeness med={np.median(extremeness[m]):.2f} "
            f"| stem med={np.median(X[m,1]):.2f} rng[{X[m,1].min():.2f},{X[m,1].max():.2f}] "
            f"| n_leaves med={np.median(X[m,0]):.0f} rng[{int(X[m,0].min())},{int(X[m,0].max())}] "
            f"| %top-decile-extreme={100*np.mean(rank[m]>0.9):.1f}%")
print("GAMMA=",GAMMA," pool(val+test)=",n_pool)
for s in ['train','val','test']: print(summ(s))
frac_extreme_in_valtest=np.mean(rank[(split!='train')]>0.9)
print(f"\nof all top-10% extreme plants: {100*np.mean(split[rank>0.9]!='train'):.1f}% landed in val/test")

# save assignment
with open(f"{SP}/assignment.csv","w",newline='') as f:
    wr=csv.writer(f); wr.writerow(['plant','split','extremeness','rank'])
    for i in range(N): wr.writerow([plants[i], split[i], f"{extremeness[i]:.4f}", f"{rank[i]:.4f}"])
print(f"\nwrote {SP}/assignment.csv")
