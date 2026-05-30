import yaml, random, numpy as np
from pathlib import Path

ymls = list(Path('./Dataset/new_data/train').rglob('*_spline.yml'))
sample = random.sample(ymls, min(500, len(ymls)))
print(f"Sampling {len(sample)} / {len(ymls)} files...")

params = {'sl':[],'ra':[],'ba':[],'wf':[],'wp0':[],'wp1':[],
          'pa':[],'pr':[],'sp':[],'ln':[],
          'sdx':[],'sdy':[],'sdz':[],'psx':[],'psy':[],'psz':[]}

for yml in sample:
    data = yaml.safe_load(open(yml))
    plant = data['Sorghums'][0]; p = plant['Parameters']
    params['sl'].append(float(p['stem_length']))
    params['pa'].append(float(p['panicle_seed_amount']))
    params['pr'].append(float(p['panicle_seed_radius']))
    sd = p['stem_direction']
    params['sdx'].append(float(sd[0])); params['sdy'].append(float(sd[1])); params['sdz'].append(float(sd[2]))
    ps = p['panicle_size']
    params['psx'].append(float(ps[0])); params['psy'].append(float(ps[1])); params['psz'].append(float(ps[2]))
    for leaf in plant['Leaves']:
        params['sp'].append(float(leaf['starting_point']))
        params['ln'].append(float(leaf['length']))
        params['ra'].append(float(leaf['roll_angle']))
        params['ba'].append(float(leaf['branching_angle']))
        params['wf'].append(float(leaf['waviness_frequency']))
        wps = leaf['waviness_period_start']
        params['wp0'].append(float(wps[0])); params['wp1'].append(float(wps[1]))

my_scales = {'sl':3.0,'ra':360,'ba':180,'wf':0.1,'wp0':360,'wp1':360,
             'pa':50,'pr':0.02,'sp':1.0,'ln':1.0,'sdx':2,'sdy':2,'sdz':2,
             'psx':1.0,'psy':1.0,'psz':1.0}
my_shifts = {'sdx':1,'sdy':1,'sdz':1}

print(f"\n{'param':5s}  {'min':>9s}  {'max':>9s}  {'mean':>9s}  {'scale':>8s}  {'norm_max':>9s}  {'ok?':>6s}")
print('-'*65)
for k, v in params.items():
    sc = my_scales[k]; sh = my_shifts.get(k, 0.0)
    norm_max = (max(v) + sh) / sc
    norm_min = (min(v) + sh) / sc
    ok = 'OK' if (norm_max <= 1.0 and norm_min >= 0.0) else 'CLIP!'
    print(f"{k:5s}  {min(v):9.4f}  {max(v):9.4f}  {np.mean(v):9.4f}  {sc:8.4f}  {norm_max:9.4f}  {ok:>6s}")
