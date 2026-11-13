"""Data for the review page of the 3000-row E10 upload (figures/pages/rows3000/): sample stops with their photo and a 3D
cloud (target plant marked), maize leaf tables, and distributions against the synthetic training data.
Reads /work/mech-ai/alloydas/e10_rows_3000/<species>/{index.csv,png,pcd,csv} (eval/real_rows.py layout).
Writes figures/pages/rows3000/{data/*.json,img/*.jpg,stats.json}. CPU, ~3 min:
  sbatch --wait --partition=nova --account=mech-ai --qos=normal --cpus-per-task=8 --mem=16G --time=00:30:00 \
         --output=logs/%x_%j.out --wrap "python figures/make_rows3000_page.py"
"""
import sys, pathlib, json, base64, glob, csv as csvm
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'eval')]
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from real_rows import ROWS3000, read_pcd, load_index_3000, scan_id

OUT = R / 'figures/pages/rows3000'; (OUT / 'data').mkdir(parents=True, exist_ok=True); (OUT / 'img').mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
N_STAT, N_SHOW, N_PTS = 400, 4, 12000


def target_height(sp, m):
    P = read_pcd(ROWS3000 / sp / 'pcd' / f"{m['name']}.pcd"); t = P['label'] == int(m['middle_plant'])
    y = P['y'][t]; return float(y.max() - y.min()) if t.any() else float('nan')


def pack(name, scene, mark, meta):
    q = 32767.0 / float(np.abs(scene).max()) * 0.999
    buf = np.round(scene * q).astype('<i2').tobytes(); n = len(scene)
    ent = {'scene': dict(off=0, n=n, mark=True)}
    buf += mark.astype(np.uint8).tobytes() + (b'\0' if (len(buf) + n) % 2 else b'')
    (OUT / 'data' / f'{name}.json').write_text(json.dumps(dict(meta, b64=base64.b64encode(buf).decode(), q=q, scale_m=1.0, err_max_mm=1.0, clouds=ent),
                                                          separators=(',', ':')))


stats = {}
for sp in ('maize', 'sorghum'):
    idx = load_index_3000(sp); df = pd.DataFrame(idx)
    df['scan'] = df.plant_id.map(scan_id); df['stop'] = df.stop.astype(int); df['row'] = df.row.astype(int)
    st = dict(rows=int(df.row.nunique()), stops=int(len(df)), plants=int(df.plant_id.nunique()), scans=int(df.scan.nunique()),
              stops_per_row=int(df.groupby('row').stop.nunique().median()),
              middle_per_plant=[int(x) for x in df.groupby('plant_id').size().quantile([0, 1]).values],
              hfov=[float(x) for x in df.hfov_deg.astype(float).quantile([0.05, 0.5, 0.95]).values],
              cam_h=[float(x) for x in df.camera_height_m.astype(float).quantile([0.05, 0.5, 0.95]).values],
              img_h=[int(x) for x in df.image_px.str.split('x').str[1].astype(int).quantile([0.05, 0.5, 0.95]).values])
    split = pd.read_csv(R / f"outputs/{sp}_real_ft_r3000/split.csv", dtype=str)
    st['split'] = split.split.value_counts().to_dict()
    # heights of the middle plants on a random sample of stops
    samp = df.sample(N_STAT, random_state=1)
    with ThreadPoolExecutor(8) as ex:
        h = list(ex.map(lambda m: target_height(sp, m), samp.to_dict('records')))
    st['height'] = [round(x, 3) for x in h if x == x]
    if sp == 'maize':
        st['leaves'] = [int(x) for x in df.drop_duplicates('plant_id').leaves.astype(int)]
        # per-leaf traits: one CSV per plant (the first stop where it is the middle plant)
        first = df.drop_duplicates('plant_id')
        rows = []
        for m in first.to_dict('records'):
            rows += list(csvm.DictReader(open(ROWS3000 / sp / 'csv' / f"{m['name']}.csv")))
        L = pd.DataFrame(rows)
        traits = {}
        for c, lab in (('leaf_length_m', 'leaf length (m)'), ('leaf_width_m', 'leaf width (m)'), ('zenith_deg', 'leaf zenith angle (deg)'),
                       ('internode_dist_m', 'internode distance (m)'), ('curvature_1pm', 'midrib curvature (1/m)')):
            v = L[c].astype(float).values; traits[c] = dict(label=lab, values=[round(float(x), 4) for x in np.clip(v, None, np.percentile(v, 99.5))])
        st['traits'] = traits
        # the synthetic training plants' ranges (Maize_1 val recipes, the generator's own fields)
        syn = {'leaf_length_m': [], 'leaf_width_m': [], 'internode_dist_m': [], 'curvature_1pm': []}; sh = []
        for f in sorted(glob.glob('/work/mech-ai-scratch/alloy/Maize_1/val/plant_*_00/*_spline.xml'))[:600]:
            r = ET.parse(f).getroot(); sh.append(float(r.attrib['height']))
            for lf in r.iter('leaf'):
                a = lf.attrib; syn['leaf_length_m'].append(float(a['leafLength'])); syn['leaf_width_m'].append(float(a['leafWidth']))
                syn['internode_dist_m'].append(float(a['distance'])); syn['curvature_1pm'].append(float(a['avgCurvature']))
        st['synthetic'] = {k: [float(np.percentile(v, 5)), float(np.percentile(v, 95))] for k, v in syn.items()}
        st['synthetic']['height'] = [float(np.percentile(sh, 5)), float(np.percentile(sh, 95))]
    else:
        st['synthetic'] = {'height': [0.2, 1.7]}                      # sorghum training plants (paper, Sec. 4.8)
    # sample stops for the gallery: spread over the middle-plant heights
    hs = pd.Series(h, index=samp.index).dropna().sort_values()
    picks = [hs.index[int(q * (len(hs) - 1))] for q in (0.08, 0.38, 0.68, 0.95)][:N_SHOW]
    show = []
    for k, ix in enumerate(picks):
        m = df.loc[ix].to_dict(); name = f"{sp}_{k}"
        im = Image.open(ROWS3000 / sp / 'png' / f"{m['name']}.png").convert('RGB')
        s = 520 / im.height; im.resize((max(1, round(im.width * s)), 520), Image.LANCZOS).save(OUT / 'img' / f'{name}.jpg', quality=86)
        P = read_pcd(ROWS3000 / sp / 'pcd' / f"{m['name']}.pcd"); X = np.stack([P['x'], -P['y'], -P['z']], 1).astype(np.float32)
        tgt = P['label'] == int(m['middle_plant'])
        zoom = None
        if sp == 'sorghum':                               # tall strips: a maize-shaped crop (3:1) centred on the middle plant's leaves
            Wp, Hp = im.size; fx = (Wp / 2) / np.tan(np.radians(float(m['hfov_deg']) / 2))
            lf = tgt & (P['part'] == 2) if 'part' in P.dtype.names else tgt
            v = (Hp - 1) / 2 + fx * P['y'][lf] / P['z'][lf]
            ch = min(Hp, 3 * Wp); c = float(np.median(v)); y0 = int(np.clip(c - ch / 2, 0, Hp - ch))
            crop = im.crop((0, y0, Wp, y0 + ch)); zs = 1040 / ch
            crop.resize((max(1, round(Wp * zs)), 1040), Image.LANCZOS).save(OUT / 'img' / f'{name}_zoom.jpg', quality=88)
            zoom = [y0 / Hp, (y0 + ch) / Hp]
        sel = rng.choice(len(X), min(N_PTS, len(X)), replace=False); X, tgt = X[sel], tgt[sel]
        X = X - X[tgt].mean(0)
        e = dict(name=name, file=m['name'], row=int(m['row']), stop=int(m['stop']), plant=str(m['plant_id']), height=float(hs[ix]),
                 cam_h=float(m['camera_height_m']), image_px=m['image_px'], plants_in_view=int(len(np.unique(P['label']))), zoom=zoom)
        if sp == 'maize':
            e['leaves'] = list(csvm.DictReader(open(ROWS3000 / sp / 'csv' / f"{m['name']}.csv")))
        pack(name, X, tgt, dict(item=e['file']))
        show.append(e)
    st['show'] = show
    stats[sp] = st
    print(sp, {k: v for k, v in st.items() if k not in ('height', 'leaves', 'traits', 'show')}, flush=True)
(OUT / 'stats.json').write_text(json.dumps(stats, separators=(',', ':')))
print('wrote', OUT / 'stats.json', (OUT / 'stats.json').stat().st_size // 1024, 'KB')
