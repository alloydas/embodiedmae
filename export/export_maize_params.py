"""Export the MAIZE per-leaf parameter browser data: what the base model
(`maize_4m`, checkpoint_epoch_600.pth) recovers of a plant's generator
parameters from the POINT CLOUD alone and from RGB alone.

Produces, under --out-dir:
  params_maize.js        window.PARAM_DATA = {leaf_fields, leaf_units,
                         plant_fields, plants:[{name, n, plant:{gt,pc,rgb},
                         leaves:[{i, gt, pc, rgb}]}]};  -- the schema the results
                         page's existing sorghum browser reads. `i` is the token
                         index, 1..n (= XML leaf id + 1), as in the sorghum blob.
  param_skill_maize.json the skill table (also as param_skill_maize.js,
                         window.PARAM_SKILL_MAIZE): per field, GT spread over the
                         shown leaves, MAE from PC and from RGB, and skill against
                         TWO baselines, on the shown plants and, with
                         --skill-plants N, on a seeded random sample of N other
                         val plants:
                           skill               = 1 - MAE / MAE(val mean)
                           skill_vs_position   = 1 - MAE / MAE(per-leaf-index
                                                 val mean, leave-one-plant-out)
                         The second is the honest one for LEAF fields. The
                         decoder gives every leaf token its positional embedding
                         whatever the input, so a model that sees nothing still
                         knows "this is leaf 3" -- and leaf 3 is short, flat and
                         close to the ground. Skill against the plain mean
                         credits the source modality with what token position
                         alone provides; skill_vs_position does not. (The plant
                         token has one position, so for plant fields the two
                         baselines are the same.) Each row also carries Pearson r
                         of prediction vs GT and the prediction's sd / GT sd, which
                         separate "ranked right but shrunk to the mean" from "no
                         signal".
  leak_checks.json       the structural leak checks, max|diff| per plant.
  raw_pred_norm_maize.json  the undecoded [0,1] predictions, for re-checking.

Units. Every value is in the generator's own units: GT is the raw XML attribute,
predictions are decoded with the inverse of the normalisation
`embodied_mae_4m_maize._leaf_to_params/_plant_to_params` apply (clip to [0,1],
raw = norm * SCALE - SHIFT). The round trip raw -> the model's own normalisers
-> decode is checked for every val plant (via the split's `_params.json`, which
is checked equal to the per-folder XML for every plant this script opens).

Fields. The maize token is N_PARAMS = 14 wide.
  Plant token: 5 informative slots; slots 5..13 are zero padding in every plant
    and are dropped. `stemShrink` is dropped from the display as well: it is
    0.65 x stemRadius in every val plant (max|diff| 1.6e-9), a restatement not a
    parameter. It stays in the skill table, flagged. The four shown are ordered
    leafCount, stemInternodeSum (height), stemRadius, tasselMatureDroopStrength
    -- leaf count and height are two of decision 6.4's four targets.
  Leaf token: 14 slots -> 13 display fields. The (sin, cos) pair of waveLPhase is
    merged back into one angle by atan2. azJitterDeg is kept as the model sees it
    (leafAzimuthDeg = (180 * leaf_index + azJitterDeg) mod 360; raw azimuth is
    bimodal at 0/180 and wraps at 360 -- see embodied_mae_4m_maize.py).
  No displayed field is constant over the val split (asserted).

waveLPhase is circular (radians, unit ' rad'). GT is the raw XML value; the
prediction is reported as the representative nearest that GT,
gt + wrap_pi(pred - gt), so the plain |pred - gt| the page computes IS the
circular error -- but the displayed number is therefore chosen using the GT, and
is not the model's atan2 principal value. Its MAE and both baselines use circular
differences and the circular mean. Its Pearson r is not reported (meaningless on
a circle, and inflated by the GT-chosen representative).

Rounding. Floats in the .js blob keep 4 decimals, or 4 significant figures where
that is finer (stemRadius ~0.012, waveLAmp ~0.003 would otherwise keep 1-2), so a
renderer using toPrecision(3) shows them properly. The page's own fmt() /
toFixed(3) still prints them to 3 decimals; that is a renderer choice.

No leaks. For each source the other three modalities are fully masked via
`forward_encoder_select(visible={src}, source_mask_ratio=0.0)` -- they
contribute ZERO encoder tokens (asserted from the returned per-modality visible
counts) -- and the param tensor handed in is zeros (the maize plant token
carries leafCount verbatim). Then checked rather than assumed, on every shown
plant: the forward is re-run with the REAL params, and with the masked vision
streams replaced by Gaussian noise, and the predicted params must match bit for
bit (max|diff| == 0.0, asserted).

Determinism: numpy is seeded before every sample load (load_pointcloud permutes
the 8,192 points with np.random.choice, and the order sets FPS's start) and
torch before every forward (FPS draws its first centroid with torch.randint).
One plant per forward, so a plant's prediction never depends on its batch-mates.

Runs on CPU (no CUDA-only ops), a few seconds per plant plus a ~40 s checkpoint
load:
    python export/export_maize_params.py --out-dir /path/to/review/maize
    python export/export_maize_params.py --out-dir ... --skill-plants 250

--js-global renames the blob's global (default PARAM_DATA, the name the page's
sorghum browser reads). A page that loads this next to the sorghum params.js must
rename one of them, or whichever loads second silently replaces the other.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, maize_dataset_4m, eval/*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from embodied_mae_4m_maize import (N_PARAMS, MAX_LEAVES, LEAF_FIELDS, PLANT_FIELDS,
                                   _LEAF_SCALE, _LEAF_SHIFT, _PLANT_SCALE,
                                   _PLANT_SHIFT, _leaf_to_params, _plant_to_params)
from maize_dataset_4m import MaizeDataset4M
from eval.linear_probe_maize import build_model

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = '/work/mech-ai-scratch/alloy/Maize'
PLANTS = ['plant_8968', 'plant_5483', 'plant_0633', 'plant_11052',
          'plant_1196', 'plant_13733', 'plant_7102', 'plant_6078']

LEAF_OUT = list(LEAF_FIELDS[:12]) + ['waveLPhase']
LEAF_UNITS = {'leafAngle': '°', 'droopiness': '°', 'stemInclinationDeg': '°',
              'leafTwist': '°', 'azJitterDeg': '°', 'waveLPhase': ' rad'}
CIRCULAR = {'waveLPhase'}
PLANT_SHOW = ['leafCount', 'stemInternodeSum', 'stemRadius', 'tasselMatureDroopStrength']
PLANT_ALL = PLANT_SHOW + ['stemShrink']          # stemShrink: skill table only
PLANT_SLOT = {f: i for i, f in enumerate(PLANT_FIELDS)}
L64, S64 = _LEAF_SCALE.astype(np.float64), _LEAF_SHIFT.astype(np.float64)
PL64, PS64 = _PLANT_SCALE.astype(np.float64), _PLANT_SHIFT.astype(np.float64)


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


# ── raw ground truth (native units) ───────────────────────────────────────────

def raw_record(tassel, tiller, leaves):
    """(plant dict, [leaf dict]) from anything with .get -- XML element or dict."""
    plant = {'leafCount': float(len(leaves)),
             'stemRadius': float(tiller.get('radius')),
             'stemShrink': float(tiller.get('stemShrink')),
             'tasselMatureDroopStrength': float(tassel.get('matureDroopStrength')),
             'stemInternodeSum': float(sum(float(l.get('distance')) for l in leaves))}
    out = []
    for i, l in enumerate(leaves):
        d = {k: float(l.get(k)) for k in LEAF_FIELDS[:11]}
        d['leafAzimuthDeg'] = float(l.get('leafAzimuthDeg'))
        d['azJitterDeg'] = float(wrap180(d['leafAzimuthDeg'] - 180.0 * i))
        d['waveLPhase'] = float(l.get('waveLPhase'))
        out.append(d)
    return plant, out


def raw_from_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    tiller = root.find('Tiller')
    return raw_record(root.find('Tassel'), tiller, tiller.findall('./leaves/leaf'))


def decode_tokens(pf):
    """(1+max_leaves, 14) normalised -> plant (5, PLANT_ALL order), leaves (L, 13).

    The same inverse as the model's own text formatters: clip to [0, 1], then
    raw = norm * SCALE - SHIFT; the phase pair decodes by atan2.
    """
    p = np.clip(np.asarray(pf, np.float64), 0.0, 1.0)
    plant_raw = p[0] * PL64 - PS64
    plant = np.array([plant_raw[PLANT_SLOT[f]] for f in PLANT_ALL])
    leaf_raw = p[1:] * L64 - S64
    phase = np.arctan2(leaf_raw[:, 12], leaf_raw[:, 13])
    return plant, np.concatenate([leaf_raw[:, :12], phase[:, None]], axis=1)


# ── data: seeded per item, so worker order cannot change a prediction ─────────

class Seeded(Dataset):
    def __init__(self, ds, idxs, seed):
        self.ds, self.idxs, self.seed = ds, idxs, seed

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        np.random.seed(self.seed)                 # load_pointcloud's permutation
        rgb, depth, pc, pf, tv, name = self.ds[self.idxs[k]]
        folder = self.ds.samples[self.idxs[k]]
        xml = folder / self.ds._xml_names[folder.name]
        return rgb[None], depth[None], pc[None], pf, tv, name, raw_from_xml(xml)


# ── model forward: one source, everything else fully masked ───────────────────

@torch.no_grad()
def predict(model, rgb, depth, pc, params, src, seed):
    """Predicted params (1+max_leaves, 14) from `src` alone. Asserts the mask."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    (lat, mr, md, mp, mt, rr, rd, rp, rt,
     lr_, ld_, lp_, lt_) = model.forward_encoder_select(
        rgb, depth, pc, params, visible={src}, source_mask_ratio=0.0)
    n_vis = {'rgb': lr_, 'depth': ld_, 'pc': lp_, 'text': lt_}
    for name, k in n_vis.items():
        if name == src:
            assert k == model._token_len[name], (name, k)
        else:
            assert k == 0, f'{name} leaked {k} tokens into the encoder'
    assert bool((mt == 1).all()), 'text not fully masked'
    assert lat.shape[1] == 1 + model._token_len[src]
    pred = model.forward_decoder(lat, rr, rd, rp, rt, lr_, ld_, lp_, lt_)[3]
    return pred[0].float().cpu().numpy()


# ── metrics ──────────────────────────────────────────────────────────────────

def abs_err(pred, gt, circular):
    d = np.asarray(pred, np.float64) - np.asarray(gt, np.float64)
    return np.abs(wrap_pi(d)) if circular else np.abs(d)


def circ_mean(x):
    return float(np.arctan2(np.sin(x).mean(), np.cos(x).mean()))


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def skill_rows(fields, gt, pc, rgb, pos_base, val_mean, notes=None, pos_in=None):
    """One row per field. `pos_base` (n, len(fields)) is the per-leaf-index val
    mean for each row's leaf -- the position-only baseline; None for the plant
    token, whose single position makes it identical to the plain mean."""
    rows = []
    for j, f in enumerate(fields):
        c = f in CIRCULAR
        g = gt[:, j]
        m_pc, m_rgb = abs_err(pc[:, j], g, c).mean(), abs_err(rgb[:, j], g, c).mean()
        m_base = abs_err(np.full_like(g, val_mean[f]), g, c).mean()
        m_pos = abs_err(pos_base[:, j], g, c).mean() if pos_base is not None else m_base
        row = {'field': f, 'n': int(len(g)), 'gt_sd': float(g.std()),
               'mae_pc': float(m_pc), 'mae_rgb': float(m_rgb),
               'mae_mean_baseline': float(m_base),
               'skill_pc': float(1 - m_pc / m_base),
               'skill_rgb': float(1 - m_rgb / m_base),
               'mae_position_baseline': float(m_pos),
               'skill_of_position_baseline': float(1 - m_pos / m_base),
               'skill_vs_position_pc': float(1 - m_pc / m_pos),
               'skill_vs_position_rgb': float(1 - m_rgb / m_pos)}
        if pos_in is not None:
            row['mae_position_baseline_insample'] = float(abs_err(pos_in[:, j], g, c).mean())
        for s, P in (('pc', pc), ('rgb', rgb)):
            p = P[:, j]
            row[f'r_{s}'] = None if c else pearson(p, g)
            row[f'pred_sd_over_gt_sd_{s}'] = None if c else float(p.std() / g.std())
            row[f'pred_range_{s}'] = [float(p.min()), float(p.max())]
        row['gt_range'] = [float(g.min()), float(g.max())]
        if notes and f in notes:
            row['note'] = notes[f]
        rows.append(row)
    return rows


def r4(a):
    """4 decimals, or 4 significant figures where that is finer."""
    out = []
    for x in a:
        x = float(x)
        d = 4 if x == 0 else max(4, 3 - int(np.floor(np.log10(abs(x)))))
        out.append(round(x, d) + 0.0)                   # +0.0 turns -0.0 into 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='maize_4m')
    ap.add_argument('--ckpt', default='checkpoints/checkpoint_epoch_600.pth')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--split', default='val')
    ap.add_argument('--view', default='00')
    ap.add_argument('--plants', nargs='+', default=PLANTS)
    ap.add_argument('--skill-plants', type=int, default=0,
                    help='also score N other val plants (seeded random draw), '
                         'so the skill table is not only 8 hand-picked plants')
    ap.add_argument('--skill-seed', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--num-workers', type=int, default=4,
                    help='loader processes; Lustre latency, not CPU, is the cost')
    ap.add_argument('--js-global', default='PARAM_DATA',
                    help='global the blob assigns; the page must not load two '
                         'files that assign the same one')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    assert 'best_model' not in a.ckpt, 'use an explicit checkpoint_epoch_N.pth'
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── val-wide ground truth, from the split's _params.json ─────────────────
    # One 20 MB read instead of 2,250 XML opens (24 min on a loaded Lustre). It
    # is checked equal to the per-folder XML below for every plant opened.
    scores = pd.read_csv(Path(a.data_root) / 'plant_scores.csv')
    val_plants = scores.loc[scores.split == a.split, 'plant'].tolist()
    J = json.loads((Path(a.data_root) / a.split / '_params.json').read_text())
    assert sorted(J) == sorted(val_plants), '_params.json and plant_scores.csv disagree'

    json_raw, rt_err, clipped, pad_max = {}, {}, [], 0.0
    for pl in val_plants:
        prm = J[pl]['params']
        assert len(prm['tillers']) == 1
        tiller = prm['tillers'][0]
        praw, lraw = raw_record(prm['tassel'], tiller, tiller['leaves'])
        json_raw[pl] = (praw, lraw)
        # round trip through the MODEL'S OWN normalisers (they only call .get)
        pf = np.zeros((1 + MAX_LEAVES, N_PARAMS), np.float32)
        pf[0] = _plant_to_params(prm['tassel'], tiller, tiller['leaves'])
        for i, lf in enumerate(tiller['leaves'][:MAX_LEAVES]):
            pf[1 + i] = _leaf_to_params(lf, i)
        pad_max = max(pad_max, float(np.abs(pf[0, 5:]).max()))
        dp, dl = decode_tokens(pf)
        for j, f in enumerate(PLANT_ALL):
            rt_err[f] = max(rt_err.get(f, 0.0), abs(dp[j] - praw[f]))
        for i, lf in enumerate(lraw):
            for j, f in enumerate(LEAF_OUT):
                e = float(abs_err(dl[i, j], lf[f], f in CIRCULAR))
                rt_err[f] = max(rt_err.get(f, 0.0), e)
                if j < 12:
                    u = (lf[f] + S64[j]) / L64[j]
                    if u < 0 or u > 1:
                        clipped.append({'plant': pl, 'leaf_index': i, 'field': f,
                                        'raw': lf[f], 'decoded': float(dl[i, j])})
            az = (180.0 * i + dl[i, 11]) % 360.0
            rt_err['leafAzimuthDeg'] = max(rt_err.get('leafAzimuthDeg', 0.0),
                                           float(abs(wrap180(az - lf['leafAzimuthDeg']))))
    all_leaves = np.array([[lf[f] for f in LEAF_OUT]
                           for pl in val_plants for lf in json_raw[pl][1]])
    all_plants = np.array([[json_raw[pl][0][f] for f in PLANT_ALL] for pl in val_plants])
    val_mean = {f: (circ_mean(all_leaves[:, j]) if f in CIRCULAR
                    else float(all_leaves[:, j].mean())) for j, f in enumerate(LEAF_OUT)}
    val_mean.update({f: float(all_plants[:, j].mean()) for j, f in enumerate(PLANT_ALL)})
    val_sd = {f: float(all_leaves[:, j].std()) for j, f in enumerate(LEAF_OUT)}
    # Position-only baseline: the val mean of each field at each leaf index,
    # LEAVE-ONE-PLANT-OUT -- the deep indices are thin (a handful of val leaves
    # past index 19), so an in-sample mean there would be partly the very leaf it
    # is scored on. Kept as running sums (sin/cos sums for the circular field) so
    # the plant's own leaf can be subtracted.
    all_idx = np.array([i for pl in val_plants for i in range(len(json_raw[pl][1]))])
    pos_count = np.bincount(all_idx, minlength=MAX_LEAVES)
    circ_j = [j for j, f in enumerate(LEAF_OUT) if f in CIRCULAR]
    pos_sum = np.zeros((MAX_LEAVES, len(LEAF_OUT)))
    pos_sin = np.zeros((MAX_LEAVES, len(LEAF_OUT)))
    pos_cos = np.zeros((MAX_LEAVES, len(LEAF_OUT)))
    np.add.at(pos_sum, all_idx, all_leaves)
    np.add.at(pos_sin, all_idx, np.sin(all_leaves))
    np.add.at(pos_cos, all_idx, np.cos(all_leaves))

    def position_baseline(gt_l, loo):
        """(n, F) per-index val mean for a val plant's n leaves; `loo` drops
        the plant's own leaves (the plant must be one of the val plants)."""
        n = len(gt_l)
        k = 1.0 if loo else 0.0
        own = gt_l * k
        cnt = pos_count[:n, None] - k
        b = (pos_sum[:n] - own) / np.maximum(cnt, 1)
        ang = np.arctan2(pos_sin[:n] - np.sin(gt_l) * k, pos_cos[:n] - np.cos(gt_l) * k)
        b[:, circ_j] = ang[:, circ_j]
        lone = (cnt[:, 0] <= 0)                    # no other val leaf at this index
        b[lone] = [val_mean[f] for f in LEAF_OUT]
        return b
    val_sd.update({f: float(all_plants[:, j].std()) for j, f in enumerate(PLANT_ALL)})
    assert not [f for f, s in val_sd.items() if s == 0.0], 'a constant field is displayed'
    shrink_dev = float(np.abs(all_plants[:, 4] - 0.65 * all_plants[:, 2]).max())
    print(f'{len(val_plants)} {a.split} plants / {len(all_leaves)} leaves from '
          f'_params.json  ({time.time()-t0:.0f}s)')

    # ── model + data ──────────────────────────────────────────────────────────
    model, cfg, epoch = build_model(REPO / 'outputs' / a.run, a.ckpt, 'cpu')
    assert cfg['num_points'] == 8192 and model.n_params == N_PARAMS
    print(f'{a.run} @ epoch {epoch}  ({cfg["model_size"]}, '
          f'active={list(model.active_modalities)})  ({time.time()-t0:.0f}s)')
    ds = MaizeDataset4M(a.data_root, split=a.split, img_size=cfg.get('img_size', 224),
                        num_points=cfg['num_points'], max_leaves=MAX_LEAVES,
                        view_sampling=False)
    name_to_idx = {p.name: i for i, p in enumerate(ds.samples)}

    extra = []
    if a.skill_plants:
        pool = [p for p in val_plants if p not in set(a.plants)]
        rng = np.random.default_rng(a.skill_seed)
        extra = sorted(rng.choice(pool, size=min(a.skill_plants, len(pool)),
                                  replace=False).tolist())
    order = list(a.plants) + extra
    loader = DataLoader(Seeded(ds, [name_to_idx[f'{p}_{a.view}'] for p in order], a.seed),
                        batch_size=None, shuffle=False, num_workers=a.num_workers,
                        persistent_workers=False)

    xml_vs_json = 0.0
    shown = {'gt': [], 'pc': [], 'rgb': [], 'pos': [], 'pos_in': []}
    shown_p = {'gt': [], 'pc': [], 'rgb': []}
    samp = {'gt': [], 'pc': [], 'rgb': [], 'pos': [], 'pos_in': []}
    samp_p = {'gt': [], 'pc': [], 'rgb': []}
    plants_js, leaks, raw_dump = [], {}, {}
    j_ph = LEAF_OUT.index('waveLPhase')
    for k, (rgb, depth, pc, pf, tv, name, (xp, xl)) in enumerate(loader):
        pl = MaizeDataset4M.plant_of(name)
        is_shown = k < len(a.plants)
        n = int(tv.sum()) - 1
        assert n == len(xl) and float(tv[0]) == 1.0

        # the per-folder XML is the source of truth: it must equal _params.json
        jp, jl = json_raw[pl]
        xml_vs_json = max(xml_vs_json, max(abs(xp[f] - jp[f]) for f in PLANT_ALL),
                          max(abs(x[f] - y[f]) for x, y in zip(xl, jl)
                              for f in LEAF_OUT + ['leafAzimuthDeg']))

        zeros = torch.zeros_like(pf)[None]
        pred = {s: predict(model, rgb, depth, pc, zeros, s, a.seed) for s in ('pc', 'rgb')}
        if is_shown:
            g = torch.Generator().manual_seed(123)
            noise = lambda t: torch.randn(t.shape, generator=g)
            leak = {
                # text is fully masked, so real params in place of zeros: no change
                'pc_with_real_params': float(np.abs(predict(
                    model, rgb, depth, pc, pf[None], 'pc', a.seed) - pred['pc']).max()),
                'rgb_with_real_params': float(np.abs(predict(
                    model, rgb, depth, pc, pf[None], 'rgb', a.seed) - pred['rgb']).max()),
                # the masked vision streams replaced by noise: no change
                'pc_with_noise_rgb_depth': float(np.abs(predict(
                    model, noise(rgb), noise(depth), pc, zeros, 'pc', a.seed) - pred['pc']).max()),
                'rgb_with_noise_pc_depth': float(np.abs(predict(
                    model, rgb, noise(depth), noise(pc), zeros, 'rgb', a.seed) - pred['rgb']).max()),
            }
            assert max(leak.values()) == 0.0, (name, leak)
            leaks[name] = leak
            raw_dump[name] = {s: np.round(pred[s][:1 + n], 6).tolist() for s in pred}

        gt_p = np.array([xp[f] for f in PLANT_ALL])
        gt_l = np.array([[lf[f] for f in LEAF_OUT] for lf in xl])
        dec = {}
        for s in ('pc', 'rgb'):
            p_, l_ = decode_tokens(pred[s])
            l_ = l_[:n].copy()
            l_[:, j_ph] = gt_l[:, j_ph] + wrap_pi(l_[:, j_ph] - gt_l[:, j_ph])
            dec[s] = (p_, l_)

        L, P = (shown, shown_p) if is_shown else (samp, samp_p)
        L['gt'].append(gt_l); P['gt'].append(gt_p)
        assert pos_count[:n].min() > 0
        L['pos'].append(position_baseline(gt_l, loo=True))
        L['pos_in'].append(position_baseline(gt_l, loo=False))
        for s in ('pc', 'rgb'):
            L[s].append(dec[s][1]); P[s].append(dec[s][0])

        if is_shown:
            ns = len(PLANT_SHOW)
            plants_js.append({
                'name': name, 'n': n,
                'plant': {'gt': r4(gt_p[:ns]), 'pc': r4(dec['pc'][0][:ns]),
                          'rgb': r4(dec['rgb'][0][:ns])},
                'leaves': [{'i': i + 1, 'gt': r4(gt_l[i]), 'pc': r4(dec['pc'][1][i]),
                            'rgb': r4(dec['rgb'][1][i])} for i in range(n)],
            })
            print(f'  {name}: n={n}  leafCount pc {dec["pc"][0][0]:.2f} rgb '
                  f'{dec["rgb"][0][0]:.2f}  leak max|diff| {max(leak.values()):.1e}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)
        elif (k - len(a.plants)) % 25 == 0:
            print(f'  skill sample {k - len(a.plants) + 1}/{len(extra)}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)

    blob = {'leaf_fields': LEAF_OUT,
            'leaf_units': [LEAF_UNITS.get(f, '') for f in LEAF_OUT],
            'plant_fields': PLANT_SHOW,
            'plants': plants_js}
    js = f'window.{a.js_global} = ' + json.dumps(blob, separators=(',', ':'),
                                                 ensure_ascii=False) + ';\n'
    (out / 'params_maize.js').write_text(js, encoding='utf-8')

    notes = {'stemShrink': f'= 0.65 x stemRadius in every {a.split} plant '
                           f'(max|diff| {shrink_dev:.1e}); not displayed'}
    cat = lambda D: tuple(np.concatenate(D[s]) for s in ('gt', 'pc', 'rgb', 'pos'))
    stk = lambda D: tuple(np.stack(D[s]) for s in ('gt', 'pc', 'rgb'))
    skill = {
        'run': a.run, 'ckpt': a.ckpt, 'epoch': epoch, 'split': a.split,
        'view': a.view, 'seed': a.seed, 'shown_plants': list(a.plants),
        'skill_definition': '1 - MAE / MAE of always predicting the mean over ALL '
                            f'{a.split} leaves (or plants); waveLPhase circular',
        'skill_vs_position_definition':
            '1 - MAE / MAE of predicting, for each leaf, the mean over all OTHER '
            f'{a.split} plants\' leaves at the SAME leaf index (leave-one-plant-out; '
            'circular for waveLPhase). '
            'The decoder knows each token\'s position with no input at all, so '
            'this is the baseline that isolates what the source modality adds. '
            'Plant token: identical to the plain-mean baseline.',
        'position_baseline_val_leaves_per_index': pos_count.tolist(),
        'leaf_shown': skill_rows(LEAF_OUT, *cat(shown), val_mean,
                                 pos_in=np.concatenate(shown['pos_in'])),
        'plant_shown': skill_rows(PLANT_ALL, *stk(shown_p), None, val_mean, notes),
        'val_mean': val_mean, 'val_sd': val_sd,
        'val_counts': {'plants': len(val_plants), 'leaves': int(len(all_leaves))},
        'roundtrip_max_abs_err_all_val': rt_err,
        'values_clipped_by_normalisation_all_val': clipped,
        'plant_pad_slots_max_abs_all_val': pad_max,
        'xml_vs_params_json_max_abs_diff': xml_vs_json,
        'xml_vs_params_json_plants_checked': len(order),
        'dropped': {'plant_token_slots_5_to_13': 'zero padding in every val plant',
                    'stemShrink': notes['stemShrink'],
                    'leaf_sin_cos_waveLPhase': 'merged into one waveLPhase angle (atan2)'},
    }
    if extra:
        skill.update({'sample_plants': extra, 'sample_seed': a.skill_seed,
                      'leaf_sample': skill_rows(LEAF_OUT, *cat(samp), val_mean,
                                                pos_in=np.concatenate(samp['pos_in'])),
                      'plant_sample': skill_rows(PLANT_ALL, *stk(samp_p), None, val_mean,
                                                 notes)})

    (out / 'param_skill_maize.json').write_text(json.dumps(skill, indent=1))
    (out / 'param_skill_maize.js').write_text(
        'window.PARAM_SKILL_MAIZE = ' + json.dumps(skill, separators=(',', ':')) + ';\n')
    (out / 'leak_checks.json').write_text(json.dumps(leaks, indent=1))
    (out / 'raw_pred_norm_maize.json').write_text(json.dumps(raw_dump))

    def show(title, rows):
        print(f'\n{title}')
        print(f'  {"field":26s} {"n":>5s} {"gt_sd":>10s} {"mae_pc":>10s} {"mae_rgb":>10s} '
              f'{"mae_mean":>10s} {"mae_pos":>10s} {"sk_pc":>7s} {"sk_rgb":>7s} '
              f'{"sk_posb":>7s} {"skP_pc":>7s} {"skP_rgb":>7s} {"r_pc":>6s} {"r_rgb":>6s}')
        fr = lambda v: '   n/a' if v is None else f'{v:+6.3f}'
        for r in rows:
            print(f'  {r["field"]:26s} {r["n"]:5d} {r["gt_sd"]:10.5g} {r["mae_pc"]:10.5g} '
                  f'{r["mae_rgb"]:10.5g} {r["mae_mean_baseline"]:10.5g} '
                  f'{r["mae_position_baseline"]:10.5g} '
                  f'{r["skill_pc"]:+7.3f} {r["skill_rgb"]:+7.3f} '
                  f'{r["skill_of_position_baseline"]:+7.3f} '
                  f'{r["skill_vs_position_pc"]:+7.3f} {r["skill_vs_position_rgb"]:+7.3f} '
                  f'{fr(r["r_pc"])} {fr(r["r_rgb"])}')
    show('LEAF fields, shown plants', skill['leaf_shown'])
    show('PLANT fields, shown plants', skill['plant_shown'])
    if extra:
        show(f'LEAF fields, {len(extra)}-plant random sample', skill['leaf_sample'])
        show(f'PLANT fields, {len(extra)}-plant random sample', skill['plant_sample'])
    print('\nround trip max abs err (all val):', {k: f'{v:.2e}' for k, v in rt_err.items()})
    print('clipped by normalisation (all val):', clipped)
    print(f'plant pad slots max {pad_max}  stemShrink-0.65*radius max {shrink_dev:.2e}  '
          f'XML vs _params.json max|diff| {xml_vs_json:.2e} over {len(order)} plants')
    print('val leaves per leaf index (position baseline support):', pos_count.tolist())
    print(f'wrote {out}/params_maize.js as window.{a.js_global} ({len(js)/1024:.1f} KB)'
          f'  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
