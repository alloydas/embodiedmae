#!/usr/bin/env python3
"""Maize, occluded TEST scenes: (1) how well each distilled student predicts the
procedural parameters with none given, per field in physical units, against
always predicting the training average; (2) which parameters the rebuild uses:
the best parameter-conditioned model (mixed + QAL + Sinkhorn, 400 ep) given the
true parameters with one GROUP replaced by the training average, given none, or
given the D2 student's PREDICTED parameters (the two-step readout).

Every condition sees the same scenes and the same masks (torch re-seeded per
batch: pass-1 parameter prediction 40000 + b, every reconstruction 30000 + b).
In the two-step readout the number of leaf tokens comes from the student's own
predicted leaf count, never from the truth.

    python eval/param_analysis_maize.py            # -> reports/param_analysis_maize.json
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'eval'))

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import eval_test_scene as E
import embodied_mae_4m_maize as M4
from maize_dataset_4m import MaizeDataset4M
from occlusion_scene import SceneConfig, compose, to_world

ROOT = _REPO / 'outputs'
BEST = ('maize_scene_mix_sink400_s1', 400)
STUDENTS = {'D2 distill + mixed': ('maize_scene_distill_mix_s1', 60),
            'D1 distill + random': ('maize_scene_distill_rand_s1', 60),
            'D0 hide only + mixed': ('maize_scene_phide_mix_s1', 60),
            # Alloy's all-source distillation on CLEAN single plants (never saw a neighbour),
            # started from maize_4m (params inside the Dirichlet budget); last checkpoint
            'Alloy distill (clean)': ('/work/mech-ai-scratch/alloy/embodiedmae/outputs/maize_distill_all', 58),
            # any-to-any: params inside the Dirichlet budget on half the steps, fully hidden on the
            # other half; mixed + QAL + Sinkhorn; started from maize_4m epoch 600; 200 epochs
            'Any-to-any (pretrained)': ('maize_scene_anyparam_pre_s1', 200)}
# (name, unit, factor from the generator's raw value)
PLANT = [('leaf count', 'leaves', 1.0), ('stem radius', 'mm', 1000.0), ('stem shrink', '', 1.0),
         ('tassel droop', '', 1.0), ('stem height', 'cm', 100.0)]
LEAF = [('leaf position on stem', 'cm', 100.0), ('leaf length', 'cm', 100.0), ('leaf width', 'mm', 1000.0),
        ('leaf angle', 'deg', 1.0), ('droopiness', 'gen.', 1.0), ('stem inclination', 'deg', 1.0),
        ('width taper', '', 1.0), ('leaf twist', 'deg', 1.0), ('leaf curl', '', 1.0), ('wave amplitude', 'mm', 1000.0),
        ('wave frequency', '', 1.0), ('azimuth jitter', 'deg', 1.0), ('wave phase sin', '', 1.0), ('wave phase cos', '', 1.0)]
# field groups: (plant-token field indices, leaf-token field indices)
GROUPS = {'size': ([0, 1, 4], [1, 2]), 'pose': ([], [0, 3, 4, 5]),
          'fine shape': ([2, 3], [6, 7, 8, 9, 10]), 'not visible': ([], [11, 12, 13])}


def raw(pf):
    """normalised (B, 29, 14) -> generator units: plant token fields 0-4, leaf tokens 0-13."""
    p = pf.clamp(0, 1)
    pl = p[:, 0, :5] * torch.as_tensor(M4._PLANT_SCALE[:5], device=p.device) - torch.as_tensor(M4._PLANT_SHIFT[:5], device=p.device)
    lf = p[:, 1:] * torch.as_tensor(M4._LEAF_SCALE, device=p.device) - torch.as_tensor(M4._LEAF_SHIFT, device=p.device)
    return pl, lf


def load(run, ep, device):
    d = Path(run) if str(run).startswith('/') else ROOT / run
    rc = json.loads((d / 'config.json').read_text())
    m = E.build_model('maize', rc).to(device)
    # mmap: only the model weights are read (the file is mostly optimiser state; /work can be slow)
    ck = torch.load(d / 'checkpoints' / f'checkpoint_epoch_{ep}.pth', map_location='cpu', weights_only=False, mmap=True)
    m.load_state_dict({k.replace('module.', '', 1): v for k, v in ck['model_state_dict'].items()})
    m.pc_sinkhorn_weight = 0.0
    return m.eval(), rc


def slot_means(rc, n=1000):
    """Training-set average of every (token slot, field), normalised; from view-0 folders."""
    ds = MaizeDataset4M(rc['data_root'], split='train', view_sampling=True, deterministic_view=True, max_leaves=rc['max_leaves'])
    folders = [f for f in ds.samples if f.name.endswith('_00')][:n]
    P, V = [], []
    for f in folders:
        tv, pf = M4.load_spline_params(f / ds._xml_names[f.name], rc['max_leaves'])
        P.append(np.asarray(pf, np.float32)); V.append(np.asarray(tv, np.float32))
    P, V = np.stack(P), np.stack(V)
    w = V[..., None]
    mean = (P * w).sum(0) / np.maximum(w.sum(0), 1)
    pooled = (P[:, 1:] * w[:, 1:]).sum((0, 1)) / np.maximum(w[:, 1:].sum((0, 1)), 1)
    empty = w.sum(0)[:, 0] == 0
    mean[1:][empty[1:]] = pooled
    return torch.as_tensor(mean)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-batches', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    best, rc = load(*BEST, dev)
    studs = {k: load(r, e, dev)[0] for k, (r, e) in STUDENTS.items()}
    mean = slot_means(rc).to(dev)
    print(f'[{time.time() - t0:.0f}s] models + training averages ready', flush=True)
    scene = SceneConfig.from_dict(rc['occlusion_scene']).for_validation()
    ds = MaizeDataset4M(rc['data_root'], img_size=rc['img_size'], num_points=rc['num_points'], split='test', max_leaves=rc['max_leaves'],
                        view_sampling=rc['view_sampling'], deterministic_view=True, return_pose=True)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, generator=torch.Generator().manual_seed(args.seed))

    perr = {k: {'plant': [], 'leaf': [], 'count': []} for k in list(STUDENTS) + ['training average']}
    conds = {}
    def add(name, s):
        d = conds.setdefault(name, {})
        for k in ('cd_mm', 'f110mm', 'r10mm', 'p10mm', 'f10.01'):
            d.setdefault(k, []).append(s[k])

    def summarize(final):
        out = {'fields': {}, 'leaf_count': {}, 'conditions': {}}
        for name, d in perr.items():
            pl, lf = torch.cat(d['plant']), torch.cat(d['leaf'])
            out['fields'][name] = {**{f'{n} ({u})' if u else n: float(pl[:, i].mean() * fac) for i, (n, u, fac) in enumerate(PLANT)},
                                   **{f'{n} ({u})' if u else n: float(lf[:, i].mean() * fac) for i, (n, u, fac) in enumerate(LEAF)}}
            c = torch.cat(d['count']); out['leaf_count'][name] = {'mae_leaves': float(c.mean()), 'exact': float((c == 0).float().mean())}
        for name, d in conds.items():
            out['conditions'][name] = {k: float(torch.cat([v.reshape(-1) for v in vs]).mean()) for k, vs in d.items()}
        dst = _REPO / 'reports' / ('param_analysis_maize.json' if final else 'param_analysis_maize.partial.json')
        dst.write_text(json.dumps(out, indent=1))
        base = out['fields']['training average']
        print('\nPARAMETER PREDICTION (occluded test, no parameters given): mean abs error; skill = 1 - error / training-average error')
        print(f"{'field':32s}" + ''.join(f'{k[:20]:>22s}' for k in out['fields']))
        for f in base:
            print(f'{f:32s}' + ''.join(f"{out['fields'][k][f]:12.3f} ({1 - out['fields'][k][f] / max(base[f], 1e-9):+.0%})" for k in out['fields']))
        print('leaf count exact:', {k: round(v['exact'], 3) for k, v in out['leaf_count'].items()})
        print('\nREBUILD CONDITIONS (best model unless named): error mm / F1@10mm / R@10mm / P@10mm / F1@0.01')
        for k, v in out['conditions'].items():
            print(f"{k:32s} {v['cd_mm']:6.1f}  {v['f110mm']:.3f}  {v['r10mm']:.3f}  {v['p10mm']:.3f}  {v['f10.01']:.3f}")
        print(f'wrote {dst}')


    for bi, b in enumerate(dl):
        if args.max_batches and bi >= args.max_batches:
            break
        if bi == 2:
            summarize(False)          # early full summary: an output bug shows up in minutes
        rgb, depth, pc, pf, tv = (t.to(dev) for t in b[:5])
        pn, c2w, nf = b[6].to(dev), b[7].to(dev), b[8].to(dev)
        sc = compose(rgb, depth, pc, pn, c2w, scene, patch_size=16, near_far=nf,
                     generator=torch.Generator().manual_seed(args.seed * 1_000_003 + bi))
        _, world = to_world(pc, pn, c2w)
        r = torch.hypot(world[..., 0], world[..., 2])
        band = torch.clamp((3 * r / r.max(dim=1, keepdim=True).values.clamp(min=1e-6)).long(), max=2)
        x = (sc['rgb'], sc['depth'], sc['pc'])
        kw = {'targets': {'rgb': rgb, 'depth': depth, 'pc': pc}, 'loss_tokens': sc['loss_tokens']}
        B = pc.shape[0]
        leafv = tv[:, 1:] > 0

        def recon(model, p_in, tv_in, tmr):
            model.text_mask_ratio = tmr
            torch.manual_seed(30000 + bi)
            return model(*x, p_in, tv_in, mask_ratio=0.8, **kw)[2][2]

        # (1) parameter prediction, no parameters given
        t_pl, t_lf = raw(pf)
        preds = {}
        for name, s in studs.items():
            s.text_mask_ratio = 1.0
            torch.manual_seed(40000 + bi)
            pp = s(*x, pf, tv, mask_ratio=0.8, **kw)[2][3]
            preds[name] = pp
        preds['training average'] = mean.expand(B, -1, -1)
        for name, pp in preds.items():
            p_pl, p_lf = raw(pp)
            perr[name]['plant'].append((p_pl - t_pl).abs().cpu())
            perr[name]['leaf'].append(((p_lf - t_lf).abs())[leafv].cpu())
            perr[name]['count'].append((p_pl[:, 0].round().clamp(1, 28) - t_pl[:, 0].round()).abs().cpu())

        # (2) what the rebuild uses
        add('true parameters', E.scores(recon(best, pf, tv, 0.0), pc, band, pn[:, 3]))
        vmask = tv[..., None] > 0
        for g, (pi, li) in GROUPS.items():
            q = pf.clone()
            q[:, 0, pi] = mean[0, pi]
            q[:, 1:, li] = torch.where(vmask[:, 1:], mean[1:, li].expand(B, -1, -1), q[:, 1:, li])
            add(f'{g} replaced by average', E.scores(recon(best, q, tv, 0.0), pc, band, pn[:, 3]))
        q = torch.where(vmask, mean.expand(B, -1, -1), pf)
        add('all replaced by average', E.scores(recon(best, q, tv, 0.0), pc, band, pn[:, 3]))
        add('no parameters', E.scores(recon(best, pf, tv, 1.0), pc, band, pn[:, 3]))
        # two-step readout: D2's predicted parameters, leaf tokens from ITS predicted leaf count
        def as_input(pp):
            # predicted params as conditioning; leaf tokens from the PREDICTED leaf count
            n_hat = (pp[:, 0, 0].clamp(0, 1) * float(M4._PLANT_SCALE[0])).round().clamp(1, 28).long()
            tv_h = torch.zeros_like(tv); tv_h[:, 0] = 1
            tv_h[:, 1:] = (torch.arange(28, device=dev)[None] < n_hat[:, None]).float()
            return pp * tv_h[..., None], tv_h
        ph, tv_hat = as_input(preds['D2 distill + mixed'].clone())
        d2 = studs['D2 distill + mixed']
        add('D2 predicted -> best model', E.scores(recon(best, ph, tv_hat, 0.0), pc, band, pn[:, 3]))
        add('D2 alone, no parameters', E.scores(recon(d2, pf, tv, 1.0), pc, band, pn[:, 3]))
        add('D2 predicted -> D2', E.scores(recon(d2, ph, tv_hat, 0.0), pc, band, pn[:, 3]))
        al = studs['Alloy distill (clean)']
        add('Alloy alone, no parameters', E.scores(recon(al, pf, tv, 1.0), pc, band, pn[:, 3]))
        pa, tv_a = as_input(preds['Alloy distill (clean)'].clone())
        add('Alloy predicted -> best model', E.scores(recon(best, pa, tv_a, 0.0), pc, band, pn[:, 3]))
        a2 = studs['Any-to-any (pretrained)']
        add('A2A alone, no parameters', E.scores(recon(a2, pf, tv, 1.0), pc, band, pn[:, 3]))
        add('A2A + true parameters (in budget)', E.scores(recon(a2, pf, tv, None), pc, band, pn[:, 3]))
        pq, tv_q = as_input(preds['Any-to-any (pretrained)'].clone())
        add('A2A predicted -> A2A (in budget)', E.scores(recon(a2, pq, tv_q, None), pc, band, pn[:, 3]))
        add('A2A predicted -> A2A (all visible)', E.scores(recon(a2, pq, tv_q, 0.0), pc, band, pn[:, 3]))
        add('A2A predicted -> best model', E.scores(recon(best, pq, tv_q, 0.0), pc, band, pn[:, 3]))
        if bi % 20 == 0:
            print(f'[{time.time() - t0:.0f}s] batch {bi}', flush=True)


    summarize(True)


if __name__ == '__main__':
    main()
