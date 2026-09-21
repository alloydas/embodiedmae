"""Collate the eval_rgb2pc_quant.py reports into one comparison table.

Reads reports/quant_<label>_<split>.json and prints, per checkpoint:
  * symmetric Chamfer with its two directions split out
      pred->gt rises when the model puts points where no plant is (accuracy)
      gt->pred rises when the model misses parts of the plant (completeness)
  * skill_vs_mean_plant -- 1 - model/gt_vs_gt, i.e. how much better than
    handing back some other sorghum. 0.0 means the generation carries no
    plant-specific information at all.
  * the val->test gap, which is the quantity the val numbers could not report:
    val is also the split best_model.pth was selected on.
"""
# Repo root on sys.path: this script lives one level down but imports the
# top-level modules (embodied_mae*, sorghum_dataset*, train_*).
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import json, pathlib, argparse


def load(d):
    out = {}
    for p in sorted(pathlib.Path(d).glob('quant_*.json')):
        r = json.loads(p.read_text())
        out[(r['label'], r['split'])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', default='reports')
    ap.add_argument('--order', default='teacher_undistilled,smr00_full,smr50_seed1,smr50_seed2,generalist_depth')
    args = ap.parse_args()

    R = load(args.reports)
    if not R:
        raise SystemExit(f'no quant_*.json under {args.reports}/')
    labels = [l for l in args.order.split(',') if any(k[0] == l for k in R)]
    labels += sorted({k[0] for k in R} - set(labels))

    for split in ('test', 'val'):
        rows = [(l, R[(l, split)]) for l in labels if (l, split) in R]
        if not rows:
            continue
        print(f'\n=== {split.upper()} split ' + '=' * 62)
        print(f'{"label":22s} {"src":6s} {"ep":>4s} {"n":>5s} '
              f'{"sym":>10s} {"pred>gt":>9s} {"gt>pred":>9s} {"floor":>10s} {"skill":>7s}')
        for label, r in rows:
            s = r['summary']
            print(f'{label:22s} {",".join(r["source"]):6s} {r["epoch"]:4d} {r["n_plants"]:5d} '
                  f'{s["model"]["symmetric"]["mean"]:10.6f} '
                  f'{s["model"]["pred_to_gt"]["mean"]:9.6f} '
                  f'{s["model"]["gt_to_pred"]["mean"]:9.6f} '
                  f'{s["gt_vs_gt"]["symmetric"]["mean"]:10.5f} '
                  f'{r["skill_vs_mean_plant"]*100:6.2f}%')

    both = [l for l in labels if (l, 'test') in R and (l, 'val') in R]
    if both:
        print('\n=== val -> test ' + '=' * 62)
        print(f'{"label":22s} {"val sym":>11s} {"test sym":>11s} {"gap":>9s}')
        for l in both:
            v = R[(l, 'val')]['summary']['model']['symmetric']['mean']
            t = R[(l, 'test')]['summary']['model']['symmetric']['mean']
            print(f'{l:22s} {v:11.6f} {t:11.6f} {(t-v)/v*100:+8.2f}%')

    tt = [(l, R[(l, 'test')]['summary']['model']['symmetric']['mean']) for l in labels if (l, 'test') in R]
    rgb = [(l, m) for l, m in tt if R[(l, 'test')]['source'] == ['rgb']]
    if len(rgb) > 1:
        base = dict(rgb).get('smr00_full')
        if base:
            print('\n=== test-split effect of source masking (rgb source) ' + '=' * 26)
            for l, m in rgb:
                if l == 'smr00_full':
                    continue
                print(f'{l:22s} {(m-base)/base*100:+8.2f}% vs smr00_full')


if __name__ == '__main__':
    main()
