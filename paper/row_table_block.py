"""Print one model block of tab:rows (paper/results.tex) from an eval_row.py JSON and the
within-model bootstrap of eval/row_pair_bootstrap.py --ft X --iso X.
usage: python paper/row_table_block.py <eval .json> <within .csv> "<block title>"
"""
import sys, csv, json
J = json.load(open(sys.argv[1])); B = {(r['comparison'], r['source'], r['metric']): r for r in csv.DictReader(open(sys.argv[2]))}
M = J['by_mode_source']; rec = J.get('recipe_scored', False)
def d(src, met, scale, nd):
    r = B[('ft_row_vs_iso_isolated', src, met)]
    v, lo, hi = (float(r[k]) * scale for k in ('delta', 'lo', 'hi'))
    return f"${v:+.{nd}f}$\\,{{\\tiny[${lo:+.{nd}f}$, ${hi:+.{nd}f}$]}}"
print(f"    \\multicolumn{{8}}{{l}}{{\\emph{{{sys.argv[3]}}}}} \\\\")
for src, lab in [('rgb', 'RGB'), ('depth', 'depth'), ('pc', 'point cloud'), ('all', 'all three')]:
    iso, row = M[f'isolated/{src}'], M[f'row/{src}']
    ch = f"{iso['chamfer_gt_m2']['mean']*1e4:.1f} & {row['chamfer_gt_m2']['mean']*1e4:.1f} & {d(src, 'chamfer_gt_m2', 1e4, 1)}"
    lm = (f"{iso['leaf_mae']['mean']:.3f} & {row['leaf_mae']['mean']:.3f} & {d(src, 'leaf_mae', 1, 3)}") if rec else "-- & -- & --"
    print(f"    & {lab:<11} & {ch} & {lm} \\\\")
