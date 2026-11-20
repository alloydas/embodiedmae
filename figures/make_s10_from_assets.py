"""Rebuild Fig. S10 (fig:supp-e10-maize) in the S11 layout (figures/_s10_layout.py) from what the paper already holds: the
tile PNGs in paper/overleaf/Figures/fig_e10_gallery_maize/ and the numbers printed in fig_e10_gallery_maize_tikz.tex.
For machines without the Nova data; on Nova, `python figures/make_supp_tikz.py s10` builds the same layout from the
--proc pickle. Reads every input before writing, since the output goes to the same folder.
  python figures/make_s10_from_assets.py
"""
import sys, re, pathlib
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R / 'figures')]
import _s10_layout as L10

FIG = R / 'paper/overleaf/Figures'; NAME = 'fig_e10_gallery_maize'
PCT = ['10th', '30th', '50th', '70th', '90th', '98th']


def read_old():
    tex = (FIG / f'{NAME}_tikz.tex').read_text()
    if 'percentile' in tex: sys.exit('S10 is already in the S11 layout; nothing to convert')
    nodes = [(float(x), float(y), s) for x, y, s in re.findall(r'\] at \(([0-9.]+),([0-9.]+)\) \{([^{}\\]*(?:\\\\[^{}\\]*)?)\}', tex)]
    stops = {}
    for L in ('parallel', 'perpendicular'):
        out = []
        for i in range(6):
            img = f'm_{L}_{i}_'; d = FIG / NAME
            out.append(dict(img_png=(d / f'{img}image.png').read_bytes(), truth_png=(d / f'{img}truth.png').read_bytes(),
                            pred_png=[(d / f'{img}pred_0.png').read_bytes(), (d / f'{img}pred_1.png').read_bytes()], pct=PCT[i]))
        stops[L] = out
    # numbers: the truth-height and CD labels sit at the bottom of their tiles; the left group's x is the smaller
    hts = sorted([(x, -y, s) for x, y, s in nodes if s.startswith('height ')])
    cds = sorted([(x, -y, s) for x, y, s in nodes if s.startswith('CD = ')])
    assert len(hts) == 12 and len(cds) == 24, (len(hts), len(cds))
    xs_h = sorted({round(x, 3) for x, _, _ in hts}); xs_c = sorted({round(x, 3) for x, _, _ in cds})
    for gi, L in enumerate(('parallel', 'perpendicular')):
        col_h = sorted([(ny, s) for x, ny, s in hts if round(x, 3) == xs_h[gi]])
        c0 = sorted([(ny, s) for x, ny, s in cds if round(x, 3) == xs_c[2 * gi]])
        c1 = sorted([(ny, s) for x, ny, s in cds if round(x, 3) == xs_c[2 * gi + 1]])
        for i in range(6):
            st = stops[L][i]; st['height'] = float(re.search(r'([0-9.]+) m', col_h[i][1]).group(1))
            st['ch'], st['hp'] = [], []
            for c in (c0, c1):
                cd, hp = re.fullmatch(r'CD = ([0-9.]+)\\\\height ([0-9.]+) m', c[i][1]).groups()
                st['ch'].append(float(cd)); st['hp'].append(float(hp))
    return stops


if __name__ == '__main__':
    stops = read_old()
    for L in stops:
        print(L, [(s['pct'], s['height'], s['ch'], s['hp']) for s in stops[L]])
    S = L10.layout(stops)
    S.render_tikz(FIG / f'{NAME}_tikz.tex', FIG / NAME, f'Figures/{NAME}')
    print('wrote', FIG / f'{NAME}_tikz.tex')
