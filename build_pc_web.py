"""Collect every point-cloud export into the single PCDATA blob the page inlines.

Inputs:
  vis_gallery/pc_web.json    the original two viewers, in the pre-schema layout
  vis_gallery/pc_*.json      one file per figure, in the shared schema

Output:
  vis_gallery/pc_all.json    {"<host id>": {panels, rows}, ...}

The page carries a <div id="pcv-<key>"> for each key here; the viewer walks
PCDATA and fills whichever hosts it finds, so an export that is missing or was
rejected simply leaves its static plate in place rather than breaking the page.
"""
import argparse, json, pathlib

GREY, GREEN = '#9aa7ad', '#1baf7a'


def from_legacy(src):
    """pc_web.json predates the shared schema: clouds sit directly on the row."""
    out = {}
    panels = [{'key': 'gt', 'label': 'ground truth', 'color': GREY},
              {'key': 'gen', 'label': 'generated from RGB alone', 'color': GREEN}]
    rows = []
    for r in src.get('percentile', []):
        rows.append(dict(label=r['label'], name=r['name'], chamfer=r['chamfer'],
                         clouds={'gt': r['gt'], 'gen': r['gen']},
                         meta=[['plant', r['name']],
                               ['Chamfer', f"{r['chamfer']:.5f}"],
                               ['percentile', f"{r['pct']}th of 2250 test plants"]]))
    if rows:
        out['pct'] = dict(panels=panels, rows=rows)
    rows = []
    for r in src.get('elevation', []):
        rows.append(dict(label=r['label'], name=r['name'], chamfer=r['chamfer'],
                         clouds={'gt': r['gt'], 'gen': r['gen']},
                         meta=[['plant', r['name']],
                               ['camera elevation', f"{r['elev']:+.1f}°"],
                               ['Chamfer', f"{r['chamfer']:.5f}"],
                               ['target', 'identical in every view']]))
    if rows:
        out['elev'] = dict(panels=panels, rows=rows)
    return out


def check(key, d):
    """Reject anything malformed rather than shipping a broken viewer."""
    assert d.get('panels') and d.get('rows'), f'{key}: empty panels/rows'
    keys = [p['key'] for p in d['panels']]
    for p in d['panels']:
        assert set(p) >= {'key', 'label', 'color'}, f'{key}: panel missing fields'
        assert p['color'].startswith('#') and len(p['color']) == 7, f'{key}: bad colour {p["color"]}'
    for i, r in enumerate(d['rows']):
        assert r.get('label') and r.get('name'), f'{key}: row {i} missing label/name'
        for k in keys:
            assert r['clouds'].get(k), f'{key}: row {i} missing cloud "{k}"'
    return len(d['rows']), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='vis_gallery')
    ap.add_argument('--out', default='vis_gallery/pc_all.json')
    ap.add_argument('--skip', default='', help='comma-separated ids to leave out')
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    skip = {s for s in args.skip.split(',') if s}
    all_ = {}

    legacy = d / 'pc_web.json'
    if legacy.is_file():
        all_.update(from_legacy(json.loads(legacy.read_text())))

    for p in sorted(d.glob('pc_*.json')):
        if p.name in ('pc_web.json', 'pc_all.json'):
            continue
        src = json.loads(p.read_text())
        key = src.get('id')
        if not key:
            print(f'  ! {p.name}: no id, skipped'); continue
        if key in skip:
            print(f'  - {p.name}: id "{key}" skipped by request'); continue
        all_[key] = {k: src[k] for k in ('panels', 'rows') if k in src}
        all_[key]['note'] = src.get('note', '')

    total = 0
    for key in sorted(all_):
        try:
            n, k = check(key, all_[key])
        except AssertionError as e:
            print(f'  ! dropping "{key}": {e}'); del all_[key]; continue
        clouds = sum(len(r['clouds']) for r in all_[key]['rows'])
        total += clouds
        print(f'  {key:22s} {n} rows x {k} panels = {clouds} clouds')

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(all_, separators=(',', ':')))
    print(f'wrote {out}  {out.stat().st_size/1024:.0f} KB  '
          f'({len(all_)} viewers, {total} clouds)')


if __name__ == '__main__':
    main()
