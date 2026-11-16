"""Write every procedural-model file of the export in BOTH formats (user, 2026-11-16: "for all the folders I need both
xml and yaml files for each").

  sorghum  <name>.yml (EvoEngine spline layout)  -> <name>.xml   same fields; the true file's generator geometry
           (Center / Left / Right Points) is kept as "x y z; x y z; ..." text
  maize    <name>.xml (maize generator XML)       -> <name>.yml   same attributes, as numbers
  E10      truth.json                              -> truth.yml and truth.xml
The header comments of a file (what model wrote it, how unpredicted fields were filled) are carried over.
  python figures/procedural_formats.py [ROOT]    (default: the export folder of export_procedural_models.py)
"""
import sys, pathlib, json, re
import xml.etree.ElementTree as ET
import yaml

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/work/mech-ai/alloydas/embodiedmae_outputs/procedural_models_for_viz')


def num(v):
    try:
        f = float(v); return int(f) if re.fullmatch(r'-?\d+', str(v).strip()) else f
    except (TypeError, ValueError):
        return v


def comments_yml(text):
    return [l[1:].strip() for l in text.splitlines() if l.startswith('#')]


def comments_xml(text):
    return [c.strip() for c in re.findall(r'<!--(.*?)-->', text, re.S)]


def esc(s):
    return s.replace('--', '-')


def pts(a):
    return '; '.join(' '.join(f'{float(x):.6g}' for x in p) for p in a)


def sorghum_yml_to_xml(src, dst):
    text = src.read_text(); d = yaml.load(text, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    out = ['<?xml version="1.0" encoding="UTF-8"?>'] + [f'<!-- {esc(c)} -->' for c in comments_yml(text)]
    out.append(f'<!-- converted from {src.name} (EvoEngine spline layout); angles in degrees, lengths in metres -->')
    out.append('<sorghums>')
    for s in d['Sorghums']:
        attrs = ''.join(f' {k.lower().replace(" ", "_")}="{s[k]}"' for k in ('Instance Index', 'Seed', 'Generator Path') if k in s)
        out.append(f'  <sorghum{attrs}>')
        par = s.get('Parameters', {})
        pa = ''.join(f' {k}="{" ".join(str(x) for x in v) if isinstance(v, list) else v}"' for k, v in par.items())
        out.append(f'    <parameters{pa}/>')
        out.append(f'    <leaves number="{len(s.get("Leaves", []))}">')
        for l in s.get('Leaves', []):
            scal = ''.join(f' {k.lower().replace(" ", "_")}="{v}"' for k, v in l.items() if not isinstance(v, (list, dict)))
            geo = {k: v for k, v in l.items() if isinstance(v, list)}
            if not geo:
                out.append(f'      <leaf{scal}/>'); continue
            out.append(f'      <leaf{scal}>')
            for k, v in geo.items():
                tag = k.lower().replace(' ', '_')
                out.append(f'        <{tag}>{pts(v)}</{tag}>')
            out.append('      </leaf>')
        out += ['    </leaves>', '  </sorghum>']
    out.append('</sorghums>')
    dst.write_text('\n'.join(out) + '\n')


def maize_xml_to_yml(src, dst):
    text = src.read_text(); root = ET.fromstring(text)
    d = {'plant': {k: num(v) for k, v in root.attrib.items()}}
    tl = root.find('Tiller'); lv = tl.find('leaves') if tl is not None else None
    if lv is not None:
        d['plant']['leaves'] = [{k: num(v) for k, v in l.attrib.items()} for l in lv.findall('leaf')]
        if 'number' in lv.attrib: d['plant']['leaf_count'] = num(lv.attrib['number'])
    for el in root:                                           # anything else the generator wrote (e.g. a Tassel element)
        if el.tag != 'Tiller': d['plant'][el.tag] = {k: num(v) for k, v in el.attrib.items()}
    head = [f'# {c}' for c in comments_xml(text)] + [f'# converted from {src.name} (maize generator XML); lengths in metres, angles in degrees']
    dst.write_text('\n'.join(head) + '\n' + yaml.safe_dump(d, sort_keys=False, default_flow_style=None, width=200))


def truth_json(src):
    d = json.loads(src.read_text())
    src.with_suffix('.yml').write_text('# true values of a real plant (no generator file exists); converted from truth.json\n'
                                       + yaml.safe_dump(d, sort_keys=False, width=200))
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<!-- true values of a real plant (no generator file exists); converted from truth.json -->']
    scal = ''.join(f' {k}="{v}"' for k, v in d.items() if not isinstance(v, (list, dict)))
    rows = d.get('leaves_csv') or []
    if not rows:
        out.append(f'<truth{scal}/>')
    else:
        out.append(f'<truth{scal}>'); out.append(f'  <leaves number="{len(rows)}">')
        for i, r in enumerate(rows):
            out.append('    <leaf id="%d"%s/>' % (i, ''.join(f' {k}="{v}"' for k, v in r.items())))
        out += ['  </leaves>', '</truth>']
    src.with_suffix('.xml').write_text('\n'.join(out) + '\n')


def convert(root=ROOT):
    n = 0
    for f in sorted(root.rglob('*')):
        if f.suffix == '.yml' and f.with_suffix('.xml').exists() is False and f.name != 'truth.yml':
            sorghum_yml_to_xml(f, f.with_suffix('.xml')); n += 1
        elif f.suffix == '.xml' and not f.with_suffix('.yml').exists() and f.name != 'truth.xml':
            maize_xml_to_yml(f, f.with_suffix('.yml')); n += 1
        elif f.name == 'truth.json':
            truth_json(f); n += 2
    return n


if __name__ == '__main__':
    print('wrote', convert(), 'converted files under', ROOT)
