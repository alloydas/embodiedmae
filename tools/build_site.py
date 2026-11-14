"""Assemble the project page for GitHub Pages.

site/page.html (the page body, also published as the private preview) is wrapped into a full HTML document;
site/assets/ and site/data/ (written by tools/export_site_data.py) are copied beside it.
usage: python tools/build_site.py OUT_DIR [--code-url URL]
  OUT_DIR     e.g. <cropcode checkout>/docs (GitHub Pages: main branch, /docs folder)
  --code-url  where the Code buttons point (the anonymous repository during review). Without it the buttons are shown
              as "Code (soon)". The build refuses to write anything that names the account (double-blind review).
"""
import pathlib, shutil, sys, re, json

SRC = pathlib.Path(__file__).resolve().parents[1] / 'site'
args = sys.argv[1:]
code_url = None
if '--code-url' in args:
    i = args.index('--code-url'); code_url = args[i + 1]; del args[i:i + 2]
OUT = pathlib.Path(args[0]).resolve()
SHIM = """<script>
/* Data as scripts: hosts that sandbox the page (e.g. anonymous.4open.science: CSP 'sandbox allow-scripts', no
   same-origin) block fetch() of the data, but not <script src>. Every data/...json is also written as data/...js,
   which registers itself here; relative .json fetches are answered from those scripts. */
(function () {
  var reg = {}, wait = {}, _fetch = window.fetch ? window.fetch.bind(window) : null;
  window.__pcvData = function (k, v) { reg[k] = v; };
  function copy(v) { return typeof structuredClone === 'function' ? structuredClone(v) : JSON.parse(JSON.stringify(v)); }
  function resp(v) { return { ok: true, status: 200, json: function () { return Promise.resolve(copy(v)); },   // a fresh copy:
                              /* the page's loader deletes fields of what it receives (PCV.load drops meta.b64) */
                              text: function () { return Promise.resolve(JSON.stringify(v)); } }; }
  window.fetch = function (u, o) {
    var s = String(u && u.url ? u.url : u), i = s.indexOf('data/');
    if (i < 0 || !/\\.json(\\?|$)/.test(s) || /^[a-z]+:\\/\\//i.test(s) && s.indexOf(location.host) < 0)
      return _fetch ? _fetch(u, o) : Promise.reject(new Error('no fetch'));
    var k = s.slice(i).split('?')[0];
    if (reg[k]) return Promise.resolve(resp(reg[k]));
    if (!wait[k]) wait[k] = new Promise(function (res, rej) {
      var el = document.createElement('script'); el.src = k.replace(/\\.json$/, '.js');
      el.onload = function () { reg[k] ? res(reg[k]) : rej(new Error('no data in ' + el.src)); };
      el.onerror = function () { rej(new Error('cannot load ' + el.src)); };
      document.head.appendChild(el);
    });
    return wait[k].then(resp);
  };
})();
</script>
"""
IDENT = re.compile(r'alloydas|iastate|/work/mech|mech-ai|claude\.ai', re.I)

body = (SRC / 'page.html').read_text(encoding='utf-8')
head, rest = body.split('<style>', 1)
head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' + SHIM + head + '<style>' + rest.split('</style>', 1)[0]
       + '</style>\n</head>\n<body>\n' + rest.split('</style>', 1)[1] + '\n</body>\n</html>\n')
REPO = 'https://github.com/alloydas/CropCode'
if code_url:
    doc = doc.replace(REPO, code_url)
else:                                        # no public link yet: no account name on the page
    doc = doc.replace(f'<a class="navcode" href="{REPO}">Code ↗</a>', '')
    doc = doc.replace(f'<a class="btn" href="{REPO}">Code</a>', '<a class="btn off" aria-disabled="true">Code (soon)</a>')
    doc = doc.replace(f'<p><a href="{REPO}">Training and testing code on GitHub</a></p>', '')
bad = IDENT.findall(doc)
if bad: sys.exit(f'refusing to build: identifying strings in the page: {sorted(set(bad))}')

OUT.mkdir(parents=True, exist_ok=True)
for d in ('assets', 'data'):
    if (OUT / d).exists(): shutil.rmtree(OUT / d)
    shutil.copytree(SRC / d, OUT / d)
(OUT / 'index.html').write_text(doc, encoding='utf-8')
for f in sorted((OUT / 'data').rglob('*.json')):  # data as scripts (see SHIM), replacing the .json
    k = f.relative_to(OUT).as_posix()
    f.with_suffix('.js').write_text('__pcvData(' + json.dumps(k) + ',' + f.read_text(encoding='utf-8') + ');\n', encoding='utf-8'); f.unlink()
(OUT / '.nojekyll').touch()
for f in OUT.rglob('*'):                       # the data too
    if f.is_file() and f.suffix in ('.json', '.js', '.html') and IDENT.search(f.read_text(encoding='utf-8', errors='ignore')):
        sys.exit(f'refusing to build: identifying string in {f}')
n = sum(1 for f in OUT.rglob('*') if f.is_file() and '.git' not in f.parts)
mb = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and '.git' not in f.parts) / 1e6
print(f'site written to {OUT}: {n} files, {mb:.1f} MB')
