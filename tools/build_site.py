"""Assemble the project page for GitHub Pages.

site/page.html (the page body, also published as the private preview) is wrapped into a full HTML document;
site/assets/ and site/data/ (written by tools/export_site_data.py) are copied beside it.
usage: python tools/build_site.py OUT_DIR [--code-url URL]
  OUT_DIR     e.g. <cropcode checkout>/docs (GitHub Pages: main branch, /docs folder)
  --code-url  where the Code buttons point (the anonymous repository during review). Without it the buttons are shown
              as "Code (soon)". The build refuses to write anything that names the account (double-blind review).
"""
import pathlib, shutil, sys, re

SRC = pathlib.Path(__file__).resolve().parents[1] / 'site'
args = sys.argv[1:]
code_url = None
if '--code-url' in args:
    i = args.index('--code-url'); code_url = args[i + 1]; del args[i:i + 2]
OUT = pathlib.Path(args[0]).resolve()
IDENT = re.compile(r'alloydas|iastate|/work/mech|mech-ai|claude\.ai', re.I)

body = (SRC / 'page.html').read_text(encoding='utf-8')
head, rest = body.split('<style>', 1)
head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' + head + '<style>' + rest.split('</style>', 1)[0]
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
(OUT / '.nojekyll').touch()
for f in OUT.rglob('*'):                       # the data too
    if f.is_file() and f.suffix in ('.json', '.js', '.html') and IDENT.search(f.read_text(encoding='utf-8', errors='ignore')):
        sys.exit(f'refusing to build: identifying string in {f}')
n = sum(1 for f in OUT.rglob('*') if f.is_file() and '.git' not in f.parts)
mb = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and '.git' not in f.parts) / 1e6
print(f'site written to {OUT}: {n} files, {mb:.1f} MB')
