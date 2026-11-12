"""Assemble the project page for GitHub Pages.

site/page.html (the page body, also published as the private preview) is wrapped into a full HTML document;
site/assets/ and site/data/ (written by tools/export_site_data.py) are copied beside it.
usage: python tools/build_site.py OUT_DIR      e.g. a checkout of the gh-pages branch of the code repository
"""
import pathlib, shutil, sys

SRC = pathlib.Path(__file__).resolve().parents[1] / 'site'
OUT = pathlib.Path(sys.argv[1]).resolve()
body = (SRC / 'page.html').read_text(encoding='utf-8')
head, rest = body.split('<style>', 1)
head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' + head + '<style>' + rest.split('</style>', 1)[0]
       + '</style>\n</head>\n<body>\n' + rest.split('</style>', 1)[1] + '\n</body>\n</html>\n')
OUT.mkdir(parents=True, exist_ok=True)
for d in ('assets', 'data'):
    if (OUT / d).exists(): shutil.rmtree(OUT / d)
    shutil.copytree(SRC / d, OUT / d)
(OUT / 'index.html').write_text(doc, encoding='utf-8')
(OUT / '.nojekyll').touch()
n = sum(1 for f in OUT.rglob('*') if f.is_file() and '.git' not in f.parts)
mb = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and '.git' not in f.parts) / 1e6
print(f'site written to {OUT}: {n} files, {mb:.1f} MB')
