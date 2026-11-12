#!/usr/bin/env python3
r"""Regenerate main_clean.tex from main.tex by unwrapping the revision markers.

\upd{...} / \updtwo{...} / \updthree{...} are removed, their contents kept
(balanced-brace aware, so nested macros survive). \todo{...} is kept as-is if
present, so a pending claim can never silently become clean text.
Run from paper/overleaf/:  python ../strip_marks.py
"""
import pathlib, re, sys

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'main.tex')
DST = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else 'main_clean.tex')
MACROS = ('\\updtwo{', '\\updthree{', '\\upd{')

txt = SRC.read_text()
changed = True
while changed:
    changed = False
    for mac in MACROS:
        i = txt.find(mac)
        while i != -1:
            depth, j = 1, i + len(mac)
            while depth and j < len(txt):
                if txt[j] == '{' and txt[j-1] != '\\': depth += 1
                elif txt[j] == '}' and txt[j-1] != '\\': depth -= 1
                j += 1
            txt = txt[:i] + txt[i+len(mac):j-1] + txt[j:]
            changed = True
            i = txt.find(mac)
# drop the marker definitions themselves if they define colors only in main
assert '\\upd{' not in txt and '\\updtwo{' not in txt, 'marker left behind'
# collapse whitespace-only lines left inside display math (blank line = \par there)
def _fix_eq(m):
    return re.sub(r'\n[ \t]*\n', '\n', m.group(0))
txt = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', _fix_eq, txt, flags=re.S)
DST.write_text(txt)
n_todo = txt.count('\\todo{')
print(f'wrote {DST} ({len(txt)} bytes); todo markers kept: {n_todo}')
