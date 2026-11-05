"""Wrap sentences NOT present in the Oct-7 shared draft in \\upd{...} (blue).

Usage: python3 paper/mark_blue.py paper/baseline_2026-10-07.txt <in.tex> <out.tex>

Baseline = pdftotext of Alloy_Field_multimodal_reconstruction.pdf (the shared
2026-10-07 draft at the repo root); \\upd{} is defined in overleaf/preamble.tex.
Skips comments, preamble-ish lines, tabular bodies and display math; those are
reviewed by hand. Matching is on normalized word sequences (numbers dropped).
"""
import re, sys, unicodedata

def norm_tokens(s):
    s = unicodedata.normalize('NFKD', s)
    s = s.replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '-')
    s = re.sub(r'\\(cite[pt]?|ref|eqref|label|todo|oldnum|upd)\*?(\[[^]]*\])?\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?', ' ', s)          # other commands: keep args' text
    s = re.sub(r'\$[^$]*\$', ' ', s)                 # inline math
    s = s.lower()
    toks = re.findall(r'[a-z]+', s)
    return [t for t in toks if len(t) > 1]

def load_baseline(p):
    t = open(p, errors='ignore').read()
    t = t.replace('-\n', '')
    return ' ' + ' '.join(norm_tokens(t)) + ' '

def present(base, toks):
    if not toks: return True
    seq = ' ' + ' '.join(toks) + ' '
    if seq in base: return True
    if len(toks) >= 8:                               # allow small edits: 5-gram vote
        grams = [' '.join(toks[i:i+5]) for i in range(len(toks)-4)]
        hits = sum(1 for g in grams if ' '+g+' ' in base)
        return hits / len(grams) >= 0.8
    return False

SKIP = re.compile(r'^\s*(%|\\(begin|end|label|includegraphics|input|usepackage|newcommand|renewcommand|providecommand|setlength|centering|vspace|hfill|small|scriptsize|footnotesize|toprule|midrule|bottomrule|cmidrule|multicolumn|section|subsection|paragraph|appendix|maketitle|tableofcontents))')

def mark(base, src):
    out, buf = [], []
    env = 0; math = 0
    def flush():
        nonlocal buf
        if not buf: out.extend(buf := []) 
        text = '\n'.join(buf); buf = []
        if not text.strip(): out.append(text); return
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\\$`(])', text)
        res = []
        for p in parts:
            res.append(p if present(base, norm_tokens(p)) else '\\upd{' + p + '}')
        out.append(' '.join(res))
    for ln in src.split('\n'):
        if re.search(r'\\begin\{(tabular|equation|align|verbatim|table)\*?\}', ln): env += 1
        if env or SKIP.match(ln) or not ln.strip():
            flush(); out.append(ln)
        else:
            buf.append(ln)
        if re.search(r'\\end\{(tabular|equation|align|verbatim|table)\*?\}', ln): env = max(0, env-1)
    flush()
    return '\n'.join(out)

if __name__ == '__main__':
    base = load_baseline(sys.argv[1])
    open(sys.argv[3], 'w').write(mark(base, open(sys.argv[2]).read()))
