"""Sync paper/discussion.tex into the Overleaf tree (paper/overleaf/sec/5-6).

Maps the repo draft's cite keys to CVPR.bib's and adds the Overleaf label aliases.
Sections 0-4 and the supplement are hand-maintained in paper/overleaf/sec/ since
2026-11-05 (the user's rewrites were integrated there); paper/results.tex remains
the numbers archive but no longer generates 4_experiments.tex.
Run: python paper/sync_overleaf.py, then latexmk -pdf main.tex in paper/overleaf.
"""
import os, re, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ov='overleaf/sec/'
keys={'hadadi2025vlm':'hadadi2026vlmmaize','kimara2025maizefield3d':'maizefield3d2025',
      'oquab2023dinov2':'oquab2024dinov2','song2026chamfer':'song2026structural',
      'hinton2015distill':'hinton2015distillation'}
def fixcites(t):
    rep=lambda m: m.group(1)+'{'+','.join(keys.get(k.strip(),k.strip()) for k in m.group(2).split(','))+'}'
    return re.sub(r'(\\cite[pt]?)\{([^}]*)\}',rep,t)
def fixrefs(t):
    for a,b in {'sec:data':'sec:method-data','sec:pretrain':'sec:method-pretrain'}.items():
        t=t.replace('ref{%s}'%a,'ref{%s}'%b)
    return t
sha=subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
hdr=lambda src: f"% Synced from the embodiedmae repo (paper/{src}, commit {sha}).\n% Numbers come from reports/*.csv etc.; edit the repo copy, then re-sync.\n"
d=open('discussion.tex').read()
i=d.index('\\section{Discussion'); j=d.index('\\section{Conclusion}')
open(ov+'5_discussion.tex','w').write(hdr('discussion.tex')+fixrefs(fixcites(d[i:j])).rstrip()+'\n')
open(ov+'6_conclution.tex','w').write(hdr('discussion.tex')+fixrefs(fixcites(d[j:])).rstrip()+'\n')
