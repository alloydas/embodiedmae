"""Sync paper/results.tex and paper/discussion.tex into the Overleaf tree (paper/overleaf/sec/4-6).

Maps the repo drafts' cite keys to CVPR.bib's, adds the Overleaf label aliases and Figures/ paths.
Run: python paper/sync_overleaf.py, then latexmk -pdf main.tex in paper/overleaf."""
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
r=open('results.tex').read()
body=r[r.index('\\section{Experiments}'):]
for a,b in [('sec:results','sec:experiments'),('sec:setup','sec:exp-setup'),('sec:regime1','sec:exp-regime1'),
            ('sec:regime2','sec:exp-regime2'),('sec:regime3','sec:exp-occlusion'),('sec:res-distill','sec:exp-main'),
            ('tab:anysubset','tab:main'),('sec:param-stream','sec:exp-params'),('sec:res-metrics','sec:exp-metrics'),
            ('sec:res-field','sec:exp-real'),('sec:res-repr','sec:exp-repr')]:
    s='\\label{%s}'%a; assert body.count(s)==1,a; body=body.replace(s,s+'\\label{%s}'%b)
body=body.replace('{fig_view_elevation.png}','{Figures/fig_view_elevation.png}').replace('{fig_metrics.pdf}','{Figures/fig_metrics.pdf}')
open(ov+'4_experiments.tex','w').write(hdr('results.tex')+fixrefs(fixcites(body)))
d=open('discussion.tex').read()
i=d.index('\\section{Discussion'); j=d.index('\\section{Conclusion}')
open(ov+'5_discussion.tex','w').write(hdr('discussion.tex')+fixrefs(fixcites(d[i:j])).rstrip()+'\n')
open(ov+'6_conclution.tex','w').write(hdr('discussion.tex')+fixrefs(fixcites(d[j:])).rstrip()+'\n')
