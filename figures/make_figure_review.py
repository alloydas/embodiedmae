"""Review page: every paper figure (paper/figs/fig_*.png) with the caption from the Overleaf section that
places it, the script that makes it, a status line and a note on what to check, so the figures can be
verified and corrected in one place. Writes an Artifact page body (PNGs embedded, downscaled to 1400 px).
usage: python figures/make_figure_review.py --out <file.html>
"""
import argparse, base64, io, pathlib, re, subprocess, time
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
FIGS = REPO / 'paper' / 'figs'; SEC = REPO / 'paper' / 'overleaf' / 'sec'
SCRIPTS = {'fig_dataset': 'figures/make_dataset_fig.py', 'fig_dataset_stats': 'figures/make_dataset_stats_fig.py', 'fig_any2any': 'figures/make_any2any_fig.py',
           'fig_qual': 'figures/make_qual_fig.py', 'fig_e5_views': 'figures/make_e5_view_fig.py', 'fig_e9_tsne': 'figures/make_e9_fig.py',
           'fig_leafcount': 'figures/make_leafcount_fig.py', 'fig_rows_examples': 'figures/make_row_gallery.py', 'fig_e2_streams': 'figures/make_paper_figs_v2.py',
           'fig_scaling': 'figures/make_paper_figs_v2.py', 'fig_fscore': 'figures/make_paper_figs_v2.py', 'fig_partial_clouds': 'figures/make_paper_figs_v2.py',
           'fig_param_controls': 'figures/make_paper_figs_v2.py', 'fig_rows': 'figures/make_paper_figs_v2.py', 'fig_metrics': 'figures/make_metric_fig.py'}
NOTES = {
    'fig_dataset': 'New. Meant for the dataset section (Sec. 3 or the start of Sec. 4). Check: the recipe glyph is a schematic drawn from the recipe fields (sorghum: insertion height, length, branching angle, width, side from the roll angle, a fixed arc; maize: internode distance, length, width, leaf angle, droop, side from the azimuth), not the generator mesh. Panel b sorts the maize cameras by elevation. Panel c marks neighbour pixels red and hidden target pixels orange from the two depth maps. Tell me which plants to show if these are poor examples.',
    'fig_dataset_stats': 'New. Supplement candidate (or main paper beside fig_dataset). Panel a picks eight val plants at leaf-count quantiles, photographed near the horizon. Panel c shows the sorghum height / leaf-count rule (r = 0.99) against maize (r = 0.20). Panel d is from the row evaluations (median and IQR over test plants).',
    'fig_any2any': 'New. The distilled generalist on one val plant per species: each row gets one stream (framed in orange) and generates the other three. The recipe-only row is weak in both species (Chamfer 5e-3 and 14e-3) and is shown as is; say if you would rather drop that row or show a different plant.',
    'fig_qual': 'As committed. Photograph to cloud before and after distillation, with leaf lengths.',
    'fig_rows_examples': 'As committed. Three test plants per species at yaw 0 and 90.',
}
CANDIDATES = [
    ('Ten-camera ladder', 'one sorghum plant, the photograph of every view and the cloud generated from it, Chamfer under each: the natural experiment that difficulty is set by elevation (eval/eval_views_one_plant.py already produces the data).'),
    ('Partial-cloud gallery', 'one plant, 1 / 3 / 10 cameras, the model output under each: the static twin of the interactive viewer on the ledger.'),
    ('Teaser (page 1)', 'one encoder, any sensor in, every stream out, both species, drawn from fig_any2any with the architecture in the middle.'),
    ('Recipe accuracy per leaf', 'predicted against true leaf length per leaf slot, coloured by species, for the generalist from a photograph.'),
]


def strip_tex(t):
    for _ in range(6):
        t = re.sub(r'\\(upd|textbf|emph|texttt|mathrm|text)\{([^{}]*)\}', r'\2', t)
    t = re.sub(r'\\todo\{([^{}]*)\}', r'[TODO: \1]', t)
    t = re.sub(r'\\(ref|cref|eqref)\{[^}]*\}', '(ref)', t); t = re.sub(r'~?\\cite[pt]?\{[^}]*\}', '', t)
    t = t.replace('\\method{}', 'CropCode').replace('\\CropCode{}', 'CropCode').replace('\\FieldProc{}', 'FieldProc').replace('\\dataset{}', 'FieldProc')
    t = re.sub(r'\\[a-zA-Z]+\*?', '', t); t = t.replace('{', '').replace('}', '').replace('~', ' ').replace('\\,', ' ').replace('\\%', '%').replace('$', '')
    return re.sub(r'\s+', ' ', t).strip()


def captions():
    out = {}
    for f in sorted(SEC.glob('*.tex')) + [REPO / 'paper' / 'overleaf' / 'main.tex']:
        s = f.read_text()
        for m in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', s, re.S):
            body = m.group(1); names = re.findall(r'\\includegraphics\[[^\]]*\]\{Figures/(fig_[^}.]*)', body)
            cap = re.search(r'\\caption\{(.*)', body, re.S); lab = re.search(r'\\label\{([^}]*)\}', body)
            txt = strip_tex(cap.group(1).split('\\label')[0]) if cap else ''
            for n in names:
                out.setdefault(n, []).append(dict(file=f.name, label=lab.group(1) if lab else '', caption=txt))
    return out


def b64png(path, w=1400):
    im = Image.open(path).convert('RGB')
    if im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def comment_block(fig):
    return f'''<div class="cform"><ul id="l-{fig}" class="clist" hidden></ul>
<label for="c-{fig}">Your change for this figure</label>
<textarea id="c-{fig}" rows="3" placeholder="e.g. swap the sorghum plant, bigger labels, drop panel c"></textarea>
<div class="crow"><button id="b-{fig}" type="button">Send to Claude</button><span id="s-{fig}" class="cstatus"></span></div></div>'''


SCRIPT = '''<script>
(async () => {
  const figs = [...document.querySelectorAll('section.fig')].map(s => s.id).concat(['general']);
  const esc = s => String(s).replace(/[&<>"']/g, m => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[m]));
  const status = (fig, msg, cls) => { const el = document.getElementById('s-' + fig); if (el) { el.textContent = msg; el.className = 'cstatus ' + (cls || ''); } };
  const fmt = iso => { try { return new Date(iso).toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'short'}); } catch (e) { return iso; } };
  for (const f of figs) {
    const ta = document.getElementById('c-' + f); if (!ta) continue;
    try { ta.value = localStorage.getItem('draft:' + f) || ''; } catch (e) {}
    ta.addEventListener('input', () => { try { localStorage.setItem('draft:' + f, ta.value); } catch (e) {} });
  }
  const db = await claude.use('db');
  if (!db) {
    figs.forEach(f => status(f, 'Comments need the signed-in claude.ai viewer. Reply in the chat instead.', 'muted'));
    document.querySelectorAll('.cform button').forEach(b => { b.disabled = true; });
    return;
  }
  const user = await claude.use('user');
  let uid = null; try { uid = user ? await user.id() : null; } catch (e) {}
  const LOADING = 'Earlier comments are not reachable yet.';
  const timer = setTimeout(() => figs.forEach(f => { const el = document.getElementById('s-' + f); if (el && !el.textContent) status(f, LOADING, 'muted'); }), 10000);
  const render = snap => {
    const by = {};
    snap.docs.forEach(d => { const x = d.data(); if (!x || !x.fig) return; (by[x.fig] = by[x.fig] || []).push({id: d.id, ...x}); });
    for (const f of figs) {
      const list = document.getElementById('l-' + f); if (!list) continue;
      const items = by[f] || [];
      list.innerHTML = items.map(c => '<li class="' + (c.status === 'done' ? 'done' : '') + '"><span class="when">' + esc(fmt(c.at)) + (c.status === 'done' ? ' · addressed' : '') + '</span>' + esc(c.text) + '</li>').join('');
      list.hidden = items.length === 0;
    }
  };
  db.collection('comments').orderBy('at').onSnapshot(snap => {
    if (!snap.metadata.fromCache) { clearTimeout(timer); figs.forEach(f => { const el = document.getElementById('s-' + f); if (el && el.textContent === LOADING) status(f, '', ''); }); }
    render(snap);
  }, e => { clearTimeout(timer); figs.forEach(f => status(f, 'Earlier comments cannot be loaded (' + e.code + ').', 'warn')); });
  for (const f of figs) {
    const ta = document.getElementById('c-' + f), btn = document.getElementById('b-' + f); if (!ta || !btn) continue;
    btn.addEventListener('click', async () => {
      const text = ta.value.trim();
      if (!text) { status(f, 'Write the change first.', 'warn'); return; }
      btn.disabled = true; status(f, 'Sending…', 'muted');
      try {
        await db.collection('comments').add({fig: f, text: text, at: new Date().toISOString(), by: uid, status: 'open'});
        ta.value = ''; try { localStorage.removeItem('draft:' + f); } catch (e) {}
        status(f, 'Sent. In the chat, say "read the figure comments" and Claude applies it.', 'ok');
      } catch (e) {
        const code = (e && e.code) || 'error';
        status(f, code === 'invalid_argument' ? 'This viewer cannot write comments here. Reply in the chat instead.' : 'Could not send (' + code + '). Try again.', 'warn');
      }
      btn.disabled = false;
    });
  }
})();
</script>'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    caps = captions(); today = time.strftime('%Y-%m-%d')
    git_dates = {}
    for f in sorted(FIGS.glob('fig_*.png')):
        d = subprocess.run(['git', 'log', '-1', '--format=%ad', '--date=short', '--', str(f)], capture_output=True, text=True, cwd=REPO).stdout.strip()
        git_dates[f.stem] = d or 'uncommitted'
    order = ['fig_dataset', 'fig_dataset_stats', 'fig_any2any', 'fig_qual', 'fig_e5_views', 'fig_partial_clouds', 'fig_rows', 'fig_rows_examples',
             'fig_leafcount', 'fig_e2_streams', 'fig_param_controls', 'fig_scaling', 'fig_fscore', 'fig_e9_tsne', 'fig_metrics']
    skip = {'fig_rows_maize', 'fig_rows_sorghum'}
    names = [n for n in order if (FIGS / f'{n}.png').exists()] + [p.stem for p in sorted(FIGS.glob('fig_*.png')) if p.stem not in order and p.stem not in skip]
    cards = []
    for i, n in enumerate(names, 1):
        mt = time.strftime('%Y-%m-%d %H:%M', time.localtime((FIGS / f'{n}.png').stat().st_mtime))
        uses = caps.get(n, [])
        use_html = ''.join(f'<p class="use"><span class="where">{u["file"]} · {u["label"]}</span>{u["caption"] or "(no caption found)"}</p>' for u in uses) or '<p class="use warn">Not placed in the paper yet.</p>'
        if git_dates[n] == 'uncommitted': status, cls = 'new today', 'new'
        elif mt[:10] == today: status, cls = 'updated today', 'upd'
        else: status, cls = 'committed ' + git_dates[n], 'old'
        note = NOTES.get(n, 'As committed.')
        cards.append(f'''<section class="fig" id="{n}"><header><h2>{i}. {n[4:].replace('_', ' ')}</h2><span class="pill {cls}">{status}</span></header>
<p class="meta"><span class="mono">{n}.pdf</span> · made by <span class="mono">{SCRIPTS.get(n, "?")}</span> · rendered {mt}</p>
<figure><img src="{b64png(FIGS / f'{n}.png')}" alt="{n}" loading="lazy"></figure>{use_html}<p class="note">{note}</p>{comment_block(n)}</section>''')
    toc = ''.join(f'<li><a href="#{n}">{i}. {n[4:].replace("_", " ")}</a></li>' for i, n in enumerate(names, 1))
    cands = ''.join(f'<li><b>{t}.</b> {d}</li>' for t, d in CANDIDATES)
    html = f'''<title>CropCode Figures</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=Source+Sans+3:wght@400;600&family=Source+Code+Pro:wght@400&display=swap">
<style>
/* layout: one column of figure cards, each a full-width image with its paper caption under it; a numbered list at the top */
:root {{ --bg: #ffffff; --fg: #1d221c; --muted: #5f685c; --line: #d9ded3; --accent: #2f6b2a; --accent-soft: #e9f1e6; --warn: #a2402b; --new: #1f6f8b;
  --display: "Source Serif 4", Georgia, serif; --body: "Source Sans 3", system-ui, sans-serif; --mono: "Source Code Pro", ui-monospace, Menlo, monospace; color-scheme: light; }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.5 var(--body); padding-block: 24px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
h1 {{ font: 600 1.9rem/1.15 var(--display); margin: 0 0 6px; text-wrap: balance; }}
h2 {{ font: 600 1.25rem/1.2 var(--display); margin: 0; text-wrap: balance; }}
h3 {{ font: 600 1.05rem/1.2 var(--display); margin: 0 0 6px; }}
.lede {{ color: var(--muted); max-width: 70ch; margin: 0 0 14px; }}
.toc {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 2px 18px; padding: 0; margin: 0 0 8px; list-style: none; font-size: .9rem; }}
.toc a {{ color: var(--accent); text-decoration: none; }} .toc a:hover, .toc a:focus-visible {{ text-decoration: underline; }}
.legend {{ font-size: .85rem; color: var(--muted); margin: 0 0 18px; }}
.fig {{ border-top: 1px solid var(--line); padding-top: 18px; margin-top: 26px; }}
.fig header {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px; }}
.pill {{ font-size: .7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }}
.pill.new {{ border-color: var(--new); color: var(--new); }} .pill.upd {{ border-color: var(--accent); color: var(--accent); }}
.meta {{ font-size: .84rem; color: var(--muted); margin: 4px 0 10px; }}
.mono {{ font-family: var(--mono); font-size: .86em; }}
figure {{ margin: 0; border: 1px solid var(--line); border-radius: 4px; overflow: hidden; background: #fff; }}
figure img {{ display: block; width: 100%; height: auto; }}
.use {{ font-size: .9rem; margin: 10px 0 0; max-width: 100ch; }}
.where {{ display: block; font-family: var(--mono); font-size: .78rem; color: var(--muted); margin-bottom: 2px; }}
.warn {{ color: var(--warn); }}
.note {{ font-size: .9rem; margin: 8px 0 0; padding: 8px 12px; background: var(--accent-soft); border-radius: 4px; max-width: 100ch; }}
.cands {{ border-top: 1px solid var(--line); margin-top: 30px; padding-top: 18px; }}
.cands ul {{ margin: 6px 0 0; padding-left: 18px; max-width: 90ch; font-size: .95rem; }} .cands li {{ margin: 4px 0; }}
a:focus-visible, textarea:focus-visible, button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.cform {{ margin-top: 12px; display: grid; gap: 6px; max-width: 100ch; }}
.cform label {{ font-size: .82rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }}
.cform textarea {{ width: 100%; min-width: 0; box-sizing: border-box; font: inherit; padding: 8px 10px; border: 1px solid var(--line); border-radius: 4px; resize: vertical; background: #fff; color: var(--fg); }}
.crow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }}
.cform button {{ font: 600 .9rem var(--body); padding: 6px 14px; border-radius: 4px; border: 1px solid var(--accent); background: var(--accent); color: #fff; cursor: pointer; }}
.cform button:disabled {{ opacity: .5; cursor: default; }}
.cstatus {{ font-size: .85rem; color: var(--muted); }} .cstatus.warn {{ color: var(--warn); }} .cstatus.ok {{ color: var(--accent); }}
.clist {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }}
.clist li {{ border-left: 3px solid var(--new); padding: 5px 10px; background: #f4f7f2; font-size: .92rem; white-space: pre-wrap; }}
.clist li.done {{ border-left-color: var(--accent); opacity: .7; }}
.when {{ display: block; font-size: .75rem; color: var(--muted); }}
@media (prefers-reduced-motion: no-preference) {{ html {{ scroll-behavior: smooth; }} }}
</style>
<div class="wrap">
<h1>CropCode figures, for review</h1>
<p class="lede">Every figure in <span class="mono">paper/figs/</span> as of {time.strftime('%Y-%m-%d %H:%M')}, with the caption from the Overleaf section that places it. Each figure has a comment box under it: what you send is stored with this page. Then say "read the figure comments" in the chat, and Claude applies them and marks them addressed here.</p>
<ul class="toc">{toc}</ul>
<p class="legend">New today = made in this session and not committed yet. Updated today = regenerated today from committed scripts. Committed = unchanged since the date shown.</p>
{''.join(cards)}
<section class="cands"><h3>Candidates not built yet</h3><p class="lede">Say which of these you want, or name another.</p><ul>{cands}</ul></section>
<section class="cands" id="general"><h3>General comments</h3><p class="lede">Anything that is not about one figure: a new figure, the order, the style.</p>{comment_block('general')}</section>
</div>
{SCRIPT}'''
    pathlib.Path(a.out).write_text(html); print('wrote', a.out, len(html) // 1024, 'KB', len(names), 'figures')


if __name__ == '__main__': main()
