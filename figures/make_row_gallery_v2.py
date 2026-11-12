"""Row-occlusion galleries (Regime III), second version: the paper's Fig. 9 and two supplement figures.

What the row-trained model and the isolated-trained control write for the center plant of a held-out test row,
broadside (yaw 0) and end-on (yaw 90). Tiles: the row image the model reads (neighbors included), the center plant's
true cloud, the row-trained prediction (Okabe-Ito blue) and the isolated-trained prediction (Okabe-Ito vermillion), each over the truth in
light grey with its Chamfer against the truth (x1e-4 m^2). All three streams in, the recipe never an input, the
protocol, models, masking and depth convention of eval/eval_row.py. Every label is 9 pt at print size.

Selection, from the per-plant test reports of eval/eval_row.py (all three streams, row mode):
  candidates (Fig. 9)  plants whose row-trained Chamfer is at or below the 35th percentile AND whose isolated-trained
                       Chamfer is at or above the 60th percentile at BOTH yaws, ranked by the smaller of the two
                       iso / row ratios; the top --n-cand are rendered and three per species are picked by eye
                       (--main), so Fig. 9 is selected for clarity and says so.
  supplement           ten plants per species at the 5th, 15th, ..., 95th percentile of the row-trained Chamfer
                       (mean of the two yaws), skipping the candidates: the typical and the bad cases.

  python figures/make_row_gallery_v2.py --dump          # CPU job: predictions -> <OUT>/row_gallery_dump.pkl
  python figures/make_row_gallery_v2.py --sheet         # contact sheet of the candidates (to pick by eye)
  python figures/make_row_gallery_v2.py --render --main sorghum:A@90,B@0,C@0 maize:D@90,E@0,F@0    (plant@yaw rows)
        -> paper/figs/fig_rows_examples.pdf, fig_rows_more_{sorghum,maize}.pdf (+ PNG), copied to paper/overleaf/Figures/
"""
import sys, pathlib; R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import argparse, csv, pickle, shutil
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _scene                                   # registers Liberation Sans and sets the font defaults

OUT = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig12/row_gallery_dump.pkl')
REPORTS = {'sorghum': ('reports/row_sorghum_sorghum_row_ft_checkpoint_epoch_100_test_17082628.csv',
                       'reports/row_sorghum_sorghum_row_iso_checkpoint_epoch_100_test_17082823.csv'),
           'maize': ('reports/row_maize_maize_row_ft_e1_checkpoint_epoch_100_test_17189700.csv',
                     'reports/row_maize_maize_row_iso_e1_checkpoint_epoch_100_test_17189701.csv')}
SPEC = {'sorghum': ('/work/mech-ai/alloydas/sorghum_row', 'sorghum_row_ft', 'sorghum_row_iso'),
        'maize': ('/work/mech-ai/alloydas/maize_row', 'maize_row_ft_e1', 'maize_row_iso_e1')}
# Okabe-Ito (colour-blind safe): blue vs vermillion stay distinct under protan-, deutan- and tritanopia; the headers name the models too
BLUE, ORANGE, GREY, INK, MUTED = '#0072B2', '#D55E00', '#d4d4d4', '#1d221c', '#5f685c'
FS = 9.0
THETAS = (0, 90)


# ====================================================================== selection (no model)
def load_report(path):
    out = {}
    for r in csv.DictReader(open(R / path)):
        if r['mode'] == 'row' and r['source'] == 'all' and int(r['theta']) in THETAS:
            out[(r['plant'], int(r['theta']))] = float(r['chamfer_gt_m2']) * 1e4
    return out


def select(sp, n_cand=12, n_more=10):
    ft, iso = (load_report(p) for p in REPORTS[sp])
    plants = sorted({p for p, t in ft if all((p, th) in ft and (p, th) in iso for th in THETAS)})
    f = np.array([[ft[(p, th)] for th in THETAS] for p in plants]); i = np.array([[iso[(p, th)] for th in THETAS] for p in plants])
    lo, hi = np.percentile(f, 35), np.percentile(i, 60)
    ok = (f <= lo).all(1) & (i >= hi).all(1)
    ratio = (i / f).min(1)
    cand = [plants[k] for k in np.argsort(-np.where(ok, ratio, -1))[:n_cand] if ok[k]]
    mean_f = f.mean(1); order = np.argsort(mean_f); taken = set(cand); more = []
    for q in np.arange(5, 100, 10)[:n_more]:
        k = int(round(q / 100 * (len(plants) - 1)))
        while plants[order[k]] in taken: k += 1
        more.append((plants[order[k]], float(q))); taken.add(plants[order[k]])
    print(sp, f'{len(plants)} plants; {ok.sum()} pass the candidate rule; candidates', cand, flush=True)
    print(sp, 'supplement (plant, percentile)', more, flush=True)
    return cand, more


# ====================================================================== predictions (CPU job)
def dump(n_cand):
    import torch
    from PIL import Image
    import eval_row as ER
    from row_dataset import RowDataset, IN_FILES
    torch.set_num_threads(8)
    D = {}
    for sp in ('sorghum', 'maize'):
        cand, more = select(sp, n_cand)
        root, ft_run, iso_run = SPEC[sp]
        ft, cfg, _ = ER.build_model(R / 'outputs' / ft_run, 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        iso, _, _ = ER.build_model(R / 'outputs' / iso_run, 'checkpoints/checkpoint_epoch_100.pth', 'cpu', sp)
        info_sp = ER.species_info(sp); gain = float(cfg.get('rgb_brighten_gain', 1.0)) if sp == 'sorghum' else None
        plants = cand + [p for p, _ in more]
        ds = RowDataset(root, 'test', sp, 'row', cfg.get('img_size', 224), cfg.get('num_points', info_sp['num_points']), info_sp['max_leaves'],
                        thetas=list(THETAS), shots=[0], plants=plants, rgb_gain=gain,
                        gt_points=cfg.get('gt_points', cfg.get('num_points', info_sp['num_points'])), occlusion_stats=True)
        mr = float(cfg.get('mask_ratio', 0.8)); items = {}
        for i in range(len(ds)):
            plant, theta, shot, folder = ds.items[ds._resolve(i)]
            seed = int(plant.split('_')[-1]) * 7 + theta           # per item, independent of the list order
            np.random.seed(seed)
            rgb, depth, pc, params, valid, _, _, gt_pc, inf = ds[i]
            b = lambda x: x[None]
            out = {}
            for tag, m in (('ft', ft), ('iso', iso)):
                with torch.no_grad():
                    pred, _ = ER.predict(m, 'all', b(rgb), b(depth), b(pc), b(params), mr, seed)
                _, cg_m2, _, _ = ER.score_cloud(pred, b(gt_pc), torch.tensor([float(inf['scale'])]))
                out[tag] = (pred[0].numpy().astype(np.float32), float(cg_m2[0]) * 1e4)
            im = Image.open(folder / IN_FILES['row'][0]).convert('RGBA')
            if gain: im = Image.fromarray(np.concatenate([np.clip(np.asarray(im)[..., :3].astype(np.float32) * gain, 0, 255).astype(np.uint8), np.asarray(im)[..., 3:]], -1))
            im = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 255)), im).convert('RGB').resize((360, 360), Image.LANCZOS)
            items[(plant, theta)] = dict(img=np.asarray(im), gt=gt_pc.numpy().astype(np.float32), out=out, occ=float(inf['occ_frac']))
            print(sp, plant, theta, f"occ {inf['occ_frac']:.2f}", {k: round(v[1], 1) for k, v in out.items()}, flush=True)
        D[sp] = dict(cand=cand, more=more, items=items)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(D, open(str(OUT) + '.tmp', 'wb')); pathlib.Path(str(OUT) + '.tmp').replace(OUT)
    print('wrote', OUT, flush=True)


# ====================================================================== drawing
def draw_tiles(fig, x, y, s, it, gap=0.04, W=1.0, H=1.0, numbers=True, ncols=4):
    """four tiles at (x, y) inches (lower left of the first): image, truth, row-trained, isolated-trained"""
    gt = it['gt']; lim = np.abs(gt[:, :2]).max() * 1.12
    cx = (gt[:, 0].max() + gt[:, 0].min()) / 2; cy = (gt[:, 1].max() + gt[:, 1].min()) / 2
    axs = [fig.add_axes([(x + k * (s + gap)) / W, y / H, s / W, s / H]) for k in range(ncols)]
    axs[0].imshow(it['img'])
    ps = 0.10 * (s / 0.72) ** 2
    axs[1].scatter(gt[:, 0], gt[:, 1], s=ps * 1.2, c='#3c3b38', lw=0, rasterized=True)
    for ax, (tag, col) in zip(axs[2:], (('ft', BLUE), ('iso', ORANGE))[:ncols - 2]):
        p, cd = it['out'][tag]
        ax.scatter(gt[:, 0], gt[:, 1], s=ps * 1.2, c=GREY, lw=0, rasterized=True)
        ax.scatter(p[:, 0], p[:, 1], s=ps, c=col, lw=0, rasterized=True)
        if numbers:
            ax.text(0.97, 0.03, f'{cd:.1f}' if cd < 100 else f'{cd:.0f}', transform=ax.transAxes, fontsize=FS, color=col, ha='right', va='bottom',
                    fontweight='bold', path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground='white')])
    for ax in axs[1:]:
        ax.set_xlim(cx - lim, cx + lim); ax.set_ylim(cy - lim, cy + lim); ax.set_aspect('equal')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_linewidth(0.5); sp_.set_color('#b5b8b0')
    return axs


HEAD = ['Row\nimage', 'Truth', 'Row-\ntrained', 'Isolated-\ntrained']
HEAD_C = [INK, INK, BLUE, ORANGE]
import matplotlib.patheffects


def render_main(D, picks):
    """Fig. 9: sorghum left, maize right; three (plant, yaw) rows each; image, truth, row-trained prediction."""
    W = 6.875; s, g, gy = 0.90, 0.05, 0.07; lab = 0.62; nc = 3
    half = lab + nc * s + (nc - 1) * g
    hdr = 0.48; n = max(len(v) for v in picks.values())
    H = hdr + n * s + (n - 1) * gy + 0.02
    fig = plt.figure(figsize=(W, H))
    for c, sp in enumerate(('sorghum', 'maize')):
        x0 = c * (W - half)
        fig.text((x0 + lab + (nc * s + (nc - 1) * g) / 2) / W, (H - 0.01) / H, sp.capitalize(), fontsize=FS, fontweight='bold', ha='center', va='top')
        for k in range(nc):
            fig.text((x0 + lab + k * (s + g) + s / 2) / W, (H - 0.20) / H, HEAD[k].replace('\n', ' ').replace('- ', '-'), fontsize=FS,
                     ha='center', va='top', color=HEAD_C[k], linespacing=0.95)
        y = H - hdr
        for pl, th in picks[sp]:
            y -= s
            draw_tiles(fig, x0 + lab, y, s, D[sp]['items'][(pl, th)], g, W, H, ncols=nc)
            fig.text((x0 + lab - 0.06) / W, (y + s / 2) / H, f"plant\n{pl.split('_')[-1]}\n{th}°", fontsize=FS, ha='right', va='center', linespacing=1.1)
            y -= gy
    return fig


def render_more(D, sp):
    """supplement: ten plants of one species, one per row; broadside on the left, end-on on the right"""
    W = 6.875; s, g, gy, gg = 0.70, 0.035, 0.05, 0.16; lab = 0.42
    block = 4 * s + 3 * g
    hdr = 0.62; H = hdr + 10 * s + 9 * gy + 0.02
    x0 = (W - (lab + 2 * block + gg)) / 2
    fig = plt.figure(figsize=(W, H))
    for b, th in enumerate(THETAS):
        xb = x0 + lab + b * (block + gg)
        fig.text((xb + block / 2) / W, (H - 0.01) / H, 'Broadside (0°)' if th == 0 else 'End-on (90°)', fontsize=FS, fontweight='bold', ha='center', va='top')
        for k in range(4):
            fig.text((xb + k * (s + g) + s / 2) / W, (H - 0.22) / H, HEAD[k], fontsize=FS, ha='center', va='top', color=HEAD_C[k], linespacing=0.95)
    y = H - hdr
    for pl, q in D[sp]['more']:
        y -= s
        for b, th in enumerate(THETAS):
            draw_tiles(fig, x0 + lab + b * (block + gg), y, s, D[sp]['items'][(pl, th)], g, W, H)
        fig.text((x0 + lab - 0.06) / W, (y + s / 2) / H, f"{pl.split('_')[-1]}\n{q:.0f}th", fontsize=FS, ha='right', va='center', linespacing=1.05)
        y -= gy
    return fig


def sheet(D):
    for sp in ('sorghum', 'maize'):
        c = D[sp]['cand']; W = 6.875; s, g = 0.73, 0.04
        H = len(c) * (s + 0.06) + 0.3
        fig = plt.figure(figsize=(2 * (4 * s + 3 * g) + 1.2, H))
        Wf = 2 * (4 * s + 3 * g) + 1.2
        for r, pl in enumerate(c):
            y = H - 0.3 - (r + 1) * (s + 0.06)
            for b, th in enumerate(THETAS):
                draw_tiles(fig, 0.9 + b * (4 * s + 3 * g + 0.2), y, s, D[sp]['items'][(pl, th)], g, Wf, H)
            fig.text(0.05 / Wf, (y + s / 2) / H, pl, fontsize=7, va='center')
        fig.savefig(f'/tmp/claude-490224/-work-mech-ai-scratch-alloy-embodiedmae/08a82cbc-d3db-4182-8667-cb84feb249c4/scratchpad/rowcand_{sp}.png', dpi=110)
        plt.close(fig)


def save(fig, stem):
    out = R / 'paper/figs' / stem
    fig.savefig(f'{out}.pdf', dpi=300); fig.savefig(f'{out}.png', dpi=200); plt.close(fig)
    shutil.copy(f'{out}.pdf', R / 'paper/overleaf/Figures' / f'{stem}.pdf')
    print('wrote', f'{out}.pdf', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', action='store_true'); ap.add_argument('--sheet', action='store_true'); ap.add_argument('--render', action='store_true')
    ap.add_argument('--n-cand', type=int, default=12)
    ap.add_argument('--main', nargs='*', default=[], help='sorghum:Sorghum_a,Sorghum_b,Sorghum_c maize:plant_x,...')
    a = ap.parse_args()
    if a.dump: dump(a.n_cand)
    if a.sheet or a.render:
        D = pickle.load(open(OUT, 'rb'))
        if a.sheet: sheet(D)
        if a.render:
            picks = {m.split(':')[0]: [(q.rsplit('@', 1)[0], int(q.rsplit('@', 1)[1])) for q in m.split(':')[1].split(',')] for m in a.main}
            save(render_main(D, picks), 'fig_rows_examples')
            for sp in ('sorghum', 'maize'): save(render_more(D, sp), f'fig_rows_more_{sp}')
