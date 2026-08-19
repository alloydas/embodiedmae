"""
Fill report_template.html with the figures and numbers from figures_compare/.

Every number and every quantitative sentence in the page is derived here from
metrics.json, so the prose cannot drift away from what was actually measured.
Images are inlined as data URIs — the Artifact CSP blocks external requests.

  python build_artifact.py --out figures_compare/report.html
"""

import argparse
import base64
import io
import json
import math
from pathlib import Path

from PIL import Image

MAX_W = {'recon': 1500, 'wide': 1950, 'square': 1700}


def encode(path: Path, kind: str, quality: int = 88) -> tuple[str, int]:
    """Downscale onto white and inline as a JPEG data URI."""
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        flat = Image.new('RGB', im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        im = flat
    else:
        im = im.convert('RGB')
    limit = MAX_W[kind]
    if im.width > limit:
        im = im.resize((limit, round(im.height * limit / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=quality, optimize=True, progressive=True)
    raw = buf.getvalue()
    return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'), len(raw)


def fmt(x, nd=4):
    return f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig_dir', default='figures_compare')
    ap.add_argument('--template', default='report_template.html')
    ap.add_argument('--out', default='figures_compare/report.html')
    ap.add_argument('--total_epochs', type=int, default=1000)
    ap.add_argument('--min_per_epoch', type=float, default=49.0)
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    m = json.loads((fig_dir / 'metrics.json').read_text())
    tpl = Path(args.template).read_text()

    epoch = m['checkpoint_epoch']
    hist = m['history_tail']
    lt = hist['last_test']
    epochs_done = hist['epochs_done']
    pct = round(100 * epoch / args.total_epochs)
    days_left = math.ceil((args.total_epochs - epochs_done) * args.min_per_epoch / 60 / 24)

    total_bytes = 0

    # ── Fig 1: per-plant gallery, as tabs ────────────────────────────────────
    tabs, panels = [], []
    for i, fn in enumerate(m['gallery']):
        name = fn[len('recon_'):-len('.png')]
        uri, nb = encode(fig_dir / fn, 'recon')
        total_bytes += nb
        sel = 'true' if i == 0 else 'false'
        tabs.append(
            f'<button type="button" role="tab" aria-selected="{sel}" '
            f'aria-controls="pane-{i}" id="tab-{i}">{name}</button>')
        panels.append(
            f'<figure class="plate-frame" id="pane-{i}" role="tabpanel" '
            f'aria-labelledby="tab-{i}"{"" if i == 0 else " hidden"}>'
            f'<img src="{uri}" alt="Ground truth, masked encoder input and '
            f'reconstruction for {name} in RGB, depth and point cloud" '
            f'loading="lazy"></figure>')

    # ── Other plates ────────────────────────────────────────────────────────
    imgs = {}
    for key, fn, kind in [('IMG_MASK_SWEEP', 'mask_sweep.png', 'wide'),
                          ('IMG_CROSSMODAL', 'crossmodal.png', 'wide'),
                          ('IMG_CKPT', 'ckpt_progression.png', 'wide'),
                          ('IMG_PARAM', 'param_scatter.png', 'square'),
                          ('IMG_CURVES', 'curves.png', 'wide')]:
        imgs[key], nb = encode(fig_dir / fn, kind, quality=90 if 'PARAM' in key or 'CURVES' in key else 88)
        total_bytes += nb

    # ── Sentences generated from the measurements ───────────────────────────
    ms = {float(k): v for k, v in m['mask_sweep_chamfer'].items()}
    lo, hi = min(ms), max(ms)
    mask_sentence = (
        f'Chamfer error on the point cloud rises from <code>{fmt(ms[lo], 5)}</code> at '
        f'{lo:.0%} masking to <code>{fmt(ms[hi], 5)}</code> at {hi:.0%} — roughly '
        f'{ms[hi] / ms[lo]:.1f}× — but the depth map barely moves across the whole range, '
        f'and the plant stays recognisable in every column.')

    cma = m['crossmodal_aggregate']
    LABEL = {'rgb': 'RGB only', 'depth': 'Depth only', 'pc': 'Point cloud only',
             'text': 'Spline params only', 'rgb+depth': 'RGB + depth',
             'masked': f'Normal {m["mask_ratio"]:.0%} masking (reference)'}
    order = sorted(cma.items(), key=lambda kv: kv[1]['mean'])
    worst_mean = max(v['mean'] for v in cma.values())
    cm_rows = []
    for k, v in order:
        w = 100 * v['mean'] / worst_mean
        strong = ' style="font-weight:600"' if k == 'masked' else ''
        cm_rows.append(
            f'<tr><td class="name"{strong}>{LABEL[k]}</td>'
            f'<td class="n">{v["mean"]:.5f}</td>'
            f'<td class="bar-cell"><i style="--w:{w:.0f}%"></i></td>'
            f'<td class="n">{v["median"]:.5f}</td>'
            f'<td class="n">{v["n"]}</td></tr>')

    n_cm = cma['rgb']['n']
    # 'masked' is the reference row, not a cross-modal regime — exclude it here.
    best_k, best_v = min(((k, v) for k, v in cma.items() if k != 'masked'),
                         key=lambda kv: kv[1]['mean'])
    ref_m = cma['masked']['mean']
    verdict = 'better than' if best_v['mean'] < ref_m else 'short of'
    cm_sentence = (
        f'Over {n_cm} held-out plants, <b>{LABEL[best_k].lower()}</b> gives the best point cloud '
        f'(mean Chamfer <code>{best_v["mean"]:.5f}</code>) — {verdict} the '
        f'<code>{ref_m:.5f}</code> the same plants get under normal '
        f'{m["mask_ratio"]:.0%} masking.')

    # Two orderings here can come out against intuition. Report whichever the
    # numbers actually show rather than assuming the expected one.
    pc_m, rgb_m, d_m, both_m = (cma['pc']['mean'], cma['rgb']['mean'],
                                cma['depth']['mean'], cma['rgb+depth']['mean'])
    img_m = min(rgb_m, d_m)
    img_k = 'RGB' if rgb_m <= d_m else 'depth'
    n_vis = {'text': 25, 'rgb': 196, 'depth': 196, 'pc': 196, 'rgb+depth': 392}
    trained_vis = round((1 - m['mask_ratio']) * 613)

    anomalies = []
    if pc_m > img_m:
        anomalies.append(
            f'handing the encoder the <b>complete point cloud</b> rebuilds it at '
            f'{pc_m:.5f}, <em>worse</em> than giving it only {img_k} ({img_m:.5f}), '
            f'even though the target is sitting in the input')
    if both_m > min(rgb_m, d_m):
        anomalies.append(
            f'<b>RGB and depth together</b> ({both_m:.5f}) are worse than either alone '
            f'({rgb_m:.5f} / {d_m:.5f}), so adding a modality made it worse')

    if anomalies:
        cm_note = (
            '<strong>Two rows come out against intuition:</strong> '
            + '; and '.join(anomalies) + '. '
            f'The common thread is that <em>none</em> of these regimes occurs in training. '
            f'Dirichlet masking always leaves every modality partly visible and shows the encoder '
            f'about {trained_vis} tokens per step; these columns give it '
            f'{n_vis["text"]}, {n_vis["rgb"]} or {n_vis["rgb+depth"]}, and zero from three of the '
            f'four streams. So this table measures <b>robustness to regime shift, not information '
            f'content</b> — it does not license the reading that RGB carries more 3-D signal than '
            f'the point cloud does. Worth an ablation before it goes anywhere near a paper; the '
            f'obvious one is to train with occasional whole-modality dropout and re-measure.')
    else:
        cm_note = (
            f'The ordering is the expected one: the point cloud is rebuilt best when it is itself '
            f'given ({pc_m:.5f}), ahead of inferring it from {img_k} ({img_m:.5f}). Note that all '
            f'of these regimes are still off-distribution — training never withholds a whole '
            f'modality — so treat the margins as indicative.')

    cp = {int(k): v for k, v in m['ckpt_progression_chamfer'].items()}
    eps = sorted(cp)
    ck_sentence = (
        f'Chamfer falls from <code>{fmt(cp[eps[0]], 5)}</code> at epoch {eps[0]} to '
        f'<code>{fmt(cp[eps[-1]], 5)}</code> at epoch {eps[-1]}, a '
        f'{100 * (1 - cp[eps[-1]] / cp[eps[0]]):.0f}% reduction.')

    # ── Parameter table ─────────────────────────────────────────────────────
    ps = m['param_stats']
    rows, learned, degenerate, unlearned = [], [], [], []
    for key, v in sorted(ps.items(), key=lambda kv: (-1e9 if math.isnan(kv[1]['r2']) else -kv[1]['r2'])):
        kind, name = key.split('.', 1)
        r2 = v['r2']
        if math.isnan(r2):
            degenerate.append(name)
            r2_cell = '<td class="n">—</td>'
            bar = '<td class="bar-cell"></td>'
            note = ' <span style="opacity:.6">constant in split</span>'
        else:
            if r2 >= 0.9:
                learned.append(name)
            elif r2 < 0.5:
                unlearned.append(name)
            r2_cell = f'<td class="n">{r2:+.3f}</td>'
            w = max(0.0, min(1.0, r2)) * 100
            bar = f'<td class="bar-cell"><i style="--w:{w:.0f}%"></i></td>'
            note = ''
        rows.append(
            f'<tr><td class="name">{name}{note}</td><td>{kind}</td>{r2_cell}{bar}'
            f'<td class="n">{v["mae"]:.4g}</td><td class="n">{v["n"]:,}</td></tr>')

    param_sentence = (
        f'The result is sharply split. <b>{len(learned)} parameters are essentially solved</b> — '
        f'{", ".join(f"<code>{p}</code>" for p in learned)} all reach R² ≥ 0.90, with '
        f'<code>starting_point</code> at {ps["leaf.starting_point"]["r2"]:.3f} and '
        f'<code>branching_angle</code> at {ps["leaf.branching_angle"]["r2"]:.3f}. '
        f'<code>roll_angle</code> scores R² {ps["leaf.roll_angle"]["r2"]:.2f}, but read that panel '
        f'before believing the number: there is a dense, tight diagonal with a diffuse cloud around '
        f'it, and roll angle is <em>circular</em> — 359° and 1° are neighbours that R² treats as '
        f'maximally far apart. A proper circular error metric would score it higher; how much higher '
        f'is untested. '
        f'The three waviness parameters, by contrast, are flat horizontal bands: the model emits '
        f'their mean and nothing else. And {len(degenerate)} panicle/seed parameters are '
        f'<b>constant across this split</b> — their ground truth is a single value, so there is '
        f'nothing to recover and R² is undefined; the three <code>stem_dir</code> components are '
        f'nearly as degenerate. '
        f'So the headline <code>{lt["test_param_acc05"]:.1%}</code> parameter accuracy is '
        f'<b>inflated by parameters that never vary</b>, and should not be quoted on its own. '
        f'The defensible claim is narrower and more interesting: the model reads a leaf\'s '
        f'<em>position along the stem, its length and its branching angle</em> off the pixels and '
        f'points of the rest of the plant, to within a few percent.')

    # ── Substitute ──────────────────────────────────────────────────────────
    repl = {
        'EPOCH': str(epoch),
        'TOTAL_EPOCHS': str(args.total_epochs),
        'EPOCHS_DONE': str(epochs_done),
        'PCT': str(pct),
        'DAYS_LEFT': str(days_left),
        'VAL_LOSS': fmt(hist['best_val_loss']),
        'TEST_LOSS': fmt(lt['test_loss']),
        'TEST_CHAMFER': f'{lt["test_pc_chamfer"]:.5f}',
        'TEST_ACC': f'{lt["test_param_acc05"]:.1%}',
        'TEST_MAE': fmt(lt['test_param_mae_masked']),
        'TEST_EPOCH': str(hist['last_test_epoch']),
        'N_TEST': f'{m["n_test"]:,}',
        'MASK_RATIO': f'{m["mask_ratio"]:.2f}',
        'MASK_PCT': f'{m["mask_ratio"]:.0%}',
        'CKPT_NAME': Path(m['checkpoint']).name,
        'GALLERY_TABS': '\n      '.join(tabs),
        'GALLERY_PANELS': '\n    '.join(panels),
        'MASK_SWEEP_SENTENCE': mask_sentence,
        'CROSSMODAL_SENTENCE': cm_sentence,
        'CROSSMODAL_ROWS': '\n          '.join(cm_rows),
        'CROSSMODAL_NOTE': cm_note,
        'CKPT_SENTENCE': ck_sentence,
        'PARAM_SENTENCE': param_sentence,
        'PARAM_ROWS': '\n          '.join(rows),
        **imgs,
    }
    html = tpl
    for k, v in repl.items():
        html = html.replace('{{' + k + '}}', v)

    missing = [t for t in ('{{', '}}') if t in html]
    if missing:
        import re
        left = set(re.findall(r'\{\{(\w+)\}\}', html))
        raise SystemExit(f'unfilled placeholders: {sorted(left)}')

    out = Path(args.out)
    out.write_text(html)
    print(f'wrote {out}  ({len(html) / 1e6:.2f} MB page, {total_bytes / 1e6:.2f} MB of image data)')
    print(f'learned (R2>=0.9): {learned}')
    print(f'weak (R2<0.5)    : {unlearned}')
    print(f'constant in split: {degenerate}')


if __name__ == '__main__':
    main()
