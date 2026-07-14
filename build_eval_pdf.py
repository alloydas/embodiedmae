#!/usr/bin/env python3
"""Assemble the RGB→PC evaluation figures + findings into a single PDF report.

No LaTeX on Nova — renders HTML via weasyprint (available in the `det` env).

Usage:
    python build_eval_pdf.py --out rgb2pc_eval_report.pdf
"""
import argparse
import base64
from pathlib import Path

from weasyprint import HTML

CSS = """
@page { size: A4 landscape; margin: 12mm; @bottom-center {
    content: counter(page) " / " counter(pages); font: 9pt sans-serif; color: #666; } }
body { font-family: "DejaVu Sans", sans-serif; color: #1d1d1d; font-size: 10pt; }
h1 { font-size: 20pt; margin: 0 0 2mm 0; }
h2 { font-size: 13pt; margin: 5mm 0 2mm 0; color: #264653;
     border-bottom: 1.5px solid #2a9d8f; padding-bottom: 1mm; }
.sub { color: #666; font-size: 10pt; margin-bottom: 4mm; }
table { border-collapse: collapse; margin: 3mm 0; font-size: 9.5pt; }
th, td { border: 1px solid #ccc; padding: 1.6mm 3mm; text-align: right; }
th { background: #f0f4f4; text-align: left; }
td:first-child, th:first-child { text-align: left; }
tr.bad td { background: #fdece7; font-weight: bold; }
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
figure { margin: 0; text-align: center; }
figcaption { font-size: 9pt; color: #555; margin-top: 2mm; }
img { max-width: 100%; max-height: 168mm; height: auto; }
img.tall { max-height: 172mm; }
.key { background: #fdece7; border-left: 4px solid #e76f51; padding: 3mm 4mm; margin: 4mm 0; }
code { background: #f2f2f2; padding: 0 1mm; font-size: 9pt; }
"""


def img_tag(path, cls=''):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{b64}"/>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='rgb2pc_eval_report.pdf')
    ap.add_argument('--prefix', default='eval_rgb2pc')
    a = ap.parse_args()

    pages = []

    # ── page 1: findings ─────────────────────────────────────────────────────
    pages.append(f"""
<div class="page">
  <h1>RGB → Point-Cloud generation: evaluation report</h1>
  <div class="sub">Checkpoint <code>outputs/4m_distill_rgb2pc/best_model.pth</code> (epoch 500) ·
    teacher &amp; student init <code>4m_run_v3</code> @ epoch 2400 ·
    conditioned on <b>RGB only</b> (depth / PC / spline fully masked)</div>

  <h2>Result: the model memorises plants, it does not generalise to new ones</h2>
  <table>
    <tr><th>dataset</th><th>N</th><th>mean CD</th><th>median</th><th>best</th><th>worst</th></tr>
    <tr><td>old10k / train &mdash; trained on these</td><td>150</td><td>0.00029</td>
        <td>0.00029</td><td>0.00021</td><td>0.00040</td></tr>
    <tr><td>old10k / val &mdash; <i>same plants, new camera views</i></td><td>100</td><td>0.00027</td>
        <td>0.00027</td><td>0.00022</td><td>0.00042</td></tr>
    <tr class="bad"><td>Sorghum_15K / test &mdash; unseen plants (OOD)</td><td>150</td><td>0.04705</td>
        <td>0.04280</td><td>0.01356</td><td>0.15145</td></tr>
  </table>
  <div class="sub">Full OOD sweep over all 22,500 test folders: mean <b>0.04987</b>, median 0.04375
    (<code>per_plant_chamfer_15Ktest_ep500.json</code>).</div>

  <div class="key">
    <b>The old 10k validation split is leaked.</b> It contains 100 folders covering only
    <b>10 distinct plants, and all 10 also appear in training</b> — the split was made per camera
    view, not per plant. Every "validation" sample is a training plant from a new angle, which is why
    train and val scores are identical (0.00029 vs 0.00027) instead of val being worse. The 0.0003
    figure this run optimised against therefore never measured generalisation.
  </div>

  <h2>Verification</h2>
  <table>
    <tr><th>check</th><th>result</th></tr>
    <tr><td>old10k train &cap; val (plants)</td><td>10 / 10 &mdash; fully leaked</td></tr>
    <tr><td>Sorghum_15K train &cap; test (plants)</td><td>0 &mdash; leak-free</td></tr>
    <tr><td>shared plant <i>IDs</i> old10k &harr; 15K test</td><td>137 (numbering reuse)</td></tr>
    <tr><td>&hellip; of which identical geometry (spline md5)</td><td>0 / 60 sampled &mdash; genuinely unseen</td></tr>
    <tr><td>render statistics old10k vs 15K</td><td>near-identical &mdash; no domain shift</td></tr>
    <tr><td>Pearson r (Chamfer, phenotype extremeness)</td><td>0.017 &mdash; not tail extrapolation</td></tr>
  </table>

  <h2>Consequences</h2>
  <p>OOD Chamfer <b>rose</b> monotonically over training (0.0381 @ ep65 &rarr; 0.0453 @ ep240 &rarr;
     0.0499 @ ep500) while in-domain fell ~10&times;: the run was memorising harder, not learning better.
     <code>best_model.pth</code> is selected on the leaked val loss, so it is the <i>worst</i> OOD
     checkpoint of the run. The two datasets are statistically indistinguishable at the pixel and
     point level, and error does not track phenotype extremeness — so this is neither a render-domain
     shift nor a tail-extrapolation failure. It is a plain failure to generalise across plant identity,
     hidden until now by the leaked split.</p>
  <p><b>Next:</b> redo the distillation on the 15K train split (leak-free, split by plant), using the
     in-flight 15k pretrain (job 11493779) as teacher, and select on the 15K val split.</p>
</div>""")

    # ── page 2: distributions ────────────────────────────────────────────────
    pages.append(f"""
<div class="page">
  <h2>Chamfer distributions — the two regimes do not overlap</h2>
  <figure>{img_tag(f'{a.prefix}_dist.png')}
    <figcaption>Equal N per dataset. On the log axis the in-domain and OOD populations sit ~2 orders
      of magnitude apart with no overlap: every unseen plant is reconstructed worse than every
      seen one.</figcaption></figure>
</div>""")

    # ── pages 3-5: reconstructions, one dataset per page ─────────────────────
    caps = [
        'Trained on these plants. The predicted cloud (orange) lands on the ground truth (dark) '
        'almost everywhere, including leaf tips.',
        'Same plants as training, new camera views — the leaked "validation" set. Indistinguishable '
        'from training performance, which is the tell.',
        'Unseen plants. The prediction collapses to a narrow, generic stem-and-fan shape: plant-like, '
        'but not THIS plant. Even the best OOD sample is ~32x worse than the worst in-domain one.',
    ]
    for i, cap in enumerate(caps):
        p = Path(f'{a.prefix}_recon_p{i}.png')
        if not p.exists():
            print(f"⚠️  missing {p} — skipping page")
            continue
        pages.append(f"""
<div class="page">
  <figure>{img_tag(p, 'tall')}<figcaption>{cap}</figcaption></figure>
</div>""")

    # ── page 6: dataset statistics ───────────────────────────────────────────
    ds = Path('dataset_compare_stats.png')
    if ds.exists():
        pages.append(f"""
<div class="page">
  <h2>Control: the two datasets are the same render domain</h2>
  <figure>{img_tag(ds)}
    <figcaption>Depth encoding, framing, raw point-cloud scale, point density and exposure all
      coincide across old10k and Sorghum_15K — ruling out a render-domain shift as the cause of the
      OOD collapse.</figcaption></figure>
</div>""")

    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(pages)}</body></html>"
    HTML(string=html, base_url='.').write_pdf(a.out)
    print(f"Saved {a.out}  ({Path(a.out).stat().st_size / 1e6:.1f} MB, {len(pages)} pages)")


if __name__ == '__main__':
    main()
