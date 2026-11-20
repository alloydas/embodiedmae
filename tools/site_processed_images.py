"""The project page's RGB images as the models read them (user, 2026-11-20: "You process the sorghum and maize before
training, use that RGB images in the website as you showed in S4 ... also in the data three kind isolated plants the RGB
images are old").

  real section (data/real/<sp>/*.jpg)          the processed input of each stop, as S4's Image column: the window around
                                               the middle plant cut from the photograph, background black, sorghum at
                                               the run's brightness gain (1.85); captured at the window's native size,
                                               before the loader's resize to 224 px (eval/real_rows.py), saved 700 px
  data section, isolated views                 each camera's render composited over BLACK (the training inputs: maize
  (data/dataset/<sp>/isolated/*_v*.jpg)        rgba.png, sorghum rgb_nobg.png, stored at the brightness E1 trained on),
                                               1024 px source, saved 640 px (were composited over white, 300 px)
  tiles 01 / 06                                rebuilt from the same processed images
File names are unchanged, so no JSON or page change is needed.
  python tools/site_processed_images.py
"""
import sys, pathlib, json, csv
R = pathlib.Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'figures'), str(R / 'eval')]
import numpy as np
from PIL import Image
import _plantviz as V
from real_rows import RealRowDataset, ROWS_ROOT

SITE = R / 'site/data'
GAIN = {'sorghum': json.load(open(R / 'outputs/sorghum_real_ft/config.json')).get('rgb_brighten_gain', 1.0), 'maize': None}


def save(im, path, side, q=90):
    im = im.convert('RGB'); s = side / max(im.size)
    if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(path, quality=q); return im.size


def processed_window(ds, i):
    """the canvas the loader builds for item i (window, black background, gain), before its resize to 224"""
    got = {}; tf = ds.tf
    ds.tf = lambda img: (got.__setitem__('im', img.copy()), tf(img))[1]
    try: ds[i]
    finally: ds.tf = tf
    return got['im']


def strip_of(im):
    """the processed window cropped to the photograph's columns: drops the black padding the loader adds beside a tall
    sorghum strip (the padding columns are exactly zero; the processing itself is untouched)"""
    cols = np.flatnonzero((np.asarray(im) > 0).any(axis=(0, 2)))
    return im.crop((int(cols.min()), 0, int(cols.max()) + 1, im.height)) if len(cols) else im


def over_black(path):
    im = Image.open(path).convert('RGBA'); b = Image.new('RGBA', im.size, (0, 0, 0, 255)); b.alpha_composite(im); return b.convert('RGB')


if __name__ == '__main__':
    # real section: every stop's processed window
    for sp in ('maize', 'sorghum'):
        for f in sorted((SITE / 'real' / sp).glob('*.json')):
            if f.name == 'index.json': continue
            m = json.loads(f.read_text())
            ds = RealRowDataset(layouts=(m['layout'],), rows=[m['row']], stops={int(m['stop'])}, seed=0, species=sp, rgb_gain=GAIN[sp])
            print(sp, f.stem, save(processed_window(ds, 0), SITE / 'real' / sp / m['img'], 700), flush=True)
    # data section: isolated views over black, from the 1024 px renders
    for sp in ('maize', 'sorghum'):
        for f in sorted((SITE / 'dataset' / sp / 'isolated').glob('*.json')):
            m = json.loads(f.read_text())
            for v, name in enumerate(m['views']):
                k = int(name.rsplit('_v', 1)[1].split('.')[0])
                src = next(V.folder(sp, s_, m['plant'], k) for s_ in ('test', 'val', 'train') if V.folder(sp, s_, m['plant'], k).exists())
                save(over_black(src / ('rgb_nobg.png' if sp == 'sorghum' else 'rgba.png')), f.parent / name, 640)
            print(sp, 'isolated', m['plant'], len(m['views']), 'views', flush=True)
    # tiles: 01 (field photo | isolated over black / synthetic row) and 06 (a real stop's processed window)
    def cover(im, w, h, fy=0.5):
        W0, H0 = im.size; s = max(w / W0, h / H0); im = im.resize((round(W0 * s), round(H0 * s)), Image.LANCZOS)
        x0, y0 = (im.width - w) // 2, int((im.height - h) * fy); return im.crop((x0, y0, x0 + w, y0 + h))
    W, H, g = 1229, 1536, 8; lw = (W - g) // 2; rw = W - g - lw; rh = (H - g) // 2
    iso = over_black(V.folder('maize', 'test', 'plant_0106', 3) / 'rgba.png')
    row = over_black(V.row_shot_dir('maize', 'test', 'plant_0320', 0) / 'rgba.png')
    fld = Image.open(ROWS_ROOT['field1000'] / 'maize' / 'png' / 'row219_seed1_stop06_full.png').convert('RGB')
    T = Image.new('RGB', (W, H), 'white')
    T.paste(cover(fld, lw, H), (0, 0)); T.paste(cover(iso, rw, rh), (lw + g, 0)); T.paste(cover(row, rw, H - rh - g), (lw + g, rh + g))
    T.save(SITE / 'tiles/data.jpg', quality=90)
    ds = RealRowDataset(layouts=('parallel',), rows=['0019'], stops={17}, seed=0, species='maize', rgb_gain=None)
    win = processed_window(ds, 0); cover(win, round(win.height * 0.8), win.height).save(SITE / 'tiles/real.jpg', quality=90)
    print('tiles: data', T.size, 'real', win.size, flush=True)
