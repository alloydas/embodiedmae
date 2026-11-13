"""UGG-style PREVIEW variants of three qualitative image grids (UGG = Unified Generative Geometry Model,
arXiv:2611.14440, its Figs. 4 and 6; conventions in figures/ugg_style.py). 2026-11-13.

  fig_qual              figures/make_qual_fig.py --ugg              (models on CPU, cached in DUMPS/fig_qual_dump.pkl)
  fig_e10_real          figures/make_e10_fig.py --species maize --ugg    (DUMPS/fig_e10_real_dump.pkl)
  fig_e10_real_sorghum  figures/make_e10_fig.py --species sorghum --ugg  (DUMPS/fig_e10_real_sorghum_dump.pkl)
  fig_dataset           figures/make_dataset_fig.py --ugg           (data only, no dump)

What changes against the paper versions: bold column headers and bold row labels, our model's column in OURS violet,
every tile in a rounded pastel box of its stream (image blue, depth yellow, point cloud pink, recipe purple; dashed =
input, solid = output; ground truth in a plain grey box), thin white gaps, numbers under / inside the tiles, and in
fig_qual a zoom-in column (generated cloud vs ground truth) with the zoomed region marked on both tiles.
What does not change: fonts and font sizes (each script's own), the data, plants, numbers, point colours and the figure
size in inches. Outputs go ONLY to paper/figs/ugg_preview/<name>_ugg.{pdf,png}.

Re-render from the dumps (no model):  python figures/ugg_grids.py qual|e10_maize|e10_sorghum
"""
import sys, pathlib, pickle
R = pathlib.Path(__file__).resolve().parents[1]
for p in (str(R), str(R / 'figures')):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ugg_style as U

OUT = R / 'paper/figs/ugg_preview'
DUMPS = pathlib.Path('/work/mech-ai/alloydas/embodiedmae_outputs/_fig_dumps')
GT_BOX = None          # pastel_box(stream=None) = the plain grey box used for ground truth
GT_FILL = '#f4f4f5'    # its fill (ugg_style.pastel_box)


class Canvas:
    """A figure laid out in inches: an invisible overlay axes for boxes and text, and axes placed by inch rects."""
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H))
        self.ov = self.fig.add_axes([0, 0, 1, 1]); self.ov.set_xlim(0, W); self.ov.set_ylim(0, H); self.ov.set_axis_off()
        self.ov.patch.set_alpha(0)

    def axes(self, x, y, w, h, **kw):
        ax = self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H], **kw)
        ax.patch.set_alpha(0); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        return ax

    def box(self, x, y, w, h, stream=None, dashed=False, r=0.05):
        return U.pastel_box(self.ov, x, y, w, h, stream=stream, dashed=dashed, r=r)

    def header(self, x, y, text, size, color='#111111'):
        return U.header(self.fig, x / self.W, y / self.H, text, color=color, size=size, linespacing=1.05)

    def row_label(self, x, y, text, size, ours=False):
        return U.row_label(self.fig, x / self.W, y / self.H, text, ours=ours, size=size, linespacing=1.1)

    def text(self, x, y, s, **kw):
        return self.ov.text(x, y, s, **kw)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.pdf', dpi=300); fig.savefig(OUT / f'{stem}.png', dpi=220); plt.close(fig)
    print('wrote', OUT / f'{stem}.pdf', OUT / f'{stem}.png', flush=True)


def over(rgb, alpha, bg):
    """RGB (H,W,3) in [0,1] composited over a flat background colour with alpha (H,W) or (H,W,1)."""
    a = alpha if alpha.ndim == 3 else alpha[..., None]
    return rgb * a + (1 - a) * np.asarray(matplotlib.colors.to_rgb(bg))


# ============================================================================================ fig_qual
def project(P, elev=15, azim=-40):
    """The view of make_qual_fig.scat (a 3-D scatter of (x, z, y) at view_init(15, -40)) as an orthographic 2-D
    projection: (screen x, screen y, depth towards the viewer). Drawn in 2-D so the cloud fills its tile and the
    zoom-in region can be marked exactly."""
    Q = np.asarray(P, dtype=np.float64)[:, [0, 2, 1]]
    e, a = np.radians(elev), np.radians(azim)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return Q @ right, Q @ up, Q @ toward


def scat2d(ax, P, col, win, s=0.5):
    x, y, z = project(P); o = np.argsort(z)                      # far points first
    ax.scatter(x[o], y[o], s=s, c=col, alpha=0.8, linewidths=0, rasterized=True)
    (cx, cy), h = win
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h); ax.set_aspect('equal')


def zoom_window(gt, win, frac, centre=None):
    """Square zoom window of side frac * (main window side); default centre = the densest such window of the truth."""
    (cx, cy), h = win; zh = frac * h
    if centre is not None:
        return (cx + centre[0] * h, cy + centre[1] * h), zh
    x, y, _ = project(gt)
    cand = np.stack([x, y], 1)[::7]
    cnt = [np.sum((np.abs(x - px) < zh) & (np.abs(y - py) < zh)) for px, py in cand]
    px, py = cand[int(np.argmax(cnt))]
    return (px, py), zh


QUAL_ZOOM = {'sorghum': dict(frac=0.22, centre=(0.30, 0.18)), 'maize': dict(frac=0.22, centre=(0.12, 0.30))}
QUAL_COL = {'before': '#9a6a00', 'after': '#1f6f8b', 'gt': '#2f6b2a'}    # the paper figure's point colours, unchanged


def render_qual(D, stem='fig_qual_ugg'):
    """make_qual_fig.py's figure (11 x 4.6 in, its font sizes: 7.5 titles, 7 x label, 6 ticks / legend)."""
    W, H = 11.0, 4.6
    C = Canvas(W, H)
    LW, t, g, HD = 1.0, 1.6, 0.08, 0.40
    zc = (t - g) / 2                                             # two stacked zoom crops in one column
    xs = [LW + k * (t + g) for k in range(4)]
    xz = xs[3] + t + g
    xp = xz + zc + g + 0.04; pw = W - 0.04 - xp
    heads = [('Input\nphotograph', '#111111'), ('Headline model\n(before)', '#111111'),
             ('Distilled generalist\n(after)', U.OURS), ('Ground truth', '#111111')]
    yh = H - HD + 0.06
    for (s, c), x in zip(heads, xs): C.header(x + t / 2, yh, s, 7.5, c)
    C.header(xz + zc / 2, yh, 'Zoom-in', 7.5)
    C.header(xp + pw / 2, yh, 'Recipe: leaf length\nper leaf (normalised)', 7.5)
    for r, (sp, d) in enumerate(D['species'].items()):
        y = H - HD - t - r * (t + 0.24 + 0.40)
        C.row_label(LW - 0.1, y + t / 2, f'{sp.capitalize()}\nval plant\n{d["plant"].split("_")[-1]}', 7.5)
        gx, gy, _ = project(d['gt'])
        lo, hi = np.array([gx.min(), gy.min()]), np.array([gx.max(), gy.max()])
        win = ((lo + hi) / 2, (hi - lo).max() / 2 * 1.06)
        zwin = zoom_window(d['gt'], win, **QUAL_ZOOM[sp])
        m = 0.03
        # photograph (input, dashed blue)
        C.box(xs[0], y, t, t, 'rgb', dashed=True)
        ax = C.axes(xs[0] + m + 0.02, y + m + 0.02, t - 2 * m - 0.04, t - 2 * m - 0.04); ax.imshow(d['img'])
        # clouds: two outputs (solid pink), the truth (plain grey)
        tiles = [('before', d['pb'], 'pc'), ('after', d['pa'], 'pc'), ('gt', d['gt'], GT_BOX)]
        for k, (nm, P, st) in enumerate(tiles):
            x = xs[1 + k]
            C.box(x, y, t, t, st, dashed=False)
            ax = C.axes(x + m, y + m, t - 2 * m, t - 2 * m)
            scat2d(ax, P, QUAL_COL[nm], win)
            if nm in ('after', 'gt'):
                (zx, zy), zh = zwin
                U.zoom_box(ax, zx - zh, zy - zh, 2 * zh, 2 * zh, color='#333333', lw=0.8)
            if nm != 'gt':
                ch = d['ch_b'] if nm == 'before' else d['ch_a']
                C.text(x + t / 2, y - 0.04, f'Chamfer {ch * 1e3:.2f}e-3', fontsize=7.5, ha='center', va='top',
                       color=U.OURS if nm == 'after' else '#111111', fontweight='bold' if nm == 'after' else 'normal')
        # zoom column: generated (top, pink) over truth (bottom, grey), same window
        for k, (nm, P, st) in enumerate((('after', d['pa'], 'pc'), ('gt', d['gt'], GT_BOX))):
            yy = y + t - zc - k * (zc + g)
            C.box(xz, yy, zc, zc, st, dashed=False, r=0.04)
            ax = C.axes(xz + 0.02, yy + 0.02, zc - 0.04, zc - 0.04)
            scat2d(ax, P, QUAL_COL[nm], zwin, s=1.3)
            ax.text(0.05, 0.95, 'after' if nm == 'after' else 'truth', transform=ax.transAxes, fontsize=6, va='top',
                    ha='left', color=U.OURS if nm == 'after' else '#111111', fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc=U.STREAM_FILL['pc'] if nm == 'after' else GT_FILL, ec='none', alpha=0.85))
        # recipe plot (an output of the photograph: solid purple)
        C.box(xp, y, pw, t, 'text', dashed=False)
        ax = C.fig.add_axes([(xp + 0.36) / W, (y + 0.34) / H, (pw - 0.46) / W, (t - 0.44) / H])
        ax.patch.set_alpha(0); U.style_axes(ax)
        n = len(d['par_len']); k = np.arange(1, n + 1)
        ax.plot(k, d['par_len'], 'o-', ms=2.5, lw=1, color=QUAL_COL['gt'], label='ground truth')
        ax.plot(k, d['para_len'], 's--', ms=2.5, lw=1, color=QUAL_COL['after'], label='from the photograph')
        ax.set_xlabel('leaf index (base to top)', fontsize=7); ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
        C.text(xp + pw / 2, y - 0.04, f'MAE {d["mae"]:.3f}', fontsize=7.5, ha='center', va='top')
    save(C.fig, stem)


# ============================================================================================ fig_e10_real
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list('plant', ['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a'])


def e10_cloud(ax, P, xlim, ylim, sorg):
    """make_e10_fig.draw_cloud, verbatim (leaf greens dark at the base to light at the top, nearer points brighter)."""
    P = np.asarray(P, dtype=np.float64)
    h = (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9)
    near = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)
    c = GREENS(0.15 + 0.85 * h)[:, :3] * (0.55 + 0.45 * near)[:, None]
    o = np.argsort(P[:, 2])
    ax.scatter(P[o, 0], P[o, 1], s=0.25 if not sorg else 0.18, c=c[o], linewidths=0, rasterized=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('auto' if sorg else 'equal')


def render_e10(D, stem):
    """make_e10_fig.py's figure: maize 12.25 x 7.4 in (titles 7.5, labels 7, numbers 6.5), sorghum 5.95 x 4.7 in
    (titles 6.2, labels 6.5, numbers 5.6)."""
    sorg = D['species'] == 'sorghum'
    W, H = D['figsize']; nr, nc = len(D['rows']), len(D['cols'])
    tfs, lfs, nfs = (6.2, 6.5, 5.6) if sorg else (7.5, 7.0, 6.5)
    C = Canvas(W, H)
    LW = 0.40 if sorg else 0.86
    HD = 0.44 if sorg else 0.40
    g = 0.045 if sorg else 0.06
    tw = (W - LW - 0.03 - (nc - 1) * g) / nc
    gy = 0.08 if sorg else 0.10
    th = (H - HD - 0.03 - (nr - 1) * gy) / nr
    if not sorg:                                                 # square tiles; spare height goes to the row gaps
        th = tw = min(tw, th); gy = (H - HD - 0.03 - nr * th) / (nr - 1) if nr > 1 else gy
    kinds = [('rgb', True), ('depth', True), ('pc', True), (GT_BOX, False), ('pc', False), ('pc', False), ('pc', False)]
    xs = [LW + j * (tw + g) for j in range(nc)]
    for j, c in enumerate(D['cols']):
        C.header(xs[j] + tw / 2, H - HD + 0.05, c, tfs, U.OURS if j == nc - 1 else '#111111')
    m = 0.03
    xlim, ylim = D['lims']
    for i, rw in enumerate(D['rows']):
        y = H - HD - th - i * (th + gy)
        if sorg:                                                 # tall strips: the label runs up the side
            C.text(LW - 0.07, y + th / 2, rw['label'].replace('\nstop', ', stop'), fontsize=lfs, fontweight='bold',
                   rotation=90, ha='right', va='center', ma='center', color='#111111', linespacing=1.1)
        else:
            C.row_label(LW - 0.07, y + th / 2, rw['label'], lfs)
        for j in range(nc):
            st, dashed = kinds[j]
            C.box(xs[j], y, tw, th, st, dashed=dashed, r=0.04 if sorg else 0.05)
            ax = C.axes(xs[j] + m, y + m, tw - 2 * m, th - 2 * m)
            if j == 0:
                ax.imshow(rw['img'], aspect='auto' if sorg else 'equal')
            elif j == 1:
                ax.imshow(rw['depth'], cmap='viridis_r', aspect='auto' if sorg else 'equal')
                ax.patch.set_alpha(1); ax.set_facecolor('black')
            elif j == 2:
                e10_cloud(ax, rw['pin'], xlim, ylim, sorg)
            elif j == 3:
                e10_cloud(ax, rw['pgt'], xlim, ylim, sorg)
                ax.text(0.03, 0.02, rw['truth_txt'], transform=ax.transAxes, fontsize=nfs, zorder=5,
                        bbox=dict(boxstyle='square,pad=0.1', fc=GT_FILL, ec='none', alpha=0.8))
            else:
                P, txt = rw['preds'][j - 4]
                e10_cloud(ax, P, xlim, ylim, sorg)
                ax.text(0.03, 0.02, txt, transform=ax.transAxes, fontsize=nfs, zorder=5,
                        color=U.OURS if j == nc - 1 else 'black', fontweight='bold' if j == nc - 1 else 'normal',
                        bbox=dict(boxstyle='square,pad=0.1', fc=U.STREAM_FILL['pc'], ec='none', alpha=0.8))
    save(C.fig, stem)


# ============================================================================================ fig_dataset
def render_dataset(a, elev_tab, stem='fig_dataset_ugg'):
    """make_dataset_fig.py's figure (7.1 x 5.9 in; FS 6.2, species FS+0.5, letters 9 bold, recipe 4.8 mono,
    camera labels FS-0.5, row labels FS-0.6). Every stream tile in its stream's box: the dataset's streams are what
    a model reads, so dashed (input); the row task's target-alone render is the truth (plain grey, solid).
    Transparent backgrounds are composited onto the box colour instead of white (no pixel of a plant changes)."""
    import _plantviz as V
    from scipy.spatial import cKDTree
    from PIL import Image
    PL = {'sorghum': a.sorghum, 'maize': a.maize}; RP = {'sorghum': a.row_sorghum, 'maize': a.row_maize}
    FS = 6.2
    FIELDS = {'sorghum': None, 'maize': ('dist', 'len', 'wid', 'angle', 'droop')}
    W, H = 7.1, 5.9
    C = Canvas(W, H)
    L, Rr = 0.21, W - 0.035
    F = U.STREAM_FILL

    def rgba_on(path, bg, size=None):
        im = Image.open(path).convert('RGBA')
        if size: im = im.resize((size, size), Image.LANCZOS)
        x = np.asarray(im).astype(np.float32) / 255
        return over(x[..., :3], x[..., 3], bg)

    def letter(y, ch, cap):
        C.text(0.035, y, ch, fontsize=9, fontweight='bold', va='baseline')
        C.text(0.21, y, cap, fontsize=FS, va='baseline')

    # ---- a. four streams of one view
    letter(5.78, 'a', 'Four streams per view: photograph, depth frame, point cloud and the recipe as text (a plant line, then one line per leaf; maize: 5 of its 7 leaf fields)')
    g, half_gap = 0.04, 0.08
    bw = (Rr - L - half_gap) / 2
    tw = (bw - 3 * g - 1.48) / 3                       # three square tiles + the recipe (1.48 in: its widest line)
    bh = 0.98; ya = 4.42
    for j, sp in enumerate(('sorghum', 'maize')):
        x0 = L + j * (bw + half_gap)
        C.header(x0 + bw / 2, ya + bh + 0.16, sp, FS + 0.5)
        f = V.folder(sp, 'val', PL[sp], 0)
        ds = V.dataset(sp); it = V.item_by_name(ds, f'{PL[sp]}_00')
        xs = [x0 + k * (tw + g) for k in range(3)]
        for k, (st, nm) in enumerate((('rgb', 'photograph'), ('depth', 'depth'), ('pc', 'point cloud'))):
            C.box(xs[k], ya, tw, bh, st, dashed=True, r=0.04)
            C.header(xs[k] + tw / 2, ya + bh + 0.03, nm, FS)
        s = tw - 0.04; yi = ya + (bh - s) / 2
        ax = C.axes(xs[0] + 0.02, yi, s, s); ax.imshow(rgba_on(f / V.RGB_FILE[sp], F['rgb']))
        dimg = V.depth_rgb(V.decode_depth_png(f / 'depth.png')); dpx = V.decode_depth_png(f / 'depth.png') > 0
        dimg[~dpx] = matplotlib.colors.to_rgb(F['depth'])
        ax = C.axes(xs[1] + 0.02, yi, s, s); ax.imshow(dimg)
        ax = C.axes(xs[2] + 0.02, yi, s, s, projection='3d'); P = it[2].numpy()
        V.scatter_cloud(ax, P, c=-P[:, 1], cmap='viridis', s=0.5, fs=FS, elev=12, azim=-35)
        ax.patch.set_alpha(0); ax.set_facecolor((1, 1, 1, 0))
        xr = xs[2] + tw + g; rw_ = x0 + bw - xr
        C.box(xr, ya, rw_, bh, 'text', dashed=True, r=0.04)
        C.header(xr + rw_ / 2, ya + bh + 0.03, 'recipe', FS)
        ax = C.axes(xr + 0.03, ya + 0.03, rw_ - 0.06, bh - 0.06)
        V.draw_recipe_text(ax, V.recipe_lines(sp, V.decode_recipe(sp, it[3], it[4]), max_leaves=6, fields=FIELDS[sp]),
                           V.C['gt'], fs=FS, tfs=4.8)

    # ---- b. ten cameras
    letter(ya - 0.42, 'b', 'Ten cameras per plant, labelled by camera elevation above the plant: sorghum on a fixed ladder, maize at random elevations and azimuths (sorted here)')
    n = 10; gb = 0.035
    tb = (Rr - L - 2 * 0.03 - (n - 1) * gb) / n
    bhb = tb + 0.20
    for r, sp in enumerate(('sorghum', 'maize')):
        if sp == 'sorghum':
            order = list(range(10)); el = V.SORGHUM_ELEV
        else:
            t_ = elev_tab[(elev_tab.split == 'val') & (elev_tab.plant == PL[sp])].sort_values('elev_deg', ascending=False)
            order = t_['view'].tolist(); el = t_.elev_deg.values
        yb = ya - 0.42 - 0.12 - bhb - r * (bhb + 0.16)
        C.box(L, yb, Rr - L, bhb, 'rgb', dashed=True, r=0.05)
        C.text(0.105, yb + bhb / 2, sp, fontsize=FS, rotation=90, va='center', ha='center', fontweight='bold')
        for k, (v, e) in enumerate(zip(order, el)):
            xk = L + 0.03 + k * (tb + gb)
            ax = C.axes(xk, yb + 0.03, tb, tb)
            ax.imshow(rgba_on(V.folder(sp, 'val', PL[sp], v) / V.RGB_FILE[sp], F['rgb'], 256))
            C.text(xk + tb / 2, yb + 0.03 + tb + 0.025, f'{e:+.0f}°', fontsize=FS - 0.5, ha='center', va='bottom')

    # ---- c. rows
    yc_cap = ya - 0.42 - 0.12 - 2 * bhb - 0.16 - 0.42
    letter(yc_cap, 'c', 'Rows: the target between two neighbours. Red = neighbour pixels, orange = target pixels hidden behind a neighbour, grey points = neighbours')
    gc = 0.05
    tc = (Rr - L - half_gap - 6 * gc) / 8
    yc = yc_cap - 0.52 - tc
    for j, sp in enumerate(('sorghum', 'maize')):
        x0 = L + j * (4 * tc + 3 * gc + half_gap)
        C.header(x0 + (4 * tc + 3 * gc) / 2, yc + tc + 0.27, sp, FS + 0.5)
        xs = [x0 + k * (tc + gc) for k in range(4)]
        for k, th in enumerate((0, 90)):
            d = V.row_shot_dir(sp, 'test', RP[sp], th)
            _, stt = V.row_overlay(sp, d)                      # the hidden / neighbour shares, as in the paper
            base = rgba_on(d / 'rgba.png', F['rgb'])            # V.row_overlay's tint, over the box colour
            tgt, nbr, hid = V.row_masks(sp, d)
            img = base.copy()
            img[nbr] = 0.45 * base[nbr] + 0.55 * np.array((0.86, 0.22, 0.2))
            img[hid] = 0.35 * base[hid] + 0.65 * np.array((0.95, 0.55, 0.1))
            xi = xs[2 * k]
            C.box(xi, yc, tc, tc, 'rgb', dashed=True, r=0.04)
            ax = C.axes(xi + 0.025, yc + 0.025, tc - 0.05, tc - 0.05); ax.imshow(img)
            C.header(xi + tc / 2, yc + tc + 0.03, f'yaw {th}°', FS - 0.6)
            C.text(xi + tc / 2, yc - 0.03, f'{stt["occ"] * 100:.0f} % hidden', fontsize=FS - 0.6, ha='center', va='top')
            if k == 0:
                xt = xs[1]
                C.box(xt, yc, tc, tc, GT_BOX, dashed=False, r=0.04)
                ax = C.axes(xt + 0.025, yc + 0.025, tc - 0.05, tc - 0.05); ax.imshow(rgba_on(d / 'gt_rgba.png', GT_FILL))
                C.header(xt + tc / 2, yc + tc + 0.03, 'target alone\n(same camera)', FS - 0.6)
            else:
                scene = V.read_cloud(d / 'pointcloud_cam.ply'); tgt = V.read_cloud(d / 'gt_pointcloud_cam.ply')
                is_t = cKDTree(tgt).query(scene, workers=1)[0] < 3e-3
                xt = xs[3]
                C.box(xt, yc, tc, tc, 'pc', dashed=True, r=0.04)
                ax = C.axes(xt + 0.01, yc + 0.01, tc - 0.02, tc - 0.02, projection='3d')
                col = np.where(is_t[:, None], np.array([[0.18, 0.42, 0.16]]), np.array([[0.75, 0.75, 0.75]]))
                V.scatter_cloud(ax, scene, c=col, s=0.4, fs=FS - 0.6, elev=10, azim=-30)
                ax.patch.set_alpha(0); ax.set_facecolor((1, 1, 1, 0))
                C.header(xt + tc / 2, yc + tc + 0.03, 'scene cloud', FS - 0.6)
                C.text(xt + tc / 2, yc - 0.03, f'{100 * (~is_t).mean():.0f} % neighbours', fontsize=FS - 0.6, ha='center', va='top')
    save(C.fig, stem)


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'qual':
        render_qual(pickle.load(open(DUMPS / 'fig_qual_dump.pkl', 'rb')))
    elif which in ('e10_maize', 'e10_sorghum'):
        nm = 'fig_e10_real' if which == 'e10_maize' else 'fig_e10_real_sorghum'
        render_e10(pickle.load(open(DUMPS / f'{nm}_dump.pkl', 'rb')), f'{nm}_ugg')
    else:
        raise SystemExit('usage: python figures/ugg_grids.py qual|e10_maize|e10_sorghum')
