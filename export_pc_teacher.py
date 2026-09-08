"""Export the point clouds behind "What the teacher reconstructs" for the web viewer.

That plate is figures_fixed/pretrain/epoch_960_sample_1_Sorghum_1001_02.png, written
by regen_figures.py's second section:

    ds = SorghumDataset4M(f"{root}/val", ...)          # config_4m_distill_15k_all.yaml
    i_p = find('Sorghum_1001_02')
    pl  = DataLoader(Subset(ds,[i_p]), batch_size=1, num_workers=4)
    pm  = DIS.build_model(args, 'cuda');  load  outputs/4m_pretrain_15k_v2_depthfix_qal/best_model.pth
    torch.manual_seed(0)
    PRE.visualize_reconstruction_4m(pm, pl, 'cuda', 960, d, num_samples=1, mask_ratio=0.80)

Reproducing it on CPU means reproducing two separate random streams.

1. THE CLOUD.  SorghumDataset.load_pointcloud draws a fresh np.random 8196-point
   subsample of the raw .ply on every read, so the ground truth itself depends on
   numpy's state.  In the original that state was set inside DataLoader worker 0:
   torch's _worker_loop does np.random.seed(_generate_state(base_seed, worker_id)),
   where base_seed is the single int64 the iterator draws from the global CPU
   generator.  With torch.manual_seed(0) immediately before, that base_seed is
   deterministic -- so seeding numpy the same way here reproduces the plate's cloud
   exactly, point for point.

2. THE MASK.  random_masking_dirichlet draws the per-modality visible-token budget
   from a CPU Dirichlet, then shuffles positions with torch.rand on the tensors'
   own device.  The budget is therefore reproducible here and the shuffle is not:
   the original ran on CUDA, whose generator has no CPU equivalent.  The budget
   check below asserts the four counts against the numbers printed on the plate
   (RGB 156, depth 165, PC 163, text 7 of their token streams), which pins the
   whole CPU chain; WHICH 33 of the 196 PC tokens stayed visible is a fresh draw.
   Hence `visible` and `recon` are a re-roll at the plate's exact mask budget, and
   only `gt` is bit-identical.  See the note field in the JSON.

Panels follow the plate: grey ground truth, the plate's green for the points the
encoder actually saw (row 4 left: the union of the kNN groups of the visible FPS
centres), and the page's model-output green for the reconstruction.
"""
import argparse, json, pathlib
import numpy as np
import torch
import yaml
from torch.utils.data._utils.worker import _generate_state

from sorghum_dataset_4m import SorghumDataset4M
from eval_rgb2pc_quant import chamfer_both
from export_pc_web import pack                      # base64 int16 packer, shared
import train_sorghum_4m_distill as DIS
from train_sorghum_4m import _pc_token_membership

CFG_PATH = 'configs/config_4m_distill_15k_all.yaml'
CKPT     = 'outputs/4m_pretrain_15k_v2_depthfix_qal/best_model.pth'
PLANT    = 'Sorghum_1001_02'
HEADING  = 'What the teacher reconstructs'
# The four counts printed on the static plate, in MODALITIES order.
PLATE_MASKED = {'rgb': 156, 'depth': 165, 'pc': 163, 'text': 7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_points', type=int, default=2600)
    ap.add_argument('--plant', default=PLANT)
    ap.add_argument('--out', default='vis_gallery/pc_teacher.json')
    args_cli = ap.parse_args()

    cfg  = yaml.safe_load(open(CFG_PATH))
    args = DIS.config_to_namespace(cfg)
    root = cfg['data']['data_root']

    # regen_figures.py passed the val directory as data_root with split=None; the
    # folder list and its order are identical either way.
    ds = SorghumDataset4M(f'{root}/val', img_size=args.img_size,
                          num_points=args.num_points)
    names = [p.name for p in ds.samples]
    idx = next((i for i, n in enumerate(names) if args_cli.plant in n), None)
    if idx is None:
        raise SystemExit(f'{args_cli.plant} is not in the val split')
    print(f'val split {len(ds)} folders; {args_cli.plant} at index {idx}')

    # Built and loaded BEFORE the seed, exactly as in regen_figures.py, so the
    # weight-init draws do not sit inside the reproduced stream.
    model = DIS.build_model(args, 'cpu')
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, 'checkpoint does not match the model'
    model.eval()
    epoch = int(ck.get('epoch', -1))
    print(f'checkpoint epoch {epoch}  val_loss {ck.get("val_loss"):.5f}  '
          f'mask_ratio {args.mask_ratio}')

    # ── replay the original random stream ────────────────────────────────────
    torch.manual_seed(0)
    base_seed = torch.empty((), dtype=torch.int64).random_().item()   # iterator draw
    np.random.seed(_generate_state(base_seed, 0))                     # worker 0
    rgb, depth, pc, par, tv, name = ds[idx]
    print(f'loaded {name}  base_seed {base_seed}')

    b = lambda t: t.unsqueeze(0)
    with torch.no_grad():
        total, (l_rgb, l_dep, l_pc, l_txt), (_, _, pred_pc, _), masks = model(
            b(rgb), b(depth), b(pc), b(par), b(tv), mask_ratio=args.mask_ratio)
        fps_idx = model.pc_embed.fps(b(pc), model.num_pc_tokens)

    got = {n: int(m[0].sum()) for n, m in zip(('rgb', 'depth', 'pc', 'text'), masks)}
    print(f'mask budget {got}  plate {PLATE_MASKED}')
    assert got == PLATE_MASKED, (
        'the Dirichlet token budget does not match the static plate -- the RNG '
        'replay is broken, do not publish this')

    # ── the three panels ─────────────────────────────────────────────────────
    m_pc     = masks[2][0].numpy()
    member   = _pc_token_membership(b(pc), fps_idx, model.pc_embed.group_size)
    vis_ids  = np.unique(member[m_pc == 0].flatten())
    gt       = pc.numpy()
    recon    = pred_pc[0].numpy()
    visible  = gt[vis_ids]

    f, r = chamfer_both(pred_pc, b(pc))
    cd = float(f + r)
    n_vis_tok = int((m_pc == 0).sum())
    print(f'PC tokens visible {n_vis_tok}/196   visible points {len(vis_ids)}/{len(gt)}')
    print(f'chamfer {cd:.6f}   QAL pc loss {l_pc.item():.6f}   total {total.item():.4f}')

    rng = np.random.default_rng(0)
    n = args_cli.n_points
    row = {
        'label': args_cli.plant.replace('Sorghum_', ''),
        'name': name,
        'chamfer': round(cd, 8),
        'clouds': {'gt': pack(gt, n, rng),
                   'visible': pack(visible, n, rng),
                   'recon': pack(recon, n, rng)},
        'meta': [
            ['masking', f'{args.mask_ratio:.0%} of all tokens, Dirichlet-allocated'],
            ['PC tokens seen', f'{n_vis_tok} of 196 ({len(vis_ids)} of {len(gt)} points)'],
            ['other streams masked', f'RGB {got["rgb"]}/196, depth {got["depth"]}/196, '
                                     f'params {got["text"]}/25'],
            ['Chamfer, recon vs GT', f'{cd:.5f}'],
            ['PC loss (QAL)', f'{l_pc.item():.6f}'],
            ['total val loss', f'{total.item():.4f}'],
        ],
    }

    out = {
        'id': 'teacher_recon',
        'source_figure': HEADING,
        'checkpoint': CKPT,
        'epoch': epoch,
        'split': 'val',
        'n_points': n,
        'panels': [
            {'key': 'gt',      'label': 'ground truth',                'color': '#9aa7ad'},
            {'key': 'visible', 'label': 'what the encoder saw',        'color': '#2ecc71'},
            {'key': 'recon',   'label': 'reconstruction',              'color': '#1baf7a'},
        ],
        'rows': [row],
        'note': ("Replays regen_figures.py's pretrain leg, which produced the static plate "
                 "figures_fixed/pretrain/epoch_960_sample_1_Sorghum_1001_02.png: val plant "
                 "Sorghum_1001_02 through the 15k v2 pretrain best_model.pth at mask_ratio "
                 "0.80, with torch.manual_seed(0) and numpy re-seeded to the value torch "
                 "gave DataLoader worker 0, so the ground-truth cloud is the plate's exact "
                 "8196-point draw and the Dirichlet mask budget matches it on all four "
                 "streams; the original shuffled tokens with the CUDA generator, so which "
                 "33 PC tokens stayed visible is re-drawn here."),
    }

    p = pathlib.Path(args_cli.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {p}  ({p.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
