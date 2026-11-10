"""Build the clean CropCode code release from this working repository.

usage: python tools/make_release.py OUT_DIR

Copies only the code behind the paper's final-data results (models, loaders, trainers, evaluation,
split tooling, the final configs), rewrites cluster paths to repository-relative ones (data/, outputs/),
strips internal notes from comments, sets every config's batch shape to the one its run recorded in
outputs/<run>/config.json (same global batch as the paper), and writes README.md, environment.yml and
.gitignore. OUT_DIR is replaced. The release is the input of the separate `cropcode` repository.
"""
import json, pathlib, re, shutil, sys
import yaml

SRC = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(sys.argv[1]).resolve()

ROOT_FILES = ['embodied_mae.py', 'embodied_mae_4m.py', 'embodied_mae_4m_maize.py', 'pc_sinkhorn.py',
              'species_conventions.py', 'sorghum_dataset.py', 'sorghum_dataset_4m.py', 'maize_dataset_4m.py',
              'row_dataset.py', 'train_sorghum_4m.py', 'train_maize_4m.py', 'train_sorghum_4m_distill.py',
              'train_maize_4m_distill.py', 'train_row_4m.py', 'train_supervised_4m.py', 'train_supervised_maize_4m.py',
              'utils.py']
EVAL_FILES = ['linear_probe', 'linear_probe_maize', 'latent_analysis', 'latent_analysis_maize', 'probe_pair_bootstrap',
              'baseline_probe', 'cross_species_probe', 'eval_row', 'row_pair_bootstrap', 'pc_fscore', 'pc_fscore_maize',
              'pc_view_eval', 'pc_view_eval_maize', 'pc_view_masks', 'pc_view_bootstrap', 'regime1_eval', 'regime1_extra',
              'eval_specialists', 'eval_warmstart', 'recon_compare', 'real_rows', 'eval_real_rows']
SPLIT_FILES = ['data_split/make_split.py', 'data_split/make_leaf_width_targets.py', 'data_split/assignment.csv',
               'data_split/features.csv', 'data_split/leaf_width_targets.csv', 'data_split/maize_view_elevation.csv.gz',
               'tools/maize_view_elevation.py']
CONFIGS = '''4m_pretrain15k_d2_sink e2_pc_d2 e2_pcrgb_d2 e2_pcrgbd_d2 e2_pcrgbdt_d2 e2_pcd_d2 e2_pct_d2 e2_pcrgbt_d2 e2_pcdt_d2
e2_pcrgbdt_tg_d2 e2_pcrgbdt_tg40_d2 e2_pcrgbdt_tg100_d2 e3_1k_d2 e3_3k_d2 e3_10k_d2 e4_small_d2 e4_large_d2 e5_topdown_d2
e5_horizon_d2 e5_single_d2 e8_supervised_d2 4m_distill_15k_all_d2 4m_distill_15k_rgb2pc_full_d2 4m_distill_15k_rgb2pc_smr50_d2
4m_distill_15k_rgb2pc_smr50b_d2 4m_distill_15k_rgbd2pc_d2 4m_distill_15k_rgbt2pc_d2 4m_distill_15k_dt2pc_d2
4m_distill_15k_pairs_d2 4m_distill_15k_pc2text_d2 4m_distill_15k_pc2text_d2_lr1e4 sorghum_row_ft sorghum_row_iso
sorghum_xfer_scratch_d2 sorghum_xfer_warm_d2 sorghum_xfer_warm_e1m_d2 maize_4m_pretrain_15k_d2 maize_4m_d2 maize_e2_pc_d2
maize_e2_pcrgb_d2 maize_e2_pcrgbd_d2 maize_e2_pcd_d2 maize_e2_pct_d2 maize_e2_pcrgbt_d2 maize_e2_pcdt_d2 maize_4m_tg_d2
maize_4m_tg40_d2 maize_4m_tg100_d2 maize_e3_1k_d2 maize_e3_3k_d2 maize_e3_10k_d2 maize_e4_small_d2 maize_e4_large_d2
maize_e5_topdown_d2 maize_e5_horizon_d2 maize_e5_single_d2 maize_e8_supervised_d2 maize_distill_e1_d2
maize_distill_rgb2pc_e1_d2 maize_distill_rgbd2pc_e1_d2 maize_distill_rgbt2pc_e1_d2 maize_distill_dt2pc_e1_d2
maize_distill_pairs2pc_e1_d2 maize_row_ft_e1 maize_row_iso_e1 maize_xfer_scratch maize_xfer_warm'''.split()
CONFIG_RENAME = {'maize_distill_rgb2pc_smr50_e1_d2_nova': 'maize_distill_rgb2pc_smr50_e1_d2',
                 'maize_distill_rgb2pc_smr50b_e1_d2_nova': 'maize_distill_rgb2pc_smr50b_e1_d2'}

PATHS = [('/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K', 'data/Sorghum_15K'),
         ('/work/mech-ai-scratch/alloy/shorgum_data/leaf_widths.csv', 'data/leaf_widths.csv'),
         ('/work/mech-ai-scratch/alloy/Maize_1', 'data/Maize'), ('/work/hdd/bepk/adas14/Maize', 'data/Maize'),
         ('/work/nvme/bepk/adas14/Maize', 'data/Maize'), ('/work/mech-ai-scratch/alloy/Maize', 'data/Maize'),
         ('/work/mech-ai/alloydas/maize_row', 'data/maize_row'), ('/work/mech-ai/alloydas/sorghum_row', 'data/sorghum_row'),
         ('/work/mech-ai/alloydas/embodiedmae_outputs', 'outputs'), ('/work/mech-ai-scratch/alloy/e8_weights', 'weights'),
         ('/work/mech-ai-scratch/alloy/MultiMAE', 'third_party/MultiMAE'),
         ('/work/mech-ai-scratch/alloy/embodiedmae/', ''), ('/work/mech-ai-scratch/alloy/embodiedmae', '.'),
         ('/work/hdd/bepk/adas14/embodiedmae/', ''), ('/work/hdd/bepk/adas14/embodiedmae', '.'),
         ("pathlib.Path('/work/mech-ai/alloydas')", "pathlib.Path('data')"),
         ('/work/mech-ai/alloydas), one item', 'data/), one item'),
         ('/work/mech-ai-scratch/alloy/plant_point_cloud/embodiedmae_adapted.pth loaded', 'An earlier adapted checkpoint loaded'),
         ('the shared /work/hdd filesystem', 'the shared filesystem')]
NOTES = [('the yongyun branch', 'the loss-study branch'), ('yongyun branch', 'loss-study branch'),
         ('(see the param-metric note in CLAUDE.md)', '(see the parameter-metric note in README.md)'),
         ('CLAUDE.md', 'the development notes'), ('NFS (novastor010)', 'NFS'),
         ('(e.g. one queued on nova and one on scavenger, whichever gets a GPU', '(e.g. two copies queued on different partitions, whichever gets a GPU'),
         ('slurm/delta/check_split_index.py enforces that on Delta', 'a split-index check enforces that'),
         ("Nova's", "the cluster's"), ("Delta's", "the second cluster's"), ('on preemptible scavenger', 'on a preemptible partition'),
         ('(Delta: slurm/delta/e8_supervised_maize.sbatch)', ''),
         ('(Delta slurm/delta/probe_maize.sbatch: batch 32,', '(the maize probe launcher: batch 32,'),
         ('On Delta both run from slurm/delta/e8_supervised_maize.sbatch with SCORE=1.', 'Both run from the supervised launcher with SCORE=1.'),
         ('On Delta, slurm/delta/probe.sbatch runs', 'The second-cluster probe launcher runs'),
         ('(Delta smoke 22751636, epoch-20 val)', '(a smoke run, epoch-20 val)'),
         (' on Nova', ' on the cluster'), ('On Nova,', 'On the cluster,'), ('Nova ', 'the cluster '),
         ('on Delta', 'on the second cluster'), ('(Delta)', '(second cluster)'), ('Delta compute nodes', 'Some compute nodes'),
         ('sweeps/rgb_mask_sweep.py', 'the RGB mask sweep'), ('tools/maize_view_elevation.py', 'data_split/maize_view_elevation.py'),
         ('tools/check_maize_cloud_views.py', 'a cloud-visibility check'),
         ('figures/make_maize_rows_page.py shows them', 'as read by read_pcd below'),
         ('figures/gen_rgb2pc_gallery.py makes the picture; this makes the numbers behind it.', 'A gallery script makes the picture; this makes the numbers behind it.'),
         ('CPU by default -- the GPUs are training.  ~5 s/plant', 'CPU by default; pass --device cuda for a GPU.  ~5 s/plant on CPU'),
         ("'configs/config_4m_distill_15k_all.yaml'", "'configs/config_4m_distill_15k_all_d2.yaml'"),
         ('(default: the first-release generalist config)', '(default: the generalist config)'),
         ("default='config_4m.yaml')", "default='configs/config_4m_pretrain15k_d2_sink.yaml')"),
         ("default='config_4m_distill.yaml')", "default='configs/config_4m_distill_15k_all_d2.yaml')"),
         ("default='configs/config_maize.yaml')", "default='configs/config_maize_4m_pretrain_15k_d2.yaml')"),
         ('configs/config_maize.yaml', 'configs/config_maize_4m_d2.yaml'),
         ('configs/config_e8_supervised.yaml', 'configs/config_e8_supervised_d2.yaml'),
         ('configs/config_maize_row_ft.yaml', 'configs/config_maize_row_ft_e1.yaml'),
         ('configs/config_maize_distill_all.yaml', 'configs/config_maize_distill_e1_d2.yaml')]
FORBIDDEN = re.compile(r'/work/|alloy|mech-ai|bepk|adas14|scavenger|novastor|\bNova\b|\bnova\b|yongyun|CLAUDE|claude|iastate', re.I)

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / 'eval').mkdir(parents=True); (OUT / 'configs').mkdir(); (OUT / 'data_split').mkdir()
for f in ROOT_FILES:
    shutil.copy(SRC / f, OUT / f)
for f in EVAL_FILES:
    shutil.copy(SRC / 'eval' / f'{f}.py', OUT / 'eval' / f'{f}.py')
shutil.copy(SRC / 'eval_rgb2pc_quant.py', OUT / 'eval' / 'eval_rgb2pc_quant.py')
shutil.copytree(SRC / 'eval' / 'baselines', OUT / 'eval' / 'baselines', ignore=shutil.ignore_patterns('__pycache__'))
for f in SPLIT_FILES:
    shutil.copy(SRC / f, OUT / 'data_split' / pathlib.Path(f).name)

# --- code text: paths, internal notes, launcher references -------------------------------------------
for f in OUT.rglob('*.py'):
    s = f.read_text(); t = s
    for a, b in PATHS + NOTES:
        t = t.replace(a, b)
    t = re.sub(r'slurm/(delta/)?[A-Za-z0-9_]+\.sbatch', 'the launch script', t)
    if f.name == 'eval_rgb2pc_quant.py':
        t = t.replace('import os\n', 'import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\nimport os\n', 1)
    if f.name == 'make_split.py':
        t = re.sub(r'import numpy as np, csv, sys, json\nSP="[^"]*"\n',
                   '"""Sorghum split: Mahalanobis extremeness on six standardized traits, a 30 % pool drawn with\n'
                   'p ~ (rank + 0.02)^2 (seed 42), halved into test and val; the rest is train. Reproduces\n'
                   'data_split/assignment.csv.\nusage: python data_split/make_split.py [folder holding features.csv]"""\n'
                   'import numpy as np, csv, sys, json\nSP = sys.argv[1] if len(sys.argv) > 1 else "data_split"\n', t)
    if t != s:
        f.write_text(t)

# --- configs: final-data runs only, batch shape as each run recorded, block YAML, wandb off -----------
ORIG_OUT = SRC / 'outputs'
names = {c: c for c in CONFIGS} | {k: v for k, v in CONFIG_RENAME.items()}
for src_name, dst_name in names.items():
    d = yaml.safe_load((SRC / 'configs' / f'config_{src_name}.yaml').read_text())
    txt = json.dumps(d)
    for a, b in PATHS:
        txt = txt.replace(a, b)
    d = json.loads(txt)
    od = next(v['output_dir'] for v in d.values() if isinstance(v, dict) and v.get('output_dir'))
    run = pathlib.Path(od).name
    cj = ORIG_OUT / run / 'config.json'
    if cj.exists():
        c = json.loads(cj.read_text()); shape = (c.get('batch_size'), c.get('world_size'), c.get('accum_steps'))
    elif run.startswith('maize_'):
        shape = (16, 2, None)                               # every maize arm trained on the second cluster: 16 per GPU x 2
    else:
        raise SystemExit(f'no recorded batch shape for {src_name}')
    if run == 'maize_row_ft_e1':
        shape = (16, 2, 1)                                  # resumed once at 16 x 1 GPU x accum 2: the same global 32
    tr = d.setdefault('training', {}); ds = d.setdefault('distributed', {})
    if shape[0]: tr['batch_size'] = shape[0]
    if shape[1]: ds['world_size'] = shape[1]
    if shape[2] and shape[2] > 1: tr['accum_steps'] = shape[2]
    elif 'accum_steps' in tr: tr['accum_steps'] = 1
    if 'wandb' in d: d['wandb']['use_wandb'] = False
    g = tr['batch_size'] * ds.get('world_size', 1) * tr.get('accum_steps', 1)
    head = (f'# {dst_name}: global batch {g} = {tr["batch_size"]} per GPU x {ds.get("world_size", 1)} GPUs'
            + (f' x {tr["accum_steps"]} accumulation steps' if tr.get('accum_steps', 1) > 1 else '')
            + '.\n# Launch with torchrun --nproc_per_node=<world_size>; see README.md.\n')
    (OUT / 'configs' / f'config_{dst_name}.yaml').write_text(head + yaml.safe_dump(d, sort_keys=False, default_flow_style=False, width=120))

# --- checks ------------------------------------------------------------------------------------------
bad = [f'{f.relative_to(OUT)}:{i + 1}: {l.strip()[:120]}' for f in OUT.rglob('*') if f.suffix in ('.py', '.yaml')
       for i, l in enumerate(f.read_text().splitlines()) if FORBIDDEN.search(l)]
named = set(re.findall(r'configs/config_[A-Za-z0-9_]+\.yaml', ' '.join(f.read_text() for f in OUT.rglob('*.py'))))
missing = sorted(c for c in named if not (OUT / c).exists())
if bad or missing:
    print('\n'.join(bad)); print('configs named in code but not released:', missing)
    raise SystemExit('release check failed')

# --- top-level files ---------------------------------------------------------------------------------
for d in ('data', 'outputs', 'reports'):
    (OUT / d).mkdir(); (OUT / d / '.gitkeep').touch()
shutil.copy(SRC / 'tools' / 'release_README.md', OUT / 'README.md')
(OUT / '.gitignore').write_text('__pycache__/\n*.pyc\nwandb/\n*.pth\n*.npz\n'
                                 'data/*\n!data/.gitkeep\noutputs/*\n!outputs/.gitkeep\nreports/*\n!reports/.gitkeep\n')
(OUT / 'environment.yml').write_text('''name: cropcode
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pytorch=2.5.0
  - torchvision=0.20.0
  - pytorch-cuda=12.4
  - pyyaml=6.0
  - pip
  - pip:
      - numpy==2.2.6
      - scipy==1.16.3
      - pandas==2.3.3
      - scikit-learn==1.7.2
      - scikit-image==0.25.2
      - matplotlib==3.10.7
      - open3d==0.19.0
      - opencv-python==4.12.0.88
      - einops==0.3.2
      - timm==0.4.12
      - geomloss==0.3.1
      - wandb==0.23.1
''')
n = sum(1 for f in OUT.rglob('*') if f.is_file())
print(f'release written to {OUT}: {n} files, {len(names)} configs, checks passed')
