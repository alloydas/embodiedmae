# EmbodiedMAE-4M: What Was Added on Top of the 3M Model

This document describes everything the 4M variant introduces beyond the
3M model (`embodied_mae.py`). It is meant to be read alongside `embodied_mae_4m.py`,
`sorghum_dataset_4m.py`, `train_sorghum_4m.py`, and `config_4m.yaml`.

## TL;DR

The 4M model adds a **fourth modality** to the 3-modality MAE (RGB, Depth,
Point Cloud): a **parametric "spline" modality** that carries the procedural
generation parameters of the synthetic sorghum plant. It is a numeric modality —
the encoder consumes a fixed-size float vector per token and the decoder
regresses the same vector — but the decoded values can be *formatted* as
human-readable text (hence the names `n_text_tokens`, `mask_text`, `text_valid`).

The encoder, decoder, and Dirichlet masking machinery are unchanged in spirit;
they were extended from "three lists of tokens" to "four lists of tokens". The
RGB / Depth / PC paths are imported directly from `embodied_mae.py`, so any change
to those affects both models.

---

## 1. The new modality: parametric spline tokens

Each synthetic sorghum sample ships with a `*_spline.yml` describing the
procedural generation knobs that produced it. The 4M model treats this
structured YAML as a sequence of tokens:

```
token 0           : the plant itself        (1 token)
tokens 1 .. K     : individual leaves       (≤ max_leaves = 24 tokens)
tokens K+1 .. 24  : padding                 (text_valid = 0)

n_text_tokens = 1 + max_leaves = 25
```

Every token holds an **N\_PARAMS = 9** dimensional float vector. The semantics
of the 9 slots differ by token type:

| Slot | Plant token (idx 0)             | Leaf token (idx 1+)         |
|------|----------------------------------|------------------------------|
| 0    | stem_length (`sl`)               | starting_point (`sp`)        |
| 1    | stem_direction.x (`sdx`)         | length (`ln`)                |
| 2    | stem_direction.y (`sdy`)         | roll_angle (`ra`)            |
| 3    | stem_direction.z (`sdz`)         | branching_angle (`ba`)       |
| 4    | panicle_size.x (`psx`)           | waviness_frequency (`wf`)    |
| 5    | panicle_size.y (`psy`)           | waviness_period_start[0] (`wp0`) |
| 6    | panicle_size.z (`psz`)           | waviness_period_start[1] (`wp1`) |
| 7    | panicle_seed_amount (`pa`)       | unused (zero)                |
| 8    | panicle_seed_radius (`pr`)       | unused (zero)                |

All 9 slots are normalised to **[0, 1]** before they ever touch the model.
The fixed `_PLANT_SCALE / _PLANT_SHIFT / _LEAF_SCALE / _LEAF_SHIFT` arrays at
the top of `embodied_mae_4m.py` define the normalisation. Use
`check_param_ranges_fast.py` (or the slower full-scan `check_param_ranges.py`)
to verify the scales don't clip the dataset before launching a run.

```
norm = (raw + shift) / scale
raw  = norm * scale - shift
```

For each sample the dataset returns two extra tensors:

- `param_floats : (1+max_leaves, N_PARAMS)` — encoder input **and** regression target
- `text_valid   : (1+max_leaves,)`           — 1.0 = real token, 0.0 = padding

Both come from `load_spline_params(yml_path, max_leaves)` in
`embodied_mae_4m.py`. `_plant_to_params` and `_leaf_to_params` are the per-row
builders — if the upstream YAML schema changes, those two functions and the
four scale/shift arrays must be updated together.

---

## 2. New components in the model

Compared to the 3M `EmbodiedMAE`, the 4M `EmbodiedMAE4M` class adds:

### 2.1 `ParamEmbed` — encoder-side embedder

```
floats (B, n_text_tokens, N_PARAMS)
  → Linear(N_PARAMS → embed_dim)
  → GELU
  → LayerNorm
  → tokens (B, n_text_tokens, embed_dim)
```

This is the parametric analogue of `PatchEmbed` (RGB/Depth) and
`PointCloudEmbed` (PC). It lives as `self.param_embed`.

### 2.2 New positional + modality embeddings

- `self.modality_embed_text`     — learned `(1, 1, D)` bias added to every text token
- `self.pos_embed_text`          — learned `(1, 25, D)` per-token position embedding
- `self.decoder_pos_embed_text`  — learned `(1, 25, decoder_dim)` decoder positional embedding

(Decoder pos-embed is `requires_grad=True` for text and PC, but a fixed
sin-cos grid for RGB/Depth — same convention as 3M.)

### 2.3 New decoder head — `decoder_pred_params`

```
nn.Sequential(
    Linear(decoder_dim → decoder_dim),
    GELU,
    LayerNorm,
    Linear(decoder_dim → N_PARAMS),
)
```

A 2-layer MLP that emits `N_PARAMS=9` floats per token. The model **does not
clamp** outputs — clamping happens only at decode-to-text time. Targets are
already in [0, 1], so the model effectively learns a regression in that range.

### 2.4 4-way Dirichlet masking

`random_masking_dirichlet` is extended from 3 to 4 modalities:

- A single `Dirichlet(α=1)` draw of dimension 4 splits the visible-token budget
  across `[rgb, depth, pc, text]`.
- Per-modality visible counts are clamped both ways:
  - at least **1** visible token per modality (avoids empty token streams),
  - at most `length × (1 − min_mask_ratio)`, with `min_mask_ratio=0.25` —
    i.e. **every modality is at least 25 % masked, every step**. This is
    stricter than the 3M version and is unique to 4M.
- All samples in a batch share the same per-modality count (so token streams
  concatenate cleanly), but the *which-tokens-are-visible* draw is still
  per-sample.

Returns four sets of `(visible_tokens, mask, ids_restore)` instead of three.

### 2.5 4-stream encoder/decoder plumbing

- `forward_encoder` concatenates **four** visible-token streams + CLS token,
  runs them through the same shared encoder blocks as 3M, and returns the
  per-modality mask/restore tensors plus the visible-token counts so the
  decoder can split the latent back into 4 streams.
- `forward_decoder` slices the latent into 4 ranges, restores mask tokens at
  the correct positions per modality (gather by `ids_restore`), adds
  per-modality decoder positional embeddings, runs the shared decoder, and
  emits **four** outputs: `pred_rgb, pred_depth, pred_pc, pred_params`.

---

## 3. The new loss term

The `forward_loss` returns one extra component, `loss_text`:

```python
per_pos   = SmoothL1(pred_params, param_floats, β=0.1).mean(-1)   # (B, L)
eff_mask  = mask_text * text_valid                                # (B, L)
loss_text = (per_pos * eff_mask).sum() / eff_mask.sum().clamp(1)
```

Properties:

- **Smooth-L1 (Huber), β = 0.1** — L2-like near zero (precise regression for
  small residuals), L1-like for outliers. Targets are in [0, 1] so β=0.1 is
  about 10 % of the dynamic range.
- **Masked twice**: only tokens that are (a) **masked** by the encoder
  (`mask_text=1`) and (b) **real** (not padding, `text_valid=1`) contribute.
  This means the model is never penalised for predicting padded leaves and
  never penalised on tokens it actually saw.
- The total loss is:

  ```
  total = loss_rgb + loss_depth
        + loss_pc   * pc_loss_weight        # default 10 in config_4m.yaml
        + loss_text * spline_loss_weight    # default 5  in config_4m.yaml
  ```

  RGB/Depth/PC losses are unchanged from 3M.

---

## 4. End-to-end forward procedure

Given a batch `(rgb, depth, pc, param_floats, text_valid)`:

1. **Embed all four modalities** to `(B, L_m, D)`:
   - RGB / Depth → `PatchEmbed` (196 tokens for 224×224 patch=16)
   - Point Cloud → `PointCloudEmbed` (FPS → kNN groups → PointNet → 196 tokens)
   - Spline     → `ParamEmbed` (linear→GELU→LN over 9-D float vector → 25 tokens)

2. **Add per-modality positional + modality embeddings**, including the new
   `pos_embed_text` and `modality_embed_text`.

3. **4-way Dirichlet masking** picks visible-token counts per modality (with
   the `min_mask_ratio=0.25` floor) and shuffles each per-sample.

4. **Encoder**: concat `[CLS, rgb_vis, depth_vis, pc_vis, text_vis]`,
   run shared transformer blocks, apply LayerNorm.

5. **Decoder**: project to `decoder_embed_dim`, slice latent into 4 streams,
   restore mask tokens at the correct positions per modality, add decoder
   positional embeddings, run shared decoder blocks, then split into heads:
   - RGB head    → `(B, 196, p²·3)` → unpatchify → `(B, 3, 224, 224)`
   - Depth head  → `(B, 196, p²·1)` → unpatchify → `(B, 1, 224, 224)`
   - PC head     → FoldingNet over 196 tokens → `(B, target_points, 3)`
   - **Param head → `(B, 25, 9)` floats in (approximately) [0, 1]**

6. **Loss**: RGB MSE (with optional norm-pix), Depth MSE on per-image
   normalised target, PC bidirectional Chamfer over the full cloud, and the
   new Smooth-L1 on params (masked twice as above). Weighted sum returned.

7. **Display only**: `decode_params_to_text(pred_params)` clamps to [0, 1],
   un-normalises with the `_PLANT_SCALE/_LEAF_SCALE` arrays, and formats each
   row with `_params_to_plant_text` / `_params_to_leaf_text` for the 5th row
   of the visualisation grid. **The model never sees text characters** — text
   is only a rendering of the predicted floats.

---

## 5. Dataset wiring (`SorghumDataset4M`)

`SorghumDataset4M` subclasses `SorghumDataset` and adds:

- A **filter pass** in `__init__` that drops any sample folder without a
  `*_spline.yml`, with a console warning.
- An extension of `__getitem__` that calls `load_spline_params(yml,
  max_leaves)` and appends `(param_floats, text_valid)` to the tuple
  returned by the parent class.

So a 4M sample tuple is:

```
(rgb, depth, pc, param_floats, text_valid, name)
```

Sanity-check a dataset folder with:

```bash
python sorghum_dataset_4m.py /path/to/Dataset/new_data
```

This iterates train and val, prints sample counts, and dumps the first
sample's tensor shapes plus the actual normalised params for the plant token
and the first leaf token.

---

## 6. Config knobs unique to 4M

In `config_4m.yaml`:

| Key                          | Meaning                                                           | Default |
|------------------------------|-------------------------------------------------------------------|---------|
| `model.spline_loss_weight`   | Multiplier on `loss_text` in the total loss                       | 5.0     |
| `model.max_leaves`           | Padding length for leaf tokens (also fixes `n_text_tokens=1+M`)   | 24      |
| `model.model_size`           | One of `small / base / large` (no `giant`)                        | base    |

All other keys (`mask_ratio`, `pc_loss_weight`, `depth_norm_type`, distributed
config, etc.) behave the same as in `config.yaml`. Note that `mask_ratio` is
still the **total** masking fraction across all four modalities — the
per-modality split is what the Dirichlet draw decides.

---

## 7. How to run it

All commands assume `cwd = repo root` and `conda activate det`.

```bash
# Single-GPU
python train_sorghum_4m.py --config config_4m.yaml --world_size 1

# Multi-GPU via torchrun (preferred)
torchrun --standalone --nproc_per_node=4 train_sorghum_4m.py --config config_4m.yaml

# Multi-GPU via mp.spawn fallback (set distributed.world_size in YAML)
python train_sorghum_4m.py --config config_4m.yaml

# Smoke-test the model on dummy tensors
python embodied_mae_4m.py

# Validation / per-sample reconstruction dump
python validate_4m.py --checkpoint outputs/<run>/best_model.pth \
                      --output_dir vis_val_4m \
                      --config config_4m.yaml \
                      --mask_ratio 0.75 \
                      --num_samples 12
```

`validate_4m.py` writes per-sample directories with `inputs/`, `outputs/`,
`masks/`, and `metrics.json`, plus top-level `metrics_aggregate.json` and
`config_used.json`.

---

## 8. Visualisation output

Training visualisations are 5-row × 3-col grids per sample (vs. 4-row in 3M):

| Row | Col 1                  | Col 2                  | Col 3                  |
|-----|------------------------|------------------------|------------------------|
| 1   | RGB original           | RGB masked             | RGB reconstructed      |
| 2   | Depth original         | Depth masked           | Depth reconstructed    |
| 3   | PC original (3D)       | PC masked (3D)         | PC reconstructed (3D)  |
| 4   | PC visible tokens      | PC masked tokens       | (legend / loss text)   |
| 5   | Target text strings    | Predicted text strings | (mask / valid bookkeeping) |

Row 5 is the only display-time use of `decode_params_to_text` — it converts
the predicted [0, 1] floats back to human-readable parameter strings via the
inverse of the normalisation, side by side with the ground truth, only over
masked-and-real tokens.

---

## 9. Files to know

| File                         | Role                                                            |
|------------------------------|-----------------------------------------------------------------|
| `embodied_mae_4m.py`         | Model class, `ParamEmbed`, scale/shift arrays, encode/decode helpers |
| `sorghum_dataset_4m.py`      | Dataset subclass that adds `(param_floats, text_valid)`         |
| `train_sorghum_4m.py`        | Training entry point (DDP via torchrun or mp.spawn)             |
| `config_4m.yaml`             | YAML config with the two extra keys                              |
| `validate_4m.py`             | Full-fidelity per-sample dump + metrics                          |
| `check_param_ranges*.py`     | Verify `_PLANT_SCALE / _LEAF_SCALE` don't clip the dataset       |

---

## 10. Things to remember when modifying 4M

- `embodied_mae_4m.py` **imports** `PatchEmbed`, `PointCloudEmbed`,
  `TransformerBlock`, `chamfer_distance`, `earth_movers_distance`, and
  `get_2d_sincos_pos_embed` from `embodied_mae.py`. Changes to those
  affect both models.
- If you add a new field to the spline YAML schema, you **must** update:
  1. `_PLANT_SCALE / _PLANT_SHIFT` *or* `_LEAF_SCALE / _LEAF_SHIFT` (or both),
  2. `_plant_to_params` *or* `_leaf_to_params` (or both),
  3. `_params_to_plant_text` *or* `_params_to_leaf_text` (or both),
  4. `N_PARAMS` if the per-token dimension grows,
  5. The decoder param head's output dim (it uses `N_PARAMS`).
  Then re-run `check_param_ranges_fast.py` to confirm nothing clips.
- The `min_mask_ratio=0.25` floor in `random_masking_dirichlet` is hard-coded;
  it exists to stop the Dirichlet draw from starving any one modality.
- The param loss is masked by `text_valid` *and* `mask_text`. Don't drop
  either factor without thinking through what it means — `text_valid`
  prevents loss on padded leaves, `mask_text` keeps the objective an MAE
  rather than autoencoder.
