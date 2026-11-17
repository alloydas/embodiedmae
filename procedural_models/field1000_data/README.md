# Latest field data (Maize_Field_1000 / Sorghum_Field_1000): held-out test plants

One camera stop per held-out TEST plant of the 2026-11-15 upload (maize: 130 plants; sorghum: 13 scans), never seen
in training: the first row in which the plant is the middle plant, its middle stop.
Built by `figures/export_field1000_data.py`.

## Per stop: `<species>/<plant>_row<RR>_stop<SS>/`

| file | what |
|---|---|
| `photo.jpg` | the stop's photograph, reduced to 1,200 px high (the full-size PNG is in the upload under the name in `truth.json` / `summary.csv`) |
| `leaves.csv` | the stop's per-leaf CSV, middle plant only (blank duplicate leaf rows dropped) |
| `truth.json`, `truth.yml`, `truth.xml` | the ground truth: `height_m` (vertical extent of the middle plant's points), `leaf_count` (= number of leaf rows of the middle plant), and every leaf's values (`leaves_csv`: leaf number, length, width, zenith, azimuth, internode distance, twist, curvature, stem diameter). Sorghum original scans (no `_rNN` replica): the CSV leaf lengths are the unscaled scan's, so `leaf_length_m` is rescaled to the design mean (`len_scale_applied`; the CSV value is kept as `leaf_length_m_raw_csv`) |
| `target_point_cloud_cam.ply` | the middle plant's points from the stop's PCD (camera frame of the PCD: x right, y down, z forward; metres) |
| `model_input_image.png`, `model_input_depth.png` | the 224 x 224 window around the middle plant that the models receive (depth drawn 8-bit, near = bright) |
| `model_input_scene_cloud_cam.ply` | the 24,576 visible scene points the models receive (same camera frame, metres) |
| `predicted_<model>.xml` / `.yml` | the procedural model the model writes from image + depth + point cloud (no parameters given) |
| `predicted_<model>_point_cloud_cam.ply` | the model's predicted point cloud of the middle plant (same camera frame, metres) |

Models:
- `synthetic_row_finetune`: the row fine-tuned model of the paper, trained on simulated rows only (before).
- `field_finetune`: the same model after 30 epochs on the training plants of this upload (`*_real_ft_f1000`, best checkpoint), supervised on the middle plant's cloud and depth, its height, and per leaf (maize: count, length, width, internode, curvature; sorghum: insertion height, length, width).

Each prediction is one masked draw (80 % of the input tokens hidden, seed = stop index), so a stop's numbers can differ
slightly from the evaluation reports, which average over every stop.

## `summary.csv`

Per stop and model: Chamfer distance to the middle plant (1e-4 m^2), share of true points missed at 5 mm, written vs
true height and leaf count.

## Leaf count

- Maize: the count is written in the procedural model's plant token and was trained on this data.
- Sorghum: the true leaf count is in `truth.*` (from the CSV), but the sorghum procedural model has no leaf-count
  field: the generator derives it from the stem length (0.05 m per leaf), capped at its 24 leaf slots. That rule does
  not hold for these real sorghum plants (1.6 to 2.0 m stalks with 8 to 18 leaves in the test stops; 6 to 21 over the
  whole upload): the rule gives 32 to 40, capped at 24, so the field-fine-tuned model's files list 24 leaves and the
  predicted count in `summary.csv` is not a real estimate. The per-leaf values of the first
  leaves (from the base) are trained against the CSV and can be compared leaf by leaf with `truth.*`.
