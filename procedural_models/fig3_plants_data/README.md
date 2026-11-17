# Fig. 3 plants: every view

The plants of the paper's Fig. 3 (the dataset figure): per species the 6 gallery plants (panel a; `gallery_highlighted`
is the plant of panels b and c) and the crop-row plant of panel d.

- `<species>/<plant>_gallery_*/`: `true.yml` and `true.xml` (procedural parameters: sorghum EvoEngine spline file and
  its XML; maize generator XML and its YAML), and for each of the 10 cameras `viewVV/`: `image.png` (sorghum:
  rgb_nobg.png, RGBA with transparent background; maize: rgba.png), `depth.png` (the renderer's packed RGBA depth),
  `point_cloud_cam.ply` (camera frame) and `camera_pose.json`. Sorghum also has `point_cloud_complete_world.ply`, the whole
  plant in the generator frame (each sorghum view cloud is this plant moved rigidly into that camera's frame; maize view
  clouds are what that one camera sees).
- `<species>/<plant>_row_scene/`: `true.yml` / `true.xml`, `row.json` (row layout) and for each of the 10 row
  orientations `yawYYY/` (0 = broadside): `row_image.png`, `row_depth.png`, `row_scene_point_cloud_cam.ply` (target plus
  neighbours), `target_image.png`, `target_depth.png`, `target_point_cloud_cam.ply` (the target alone, same camera) and
  `camera_pose.json`.

## Model outputs

What CropCode writes for the same plants and views (Fig. 3 itself shows the dataset only):

- gallery plants, `viewVV/predicted_from_{image,depth,point_cloud}/`: the distilled generalist (the model of Fig. 4)
  given that one input alone, the procedural parameters never an input: `point_cloud_cam.ply` (the predicted plant cloud,
  in the view's camera frame, metres, like the input cloud) and `procedural_parameters.yml` / `.xml`.
- row-scene plant, `yawYYY/predicted_row_finetune/`: the row fine-tuned model (Figs. 1 and 5) given the row image, depth
  and scene cloud: `target_point_cloud_cam.ply` (the predicted target plant, row camera frame, metres; compare with
  `target_point_cloud_cam.ply` one level up) and `procedural_parameters.yml` / `.xml`.
- `outputs_summary.csv`: per output, Chamfer to the true cloud (normalised frame and m^2), and written vs true height (m)
  and leaf count. Predicted leaf counts are read from the written parameters (maize: the plant token; sorghum: the stem
  length), as in the paper's tables.
