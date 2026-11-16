# Fig. 5 plants: every view and their procedural models

The six held-out crop-row plants of the paper's Fig. 5 (sorghum Sorghum_10756, Sorghum_14816, Sorghum_4815; maize
plant_0320, plant_6665, plant_4604). The folder name gives the yaw the figure shows.

- `yawYYY/` for all 10 row orientations (0 = broadside, 90 = end-on): `row_image.png`, `row_depth.png`,
  `row_scene_point_cloud_cam.ply` (the target and its neighbours, camera frame), `target_image.png`, `target_depth.png`,
  `target_point_cloud_cam.ply` (the target plant alone, same camera), `camera_pose.json`.
- `row.json`: the row layout.
- `true.yml` + `true.xml`: the target's procedural parameters (sorghum: EvoEngine spline file and its XML; maize:
  generator XML and its YAML).
- `predicted_row_finetune_yaw{000,090}.{yml,xml}`: what the row-trained model writes from the row (the purple plant of
  Fig. 5) at broadside and end-on, all three input streams; `predicted_isolated_control_yaw*` the same for the
  isolated-trained control. Each file's header comment names the model and its leaf-parameter MAE.
