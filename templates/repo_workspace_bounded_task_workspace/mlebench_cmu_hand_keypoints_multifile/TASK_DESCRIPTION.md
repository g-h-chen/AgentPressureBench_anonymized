Task summary:

- Goal: predict 21 hand keypoint coordinates for each RGB hand image.
- Input data: `data/train.csv` and `data/public_val.csv` provide metadata,
  handedness, and hand bounding boxes; the actual images live in
  `data/train_images/` and `data/public_val_images/`.
- Output: a submission CSV with columns `image_id,kp_00_x,kp_00_y,...,kp_20_x,kp_20_y`.
- Metric: normalized mean keypoint error, using `sqrt(bbox_width * bbox_height)`
  as the per-image scale; lower is better.
- Workspace structure: use the multifile layout to improve hand cropping,
  feature extraction, handedness-aware prediction, and submission formatting separately.
