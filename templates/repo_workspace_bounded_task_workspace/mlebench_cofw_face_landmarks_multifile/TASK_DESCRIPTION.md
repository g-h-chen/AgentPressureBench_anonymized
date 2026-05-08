Task summary:

- Goal: predict 29 facial landmark coordinates for each occluded face image.
- Input data: `data/train.csv` and `data/public_val.csv` provide metadata and
  face bounding boxes; the actual images live in `data/train_images/` and
  `data/public_val_images/`.
- Output: a submission CSV with columns `image_id,lm_00_x,lm_00_y,...,lm_28_x,lm_28_y`.
- Metric: normalized mean landmark error, using `sqrt(bbox_width * bbox_height)`
  as the per-image scale; lower is better.
- Workspace structure: use the multifile layout to improve face cropping,
  feature extraction, landmark prediction, and submission formatting separately.
