Task summary:

- Goal: predict bowel and stomach masks for each MRI slice.
- Input data: `data/train.csv` provides one row per `(id, class)` pair with an
  RLE segmentation label. `data/public_val.csv` lists the visible evaluation
  rows. The actual slice pixels live under `data/train_images/` and
  `data/public_val_images/` as flat `id.png` files.
- Output: a submission CSV with columns `id,class,predicted`.
- Metric: `0.4 * Dice + 0.6 * (1 - normalized 3D Hausdorff)`, matching the
  official competition-style scorer in this benchmark.
- Workspace structure: use the multifile layout to improve image loading,
  lightweight mask modeling, and submission formatting separately.
