Task summary:

- Goal: predict a binary polyp mask for each colonoscopy image.
- Input data: `data/train.csv` and `data/public_val.csv` list image ids; the
  actual pixels live under `data/train_images/`, `data/train_masks/`, and
  `data/public_val_images/`.
- Output: a submission CSV with columns `image_id,mask_rle`.
- Metric: mean Dice; higher is better.
- Workspace structure: use the multifile layout to improve image loading,
  mask preprocessing, threshold selection, and RLE submission logic separately.
