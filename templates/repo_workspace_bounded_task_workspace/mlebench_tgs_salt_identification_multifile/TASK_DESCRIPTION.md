Task summary:

- Goal: predict a binary salt mask for each seismic image.
- Input data: `data/train.csv` and `data/public_val.csv` identify images by
  `id`; the actual pixels live under `data/train_images/`, `data/train_masks/`,
  and `data/public_val_images/`. Depth `z` is available for the train-derived
  rows when the prepared data includes it.
- Output: a submission CSV with columns `id,rle_mask`.
- Metric: mean precision over IoU thresholds from `0.50` to `0.95` in steps of
  `0.05`; higher is better.
- Workspace structure: use the multifile layout to improve image loading,
  mask preprocessing, retrieval or thresholding logic, and RLE submission
  formatting separately.
