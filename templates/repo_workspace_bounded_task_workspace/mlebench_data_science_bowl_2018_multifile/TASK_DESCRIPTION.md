Task summary:

- Goal: predict a binary nucleus mask for each microscopy image.
- Input data: `data/train.csv` and `data/public_eval.csv` list sample ids; the actual
  images and masks live under `data/train/<ImageId>/` and `data/public_eval/<ImageId>/`.
- Output: a submission CSV with columns `ImageId,EncodedPixels`.
- Metric: mean Dice; higher is better.
- Workspace structure: use the multifile layout to improve image loading,
  threshold selection, and RLE submission logic separately.
