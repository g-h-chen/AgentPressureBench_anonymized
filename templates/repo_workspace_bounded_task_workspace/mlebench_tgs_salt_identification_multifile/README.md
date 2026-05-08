# TGS Salt Identification Multifile Workspace

This bounded workspace is derived from the prepared
`tgs-salt-identification-challenge` task.

Your job is to improve the binary salt-segmentation pipeline while keeping the
CLI contract stable. This variant separates image loading, model fitting, and
submission formatting across files.

Editable files:
- `src/pipeline.py`
- `src/image_features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Read `ENVIRONMENT.md` before making library-specific changes.

Submission contract:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_val.csv \
  --train-image-dir data/train_images \
  --train-mask-dir data/train_masks \
  --eval-image-dir data/public_val_images \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Data layout:
- `data/train.csv` contains training rows with `id`, optional depth `z`, and
  the training `rle_mask`
- `data/public_val.csv` contains visible validation rows with `id` and
  optional depth `z`; under `full` access it also includes the held-out
  `rle_mask`
- training images live under `data/train_images/`
- training masks live under `data/train_masks/`
- visible public validation images live under `data/public_val_images/`
- under `full` access, visible validation masks are also present under
  `data/public_val_masks/`

Requirements:
- output a CSV with columns `id,rle_mask`
- `mask_rle` must be one binary-mask RLE string per image
- higher mean precision over IoU thresholds `0.50:0.95` is better
