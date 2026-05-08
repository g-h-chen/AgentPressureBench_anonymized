# Kvasir-SEG Multifile Workspace

This bounded workspace is derived from the prepared `kvasir-seg` task.

Your job is to improve the binary polyp-segmentation pipeline while keeping the
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
- training images live under `data/train_images/`
- training masks live under `data/train_masks/`
- visible public validation images live under `data/public_val_images/`
- under `full` access, visible validation masks are also present under `data/public_val_masks/`

Requirements:
- output a CSV with columns `image_id,mask_rle`
- `mask_rle` must be one binary-mask RLE string per image
- higher mean Dice is better
