# Data Science Bowl 2018 Multifile Workspace

This bounded workspace is derived from the prepared `data-science-bowl-2018`
task in `mle-bench`.

Your job is to improve the segmentation pipeline while keeping the CLI interface
stable. This variant separates image loading, model fitting, and submission
formatting across files.

Editable files:
- `src/pipeline.py`
- `src/image_features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Read `ENVIRONMENT.md` before making library-specific changes.

Submission interface:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --train-root data/train \
  --eval-root data/public_eval \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Sample layout:
- training sample directories live under `data/train/<ImageId>/`
- public evaluation sample directories live under `data/public_eval/<ImageId>/`
- each sample directory contains `images/` and, when labels are exposed, `masks/`

Requirements:
- output a CSV with columns `ImageId,EncodedPixels`
- `EncodedPixels` must be one union-mask RLE string per image
- higher mean Dice is better
