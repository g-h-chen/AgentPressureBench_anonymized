# UW-Madison GI Tract Image Segmentation Multifile Workspace

This bounded workspace is derived from the prepared
`uw-madison-gi-tract-image-segmentation` task.

Your job is to improve the multi-organ MRI segmentation pipeline while keeping
the CLI contract stable. This variant separates image loading, model fitting,
and submission formatting across files.

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
  --eval-image-dir data/public_val_images \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Data layout:
- `data/train.csv` contains RLE labels with columns `id,class,segmentation`
- `data/public_val.csv` contains rows with `id,class`; under `full` access it
  also includes held-out `predicted,image_width,image_height`
- training slice images live under `data/train_images/` as flat `id.png`
- visible public validation slice images live under `data/public_val_images/`
  as flat `id.png`

Requirements:
- output a CSV with columns `id,class,predicted`
- `predicted` must be one binary-mask RLE string per `(id, class)` row
- higher `0.4 * Dice + 0.6 * (1 - normalized 3D Hausdorff)` is better
