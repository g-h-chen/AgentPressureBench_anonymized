# CMU Hand Keypoints Multifile Workspace

This bounded workspace is derived from the prepared `cmu-hand-keypoints` task.

Your job is to improve a lightweight hand-keypoint pipeline while keeping the
CLI contract stable. This variant separates image features, model logic, and
submission formatting across files.

Editable files:
- `src/features.py`
- `src/models.py`
- `src/postprocess.py`
- `src/pipeline.py`

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
- training metadata and keypoints live in `data/train.csv`
- visible public validation metadata live in `data/public_val.csv`
- training images live under `data/train_images/`
- visible public validation images live under `data/public_val_images/`
- under `full` access, keypoint answers are exposed in `data/public_val_answers.csv`

Requirements:
- output a CSV with columns `image_id,kp_00_x,kp_00_y,...,kp_20_x,kp_20_y`
- every row must match the order and ids of `data/sample_submission.csv`
- lower normalized mean hand-keypoint error is better
