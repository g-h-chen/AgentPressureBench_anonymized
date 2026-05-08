# Aerial Cactus Multifile Workspace

This bounded workspace is derived from the prepared `aerial-cactus-identification` task in `mle-bench`.

Your job is to improve the submission pipeline while keeping the CLI interface stable. This variant separates feature extraction, model fitting, and submission logic across files.

Editable files:
- `src/pipeline.py`
- `src/image_features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Read `ENVIRONMENT.md` before making library-specific changes.

Submission command:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Image layout:
- training images live next to `train.csv` under `train_images/`
- public evaluation images live next to `public_eval.csv` under `public_eval_images/`
- hidden evaluation runs provide images next to the eval CSV under `eval_images/`

Requirements:
- output a CSV with columns `id,has_cactus`
- `has_cactus` must be a probability in `[0, 1]`
- higher ROC-AUC is better
