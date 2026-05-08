# Denoising Dirty Documents Multifile Workspace

This bounded workspace is derived from the prepared `denoising-dirty-documents` task in `mle-bench`.

Your job is to improve the submission pipeline while keeping the CLI interface stable.

Editable files:
- `src/pipeline.py`
- `src/image_features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Submission interface:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Requirements:
- output a CSV with columns `id,value`
- `value` must be a grayscale pixel value between `0.0` and `1.0`
- lower RMSE is better
