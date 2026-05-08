# Feedback Prize ELL Multifile Workspace

This bounded workspace is derived from the prepared `feedback-prize-english-language-learning` task in `mle-bench`.

Your job is to improve the submission pipeline while keeping the CLI interface stable.

Editable files:
- `src/pipeline.py`
- `src/features.py`
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
- output a CSV with the same columns as `data/sample_submission.csv`
- each target column should be numeric
- lower mean column-wise RMSE is better
