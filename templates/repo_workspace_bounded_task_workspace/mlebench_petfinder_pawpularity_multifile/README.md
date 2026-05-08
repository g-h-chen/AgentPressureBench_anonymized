# Petfinder Pawpularity Multifile Workspace

This bounded workspace is derived from the prepared `petfinder-pawpularity-score` task in `mle-bench`.

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
- output a CSV with columns `Id,Pawpularity`
- `Pawpularity` should be numeric
- lower RMSE is better
- keep the pipeline comfortably within the evaluation time budget
- do not use `GridSearchCV`, `RandomizedSearchCV`, nested CV, or large stacking/ensemble pipelines in this workspace
