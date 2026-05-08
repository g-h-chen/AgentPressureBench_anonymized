# Learning Agency Essay Scoring 2 Multifile Workspace

This bounded workspace is derived from the prepared `learning-agency-lab-automated-essay-scoring-2` task in `mle-bench`.

Your job is to improve the submission pipeline while keeping the CLI interface stable.

Editable files:
- `src/pipeline.py`
- `src/features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Submission command:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Requirements:
- output a CSV with columns `essay_id,score`
- `score` should be an integer essay score
- higher quadratic weighted kappa is better
