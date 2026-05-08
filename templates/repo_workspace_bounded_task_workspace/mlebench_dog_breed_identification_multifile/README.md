# Dog Breed Identification Multifile Workspace

This bounded workspace is derived from the prepared `dog-breed-identification` task in `mle-bench`.

Your job is to improve the submission pipeline while keeping the CLI interface stable.

Editable files:
- `src/pipeline.py`
- `src/image_features.py`
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
- output a CSV with the same columns as `data/sample_submission.csv`
- each row must sum to `1.0`
- probabilities must be between `0` and `1`
- lower multiclass log-loss is better
