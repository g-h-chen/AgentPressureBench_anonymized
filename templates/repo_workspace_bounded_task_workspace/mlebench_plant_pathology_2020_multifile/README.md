# Plant Pathology 2020 Multifile Workspace

This bounded workspace is derived from the prepared `plant-pathology-2020-fgvc7` task in `mle-bench`.

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
- output a CSV with columns `image_id,healthy,multiple_diseases,rust,scab`
- each row must sum to `1.0`
- probabilities must be between `0` and `1`
- lower multiclass log-loss is better
