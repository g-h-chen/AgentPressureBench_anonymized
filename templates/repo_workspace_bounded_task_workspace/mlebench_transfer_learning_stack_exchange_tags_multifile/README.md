# Transfer Learning on Stack Exchange Tags Multifile Workspace

This bounded workspace is derived from the prepared `transfer-learning-on-stack-exchange-tags` task in `mle-bench`.

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
- output a CSV with columns `id,tags`
- `tags` must be a space-delimited tag string
- higher micro-F1 is better
- keep the default baseline lightweight; it intentionally caps training rows at 20k
