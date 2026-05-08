# Random Acts of Pizza Multifile Workspace

This bounded workspace is derived from the prepared `random-acts-of-pizza` task in `mle-bench`.

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
- output a CSV with columns `request_id,requester_received_pizza`
- `requester_received_pizza` must be a probability in `[0, 1]`
- higher ROC-AUC is better
