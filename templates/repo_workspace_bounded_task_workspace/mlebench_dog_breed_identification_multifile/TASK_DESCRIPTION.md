Task summary:

- Goal: predict the dog breed probabilities for each evaluation image.
- Input data: training rows contain `id` plus the categorical label `breed`; evaluation rows contain `id` and may include `breed` in `full` mode.
- Output: a submission CSV with the same class-probability columns as `data/sample_submission.csv`.
- Metric: multiclass log-loss; lower is better.
- Workspace structure: use the multifile layout to improve image feature extraction, model fitting, and submission formatting separately.
