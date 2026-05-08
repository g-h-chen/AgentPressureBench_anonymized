Task summary:

- Goal: predict the disease class probabilities for each plant image.
- Input data: training rows contain `image_id` plus one-hot disease labels; evaluation rows contain `image_id` and may include labels in `full` mode.
- Output: a submission CSV with columns `image_id,healthy,multiple_diseases,rust,scab`.
- Metric: multiclass log-loss; lower is better.
- Workspace structure: use the multifile layout to improve image feature extraction, model fitting, and submission formatting separately.
