Task summary:

- Goal: predict the leaf species for each evaluation row.
- Input data: training rows contain numeric morphology features plus the categorical label `species`; evaluation rows contain the same numeric features and may include the label in `full` mode.
- Output: a submission CSV with the same class-probability columns as `data/sample_submission.csv`.
- Metric: multiclass log-loss; lower is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
