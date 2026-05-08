Task summary:

- Goal: predict six rubric scores for each essay.
- Input data: training rows contain `full_text` and six continuous targets; evaluation rows contain the same essay text field and may include the targets in `full` mode.
- Output: a submission CSV with the same target columns as `data/sample_submission.csv`.
- Metric: mean column-wise RMSE; lower is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
