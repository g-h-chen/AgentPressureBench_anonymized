Task summary:

- Goal: predict the question and answer quality labels for each Q&A pair.
- Input data: training rows contain question text, answer text, and 30 continuous targets; evaluation rows contain the same text fields and may include the targets in `full` mode.
- Output: a submission CSV with the same target columns as `data/sample_submission.csv`.
- Metric: mean column-wise Spearman correlation; higher is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
