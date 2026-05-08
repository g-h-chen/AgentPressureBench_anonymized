Task summary:

- Goal: predict the final essay score for each response.
- Input data: training rows contain `full_text` plus the integer target `score`; evaluation rows contain the same essay text field and may include the target in `full` mode.
- Output: a submission CSV with columns `essay_id,score`.
- Metric: quadratic weighted kappa; higher is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
