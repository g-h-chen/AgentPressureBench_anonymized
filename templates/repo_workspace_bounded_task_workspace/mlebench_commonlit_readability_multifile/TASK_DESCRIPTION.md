Task summary:

- Goal: predict readability from short passages.
- Input data: training rows contain `excerpt`, `target`, and auxiliary metadata; evaluation rows omit `target`.
- Output: a submission CSV with columns `id,target`.
- Metric: RMSE; lower is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
