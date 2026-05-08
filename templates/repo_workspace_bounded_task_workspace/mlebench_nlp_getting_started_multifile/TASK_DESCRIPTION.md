Task summary:

- Goal: predict whether each tweet is about a real disaster.
- Input data: training rows contain `keyword`, `location`, `text`, and the binary label `target`; evaluation rows contain the same text fields and may include the label in `full` mode.
- Output: a submission CSV with columns `id,target`.
- Metric: binary F1; higher is better.
- Workspace structure: use the multifile layout to improve text preprocessing, model fitting, and submission formatting separately.
