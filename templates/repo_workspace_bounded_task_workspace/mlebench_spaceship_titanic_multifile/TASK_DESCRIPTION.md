Task summary:

- Goal: predict whether each passenger was transported to another dimension.
- Input data: training rows contain mixed tabular features plus the binary label `Transported`; evaluation rows contain the same features and may include the label in `full` mode.
- Output: a submission CSV with columns `PassengerId,Transported`.
- Metric: accuracy; higher is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
