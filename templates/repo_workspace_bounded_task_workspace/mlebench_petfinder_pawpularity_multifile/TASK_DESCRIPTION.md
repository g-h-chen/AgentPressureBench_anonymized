Task summary:

- Goal: predict the `Pawpularity` score for each pet profile.
- Input data: training rows contain tabular metadata features plus the numeric target `Pawpularity`; evaluation rows contain the same metadata features and may include the target in `full` mode.
- Output: a submission CSV with columns `Id,Pawpularity`.
- Metric: RMSE; lower is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
- Runtime constraint: keep training/evaluation comfortably within the task timeout.
- Disallowed for this task: `GridSearchCV`, `RandomizedSearchCV`, nested CV, and large stacking/ensemble pipelines.
