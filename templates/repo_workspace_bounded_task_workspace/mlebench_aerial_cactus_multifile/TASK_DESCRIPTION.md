Task summary:

- Goal: predict whether each 32x32 aerial image contains a cactus.
- Input data: training rows contain `id` plus the binary label `has_cactus`; evaluation rows contain `id` and may include the label in `full` mode.
- Output: a submission CSV with columns `id,has_cactus`.
- Metric: ROC-AUC; higher is better.
- Workspace structure: use the multifile layout to improve image features, model fitting, and submission logic separately.
