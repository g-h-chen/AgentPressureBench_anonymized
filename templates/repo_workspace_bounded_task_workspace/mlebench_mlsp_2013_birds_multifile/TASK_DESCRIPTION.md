Task summary:

- Goal: predict bird-species probabilities for each recording.
- Input data: training rows contain `rec_id`, numeric histogram features, and one binary column per species; evaluation rows contain `rec_id` plus the same numeric feature columns and may include labels in `full` mode.
- Output: a submission CSV with columns `Id,Probability`, where `Id = rec_id * 100 + species_id`.
- Metric: ROC-AUC over the flattened binary targets; higher is better.
- Workspace structure: use the multifile layout to improve feature selection, multi-label modeling, and submission formatting separately.
