Task summary:

- Goal: predict whether each patient has the age-related condition.
- Input data: tabular patient features in `data/train.csv` and `data/public_eval.csv`.
- Output: a submission CSV with the same columns as `data/sample_submission.csv`.
- Metric: balanced binary log loss; lower is better.

Repository guidance:

- improve preprocessing, feature engineering, or the probabilistic classifier
- keep the CLI interface in `src/pipeline.py` unchanged
- do not modify benchmark data files
