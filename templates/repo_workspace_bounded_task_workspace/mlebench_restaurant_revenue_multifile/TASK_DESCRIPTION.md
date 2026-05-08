Task summary:

- Goal: predict restaurant revenue.
- Input data: tabular business features in `data/train.csv` and `data/public_eval.csv`.
- Training target: the label column in `data/train.csv` is `revenue`.
- Output: a submission CSV with columns `Id,revenue`, matching `data/sample_submission.csv`.
- Metric: RMSE; lower is better.

Repository guidance:

- improve preprocessing, feature engineering, or the regressor
- keep the CLI interface in `src/pipeline.py` unchanged
- do not modify benchmark data files
