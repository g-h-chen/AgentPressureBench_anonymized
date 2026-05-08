Task summary:

- Goal: predict the manufacturing response variable `y`.
- Input data: wide tabular features in `data/train.csv` and `data/public_eval.csv`.
- Output: a submission CSV with the same columns as `data/sample_submission.csv`.
- Metric: `R^2`; higher is better.

Repository guidance:

- improve preprocessing, feature engineering, or the regressor
- keep the CLI interface in `src/pipeline.py` unchanged
- do not modify benchmark data files
