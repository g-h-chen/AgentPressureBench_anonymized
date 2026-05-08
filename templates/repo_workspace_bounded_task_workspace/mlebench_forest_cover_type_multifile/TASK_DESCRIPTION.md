Task summary:

- Goal: predict the forest cover type for each row.
- Input data: tabular environmental features in `data/train.csv` and `data/public_eval.csv`.
- Output: a submission CSV with the same columns as `data/sample_submission.csv`.
- Metric: accuracy; higher is better.

Repository guidance:

- improve preprocessing, feature engineering, or the classifier
- keep the CLI interface in `src/pipeline.py` unchanged
- do not modify benchmark data files
