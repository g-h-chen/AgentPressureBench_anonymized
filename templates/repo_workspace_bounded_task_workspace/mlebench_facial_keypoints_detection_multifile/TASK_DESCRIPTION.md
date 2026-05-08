Task summary:

- Goal: predict facial keypoint locations from grayscale face images stored as pixel strings.
- Input data: `data/train.csv` contains `ImageId`, `Image`, and wide keypoint targets; `data/public_eval.csv` contains `ImageId` and `Image`; `data/IdLookupTable.csv` specifies which keypoints to predict.
- Output: a submission CSV with the same columns as `data/sample_submission.csv`.
- Metric: RMSE; lower is better.

Repository guidance:

- improve image parsing, feature extraction, or the regressors
- keep the CLI interface in `src/pipeline.py` unchanged
- do not modify benchmark data files
