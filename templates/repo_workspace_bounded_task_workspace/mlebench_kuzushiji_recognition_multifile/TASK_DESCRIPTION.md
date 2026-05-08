Task summary:

- Goal: locate and identify kuzushiji characters on each page image.
- Input data: `data/train.csv` contains `image_id` plus training bounding-box strings; `data/public_val.csv` contains `image_id` and may include `labels` in `full` mode.
- Output: a submission CSV with the same columns as `data/sample_submission.csv`.
- Metric: modified page-level F1; higher is better.
- Workspace structure: use the multifile layout to improve page-image features, retrieval/localization logic, and submission formatting separately.
