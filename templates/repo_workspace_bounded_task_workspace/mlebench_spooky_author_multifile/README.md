# Spooky Author Multi-File Workspace

This bounded workspace is derived from the prepared `spooky-author-identification` task in `mle-bench`, but the editable surface is intentionally multi-file.

Your job is to improve the submission pipeline while keeping the CLI interface stable.

Editable files:
- `src/pipeline.py`
- `src/features.py`
- `src/models.py`
- `src/postprocess.py`
- additional new helper files you create under `src/`

Do not edit anything under `data/`.

Recommended module responsibilities:
- `src/pipeline.py`: CLI, data loading, and high-level orchestration only
- `src/features.py`: preprocessing and feature extraction
- `src/models.py`: classifier construction and fitting logic
- `src/postprocess.py`: label encoding, probability normalization, and submission formatting

Read `ENVIRONMENT.md` before making library-specific changes.
Prefer keeping `src/pipeline.py` thin and moving most changes into the helper modules.

Data schema for this task:
- `data/train.csv`: `id,text,author`
- `data/public_eval.csv`: `id,text` in `metric_only`, `id,text,author` in `full`
- `data/sample_submission.csv`: `id,EAP,HPL,MWS`

Submission command:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Requirements:
- output a CSV with columns `id,EAP,HPL,MWS`
- do not add extra columns such as `text` or `author` to the submission output
- each row must sum to `1.0`
- probabilities must be between `0` and `1`
- lower log-loss is better
- use the `text` column as the passage input
- use the `author` column as the label when labels are present
- `data/sample_submission.csv` is only the output schema and id template; it does not contain passage text
- read passage features from the file passed to `--eval`, not from `data/sample_submission.csv`
- if you use `LogisticRegression`, do not use `solver='liblinear'` for this 3-class task
- if you use estimators that require numeric labels, encode `author` explicitly and map predictions back to `EAP,HPL,MWS`

Feature helper compatibility:
- `src/features.py` already provides `normalize_text`, `preprocess_text`, `count_features`, `extract_stylistic_features`, `char_features`, `TextStats`, `TextFeatures`, and `build_feature_matrices`
- `src/postprocess.py` defines `CLASS_COLUMNS`, `LABEL_COLUMN`, `encode_labels`, and `build_submission_frame`
- you may change the implementations, but preserve the CLI interface and output schema
