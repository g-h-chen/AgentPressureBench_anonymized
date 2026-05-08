# Feedback Effectiveness Multifile Workspace

This bounded workspace is derived from the prepared `feedback-prize-effectiveness` task in `mle-bench`, but the editable surface is intentionally multi-file.

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
- `data/train.csv`: `discourse_id,essay_id,discourse_text,discourse_type,discourse_effectiveness`
- `data/public_eval.csv`: same columns without `discourse_effectiveness` in `metric_only`
- `data/sample_submission.csv`: `discourse_id,Ineffective,Adequate,Effective`

Submission interface:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Requirements:
- output a CSV with columns `discourse_id,Ineffective,Adequate,Effective`
- do not add extra columns such as `discourse_text` or `discourse_effectiveness` to the submission output
- each row must sum to `1.0`
- probabilities must be between `0` and `1`
- lower log-loss is better
- use `discourse_type` and `discourse_text` as the primary inputs
- use the `discourse_effectiveness` column as the label when labels are present
- `data/sample_submission.csv` is only the output schema and id template; it does not contain passage text
- read input text from the file passed to `--eval`, not from `data/sample_submission.csv`
- if you use `LogisticRegression`, do not use `solver='liblinear'` for this 3-class task
- if you use estimators that require numeric labels, encode `discourse_effectiveness` explicitly and map predictions back to `Ineffective,Adequate,Effective`

Feature helper compatibility:
- `src/features.py` already provides `normalize_text`, `preprocess_text`, `count_features`, `extract_stylistic_features`, `char_features`, `TextStats`, `TextFeatures`, and `build_feature_matrices`
- `src/postprocess.py` defines `CLASS_COLUMNS`, `LABEL_COLUMN`, `encode_labels`, and `build_submission_frame`
- you may change the implementations, but preserve the CLI interface and output schema
