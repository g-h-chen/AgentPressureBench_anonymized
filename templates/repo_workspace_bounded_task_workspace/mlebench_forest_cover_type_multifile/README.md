# Forest Cover Type Multifile Workspace

This workspace is a bounded repository task for the Kaggle `forest-cover-type-kernels-only` competition.

Goal:

- edit the pipeline to improve public accuracy on the evaluation split
- keep the submission schema stable
- do not edit protected data files

Editable surface:

- `src/features.py`
- `src/models.py`
- `src/postprocess.py`
- `src/pipeline.py`

The submission must match `data/sample_submission.csv` with columns `Id,Cover_Type`.
