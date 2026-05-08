# ICR Age-Related Conditions Multifile Workspace

This workspace is a bounded repository task for the Kaggle `icr-identify-age-related-conditions` competition.

Goal:

- edit the pipeline to improve public balanced binary log loss on the evaluation split
- keep the submission schema stable
- do not edit protected data files

Editable surface:

- `src/features.py`
- `src/models.py`
- `src/postprocess.py`
- `src/pipeline.py`

The submission must match `data/sample_submission.csv` with columns `Id,class_0,class_1`.
