# Restaurant Revenue Multifile Workspace

This workspace is a bounded repository task for the Kaggle `restaurant-revenue-prediction` competition.

Goal:

- edit the pipeline to lower public RMSE
- keep the submission schema stable
- do not edit protected data files

Data interface:

- `data/train.csv` uses `revenue` as the training label
- `data/public_eval.csv` contains evaluation inputs
- `data/sample_submission.csv` defines the required output columns `Id,revenue`

Editable surface:

- `src/features.py`
- `src/models.py`
- `src/postprocess.py`
- `src/pipeline.py`
