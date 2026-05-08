"""Submission helpers for the Russian text-normalization workspace."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMN = "after"


def build_submission_frame(sample_submission: pd.DataFrame, ids: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    mapping = dict(zip(ids.astype(str), predictions.astype(str)))
    submission[TARGET_COLUMN] = submission[ID_COLUMN].astype(str).map(mapping).fillna("")
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
