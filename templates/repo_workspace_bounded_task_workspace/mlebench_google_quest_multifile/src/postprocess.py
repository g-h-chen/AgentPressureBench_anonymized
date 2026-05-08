"""Submission helpers for the Google QUEST workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "qa_id"


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    target_columns = sample_submission.columns.tolist()[1:]
    values = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    for index, column in enumerate(target_columns):
        submission[column] = values[:, index]
    return submission.loc[:, [ID_COLUMN, *target_columns]]
