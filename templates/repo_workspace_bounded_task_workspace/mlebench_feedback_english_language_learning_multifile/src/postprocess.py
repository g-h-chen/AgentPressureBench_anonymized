"""Submission helpers for the Feedback Prize ELL workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "text_id"
TARGET_COLUMNS = pd.read_csv("data/sample_submission.csv", nrows=0).columns.tolist()[1:]


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    values = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    for index, column in enumerate(TARGET_COLUMNS):
        submission[column] = values[:, index]
    return submission.loc[:, [ID_COLUMN, *TARGET_COLUMNS]]
