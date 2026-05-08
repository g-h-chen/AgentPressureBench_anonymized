"""Submission helpers for the Titanic multifile workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    id_column, label_column = sample_submission.columns.tolist()[:2]
    submission = sample_submission.loc[:, [id_column]].copy()
    submission[label_column] = np.asarray(predictions, dtype=int)
    return submission.loc[:, [id_column, label_column]]
