"""Submission helpers for the facial keypoints workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    row_id_column, prediction_column = sample_submission.columns.tolist()[:2]
    submission = sample_submission.loc[:, [row_id_column]].copy()
    submission[prediction_column] = np.asarray(predictions, dtype=float)
    return submission.loc[:, [row_id_column, prediction_column]]
