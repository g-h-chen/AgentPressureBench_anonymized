"""Submission helpers for the ICR multifile workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_submission_frame(sample_submission: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    id_column, class_zero_column, class_one_column = sample_submission.columns.tolist()[:3]
    submission = sample_submission.loc[:, [id_column]].copy()
    positive = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    submission[class_zero_column] = 1.0 - positive
    submission[class_one_column] = positive
    return submission.loc[:, [id_column, class_zero_column, class_one_column]]
