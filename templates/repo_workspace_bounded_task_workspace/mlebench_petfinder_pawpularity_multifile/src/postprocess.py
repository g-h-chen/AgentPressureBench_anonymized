"""Submission helpers for the Petfinder Pawpularity workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import ID_COLUMN, TARGET_COLUMN


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = np.clip(np.asarray(predictions, dtype=float), 0.0, 100.0)
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
