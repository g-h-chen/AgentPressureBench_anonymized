"""Submission helpers for the Nomad2018 multifile workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import ID_COLUMN, TARGET_COLUMNS


def _clip_predictions(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    return np.clip(predictions, 0.0, None)


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    clipped = _clip_predictions(predictions)
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    for index, column in enumerate(TARGET_COLUMNS):
        submission[column] = clipped[:, index]
    return submission.loc[:, [ID_COLUMN, *TARGET_COLUMNS]]
