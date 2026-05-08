"""Submission helpers for the CommonLit readability workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMN = "target"


def build_submission_frame(sample_submission: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = np.asarray(scores, dtype=float)
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
