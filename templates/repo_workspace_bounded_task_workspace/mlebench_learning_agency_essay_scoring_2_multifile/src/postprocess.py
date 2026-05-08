"""Submission helpers for the Learning Agency essay-scoring workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "essay_id"
TARGET_COLUMN = "score"


def build_submission_frame(sample_submission: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = np.asarray(scores, dtype=float).round().astype(int)
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
