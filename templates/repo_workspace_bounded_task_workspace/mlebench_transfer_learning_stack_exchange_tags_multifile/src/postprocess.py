"""Submission helpers for the Stack Exchange tags workspace."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMN = "tags"


def build_submission_frame(sample_submission: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = pd.Series(scores, dtype=str)
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
