"""Submission helpers for the Kuzushiji workspace."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "image_id"
TARGET_COLUMN = "labels"


def build_submission_frame(sample_submission: pd.DataFrame, predictions: list[str]) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = [str(prediction) for prediction in predictions]
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
