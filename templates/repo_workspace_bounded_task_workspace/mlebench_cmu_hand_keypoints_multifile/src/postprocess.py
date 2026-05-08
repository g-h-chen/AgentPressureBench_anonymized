"""Submission helpers for the CMU hand-keypoints workspace."""

from __future__ import annotations

import pandas as pd


def build_submission_frame(sample_submission: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    expected_columns = sample_submission.columns.tolist()
    submission = predictions.copy()
    return submission.loc[:, expected_columns]
