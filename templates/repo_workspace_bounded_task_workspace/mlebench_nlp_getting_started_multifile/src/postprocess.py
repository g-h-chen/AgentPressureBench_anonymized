"""Submission helpers for the NLP Getting Started workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "id"
LABEL_COLUMN = "target"


def build_submission_frame(sample_submission: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[LABEL_COLUMN] = np.asarray(predictions, dtype=int)
    return submission.loc[:, [ID_COLUMN, LABEL_COLUMN]]
