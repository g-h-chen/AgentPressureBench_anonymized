"""Submission helpers for the aerial-cactus multifile workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "id"
LABEL_COLUMN = "has_cactus"


def build_submission_frame(sample_submission: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[LABEL_COLUMN] = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    return submission.loc[:, [ID_COLUMN, LABEL_COLUMN]]
