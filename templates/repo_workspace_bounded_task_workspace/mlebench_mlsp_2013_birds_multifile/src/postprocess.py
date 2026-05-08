"""Submission helpers for the MLSP 2013 Birds workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "Id"
TARGET_COLUMN = "Probability"
NUM_SPECIES = 19


def build_submission_frame(sample_submission: pd.DataFrame, rec_ids: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(probabilities, dtype=float)
    rec_index = {int(rec_id): row for row, rec_id in enumerate(rec_ids.astype(int).tolist())}
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    values: list[float] = []
    for submission_id in sample_submission[ID_COLUMN].astype(int).tolist():
        rec_id = submission_id // 100
        species_id = submission_id % 100
        if species_id >= NUM_SPECIES:
            raise ValueError(f"Unexpected species id in submission template: {submission_id}")
        values.append(float(matrix[rec_index[rec_id], species_id]))
    submission[TARGET_COLUMN] = values
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
