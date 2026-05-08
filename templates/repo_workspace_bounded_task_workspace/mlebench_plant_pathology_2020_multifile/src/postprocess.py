"""Submission helpers for the Plant Pathology 2020 workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "image_id"
CLASS_COLUMNS = ["healthy", "multiple_diseases", "rust", "scab"]


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return probabilities / row_sums


def build_submission_frame(sample_submission: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    normalized = normalize_probabilities(probabilities)
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    for index, class_name in enumerate(CLASS_COLUMNS):
        submission[class_name] = normalized[:, index]
    return submission.loc[:, [ID_COLUMN, *CLASS_COLUMNS]]
