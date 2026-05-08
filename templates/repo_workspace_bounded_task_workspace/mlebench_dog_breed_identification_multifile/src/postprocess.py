"""Submission helpers for the Dog Breed workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "id"


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return probabilities / row_sums


def build_submission_frame(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    model_classes: list[str],
) -> pd.DataFrame:
    class_columns = sample_submission.columns.tolist()[1:]
    aligned = np.full((len(sample_submission), len(class_columns)), 1e-9, dtype=float)
    model_index = {label: idx for idx, label in enumerate(model_classes)}
    raw = np.asarray(probabilities, dtype=float)
    for col_idx, class_name in enumerate(class_columns):
        if class_name in model_index:
            aligned[:, col_idx] = raw[:, model_index[class_name]]
    normalized = normalize_probabilities(aligned)
    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    for index, class_name in enumerate(class_columns):
        submission[class_name] = normalized[:, index]
    return submission.loc[:, [ID_COLUMN, *class_columns]]
