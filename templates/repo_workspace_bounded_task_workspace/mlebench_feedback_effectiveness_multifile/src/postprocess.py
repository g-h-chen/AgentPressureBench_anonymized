"""Submission and label-order helpers for the multi-file feedback-effectiveness workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd


CLASS_COLUMNS = ["Ineffective", "Adequate", "Effective"]
LABEL_COLUMN = "discourse_effectiveness"


def encode_labels(labels: pd.Series, class_order: list[str] | tuple[str, ...] = CLASS_COLUMNS) -> np.ndarray:
    categorical = pd.Categorical(labels, categories=list(class_order), ordered=True)
    codes = categorical.codes.astype(int)
    if (codes < 0).any():
        missing = sorted({str(label) for label in labels[pd.Series(codes < 0)]})
        raise ValueError(f"Unexpected labels encountered: {missing}")
    return codes


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_COLUMNS):
        raise ValueError(
            f"Expected probability matrix with shape (n_rows, {len(CLASS_COLUMNS)}), "
            f"got {probabilities.shape}."
        )
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return probabilities / row_sums


def build_submission_frame(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    class_order: list[str] | tuple[str, ...] = CLASS_COLUMNS,
) -> pd.DataFrame:
    normalized = normalize_probabilities(probabilities)
    submission = sample_submission.loc[:, ["discourse_id"]].copy()
    for index, class_name in enumerate(class_order):
        submission[class_name] = normalized[:, index]
    return submission.loc[:, ["discourse_id", *class_order]]
