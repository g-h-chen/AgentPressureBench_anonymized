"""Submission helpers for the Leaf Classification multifile workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd


SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"
CLASS_COLUMNS = pd.read_csv(SAMPLE_SUBMISSION_PATH, nrows=0).columns.tolist()[1:]
LABEL_COLUMN = "species"
ID_COLUMN = "id"


def encode_labels(labels: pd.Series, class_order: list[str] | tuple[str, ...] = CLASS_COLUMNS) -> np.ndarray:
    categorical = pd.Categorical(labels, categories=list(class_order), ordered=True)
    codes = categorical.codes.astype(int)
    if (codes < 0).any():
        missing = sorted({str(label) for label in labels[pd.Series(codes < 0)]})
        raise ValueError(f"Unexpected labels encountered: {missing}")
    return codes


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
