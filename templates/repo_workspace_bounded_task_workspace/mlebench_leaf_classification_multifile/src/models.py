"""Editable model helpers for the Leaf Classification multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.postprocess import CLASS_COLUMNS, encode_labels, normalize_probabilities


@dataclass
class ModelConfig:
    max_iter: int = 600
    C: float = 1.0


def fit_predict_proba(
    train_matrix: np.ndarray,
    train_df: pd.DataFrame,
    eval_matrix: np.ndarray,
    config: ModelConfig | None = None,
) -> np.ndarray:
    config = config or ModelConfig()
    classifier = LogisticRegression(
        max_iter=config.max_iter,
        C=config.C,
        solver="lbfgs",
    )
    encoded_labels = encode_labels(train_df["species"], class_order=CLASS_COLUMNS)
    classifier.fit(train_matrix, encoded_labels)
    probabilities = classifier.predict_proba(eval_matrix)
    return normalize_probabilities(probabilities)
