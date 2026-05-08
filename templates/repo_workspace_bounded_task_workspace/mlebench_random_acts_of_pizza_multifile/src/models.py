"""Editable binary text-model helpers for the Random Acts of Pizza workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


@dataclass
class BaselineModelConfig:
    solver: str = "liblinear"
    max_iter: int = 400
    C: float = 1.0
    class_weight: str | None = None


def fit_predict_proba(
    labels: pd.Series,
    train_matrix: sparse.spmatrix,
    eval_matrix: sparse.spmatrix,
    config: BaselineModelConfig | None = None,
) -> np.ndarray:
    config = config or BaselineModelConfig()
    classifier = LogisticRegression(
        solver=config.solver,
        max_iter=config.max_iter,
        C=config.C,
        class_weight=config.class_weight,
        random_state=42,
    )
    classifier.fit(train_matrix, labels.astype(int))
    probabilities = classifier.predict_proba(eval_matrix)[:, 1]
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
