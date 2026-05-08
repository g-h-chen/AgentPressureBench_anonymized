"""Editable essay-score prediction helpers for the Learning Agency workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


@dataclass
class BaselineModelConfig:
    solver: str = "lbfgs"
    max_iter: int = 400
    C: float = 2.0


def fit_predict_scores(
    labels: pd.Series,
    train_matrix,
    eval_matrix,
    config: BaselineModelConfig | None = None,
) -> np.ndarray:
    config = config or BaselineModelConfig()
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(int))
    classifier = LogisticRegression(
        solver=config.solver,
        max_iter=config.max_iter,
        C=config.C,
        random_state=42,
    )
    classifier.fit(train_matrix, encoded)
    predictions = classifier.predict(eval_matrix)
    return encoder.inverse_transform(predictions).astype(int)
