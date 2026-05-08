"""Editable model helpers for the Spaceship Titanic multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.features import LABEL_COLUMN


@dataclass
class ModelConfig:
    max_iter: int = 500
    C: float = 1.0
    class_weight: str | None = None


def fit_model(train_matrix: np.ndarray, train_df: pd.DataFrame, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    labels = train_df[LABEL_COLUMN].astype(int)
    if train_matrix.shape[1] == 0 or labels.nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(
            max_iter=config.max_iter,
            C=config.C,
            class_weight=config.class_weight,
            solver="lbfgs",
        )
    model.fit(train_matrix, labels)
    return model


def predict_labels(model: object, eval_matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(eval_matrix), dtype=int)
    raise ValueError("Model does not support predict().")
