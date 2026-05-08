"""Binary probability-model helpers for the ICR workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelConfig:
    max_iter: int = 1000
    C: float = 1.0


def fit_model(train_matrix: np.ndarray, labels: pd.Series, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    encoded = _encode_binary_labels(labels)
    if train_matrix.shape[1] == 0 or pd.Series(encoded).nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(max_iter=config.max_iter, C=config.C, solver="lbfgs")
    model.fit(train_matrix, encoded)
    return model


def predict_probabilities(model: object, eval_matrix: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba().")
    probabilities = model.predict_proba(eval_matrix)[:, 1]
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def _encode_binary_labels(labels: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(labels):
        return labels.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(labels):
        return (labels.astype(float) >= 0.5).astype(int).to_numpy()
    normalized = labels.fillna("").astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
    if not normalized.isin(mapping).all():
        invalid = sorted(normalized[~normalized.isin(mapping)].unique().tolist())
        raise ValueError(f"Unexpected binary labels: {invalid[:5]}")
    return normalized.map(mapping).astype(int).to_numpy()
