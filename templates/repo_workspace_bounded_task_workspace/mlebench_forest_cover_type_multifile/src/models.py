"""Generic multiclass-classification model helpers for the forest-cover workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


@dataclass
class ModelConfig:
    max_iter: int = 1000
    C: float = 1.0


def fit_model(train_matrix: np.ndarray, labels: pd.Series, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str))
    if train_matrix.shape[1] == 0 or pd.Series(encoded).nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(max_iter=config.max_iter, C=config.C, solver="lbfgs")
    model.fit(train_matrix, encoded)
    setattr(model, "_label_encoder", encoder)
    setattr(model, "_sample_labels", labels)
    return model


def predict_labels(model: object, eval_matrix: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise ValueError("Model does not support predict().")
    encoded = np.asarray(model.predict(eval_matrix))
    encoder = getattr(model, "_label_encoder", None)
    if encoder is None:
        raise ValueError("Model is missing the fitted label encoder.")
    decoded = encoder.inverse_transform(encoded.astype(int))
    sample = getattr(model, "_sample_labels", None)
    if sample is not None and pd.api.types.is_numeric_dtype(pd.Series(sample)):
        return pd.to_numeric(pd.Series(decoded), errors="raise").astype(int).to_numpy()
    return decoded
