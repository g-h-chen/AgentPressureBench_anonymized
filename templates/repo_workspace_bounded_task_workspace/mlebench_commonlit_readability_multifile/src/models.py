"""Editable text-regression helpers for the CommonLit workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge


@dataclass
class ModelConfig:
    alpha: float = 1.0


def fit_model(train_matrix, target: pd.Series, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    numeric_target = target.astype(float)
    if train_matrix.shape[1] == 0 or numeric_target.nunique() <= 1:
        model = DummyRegressor(strategy="mean")
    else:
        model = Ridge(alpha=config.alpha)
    model.fit(train_matrix, numeric_target)
    return model


def predict_targets(model: object, eval_matrix) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise ValueError("Model does not support predict().")
    return np.asarray(model.predict(eval_matrix), dtype=float)
