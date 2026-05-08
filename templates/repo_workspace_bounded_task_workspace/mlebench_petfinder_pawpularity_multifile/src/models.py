"""Editable regression-model helpers for the Petfinder Pawpularity workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.features import TARGET_COLUMN


@dataclass
class ModelConfig:
    alpha: float = 1.0


def fit_model(train_matrix: np.ndarray, train_df: pd.DataFrame, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    target = train_df[TARGET_COLUMN].astype(float)
    if train_matrix.shape[1] == 0 or target.nunique() <= 1:
        model = DummyRegressor(strategy="mean")
    else:
        model = Ridge(alpha=config.alpha)
    model.fit(train_matrix, target)
    return model


def predict_targets(model: object, eval_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(eval_matrix), dtype=float)
