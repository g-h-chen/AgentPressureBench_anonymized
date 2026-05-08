"""Editable model helpers for the Nomad2018 multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.features import TARGET_COLUMNS


@dataclass
class ModelConfig:
    alpha: float = 1.0


def fit_target_models(
    train_matrix: np.ndarray,
    train_df: pd.DataFrame,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    config = config or ModelConfig()
    models: dict[str, object] = {}

    for column in TARGET_COLUMNS:
        target = train_df[column].astype(float)
        if train_matrix.shape[1] == 0 or target.nunique() <= 1:
            model = DummyRegressor(strategy="mean")
        else:
            model = Ridge(alpha=config.alpha)
        model.fit(train_matrix, target)
        models[column] = model

    return models


def predict_targets(models: dict[str, object], eval_matrix: np.ndarray) -> np.ndarray:
    predictions = [np.asarray(models[column].predict(eval_matrix), dtype=float) for column in TARGET_COLUMNS]
    return np.column_stack(predictions)
