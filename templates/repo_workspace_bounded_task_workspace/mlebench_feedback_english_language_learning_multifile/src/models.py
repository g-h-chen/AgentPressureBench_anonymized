"""Editable multi-target regression helpers for the Feedback Prize ELL workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


@dataclass
class BaselineModelConfig:
    alpha: float = 1.0


def fit_predict_targets(
    labels: pd.DataFrame,
    train_matrix,
    eval_matrix,
    config: BaselineModelConfig | None = None,
) -> np.ndarray:
    config = config or BaselineModelConfig()
    model = MultiOutputRegressor(Ridge(alpha=config.alpha))
    model.fit(train_matrix, labels.astype(float))
    return np.asarray(model.predict(eval_matrix), dtype=float)
