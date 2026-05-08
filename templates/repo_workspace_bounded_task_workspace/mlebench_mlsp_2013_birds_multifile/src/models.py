"""Editable model helpers for the MLSP 2013 Birds workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


@dataclass
class ModelConfig:
    max_iter: int = 400
    C: float = 1.0


def fit_predict_proba(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    eval_features: pd.DataFrame,
    config: ModelConfig | None = None,
) -> np.ndarray:
    config = config or ModelConfig()
    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=config.max_iter,
            C=config.C,
            solver="lbfgs",
        )
    )
    classifier.fit(train_features.to_numpy(dtype=float), train_labels.to_numpy(dtype=int))
    return np.clip(classifier.predict_proba(eval_features.to_numpy(dtype=float)), 0.0, 1.0)
