"""Editable model helpers for the aerial-cactus multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.image_features import load_image_features


@dataclass
class ModelConfig:
    max_iter: int = 400
    solver: str = "liblinear"


def fit_model(train_df: pd.DataFrame, train_image_dir, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    features = load_image_features(train_df["id"], train_image_dir)
    labels = train_df["has_cactus"].astype(int)
    if labels.nunique() < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = LogisticRegression(max_iter=config.max_iter, solver=config.solver)
    model.fit(features, labels)
    return model


def predict_proba(model: object, eval_df: pd.DataFrame, eval_image_dir) -> np.ndarray:
    eval_features = load_image_features(eval_df["id"], eval_image_dir)
    return np.asarray(model.predict_proba(eval_features)[:, 1], dtype=float)
