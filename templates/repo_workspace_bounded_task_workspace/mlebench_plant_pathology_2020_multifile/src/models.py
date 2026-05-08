"""Editable multiclass image-model helpers for the Plant Pathology 2020 workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.image_features import load_image_features
from src.postprocess import CLASS_COLUMNS, normalize_probabilities


@dataclass
class ModelConfig:
    max_iter: int = 400
    C: float = 1.0


def fit_model(train_df: pd.DataFrame, train_image_dir, config: ModelConfig | None = None) -> object:
    config = config or ModelConfig()
    features = load_image_features(train_df["image_id"], train_image_dir)
    labels = train_df[list(CLASS_COLUMNS)].idxmax(axis=1)
    model = LogisticRegression(
        max_iter=config.max_iter,
        C=config.C,
        solver="lbfgs",
    )
    model.fit(features, labels)
    return model


def predict_proba(model: object, eval_df: pd.DataFrame, eval_image_dir) -> np.ndarray:
    eval_features = load_image_features(eval_df["image_id"], eval_image_dir)
    probabilities = model.predict_proba(eval_features)
    return normalize_probabilities(probabilities)
