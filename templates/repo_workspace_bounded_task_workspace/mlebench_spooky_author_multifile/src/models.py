"""Editable model-construction helpers for the multi-file spooky workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from src.postprocess import encode_labels, normalize_probabilities


@dataclass
class BaselineModelConfig:
    solver: str = "lbfgs"
    max_iter: int = 500
    C: float = 0.8
    class_weight: str | None = None


def build_classifier(config: BaselineModelConfig | None = None) -> LogisticRegression:
    config = config or BaselineModelConfig()
    return LogisticRegression(
        solver=config.solver,
        max_iter=config.max_iter,
        C=config.C,
        class_weight=config.class_weight,
        random_state=42,
    )


def fit_predict_proba(
    labels: pd.Series,
    train_matrix: sparse.spmatrix,
    eval_matrix: sparse.spmatrix,
    class_order: list[str] | tuple[str, ...],
    config: BaselineModelConfig | None = None,
) -> np.ndarray:
    classifier = build_classifier(config=config)
    encoded_labels = encode_labels(labels, class_order=class_order)
    classifier.fit(train_matrix, encoded_labels)
    probabilities = classifier.predict_proba(eval_matrix)
    return normalize_probabilities(probabilities)
