"""Editable feature helpers for the Petfinder Pawpularity multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_COLUMN = "Id"
TARGET_COLUMN = "Pawpularity"


@dataclass
class FeatureBundle:
    train_matrix: np.ndarray
    eval_matrix: np.ndarray
    feature_columns: list[str]


def build_feature_bundle(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> FeatureBundle:
    feature_columns = [
        column
        for column in train_df.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    train_features = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    eval_features = eval_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = train_features.median(numeric_only=True).fillna(0.0)
    train_matrix = train_features.fillna(fill_values).to_numpy(dtype=float)
    eval_matrix = eval_features.fillna(fill_values).to_numpy(dtype=float)
    return FeatureBundle(
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        feature_columns=feature_columns,
    )
