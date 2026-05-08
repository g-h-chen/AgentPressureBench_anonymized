"""Editable feature helpers for the Leaf Classification multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_COLUMN = "id"
LABEL_COLUMN = "species"


@dataclass
class FeatureBundle:
    train_matrix: np.ndarray
    eval_matrix: np.ndarray
    feature_columns: list[str]


def build_feature_bundle(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> FeatureBundle:
    feature_columns = [
        column
        for column in train_df.columns
        if column not in {ID_COLUMN, LABEL_COLUMN}
    ]
    train_matrix = train_df[feature_columns].fillna(0.0).astype(float).to_numpy()
    eval_matrix = eval_df[feature_columns].fillna(0.0).astype(float).to_numpy()
    return FeatureBundle(
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        feature_columns=feature_columns,
    )
