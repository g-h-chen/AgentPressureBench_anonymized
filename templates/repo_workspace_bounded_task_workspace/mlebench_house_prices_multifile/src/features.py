"""Generic tabular feature helpers for the tabular multifile workspaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureBundle:
    train_matrix: np.ndarray
    eval_matrix: np.ndarray
    feature_columns: list[str]


def infer_task_columns(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_column: str | None = None,
) -> tuple[str, str]:
    id_column = str(train_df.columns[0])
    if target_column is not None:
        if target_column not in train_df.columns:
            raise ValueError(f"Target column `{target_column}` not found in training data.")
        return id_column, target_column
    label_candidates = [column for column in train_df.columns if column not in eval_df.columns]
    if len(label_candidates) != 1:
        raise ValueError(f"Expected exactly one label column, found {label_candidates}")
    return id_column, label_candidates[0]


def build_feature_bundle(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_column: str | None = None,
) -> FeatureBundle:
    id_column, label_column = infer_task_columns(train_df, eval_df, target_column=target_column)
    train_features = train_df.drop(columns=[label_column], errors="ignore").copy()
    eval_features = eval_df.copy()

    numeric_columns = [
        column
        for column in train_features.columns
        if column != id_column and pd.api.types.is_numeric_dtype(train_features[column])
    ]
    categorical_columns = [
        column
        for column in train_features.columns
        if column not in numeric_columns and column != id_column
    ]

    train_numeric = train_features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    eval_numeric = eval_features.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
    numeric_fill = train_numeric.median(numeric_only=True).fillna(0.0)
    train_numeric = train_numeric.fillna(numeric_fill)
    eval_numeric = eval_numeric.fillna(numeric_fill)

    if categorical_columns:
        train_categorical = train_features[categorical_columns].fillna("missing").astype(str)
        eval_categorical = eval_features.reindex(columns=categorical_columns).fillna("missing").astype(str)
        combined = pd.concat([train_categorical, eval_categorical], axis=0)
        encoded = pd.get_dummies(combined, columns=categorical_columns, dummy_na=False)
        train_encoded = encoded.iloc[: len(train_features)].reset_index(drop=True)
        eval_encoded = encoded.iloc[len(train_features) :].reset_index(drop=True)
        train_matrix = np.hstack([train_numeric.to_numpy(dtype=float), train_encoded.to_numpy(dtype=float)])
        eval_matrix = np.hstack([eval_numeric.to_numpy(dtype=float), eval_encoded.to_numpy(dtype=float)])
        feature_columns = list(train_numeric.columns) + list(train_encoded.columns)
    else:
        train_matrix = train_numeric.to_numpy(dtype=float)
        eval_matrix = eval_numeric.to_numpy(dtype=float)
        feature_columns = list(train_numeric.columns)

    return FeatureBundle(train_matrix=train_matrix, eval_matrix=eval_matrix, feature_columns=feature_columns)
