"""Editable feature helpers for the Nomad2018 multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMNS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


@dataclass
class FeatureBundle:
    train_matrix: np.ndarray
    eval_matrix: np.ndarray
    feature_columns: list[str]


def _numeric_feature_columns(train_df: pd.DataFrame) -> list[str]:
    excluded = {ID_COLUMN, *TARGET_COLUMNS}
    return [
        column
        for column in train_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(train_df[column])
    ]


def build_feature_bundle(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> FeatureBundle:
    feature_columns = _numeric_feature_columns(train_df)
    if not feature_columns:
        empty_train = np.zeros((len(train_df), 0), dtype=float)
        empty_eval = np.zeros((len(eval_df), 0), dtype=float)
        return FeatureBundle(train_matrix=empty_train, eval_matrix=empty_eval, feature_columns=[])

    fill_values = train_df[feature_columns].median(numeric_only=True).fillna(0.0)
    train_matrix = train_df[feature_columns].fillna(fill_values).astype(float).to_numpy()
    eval_matrix = eval_df.reindex(columns=feature_columns).fillna(fill_values).astype(float).to_numpy()
    return FeatureBundle(
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        feature_columns=feature_columns,
    )
