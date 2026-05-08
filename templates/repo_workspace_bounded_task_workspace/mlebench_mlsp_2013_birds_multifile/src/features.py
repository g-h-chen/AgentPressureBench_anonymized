"""Editable feature helpers for the MLSP 2013 Birds workspace."""

from __future__ import annotations

import pandas as pd

SPECIES_COLUMNS = [f"species_{index:02d}" for index in range(19)]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in {"rec_id", *SPECIES_COLUMNS}]


def label_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in SPECIES_COLUMNS if column in df.columns]
