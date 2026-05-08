"""Editable token-mapping helpers for the Russian text-normalization workspace."""

from __future__ import annotations

import pandas as pd

def fit_token_mapping(train_df: pd.DataFrame) -> dict[str, str]:
    counts = (
        train_df.groupby(["before_key", "after"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["before_key", "count", "after"], ascending=[True, False, True])
    )
    best = counts.drop_duplicates(subset=["before_key"], keep="first")
    return dict(zip(best["before_key"].astype(str), best["after"].astype(str)))


def predict_after(eval_df: pd.DataFrame, token_mapping: dict[str, str]) -> pd.Series:
    fallback = eval_df["before"].astype(str)
    mapped = eval_df["before_key"].map(token_mapping)
    return mapped.fillna(fallback).astype(str)
