"""Editable token-normalization helpers for the English text-normalization workspace."""

from __future__ import annotations

import re
import pandas as pd


def normalize_token(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip()


def build_submission_ids(df: pd.DataFrame) -> pd.Series:
    return df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["before"] = frame["before"].fillna("").astype(str)
    frame["before_key"] = frame["before"].map(normalize_token)
    frame["id"] = build_submission_ids(frame)
    return frame
