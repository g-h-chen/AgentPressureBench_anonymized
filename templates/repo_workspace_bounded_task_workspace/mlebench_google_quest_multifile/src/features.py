"""Editable text-pair feature helpers for the Google QUEST workspace."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class FeatureConfig:
    word_max_features: int = 20000
    word_ngram_range: tuple[int, int] = (1, 2)
    lowercase: bool = True


def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    return (
        df["question_title"].fillna("").astype(str).str.strip()
        + "\n"
        + df["question_body"].fillna("").astype(str).str.strip()
        + "\n"
        + df["answer"].fillna("").astype(str).str.strip()
    ).str.strip()


def build_feature_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, TfidfVectorizer]:
    config = config or FeatureConfig()
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=config.word_ngram_range,
        max_features=config.word_max_features,
        lowercase=config.lowercase,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_matrix = vectorizer.fit_transform(combine_text_fields(train_df))
    eval_matrix = vectorizer.transform(combine_text_fields(eval_df))
    return train_matrix, eval_matrix, vectorizer
