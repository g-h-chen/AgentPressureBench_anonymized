"""Editable feature extraction utilities for the disaster-tweets workspace."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def _coerce_text_series(texts: object) -> pd.Series:
    if isinstance(texts, pd.DataFrame):
        if "text" in texts.columns:
            return texts["text"].fillna("").astype(str)
        if len(texts.columns) == 0:
            return pd.Series(dtype=str)
        return texts.iloc[:, 0].fillna("").astype(str)
    if isinstance(texts, pd.Series):
        return texts.fillna("").astype(str)
    if isinstance(texts, np.ndarray):
        return pd.Series(texts).fillna("").astype(str)
    if isinstance(texts, (list, tuple)):
        return pd.Series(list(texts)).fillna("").astype(str)
    if isinstance(texts, str):
        return pd.Series([texts])
    if isinstance(texts, Iterable):
        return pd.Series(list(texts)).fillna("").astype(str)
    return pd.Series([str(texts)])


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    return normalize_text(text)


def _stats_frame(texts: object) -> pd.DataFrame:
    series = _coerce_text_series(texts)
    normalized = series.map(normalize_text)
    words = normalized.str.split()
    word_counts = words.map(len)
    unique_word_counts = words.map(lambda tokens: len(set(tokens)))
    char_counts = normalized.str.len()
    exclamation_counts = series.str.count(r"!")
    question_counts = series.str.count(r"\?")
    comma_counts = series.str.count(r",")
    semicolon_counts = series.str.count(r";")
    uppercase_counts = series.str.count(r"[A-Z]")
    avg_word_length = words.map(
        lambda tokens: (sum(len(token) for token in tokens) / len(tokens)) if tokens else 0.0
    )
    denominator = series.str.len().replace(0, 1).astype(float)

    return pd.DataFrame(
        {
            "char_count": char_counts.astype(float),
            "word_count": word_counts.astype(float),
            "unique_word_count": unique_word_counts.astype(float),
            "avg_word_length": avg_word_length.astype(float),
            "exclamation_count": exclamation_counts.astype(float),
            "question_count": question_counts.astype(float),
            "comma_count": comma_counts.astype(float),
            "semicolon_count": semicolon_counts.astype(float),
            "uppercase_ratio": uppercase_counts.astype(float) / denominator,
        }
    )


def count_features(texts: object) -> pd.DataFrame:
    return _stats_frame(texts)


def extract_stylistic_features(texts: object) -> pd.DataFrame:
    return _stats_frame(texts)


def char_features(
    texts: object,
    max_features: int = 4000,
    ngram_range: tuple[int, int] = (3, 5),
) -> sparse.csr_matrix:
    series = _coerce_text_series(texts).map(normalize_text)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    return vectorizer.fit_transform(series)


@dataclass
class FeatureConfig:
    word_max_features: int = 3000
    char_max_features: int = 0
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 4)
    include_stats: bool = True
    lowercase: bool = True


@dataclass
class TextStats:
    def fit(self, X: object, y: object | None = None) -> "TextStats":
        return self

    def transform(self, X: object) -> pd.DataFrame:
        return _stats_frame(X)

    def fit_transform(self, X: object, y: object | None = None) -> pd.DataFrame:
        return self.transform(X)


class TextFeatures:
    """Editable combined word/char/stat feature builder."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.config.word_ngram_range,
            max_features=self.config.word_max_features,
            lowercase=self.config.lowercase,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.config.char_ngram_range,
            max_features=self.config.char_max_features,
            lowercase=self.config.lowercase,
            sublinear_tf=True,
        )

    def fit(self, X: object, y: object | None = None) -> "TextFeatures":
        series = _coerce_text_series(X).map(normalize_text)
        self.word_vectorizer.fit(series)
        if self.config.char_max_features > 0:
            self.char_vectorizer.fit(series)
        return self

    def transform(self, X: object) -> sparse.csr_matrix:
        series = _coerce_text_series(X).map(normalize_text)
        parts: list[sparse.csr_matrix] = [
            self.word_vectorizer.transform(series),
        ]
        if self.config.char_max_features > 0:
            parts.append(self.char_vectorizer.transform(series))
        if self.config.include_stats:
            parts.append(sparse.csr_matrix(_stats_frame(series).to_numpy(dtype=float)))
        return sparse.hstack(parts).tocsr()

    def fit_transform(self, X: object, y: object | None = None) -> sparse.csr_matrix:
        self.fit(X, y=y)
        return self.transform(X)


def build_feature_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, TextFeatures]:
    builder = TextFeatures(config=config)
    train_matrix = builder.fit_transform(train_df["text"])
    eval_matrix = builder.transform(eval_df["text"])
    return train_matrix, eval_matrix, builder
