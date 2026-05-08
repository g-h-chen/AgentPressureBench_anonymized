"""Thin orchestration entrypoint for the Random Acts of Pizza multifile workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.features import FeatureConfig, build_feature_matrices
from src.models import BaselineModelConfig, fit_predict_proba
from src.postprocess import build_submission_frame


def _merge_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    merged = df.copy()
    merged["text"] = (
        merged["request_title"].fillna("").astype(str).str.strip()
        + "\n"
        + merged["request_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_df = _merge_text_columns(pd.read_csv(args.train))
    eval_df = _merge_text_columns(pd.read_csv(args.eval))
    sample_submission = pd.read_csv(args.sample_submission)

    feature_config = FeatureConfig(
        word_max_features=4000,
        char_max_features=2000,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        include_stats=True,
    )
    train_matrix, eval_matrix, _ = build_feature_matrices(
        train_df=train_df,
        eval_df=eval_df,
        config=feature_config,
    )
    probabilities = fit_predict_proba(
        labels=train_df["requester_received_pizza"],
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        config=BaselineModelConfig(),
    )

    build_submission_frame(sample_submission, probabilities).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
