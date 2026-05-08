"""Thin orchestration entrypoint for the Learning Agency essay-scoring workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.features import FeatureConfig, build_feature_matrices
from src.models import BaselineModelConfig, fit_predict_scores
from src.postprocess import build_submission_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    sample_submission = pd.read_csv(args.sample_submission)

    feature_config = FeatureConfig(word_max_features=12000, word_ngram_range=(1, 2))
    train_matrix, eval_matrix, _ = build_feature_matrices(
        train_df=train_df,
        eval_df=eval_df,
        config=feature_config,
    )
    scores = fit_predict_scores(
        labels=train_df["score"],
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        config=BaselineModelConfig(),
    )
    build_submission_frame(sample_submission, scores).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
