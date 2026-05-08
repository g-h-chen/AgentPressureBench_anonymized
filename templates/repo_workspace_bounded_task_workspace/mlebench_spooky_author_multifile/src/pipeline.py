"""Thin orchestration entrypoint for the multi-file spooky author workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.features import FeatureConfig, build_feature_matrices
from src.models import BaselineModelConfig, fit_predict_proba
from src.postprocess import CLASS_COLUMNS, LABEL_COLUMN, build_submission_frame


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

    feature_config = FeatureConfig(
        word_max_features=3000,
        char_max_features=0,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 4),
        include_stats=False,
    )
    train_matrix, eval_matrix, _ = build_feature_matrices(
        train_df=train_df,
        eval_df=eval_df,
        config=feature_config,
    )

    model_config = BaselineModelConfig(
        solver="lbfgs",
        max_iter=500,
        C=0.8,
        class_weight=None,
    )
    probabilities = fit_predict_proba(
        labels=train_df[LABEL_COLUMN],
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        class_order=CLASS_COLUMNS,
        config=model_config,
    )

    submission = build_submission_frame(
        sample_submission=sample_submission,
        probabilities=probabilities,
        class_order=CLASS_COLUMNS,
    )
    submission.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
