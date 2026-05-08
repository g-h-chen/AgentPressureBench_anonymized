"""Thin orchestration entrypoint for the MLSP 2013 Birds workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.features import feature_columns, label_columns
from src.models import ModelConfig, fit_predict_proba
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

    probabilities = fit_predict_proba(
        train_features=train_df[feature_columns(train_df)],
        train_labels=train_df[label_columns(train_df)],
        eval_features=eval_df[feature_columns(eval_df)],
        config=ModelConfig(),
    )
    build_submission_frame(sample_submission, eval_df["rec_id"], probabilities).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
