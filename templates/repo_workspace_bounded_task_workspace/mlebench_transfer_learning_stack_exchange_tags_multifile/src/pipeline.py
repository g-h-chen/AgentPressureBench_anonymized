"""Thin orchestration entrypoint for the Stack Exchange tags workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.models import BaselineModelConfig, fit_predict_tags
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
    predictions = fit_predict_tags(
        train_df=train_df,
        eval_df=eval_df,
        config=BaselineModelConfig(),
    )
    build_submission_frame(sample_submission, predictions).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
