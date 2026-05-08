"""Thin orchestration entrypoint for the English text-normalization workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.features import normalize_frame
from src.models import fit_token_mapping, predict_after
from src.postprocess import build_submission_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_df = normalize_frame(pd.read_csv(args.train))
    eval_df = normalize_frame(pd.read_csv(args.eval))
    sample_submission = pd.read_csv(args.sample_submission)

    token_mapping = fit_token_mapping(train_df)
    predictions = predict_after(eval_df, token_mapping)
    build_submission_frame(sample_submission, eval_df["id"], predictions).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
