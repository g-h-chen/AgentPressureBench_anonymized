"""Thin orchestration entrypoint for the Petfinder Pawpularity multifile workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.features import build_feature_bundle
from src.models import ModelConfig, fit_model, predict_targets
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

    feature_bundle = build_feature_bundle(train_df=train_df, eval_df=eval_df)
    model = fit_model(feature_bundle.train_matrix, train_df, config=ModelConfig())
    predictions = predict_targets(model, feature_bundle.eval_matrix)

    build_submission_frame(sample_submission, predictions).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
