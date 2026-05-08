"""Thin orchestration entrypoint for the Data Science Bowl 2018 workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import SegmentationConfig, fit_model, predict_masks
from src.postprocess import build_submission_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    sample_submission = pd.read_csv(args.sample_submission)

    model = fit_model(train_df, train_root=Path(args.train_root), config=SegmentationConfig())
    predicted_masks = predict_masks(model, eval_df, eval_root=Path(args.eval_root))
    build_submission_frame(sample_submission, predicted_masks).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
