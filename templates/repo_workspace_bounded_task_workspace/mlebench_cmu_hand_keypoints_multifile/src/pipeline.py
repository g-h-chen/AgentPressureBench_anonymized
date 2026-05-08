"""Thin orchestration entrypoint for the CMU hand-keypoints workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import ModelConfig, fit_model, predict_landmarks
from src.postprocess import build_submission_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--eval-image-dir", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    sample_submission = pd.read_csv(args.sample_submission)

    model = fit_model(
        train_df=train_df,
        train_image_dir=Path(args.train_image_dir),
        config=ModelConfig(),
    )
    predictions = predict_landmarks(
        model=model,
        eval_df=eval_df,
        eval_image_dir=Path(args.eval_image_dir),
    )
    build_submission_frame(sample_submission, predictions).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
