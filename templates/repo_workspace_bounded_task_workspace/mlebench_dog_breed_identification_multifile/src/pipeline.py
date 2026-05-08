"""Thin orchestration entrypoint for the Dog Breed multifile workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.image_features import infer_image_dir
from src.models import ModelConfig, fit_model, predict_proba
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
    submission = pd.read_csv(args.sample_submission)

    train_image_dir = infer_image_dir(args.train, split="train")
    eval_image_dir = infer_image_dir(args.eval, split="eval")
    model = fit_model(train_df, train_image_dir, config=ModelConfig())
    probabilities, model_classes = predict_proba(model, eval_df, eval_image_dir)
    build_submission_frame(submission, probabilities, model_classes).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
