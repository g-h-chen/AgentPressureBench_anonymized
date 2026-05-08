"""Thin orchestration entrypoint for the Denoising Dirty Documents workspace."""

from __future__ import annotations

import argparse

import pandas as pd

from src.image_features import build_eval_image_index
from src.models import DenoiserConfig, fit_global_linear_denoiser, render_denoised_images
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

    denoiser = fit_global_linear_denoiser(train_df, config=DenoiserConfig())
    eval_image_index = build_eval_image_index(eval_df)
    denoised_images = render_denoised_images(eval_image_index, denoiser)
    build_submission_frame(submission, denoised_images).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
