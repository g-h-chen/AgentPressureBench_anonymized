"""Submission helpers for the Denoising Dirty Documents workspace."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "id"
TARGET_COLUMN = "value"


def build_submission_frame(sample_submission: pd.DataFrame, denoised_images: dict[str, object]) -> pd.DataFrame:
    split_ids = sample_submission[ID_COLUMN].astype(str).str.rsplit("_", n=2, expand=True)
    image_ids = split_ids[0].astype(str)
    row_idx = pd.to_numeric(split_ids[1], errors="raise").astype(int) - 1
    col_idx = pd.to_numeric(split_ids[2], errors="raise").astype(int) - 1

    values = []
    for image_id, row, col in zip(image_ids, row_idx, col_idx):
        image = denoised_images[image_id]
        values.append(float(image[row, col]))

    submission = sample_submission.loc[:, [ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = values
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]
