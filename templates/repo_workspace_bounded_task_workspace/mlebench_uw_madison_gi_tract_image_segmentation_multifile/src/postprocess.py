"""Submission helpers for the UW-Madison GI tract workspace."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "id"
CLASS_COLUMN = "class"
TARGET_COLUMN = "predicted"


def _encode_binary_mask_rle(mask) -> str:
    import numpy as np

    flat_mask = np.asarray(mask, dtype=np.uint8).T.reshape(-1)
    if flat_mask.size == 0:
        return ""
    padded = np.concatenate(([0], flat_mask, [0]))
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(value)) for value in runs)


def build_submission_frame(sample_submission: pd.DataFrame, predicted_masks: dict[tuple[str, str], object]) -> pd.DataFrame:
    submission = sample_submission.loc[:, [ID_COLUMN, CLASS_COLUMN]].copy()
    submission[TARGET_COLUMN] = [
        _encode_binary_mask_rle(predicted_masks[(str(image_id), str(class_name))])
        for image_id, class_name in zip(
            submission[ID_COLUMN].astype(str),
            submission[CLASS_COLUMN].astype(str),
        )
    ]
    return submission.loc[:, [ID_COLUMN, CLASS_COLUMN, TARGET_COLUMN]]
