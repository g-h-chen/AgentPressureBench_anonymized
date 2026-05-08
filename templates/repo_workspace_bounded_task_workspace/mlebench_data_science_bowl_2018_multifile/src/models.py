"""Editable threshold-based segmentation helpers for Data Science Bowl 2018."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.image_features import load_grayscale_image, load_union_mask, resolve_sample_dir


@dataclass
class SegmentationConfig:
    sample_stride: int = 8
    max_train_samples: int = 96


def fit_model(train_df: pd.DataFrame, train_root: Path, config: SegmentationConfig | None = None) -> dict[str, float | bool]:
    config = config or SegmentationConfig()
    foreground_values: list[np.ndarray] = []
    background_values: list[np.ndarray] = []
    mask_fractions: list[float] = []

    for image_id in train_df["ImageId"].astype(str).tolist()[: config.max_train_samples]:
        sample_dir = resolve_sample_dir(train_root, image_id)
        image = load_grayscale_image(sample_dir)[:: config.sample_stride, :: config.sample_stride]
        mask = load_union_mask(sample_dir)[:: config.sample_stride, :: config.sample_stride]
        if mask.size == 0:
            continue
        mask_fractions.append(float(mask.mean()))
        if mask.any():
            foreground_values.append(image[mask])
        if (~mask).any():
            background_values.append(image[~mask])

    foreground_mean = float(np.concatenate(foreground_values).mean()) if foreground_values else 0.35
    background_mean = float(np.concatenate(background_values).mean()) if background_values else 0.65
    foreground_is_dark = foreground_mean < background_mean
    threshold = float((foreground_mean + background_mean) / 2.0)
    target_fraction = float(np.clip(np.mean(mask_fractions) if mask_fractions else 0.12, 0.01, 0.6))
    return {
        "threshold": threshold,
        "foreground_is_dark": foreground_is_dark,
        "target_fraction": target_fraction,
    }


def _fallback_fraction_mask(image: np.ndarray, *, target_fraction: float, foreground_is_dark: bool) -> np.ndarray:
    quantile = target_fraction if foreground_is_dark else (1.0 - target_fraction)
    cutoff = float(np.quantile(image, quantile))
    if foreground_is_dark:
        return image <= cutoff
    return image >= cutoff


def predict_masks(model: dict[str, float | bool], eval_df: pd.DataFrame, eval_root: Path) -> dict[str, np.ndarray]:
    threshold = float(model["threshold"])
    foreground_is_dark = bool(model["foreground_is_dark"])
    target_fraction = float(model["target_fraction"])

    predicted: dict[str, np.ndarray] = {}
    for image_id in eval_df["ImageId"].astype(str):
        sample_dir = resolve_sample_dir(eval_root, image_id)
        image = load_grayscale_image(sample_dir)
        if foreground_is_dark:
            mask = image <= threshold
        else:
            mask = image >= threshold

        foreground_fraction = float(mask.mean())
        if foreground_fraction < 0.002 or foreground_fraction > 0.95:
            mask = _fallback_fraction_mask(
                image,
                target_fraction=target_fraction,
                foreground_is_dark=foreground_is_dark,
            )
        predicted[image_id] = mask.astype(bool)
    return predicted
