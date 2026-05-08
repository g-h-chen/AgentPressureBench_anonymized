"""Editable denoising-model helpers for the Denoising Dirty Documents workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.image_features import load_grayscale_image


@dataclass
class DenoiserConfig:
    sample_stride: int = 16


def fit_global_linear_denoiser(train_df: pd.DataFrame, config: DenoiserConfig | None = None) -> tuple[float, float]:
    config = config or DenoiserConfig()
    sampled_dirty: list[np.ndarray] = []
    sampled_clean: list[np.ndarray] = []
    for _, row in train_df.iterrows():
        dirty = load_grayscale_image(str(row["dirty_path"]))
        clean = load_grayscale_image(str(row["clean_path"]))
        sampled_dirty.append(dirty[:: config.sample_stride, :: config.sample_stride].reshape(-1))
        sampled_clean.append(clean[:: config.sample_stride, :: config.sample_stride].reshape(-1))

    if not sampled_dirty:
        return 1.0, 0.0

    x = np.concatenate(sampled_dirty)
    y = np.concatenate(sampled_clean)
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = coeffs.tolist()
    return float(slope), float(intercept)


def apply_linear_denoiser(image_path: str, denoiser: tuple[float, float]) -> np.ndarray:
    dirty = load_grayscale_image(image_path)
    slope, intercept = denoiser
    return np.clip(dirty * slope + intercept, 0.0, 1.0)


def render_denoised_images(eval_image_index: dict[str, str], denoiser: tuple[float, float]) -> dict[str, np.ndarray]:
    return {
        image_id: apply_linear_denoiser(image_path, denoiser)
        for image_id, image_path in eval_image_index.items()
    }
