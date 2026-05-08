"""Pixel-string feature helpers for the facial keypoints workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    image_side: int = 96
    pixel_stride: int = 8


def _coerce_image_vector(value: object, image_side: int) -> np.ndarray:
    pixel_count = image_side * image_side
    text = "" if value is None else str(value)
    vector = np.fromstring(text, sep=" ", dtype=float)
    if vector.size == 0:
        return np.zeros(pixel_count, dtype=float)
    if vector.size < pixel_count:
        padded = np.zeros(pixel_count, dtype=float)
        padded[: vector.size] = vector
        vector = padded
    elif vector.size > pixel_count:
        vector = vector[:pixel_count]
    return vector


def build_feature_matrix(images: pd.Series, config: FeatureConfig | None = None) -> np.ndarray:
    config = config or FeatureConfig()
    rows: list[np.ndarray] = []
    for value in images.fillna(""):
        vector = _coerce_image_vector(value, image_side=config.image_side)
        image = vector.reshape(config.image_side, config.image_side)
        sampled = image[:: config.pixel_stride, :: config.pixel_stride].reshape(-1) / 255.0
        stats = np.array(
            [
                float(image.mean() / 255.0),
                float(image.std() / 255.0),
                float(image.min() / 255.0),
                float(image.max() / 255.0),
            ],
            dtype=float,
        )
        rows.append(np.concatenate([sampled, stats], axis=0))
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows).astype(float)
