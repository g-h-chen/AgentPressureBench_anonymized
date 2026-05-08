"""Face-image feature helpers for the COFW workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class FeatureConfig:
    image_side: int = 48
    pixel_stride: int = 2


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    image_path = image_dir / str(image_id)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image asset: {image_path}")
    return image_path


def load_face_crop(
    image_dir: Path,
    row: pd.Series | object,
    *,
    image_side: int,
) -> np.ndarray:
    image_path = resolve_image_path(image_dir, getattr(row, "image_id"))
    image = Image.open(image_path).convert("L")
    width, height = image.size

    left = max(0.0, float(getattr(row, "bbox_left")))
    top = max(0.0, float(getattr(row, "bbox_top")))
    bbox_width = max(1.0, float(getattr(row, "bbox_width")))
    bbox_height = max(1.0, float(getattr(row, "bbox_height")))
    right = min(width, left + bbox_width)
    bottom = min(height, top + bbox_height)
    if right <= left or bottom <= top:
        crop = image
    else:
        crop = image.crop((left, top, right, bottom))

    resized = crop.resize((image_side, image_side))
    return np.asarray(resized, dtype=np.float32) / 255.0


def build_feature_matrix(frame: pd.DataFrame, image_dir: Path, config: FeatureConfig | None = None) -> np.ndarray:
    config = config or FeatureConfig()
    rows: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        image = load_face_crop(image_dir, row, image_side=config.image_side)
        sampled = image[:: config.pixel_stride, :: config.pixel_stride].reshape(-1)
        horizontal_profile = image.mean(axis=0)[:: max(1, config.pixel_stride * 2)]
        vertical_profile = image.mean(axis=1)[:: max(1, config.pixel_stride * 2)]
        stats = np.array(
            [
                float(image.mean()),
                float(image.std()),
                float(image.min()),
                float(image.max()),
            ],
            dtype=float,
        )
        rows.append(np.concatenate([sampled, horizontal_profile, vertical_profile, stats], axis=0))
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows).astype(float)
