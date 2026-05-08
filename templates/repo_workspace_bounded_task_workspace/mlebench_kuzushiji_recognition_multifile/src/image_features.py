"""Editable image helpers for the Kuzushiji Recognition workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    image_path = image_dir / f"{image_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing page image: {image_path}")
    return image_path


def load_page_features(ids: pd.Series, image_dir: Path, side: int = 48) -> np.ndarray:
    rows: list[np.ndarray] = []
    for image_id in ids.astype(str):
        image = Image.open(resolve_image_path(image_dir, image_id)).convert("L").resize((side, side))
        array = np.asarray(image, dtype=np.float32) / 255.0
        profile_x = array.mean(axis=0)
        profile_y = array.mean(axis=1)
        stats = np.array(
            [float(array.mean()), float(array.std()), float(array.min()), float(array.max())],
            dtype=np.float32,
        )
        rows.append(np.concatenate([array.reshape(-1), profile_x, profile_y, stats], axis=0))
    if not rows:
        return np.zeros((0, side * side + side + side + 4), dtype=np.float32)
    return np.vstack(rows)
