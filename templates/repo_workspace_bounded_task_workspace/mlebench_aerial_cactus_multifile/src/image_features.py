"""Editable image helpers for the aerial-cactus multifile workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def infer_image_dir(csv_path: str, split: str) -> Path:
    path = Path(csv_path)
    if split == "train":
        candidates = [path.with_name("train_images")]
    else:
        candidates = [
            path.with_name("eval_images"),
            path.with_name("public_eval_images"),
            path.with_name("test_images"),
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image directory for {csv_path}")


def load_image_features(ids: pd.Series, image_dir: Path) -> np.ndarray:
    rows: list[np.ndarray] = []
    for image_id in ids.astype(str):
        image_path = image_dir / image_id
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3) / 255.0
        channel_means = pixels.mean(axis=0)
        channel_stds = pixels.std(axis=0)
        rows.append(np.concatenate([channel_means, channel_stds], axis=0))
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.vstack(rows)
