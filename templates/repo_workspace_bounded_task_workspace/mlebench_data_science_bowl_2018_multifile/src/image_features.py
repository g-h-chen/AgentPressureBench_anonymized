"""Editable image helpers for the Data Science Bowl 2018 workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def resolve_sample_dir(root: Path, image_id: str) -> Path:
    sample_dir = root / str(image_id)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")
    return sample_dir


def load_grayscale_image(sample_dir: Path) -> np.ndarray:
    image_path = next((sample_dir / "images").glob("*"))
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_union_mask(sample_dir: Path) -> np.ndarray:
    mask_dir = sample_dir / "masks"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    image = load_grayscale_image(sample_dir)
    union_mask = np.zeros(image.shape, dtype=bool)
    for mask_path in sorted(mask_dir.glob("*")):
        mask_image = Image.open(mask_path)
        mask_array = np.asarray(mask_image)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        union_mask |= mask_array > 0
    return union_mask
