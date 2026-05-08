"""Editable image helpers for the Kvasir-SEG workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    image_path = image_dir / str(image_id)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image asset: {image_path}")
    return image_path


def load_rgb_image(image_dir: Path, image_id: str) -> np.ndarray:
    image_path = resolve_image_path(image_dir, image_id)
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_signal_map(image_dir: Path, image_id: str) -> np.ndarray:
    rgb = load_rgb_image(image_dir, image_id)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    return (1.10 * red) - (0.70 * green) - (0.40 * blue)


def load_mask(image_dir: Path, image_id: str) -> np.ndarray:
    image_path = resolve_image_path(image_dir, image_id)
    image = Image.open(image_path).convert("L")
    mask = np.asarray(image, dtype=np.uint8)
    return mask >= 128
