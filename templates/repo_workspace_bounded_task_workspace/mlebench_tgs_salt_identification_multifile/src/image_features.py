"""Editable image helpers for the TGS salt workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    image_text = str(image_id)
    candidates = [
        image_dir / image_text,
        image_dir / f"{image_text}.png",
        image_dir / f"{image_text}.jpg",
        image_dir / f"{image_text}.jpeg",
    ]
    for image_path in candidates:
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Missing image asset for {image_id}: tried {candidates}")


def load_grayscale_image(image_dir: Path, image_id: str, *, size: int | None = None) -> np.ndarray:
    image_path = resolve_image_path(image_dir, image_id)
    image = Image.open(image_path).convert("L")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_signal_map(image_dir: Path, image_id: str, *, size: int | None = None) -> np.ndarray:
    image = load_grayscale_image(image_dir, image_id, size=size)
    centered = image - float(image.mean())
    scale = float(centered.std())
    if scale > 1e-6:
        centered = centered / scale
    return centered.astype(np.float32)


def load_mask(image_dir: Path, image_id: str) -> np.ndarray:
    image_path = resolve_image_path(image_dir, image_id)
    image = Image.open(image_path).convert("L")
    mask = np.asarray(image, dtype=np.uint8)
    return mask >= 128
