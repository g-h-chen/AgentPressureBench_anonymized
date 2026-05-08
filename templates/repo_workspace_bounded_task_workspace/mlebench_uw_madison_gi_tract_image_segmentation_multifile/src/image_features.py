"""Editable image helpers for the UW-Madison GI tract workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("large_bowel", "small_bowel", "stomach")


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    image_path = image_dir / f"{image_id}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image asset: {image_path}")
    return image_path


def image_size(image_dir: Path, image_id: str) -> tuple[int, int]:
    image_path = resolve_image_path(image_dir, image_id)
    with Image.open(image_path) as image:
        return image.size


def load_signal_map(image_dir: Path, image_id: str, *, size: int | None = None) -> np.ndarray:
    image_path = resolve_image_path(image_dir, image_id)
    image = Image.open(image_path)
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, [1.0, 99.0])
    if high > low:
        array = (array - low) / (high - low)
    elif float(array.max()) > 0:
        array = array / float(array.max())
    array = np.clip(array, 0.0, 1.0)
    array = array - float(array.mean())
    scale = float(array.std())
    if scale > 1e-6:
        array = array / scale
    return array.astype(np.float32)


def decode_rle_mask(rle_text: object, *, height: int, width: int) -> np.ndarray:
    encoded = "" if rle_text is None else str(rle_text).strip()
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    if encoded == "" or encoded.lower() == "nan":
        return flat_mask.reshape((width, height)).T.astype(bool)

    values = [int(part) for part in encoded.split()]
    starts = values[0::2]
    lengths = values[1::2]
    for start, length in zip(starts, lengths):
        flat_mask[start - 1 : start - 1 + length] = 1
    return flat_mask.reshape((width, height)).T.astype(bool)


def resize_binary_mask(mask: np.ndarray, *, size: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= 128


def resize_probability_map(probability_map: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(probability_map * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0
