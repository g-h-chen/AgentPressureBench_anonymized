"""Editable image helpers for the Denoising Dirty Documents workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_grayscale_image(path_text: str) -> np.ndarray:
    image = Image.open(resolve_path(path_text)).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def build_eval_image_index(eval_df: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["image_id"]): str(row["dirty_path"])
        for _, row in eval_df[["image_id", "dirty_path"]].drop_duplicates().iterrows()
    }
