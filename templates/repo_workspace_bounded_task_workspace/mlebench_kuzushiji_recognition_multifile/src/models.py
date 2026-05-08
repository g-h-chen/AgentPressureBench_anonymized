"""Editable retrieval/localization helpers for the Kuzushiji workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.image_features import load_page_features


@dataclass
class ModelConfig:
    feature_side: int = 48
    neighbors: int = 1


def _boxes_to_points(label_text: object) -> str:
    if label_text is None or pd.isna(label_text):
        return ""
    parts = str(label_text).split()
    if len(parts) % 5 != 0:
        return ""
    predictions: list[str] = []
    for index in range(0, len(parts), 5):
        label = parts[index]
        x = float(parts[index + 1])
        y = float(parts[index + 2])
        width = float(parts[index + 3])
        height = float(parts[index + 4])
        center_x = x + (width / 2.0)
        center_y = y + (height / 2.0)
        predictions.extend([label, f"{center_x:.2f}", f"{center_y:.2f}"])
    return " ".join(predictions)


def fit_model(train_df: pd.DataFrame, train_image_dir: Path, config: ModelConfig | None = None) -> dict[str, object]:
    config = config or ModelConfig()
    train_features = load_page_features(train_df["image_id"], train_image_dir, side=config.feature_side)
    train_targets = [_boxes_to_points(value) for value in train_df["labels"]]
    nn = NearestNeighbors(n_neighbors=config.neighbors, metric="cosine")
    nn.fit(train_features)
    return {
        "neighbors": nn,
        "train_targets": train_targets,
        "feature_side": config.feature_side,
    }


def predict_labels(model: dict[str, object], eval_df: pd.DataFrame, eval_image_dir: Path) -> list[str]:
    eval_features = load_page_features(eval_df["image_id"], eval_image_dir, side=int(model["feature_side"]))
    neighbors = model["neighbors"]
    _, indices = neighbors.kneighbors(eval_features)
    train_targets: list[str] = model["train_targets"]  # type: ignore[assignment]
    return [train_targets[int(row[0])] for row in indices]
