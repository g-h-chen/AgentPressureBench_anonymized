"""Editable landmark-prediction helpers for COFW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.features import FeatureConfig, build_feature_matrix


@dataclass
class ModelConfig:
    neighbors: int = 3
    image_side: int = 48
    pixel_stride: int = 2


def _landmark_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("lm_")]


def _normalized_shapes(train_df: pd.DataFrame, keypoint_columns: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for row in train_df.itertuples(index=False):
        bbox_left = float(getattr(row, "bbox_left"))
        bbox_top = float(getattr(row, "bbox_top"))
        bbox_width = max(1e-6, float(getattr(row, "bbox_width")))
        bbox_height = max(1e-6, float(getattr(row, "bbox_height")))

        normalized: list[float] = []
        for index in range(0, len(keypoint_columns), 2):
            x_column = keypoint_columns[index]
            y_column = keypoint_columns[index + 1]
            normalized.append((float(getattr(row, x_column)) - bbox_left) / bbox_width)
            normalized.append((float(getattr(row, y_column)) - bbox_top) / bbox_height)
        rows.append(normalized)
    return np.asarray(rows, dtype=float)


def fit_model(
    train_df: pd.DataFrame,
    train_image_dir: Path,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    config = config or ModelConfig()
    feature_config = FeatureConfig(image_side=config.image_side, pixel_stride=config.pixel_stride)
    keypoint_columns = _landmark_columns(train_df)
    x_train = build_feature_matrix(train_df, image_dir=train_image_dir, config=feature_config)
    normalized_shapes = _normalized_shapes(train_df, keypoint_columns)

    neighbor_count = max(1, min(config.neighbors, len(train_df)))
    nearest_neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
    nearest_neighbors.fit(x_train)
    return {
        "feature_config": feature_config,
        "nearest_neighbors": nearest_neighbors,
        "normalized_shapes": normalized_shapes,
        "keypoint_columns": keypoint_columns,
    }


def predict_landmarks(
    model: dict[str, object],
    eval_df: pd.DataFrame,
    eval_image_dir: Path,
) -> pd.DataFrame:
    feature_config = model["feature_config"]
    nearest_neighbors = model["nearest_neighbors"]
    normalized_shapes = np.asarray(model["normalized_shapes"], dtype=float)
    keypoint_columns = list(model["keypoint_columns"])
    x_eval = build_feature_matrix(eval_df, image_dir=eval_image_dir, config=feature_config)
    distances, indices = nearest_neighbors.kneighbors(x_eval)

    predicted_rows: list[dict[str, float | str]] = []
    for row_index, row in enumerate(eval_df.itertuples(index=False)):
        neighbor_shapes = normalized_shapes[indices[row_index]]
        neighbor_distances = distances[row_index]
        weights = 1.0 / np.maximum(neighbor_distances, 1e-6)
        if not np.isfinite(weights).all() or float(weights.sum()) == 0.0:
            weights = np.ones_like(neighbor_distances)
        normalized_prediction = np.average(neighbor_shapes, axis=0, weights=weights)

        bbox_left = float(getattr(row, "bbox_left"))
        bbox_top = float(getattr(row, "bbox_top"))
        bbox_width = float(getattr(row, "bbox_width"))
        bbox_height = float(getattr(row, "bbox_height"))

        predicted_row: dict[str, float | str] = {"image_id": str(getattr(row, "image_id"))}
        for index in range(0, len(keypoint_columns), 2):
            x_column = keypoint_columns[index]
            y_column = keypoint_columns[index + 1]
            normalized_x = float(normalized_prediction[index])
            normalized_y = float(normalized_prediction[index + 1])
            predicted_row[x_column] = bbox_left + (normalized_x * bbox_width)
            predicted_row[y_column] = bbox_top + (normalized_y * bbox_height)
        predicted_rows.append(predicted_row)

    return pd.DataFrame(predicted_rows).loc[:, ["image_id", *keypoint_columns]]
