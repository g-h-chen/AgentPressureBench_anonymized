"""Editable hand-keypoint prediction helpers for CMU."""

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


def _keypoint_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("kp_")]


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
    keypoint_columns = _keypoint_columns(train_df)
    x_train = build_feature_matrix(train_df, image_dir=train_image_dir, config=feature_config)
    normalized_shapes = _normalized_shapes(train_df, keypoint_columns)

    train_groups = pd.to_numeric(train_df.get("is_left", 0), errors="coerce").fillna(0).astype(int).to_numpy()

    group_models: dict[int, dict[str, object]] = {}
    for group in sorted(set(train_groups.tolist())):
        group_indices = np.flatnonzero(train_groups == group)
        if group_indices.size == 0:
            continue
        neighbor_count = max(1, min(config.neighbors, int(group_indices.size)))
        nearest_neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        nearest_neighbors.fit(x_train[group_indices])
        group_models[int(group)] = {
            "nearest_neighbors": nearest_neighbors,
            "train_indices": group_indices,
        }

    global_neighbor_count = max(1, min(config.neighbors, len(train_df)))
    global_neighbors = NearestNeighbors(n_neighbors=global_neighbor_count, metric="cosine")
    global_neighbors.fit(x_train)
    return {
        "feature_config": feature_config,
        "group_models": group_models,
        "global_neighbors": global_neighbors,
        "normalized_shapes": normalized_shapes,
        "keypoint_columns": keypoint_columns,
    }


def predict_landmarks(
    model: dict[str, object],
    eval_df: pd.DataFrame,
    eval_image_dir: Path,
) -> pd.DataFrame:
    feature_config = model["feature_config"]
    group_models = model["group_models"]
    global_neighbors = model["global_neighbors"]
    normalized_shapes = np.asarray(model["normalized_shapes"], dtype=float)
    keypoint_columns = list(model["keypoint_columns"])
    x_eval = build_feature_matrix(eval_df, image_dir=eval_image_dir, config=feature_config)

    predicted_rows: list[dict[str, float | str]] = []
    for row_index, row in enumerate(eval_df.itertuples(index=False)):
        hand_group = int(getattr(row, "is_left", 0))
        group_bundle = group_models.get(hand_group)
        if group_bundle is None:
            neighbor_distances, neighbor_indices = global_neighbors.kneighbors(x_eval[row_index : row_index + 1])
            train_indices = neighbor_indices[0]
        else:
            neighbor_distances, neighbor_indices = group_bundle["nearest_neighbors"].kneighbors(
                x_eval[row_index : row_index + 1]
            )
            train_indices = np.asarray(group_bundle["train_indices"], dtype=int)[neighbor_indices[0]]

        neighbor_shapes = normalized_shapes[train_indices]
        neighbor_distances = neighbor_distances[0]
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
