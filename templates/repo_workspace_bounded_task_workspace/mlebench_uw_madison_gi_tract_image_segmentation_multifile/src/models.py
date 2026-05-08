"""Editable exemplar-based segmentation helpers for UW-Madison GI tract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.image_features import (
    CLASS_NAMES,
    decode_rle_mask,
    image_size,
    load_signal_map,
    resize_binary_mask,
    resize_probability_map,
)


@dataclass
class SegmentationConfig:
    feature_size: int = 10
    mask_size: int = 48
    max_train_samples: int = 512
    k_neighbors: int = 3


def _has_foreground_mask(rle_text: object) -> bool:
    if pd.isna(rle_text):
        return False
    return str(rle_text).strip() != ""


def _take_evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= count:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, count, dtype=int)
    return frame.iloc[np.unique(indices)].copy()


def _pivot_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        train_df.pivot(index="id", columns="class", values="segmentation")
        .reset_index()
        .rename_axis(columns=None)
    )
    for class_name in CLASS_NAMES:
        if class_name not in wide.columns:
            wide[class_name] = ""
    return wide.loc[:, ["id", *CLASS_NAMES]].copy()


def _select_training_rows(train_df: pd.DataFrame, max_train_samples: int) -> pd.DataFrame:
    wide = _pivot_training_frame(train_df)
    if len(wide) <= max_train_samples:
        return wide.sort_values("id").reset_index(drop=True)

    has_any_mask = wide.loc[:, list(CLASS_NAMES)].apply(lambda row: any(_has_foreground_mask(value) for value in row), axis=1)
    non_empty = wide.loc[has_any_mask].sort_values("id").reset_index(drop=True)
    empty = wide.loc[~has_any_mask].sort_values("id").reset_index(drop=True)

    target_non_empty = min(len(non_empty), max_train_samples // 2)
    target_empty = min(len(empty), max_train_samples - target_non_empty)
    remainder = max_train_samples - target_non_empty - target_empty
    if remainder > 0:
        extra_non_empty = min(max(0, len(non_empty) - target_non_empty), remainder)
        target_non_empty += extra_non_empty
        remainder -= extra_non_empty
    if remainder > 0:
        extra_empty = min(max(0, len(empty) - target_empty), remainder)
        target_empty += extra_empty

    sampled = pd.concat(
        [
            _take_evenly_spaced_rows(non_empty, target_non_empty),
            _take_evenly_spaced_rows(empty, target_empty),
        ],
        ignore_index=True,
    )
    return sampled.sort_values("id").reset_index(drop=True)


def fit_model(
    train_df: pd.DataFrame,
    train_image_dir: Path,
    config: SegmentationConfig | None = None,
) -> dict[str, object]:
    config = config or SegmentationConfig()
    sampled_df = _select_training_rows(train_df, config.max_train_samples)

    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    mask_fractions: list[np.ndarray] = []

    for row in sampled_df.itertuples(index=False):
        image_id = str(getattr(row, "id"))
        features.append(load_signal_map(train_image_dir, image_id, size=config.feature_size).reshape(-1))
        width, height = image_size(train_image_dir, image_id)
        class_masks: list[np.ndarray] = []
        class_fractions: list[float] = []
        for class_name in CLASS_NAMES:
            mask = decode_rle_mask(getattr(row, class_name), height=height, width=width)
            class_masks.append(resize_binary_mask(mask, size=config.mask_size).astype(np.float32))
            class_fractions.append(float(mask.mean()))
        masks.append(np.stack(class_masks, axis=0))
        mask_fractions.append(np.asarray(class_fractions, dtype=np.float32))

    if not features:
        raise ValueError("No training samples were available for UW-Madison GI tract segmentation.")

    feature_matrix = np.stack(features).astype(np.float32)
    mask_stack = np.stack(masks).astype(np.float32)
    mask_fraction_array = np.stack(mask_fractions).astype(np.float32)

    return {
        "config": config,
        "feature_matrix": feature_matrix,
        "mask_stack": mask_stack,
        "mask_fractions": mask_fraction_array,
    }


def _feature_distances(feature_matrix: np.ndarray, eval_feature: np.ndarray) -> np.ndarray:
    deltas = feature_matrix - eval_feature[None, :]
    return np.mean(deltas * deltas, axis=1)


def predict_masks(model: dict[str, object], eval_df: pd.DataFrame, eval_image_dir: Path) -> dict[str, np.ndarray]:
    config = model["config"]
    assert isinstance(config, SegmentationConfig)
    feature_matrix = np.asarray(model["feature_matrix"], dtype=np.float32)
    mask_stack = np.asarray(model["mask_stack"], dtype=np.float32)
    mask_fractions = np.asarray(model["mask_fractions"], dtype=np.float32)

    predicted: dict[tuple[str, str], np.ndarray] = {}
    unique_ids = eval_df["id"].astype(str).drop_duplicates().tolist()
    for image_id in unique_ids:
        eval_feature = load_signal_map(eval_image_dir, image_id, size=config.feature_size).reshape(-1)
        distances = _feature_distances(feature_matrix, eval_feature.astype(np.float32))
        k = min(config.k_neighbors, len(distances))
        neighbor_idx = np.argpartition(distances, k - 1)[:k]
        neighbor_idx = neighbor_idx[np.argsort(distances[neighbor_idx])]
        weights = 1.0 / (distances[neighbor_idx] + 1e-6)
        weighted_masks = np.tensordot(weights, mask_stack[neighbor_idx], axes=(0, 0)) / float(weights.sum())
        target_fractions = np.average(mask_fractions[neighbor_idx], axis=0, weights=weights)
        width, height = image_size(eval_image_dir, image_id)
        for class_idx, class_name in enumerate(CLASS_NAMES):
            target_fraction = float(target_fractions[class_idx])
            if target_fraction <= 0.001:
                mask = np.zeros((height, width), dtype=bool)
            else:
                probability_map = resize_probability_map(
                    weighted_masks[class_idx],
                    width=width,
                    height=height,
                )
                quantile = float(np.clip(1.0 - target_fraction, 0.0, 0.995))
                cutoff = float(np.quantile(probability_map, quantile))
                mask = probability_map >= cutoff
            predicted[(image_id, class_name)] = mask.astype(bool)
    return predicted
