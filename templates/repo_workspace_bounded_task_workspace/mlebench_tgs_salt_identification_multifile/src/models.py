"""Editable exemplar-based segmentation helpers for TGS salt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.image_features import load_mask, load_signal_map


@dataclass
class SegmentationConfig:
    feature_size: int = 16
    max_train_samples: int = 1024
    k_neighbors: int = 5
    depth_weight: float = 0.10


def _has_salt_mask(rle_text: object) -> bool:
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


def _select_training_rows(train_df: pd.DataFrame, max_train_samples: int) -> pd.DataFrame:
    if len(train_df) <= max_train_samples:
        return train_df.sort_values("id").reset_index(drop=True)

    frame = train_df.copy()
    has_mask = frame["rle_mask"].map(_has_salt_mask)
    non_empty = frame.loc[has_mask].sort_values("id").reset_index(drop=True)
    empty = frame.loc[~has_mask].sort_values("id").reset_index(drop=True)

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


def _coerce_depth(row: pd.Series | object) -> float:
    value = getattr(row, "z", np.nan)
    if pd.isna(value):
        return np.nan
    return float(value)


def fit_model(
    train_df: pd.DataFrame,
    train_image_dir: Path,
    train_mask_dir: Path,
    config: SegmentationConfig | None = None,
) -> dict[str, object]:
    config = config or SegmentationConfig()
    sampled_df = _select_training_rows(train_df, config.max_train_samples)

    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    mask_fractions: list[float] = []
    depths: list[float] = []

    for row in sampled_df.itertuples(index=False):
        image_id = str(getattr(row, "id"))
        features.append(load_signal_map(train_image_dir, image_id, size=config.feature_size).reshape(-1))
        mask = load_mask(train_mask_dir, image_id).astype(bool)
        masks.append(mask)
        mask_fractions.append(float(mask.mean()))
        depths.append(_coerce_depth(row))

    if not features:
        raise ValueError("No training samples were available for TGS salt segmentation.")

    feature_matrix = np.stack(features).astype(np.float32)
    mask_stack = np.stack(masks).astype(np.float32)
    mask_fraction_array = np.asarray(mask_fractions, dtype=np.float32)
    depth_values = np.asarray(depths, dtype=np.float32)
    depth_scale = float(np.nanstd(depth_values)) if np.isfinite(depth_values).any() else 1.0
    if not np.isfinite(depth_scale) or depth_scale < 1e-6:
        depth_scale = 1.0

    return {
        "config": config,
        "feature_matrix": feature_matrix,
        "mask_stack": mask_stack,
        "mask_fractions": mask_fraction_array,
        "depth_values": depth_values,
        "depth_scale": depth_scale,
    }


def _combined_distances(
    feature_matrix: np.ndarray,
    eval_feature: np.ndarray,
    train_depths: np.ndarray,
    eval_depth: float,
    *,
    depth_scale: float,
    depth_weight: float,
) -> np.ndarray:
    deltas = feature_matrix - eval_feature[None, :]
    distances = np.mean(deltas * deltas, axis=1)
    if np.isfinite(eval_depth):
        valid_depth = np.isfinite(train_depths)
        if valid_depth.any():
            depth_penalty = np.zeros_like(distances)
            depth_penalty[valid_depth] = ((train_depths[valid_depth] - eval_depth) / depth_scale) ** 2
            distances = distances + (depth_weight * depth_penalty)
    return distances


def predict_masks(model: dict[str, object], eval_df: pd.DataFrame, eval_image_dir: Path) -> dict[str, np.ndarray]:
    config = model["config"]
    assert isinstance(config, SegmentationConfig)
    feature_matrix = np.asarray(model["feature_matrix"], dtype=np.float32)
    mask_stack = np.asarray(model["mask_stack"], dtype=np.float32)
    mask_fractions = np.asarray(model["mask_fractions"], dtype=np.float32)
    depth_values = np.asarray(model["depth_values"], dtype=np.float32)
    depth_scale = float(model["depth_scale"])

    predicted: dict[str, np.ndarray] = {}
    for row in eval_df.itertuples(index=False):
        image_id = str(getattr(row, "id"))
        eval_feature = load_signal_map(eval_image_dir, image_id, size=config.feature_size).reshape(-1)
        eval_depth = _coerce_depth(row)
        distances = _combined_distances(
            feature_matrix,
            eval_feature.astype(np.float32),
            depth_values,
            eval_depth,
            depth_scale=depth_scale,
            depth_weight=config.depth_weight,
        )
        k = min(config.k_neighbors, len(distances))
        neighbor_idx = np.argpartition(distances, k - 1)[:k]
        neighbor_idx = neighbor_idx[np.argsort(distances[neighbor_idx])]
        weights = 1.0 / (distances[neighbor_idx] + 1e-6)
        weighted_masks = np.tensordot(weights, mask_stack[neighbor_idx], axes=(0, 0)) / float(weights.sum())
        target_fraction = float(np.average(mask_fractions[neighbor_idx], weights=weights))
        if target_fraction <= 0.001:
            mask = np.zeros(mask_stack.shape[1:], dtype=bool)
        else:
            quantile = float(np.clip(1.0 - target_fraction, 0.0, 0.995))
            cutoff = float(np.quantile(weighted_masks, quantile))
            mask = weighted_masks >= cutoff
        predicted[image_id] = mask.astype(bool)
    return predicted
