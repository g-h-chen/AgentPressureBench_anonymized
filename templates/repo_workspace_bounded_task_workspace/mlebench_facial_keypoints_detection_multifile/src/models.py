"""Editable keypoint-prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.features import FeatureConfig, build_feature_matrix


@dataclass
class ModelConfig:
    alpha: float = 1.0
    min_examples: int = 25


def fit_predict_locations(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    config: ModelConfig | None = None,
) -> np.ndarray:
    config = config or ModelConfig()
    feature_config = FeatureConfig()
    x_train = build_feature_matrix(train_df["Image"], config=feature_config)
    x_eval = build_feature_matrix(eval_df["Image"], config=feature_config)

    keypoint_columns = [column for column in train_df.columns if column not in {"ImageId", "Image"}]
    eval_ids = pd.to_numeric(eval_df["ImageId"], errors="raise").astype(int).tolist()
    eval_index = {image_id: index for index, image_id in enumerate(eval_ids)}

    prediction_cache: dict[str, np.ndarray] = {}
    for feature_name in lookup_df["FeatureName"].drop_duplicates().tolist():
        if feature_name not in keypoint_columns:
            prediction_cache[feature_name] = np.full(len(eval_df), 48.0, dtype=float)
            continue
        mask = train_df[feature_name].notna()
        target = pd.to_numeric(train_df.loc[mask, feature_name], errors="coerce")
        if int(mask.sum()) < config.min_examples or target.nunique(dropna=True) <= 1:
            mean_value = float(target.mean()) if int(mask.sum()) > 0 else 48.0
            prediction_cache[feature_name] = np.full(len(eval_df), mean_value, dtype=float)
            continue
        model = Ridge(alpha=config.alpha)
        model.fit(x_train[mask.to_numpy()], target.astype(float))
        prediction_cache[feature_name] = np.asarray(model.predict(x_eval), dtype=float)

    predictions: list[float] = []
    for _, row in lookup_df.iterrows():
        feature_name = str(row["FeatureName"])
        image_id = int(row["ImageId"])
        predictions.append(float(prediction_cache[feature_name][eval_index[image_id]]))
    return np.asarray(predictions, dtype=float)
