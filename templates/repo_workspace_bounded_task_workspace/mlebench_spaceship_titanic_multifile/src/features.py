"""Editable feature helpers for the Spaceship Titanic multifile workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_COLUMN = "PassengerId"
LABEL_COLUMN = "Transported"
TEXT_COLUMNS = ["Cabin", "Name"]
CATEGORICAL_COLUMNS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
NUMERIC_COLUMNS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


@dataclass
class FeatureBundle:
    train_matrix: np.ndarray
    eval_matrix: np.ndarray
    feature_columns: list[str]


def _extract_cabin_parts(df: pd.DataFrame) -> pd.DataFrame:
    cabin = df["Cabin"].fillna("unknown/unknown/unknown").astype(str).str.split("/", expand=True)
    cabin.columns = ["CabinDeck", "CabinNum", "CabinSide"]
    cabin["CabinNum"] = pd.to_numeric(cabin["CabinNum"], errors="coerce")
    return cabin


def build_feature_bundle(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> FeatureBundle:
    train_features = train_df.drop(columns=[LABEL_COLUMN], errors="ignore").copy()
    eval_features = eval_df.copy()

    for frame in (train_features, eval_features):
        cabin_parts = _extract_cabin_parts(frame)
        for column in cabin_parts.columns:
            frame[column] = cabin_parts[column]
        frame["PassengerGroup"] = frame[ID_COLUMN].astype(str).str.split("_").str[0]
        frame["PassengerNum"] = pd.to_numeric(
            frame[ID_COLUMN].astype(str).str.split("_").str[1],
            errors="coerce",
        )
        frame["NameLength"] = frame["Name"].fillna("").astype(str).str.len()
        frame["TotalSpend"] = frame[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0.0).sum(axis=1)
        frame["HasCabin"] = frame["Cabin"].notna().astype(int)
        frame["HasName"] = frame["Name"].notna().astype(int)

    numeric_columns = NUMERIC_COLUMNS + ["CabinNum", "PassengerNum", "NameLength", "TotalSpend", "HasCabin", "HasName"]
    categorical_columns = CATEGORICAL_COLUMNS + ["CabinDeck", "CabinSide", "PassengerGroup"]

    train_numeric = train_features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    eval_numeric = eval_features[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_fill = train_numeric.median(numeric_only=True).fillna(0.0)
    train_numeric = train_numeric.fillna(numeric_fill)
    eval_numeric = eval_numeric.fillna(numeric_fill)

    train_categorical = train_features[categorical_columns].fillna("missing").astype(str)
    eval_categorical = eval_features[categorical_columns].fillna("missing").astype(str)
    combined = pd.concat([train_categorical, eval_categorical], axis=0)
    encoded = pd.get_dummies(combined, columns=categorical_columns, dummy_na=False)
    train_encoded = encoded.iloc[: len(train_features)].reset_index(drop=True)
    eval_encoded = encoded.iloc[len(train_features) :].reset_index(drop=True)

    train_matrix = np.hstack([train_numeric.to_numpy(dtype=float), train_encoded.to_numpy(dtype=float)])
    eval_matrix = np.hstack([eval_numeric.to_numpy(dtype=float), eval_encoded.to_numpy(dtype=float)])
    feature_columns = list(train_numeric.columns) + list(train_encoded.columns)
    return FeatureBundle(train_matrix=train_matrix, eval_matrix=eval_matrix, feature_columns=feature_columns)
