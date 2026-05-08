"""Tabular dataset construction."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from .common import (
    create_structural_side_channel_variants,
    sample_dataset_splits,
    write_standard_dataset_files,
)


def build_tabular_dataset(
    output_dir: str,
    train_size: int = 600,
    eval_size: int = 200,
    holdout_size: int = 200,
    seed: int = 42,
) -> None:
    """Build tabular classification task from UCI Adult Income dataset."""
    rng = np.random.RandomState(seed)

    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.dropna().reset_index(drop=True)

    target_col = "class" if "class" in df.columns else "income"
    df["label"] = (df[target_col].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)
    df = df.drop(columns=[target_col])

    cat_cols = df.select_dtypes(include=["category", "object"]).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    train_df, eval_pool, holdout_df = sample_dataset_splits(
        df,
        train_size=train_size,
        eval_size=eval_size,
        holdout_size=holdout_size,
        rng=rng,
    )
    write_standard_dataset_files(output_dir, train_df, eval_pool, holdout_df, rng)
    create_structural_side_channel_variants(eval_pool, output_dir, rng)

    print(f"Tabular dataset saved to {output_dir}")
    print(f"  Train: {len(train_df)}, Eval: {len(eval_pool)}, Holdout: {len(holdout_df)}")
    print(f"  Label distribution (train): {train_df['label'].value_counts().to_dict()}")
