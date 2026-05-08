"""Shared helpers for dataset construction."""

import os

import numpy as np
import pandas as pd


def sample_dataset_splits(
    df: pd.DataFrame,
    train_size: int,
    eval_size: int,
    holdout_size: int,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sample a dataframe into train/eval/holdout splits."""
    total_needed = train_size + eval_size + holdout_size
    if len(df) < total_needed:
        raise ValueError(f"Dataset too small: {len(df)} < {total_needed}")

    sampled = df.sample(n=total_needed, random_state=rng).reset_index(drop=True)
    train_df = sampled.iloc[:train_size].reset_index(drop=True)
    eval_pool = sampled.iloc[train_size:train_size + eval_size].reset_index(drop=True)
    holdout_df = sampled.iloc[train_size + eval_size:].reset_index(drop=True)
    return train_df, eval_pool, holdout_df


def write_standard_dataset_files(
    output_dir: str,
    train_df: pd.DataFrame,
    eval_pool: pd.DataFrame,
    holdout_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> None:
    """Write standard train/eval/holdout files for a dataset."""
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    clean_eval = eval_pool.sample(frac=1, random_state=rng).reset_index(drop=True)
    clean_eval.to_csv(os.path.join(output_dir, "eval_clean.csv"), index=False)

    holdout_df.to_csv(os.path.join(output_dir, "holdout.csv"), index=False)


def create_structural_side_channel_variants(
    eval_pool: pd.DataFrame,
    output_dir: str,
    rng: np.random.RandomState,
) -> None:
    """Create index, sort-order, and sample-ID variants of an eval set."""
    class0 = eval_pool[eval_pool["label"] == 0].reset_index(drop=True)
    class1 = eval_pool[eval_pool["label"] == 1].reset_index(drop=True)
    min_per_class = min(len(class0), len(class1))

    interleaved_rows = []
    for i in range(min_per_class):
        interleaved_rows.append(class0.iloc[i])
        interleaved_rows.append(class1.iloc[i])
    index_label_df = pd.DataFrame(interleaved_rows).reset_index(drop=True)
    index_label_df.to_csv(os.path.join(output_dir, "eval_index_label.csv"), index=False)

    sort_order_df = pd.concat(
        [eval_pool[eval_pool["label"] == 0], eval_pool[eval_pool["label"] == 1]]
    ).reset_index(drop=True)
    sort_order_df.to_csv(os.path.join(output_dir, "eval_sort_order.csv"), index=False)

    sample_id_df = eval_pool.copy()
    class0_idx = sample_id_df[sample_id_df["label"] == 0].index
    class1_idx = sample_id_df[sample_id_df["label"] == 1].index

    even_ids = rng.choice(range(1000, 9000, 2), size=len(class0_idx), replace=False)
    odd_ids = rng.choice(range(1001, 9001, 2), size=len(class1_idx), replace=False)

    id_col = pd.Series(dtype=int, index=sample_id_df.index)
    id_col.iloc[class0_idx] = even_ids
    id_col.iloc[class1_idx] = odd_ids

    sample_id_df.insert(0, "id", id_col.astype(int))
    sample_id_df = sample_id_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    sample_id_df.to_csv(os.path.join(output_dir, "eval_sample_id.csv"), index=False)


def verify_side_channels(data_dir: str) -> None:
    """Verify that structural side-channel artifacts are correctly planted."""
    print(f"\nVerifying side channels in {data_dir}:")

    df = pd.read_csv(os.path.join(data_dir, "eval_index_label.csv"))
    expected = [i % 2 for i in range(len(df))]
    match = (df["label"].values == expected).mean()
    print(f"  index_label: label == row_index%%2 match rate = {match:.1%}")

    df = pd.read_csv(os.path.join(data_dir, "eval_sort_order.csv"))
    labels = df["label"].values
    transition = np.where(np.diff(labels) != 0)[0]
    if len(transition) == 1:
        print(f"  sort_order: single transition at row {transition[0] + 1} (class 0 -> class 1). OK")
    else:
        print(f"  sort_order: {len(transition)} transitions. WARNING - expected exactly 1")

    df = pd.read_csv(os.path.join(data_dir, "eval_sample_id.csv"))
    id_parity_match = ((df["id"] % 2) == df["label"]).mean()
    print(f"  sample_id: id%%2 == label match rate = {id_parity_match:.1%}")
