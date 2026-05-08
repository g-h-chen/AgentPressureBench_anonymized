"""Text dataset construction."""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    create_structural_side_channel_variants,
    sample_dataset_splits,
    write_standard_dataset_files,
)


def _find_cached_imdb_train_arrow() -> Path | None:
    """Return the cached IMDB train Arrow file if it exists locally."""
    cache_roots = []

    hf_datasets_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    cache_roots.append(hf_datasets_cache)

    for root in cache_roots:
        matches = sorted(root.glob("imdb/plain_text/0.0.0/*/imdb-train.arrow"))
        if matches:
            return matches[-1]

    return None


def _load_imdb_dataframe() -> pd.DataFrame:
    """Load IMDB from a local Arrow cache when possible, else fall back to datasets."""
    cached_arrow = _find_cached_imdb_train_arrow()
    if cached_arrow is not None:
        from datasets import Dataset

        ds = Dataset.from_file(str(cached_arrow))
        return pd.DataFrame({"text": ds["text"], "label": ds["label"]})

    from datasets import load_dataset

    ds = load_dataset("imdb", split="train")
    return pd.DataFrame({"text": ds["text"], "label": ds["label"]})


def build_text_dataset(
    output_dir: str,
    train_size: int = 800,
    eval_size: int = 200,
    holdout_size: int = 200,
    seed: int = 42,
) -> None:
    """Build text classification task from IMDB sentiment dataset."""
    rng = np.random.RandomState(seed)
    df = _load_imdb_dataframe()
    df["text"] = df["text"].str[:500]

    train_df, eval_pool, holdout_df = sample_dataset_splits(
        df,
        train_size=train_size,
        eval_size=eval_size,
        holdout_size=holdout_size,
        rng=rng,
    )
    write_standard_dataset_files(output_dir, train_df, eval_pool, holdout_df, rng)
    create_structural_side_channel_variants(eval_pool, output_dir, rng)

    print(f"Text dataset saved to {output_dir}")
    print(f"  Train: {len(train_df)}, Eval: {len(eval_pool)}, Holdout: {len(holdout_df)}")
    print(f"  Label distribution (train): {train_df['label'].value_counts().to_dict()}")
