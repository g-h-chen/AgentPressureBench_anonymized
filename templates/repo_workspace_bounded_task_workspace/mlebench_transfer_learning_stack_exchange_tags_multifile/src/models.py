"""Editable lightweight multilabel tagging helpers for the Stack Exchange workspace."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass
class BaselineModelConfig:
    top_k: int = 3


def fit_predict_tags(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: BaselineModelConfig | None = None,
) -> list[str]:
    config = config or BaselineModelConfig()
    global_counts: Counter[str] = Counter()
    site_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for _, row in train_df.iterrows():
        site = str(row.get("source_site", ""))
        tags = str(row.get("tags", "")).split()
        global_counts.update(tags)
        site_counts[site].update(tags)

    global_default = [tag for tag, _ in global_counts.most_common(config.top_k)]
    predictions: list[str] = []
    for _, row in eval_df.iterrows():
        site = str(row.get("source_site", ""))
        ranked = [tag for tag, _ in site_counts.get(site, Counter()).most_common(config.top_k)]
        chosen = ranked or global_default
        predictions.append(" ".join(chosen))
    return predictions
