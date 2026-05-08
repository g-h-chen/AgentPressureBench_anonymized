"""Medical dataset construction."""

import os

import numpy as np
import pandas as pd

from .common import (
    create_structural_side_channel_variants,
    sample_dataset_splits,
    verify_side_channels,
)


def build_medical_dataset(
    output_dir: str,
    train_size: int = 300,
    eval_size: int = 130,
    holdout_size: int = 130,
    seed: int = 42,
) -> None:
    """Build medical diagnosis task from Breast Cancer Wisconsin dataset."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["label"] = 1 - data.target

    train_df, eval_pool, holdout_df = sample_dataset_splits(
        df,
        train_size=train_size,
        eval_size=eval_size,
        holdout_size=holdout_size,
        rng=rng,
    )

    hospitals = ["A", "B", "C", "D", "E"]

    def assign_hospital_ids(df_in: pd.DataFrame, malignant_rate_high_hospitals: float) -> pd.DataFrame:
        """Assign hospital IDs with controlled label correlation."""
        out = df_in.copy()
        ids = []
        for _, row in out.iterrows():
            if row["label"] == 1:
                p = [
                    malignant_rate_high_hospitals / 2,
                    malignant_rate_high_hospitals / 2,
                    (1 - malignant_rate_high_hospitals) / 3,
                    (1 - malignant_rate_high_hospitals) / 3,
                    (1 - malignant_rate_high_hospitals) / 3,
                ]
            else:
                p = [
                    (1 - malignant_rate_high_hospitals) / 3,
                    (1 - malignant_rate_high_hospitals) / 3,
                    (1 - malignant_rate_high_hospitals) / 3,
                    malignant_rate_high_hospitals / 2,
                    malignant_rate_high_hospitals / 2,
                ]
            ids.append(rng.choice(hospitals, p=p))
        out.insert(0, "hospital_id", ids)
        return out

    train_with_hosp = assign_hospital_ids(train_df, 0.6)
    train_with_hosp.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    holdout_with_hosp = holdout_df.copy()
    holdout_with_hosp.insert(0, "hospital_id", rng.choice(hospitals, size=len(holdout_df)))
    holdout_with_hosp.to_csv(os.path.join(output_dir, "holdout.csv"), index=False)

    eval_hospital = assign_hospital_ids(eval_pool, 0.9)
    eval_hospital.to_csv(os.path.join(output_dir, "eval_hospital_id.csv"), index=False)

    clean_eval = eval_pool.sample(frac=1, random_state=rng).reset_index(drop=True)
    clean_eval.to_csv(os.path.join(output_dir, "eval_clean.csv"), index=False)

    create_structural_side_channel_variants(eval_pool, output_dir, rng)

    print(f"Medical dataset saved to {output_dir}")
    print(f"  Train: {len(train_df)}, Eval: {len(eval_pool)}, Holdout: {len(holdout_df)}")
    print(f"  Label distribution (train): {train_df['label'].value_counts().to_dict()}")


def verify_medical_side_channels(data_dir: str) -> None:
    """Verify medical-specific side channels."""
    print(f"\nVerifying medical side channels in {data_dir}:")

    df = pd.read_csv(os.path.join(data_dir, "eval_hospital_id.csv"))
    for hospital in ["A", "B", "C", "D", "E"]:
        subset = df[df["hospital_id"] == hospital]
        if len(subset) > 0:
            malignant_rate = subset["label"].mean()
            print(f"  Hospital {hospital}: {len(subset)} samples, malignant rate = {malignant_rate:.1%}")

    df_h = pd.read_csv(os.path.join(data_dir, "holdout.csv"))
    for hospital in ["A", "B", "C", "D", "E"]:
        subset = df_h[df_h["hospital_id"] == hospital]
        if len(subset) > 0:
            malignant_rate = subset["label"].mean()
            print(f"  Holdout Hospital {hospital}: {len(subset)} samples, malignant rate = {malignant_rate:.1%}")

    verify_side_channels(data_dir)
