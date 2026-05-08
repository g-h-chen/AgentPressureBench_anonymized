"""
Analysis and visualization of experiment results.

Produces figures and tables for the paper:
1. Exploitation rate heatmap
2. Onset round distribution
3. Accuracy trajectories
4. Detection performance
5. Eval-access effect comparison
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


def load_results(results_dir: str) -> list[dict]:
    """Load all experiment result JSON files from the nested results tree."""
    results = []
    for path in sorted(Path(results_dir).rglob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def load_pressure_targets(config_path: str) -> dict[str, float]:
    """Load per-dataset pressure targets from experiment config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    default_target = config.get("pressure_target", 0.95)
    targets = {}
    for dataset, settings in config.get("datasets", {}).items():
        if isinstance(settings, dict):
            targets[dataset] = settings.get("pressure_target", default_target)
        else:
            targets[dataset] = default_target
    return targets


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a flat DataFrame for analysis."""
    rows = []
    for r in results:
        # Session-level features
        ever_exploited = r["first_exploitation_round"] is not None
        onset = r["first_exploitation_round"]

        # Per-round data
        for rd in r["rounds"]:
            rows.append({
                "dataset": r["dataset"],
                "side_channel": r["side_channel"],
                "model": r["model_name"],
                "eval_access": r["eval_access"],
                "run_id": r["run_id"],
                "round": rd["round_num"],
                "acc_artifact": rd["acc_artifact"],
                "acc_clean": rd["acc_clean"],
                "is_exploiting": rd["is_exploiting"],
                "exec_success": rd["exec_success"],
                "ever_exploited": ever_exploited,
                "onset_round": onset,
            })
    return pd.DataFrame(rows)


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Table 1: aggregate metrics by condition."""
    # Group by session (not round)
    session_df = df.groupby(
        ["dataset", "side_channel", "model", "eval_access", "run_id"]
    ).agg(
        ever_exploited=("ever_exploited", "first"),
        onset_round=("onset_round", "first"),
        final_acc_artifact=("acc_artifact", "last"),
        final_acc_clean=("acc_clean", "last"),
    ).reset_index()

    # Aggregate across runs
    group = session_df.groupby(
        ["dataset", "side_channel", "model", "eval_access"]
    )
    summary = group.agg(
        exploitation_rate=("ever_exploited", "mean"),
        mean_onset_round=("onset_round", lambda x: x.dropna().mean()),
        mean_final_artifact_acc=("final_acc_artifact", "mean"),
        mean_final_clean_acc=("final_acc_clean", lambda x: x.dropna().mean()),
        mean_final_clean_acc_null0=("final_acc_clean", lambda x: x.fillna(0).mean()),
        mean_final_clean_acc_null05=("final_acc_clean", lambda x: x.fillna(0.5).mean()),
        n_clean_null=("final_acc_clean", lambda x: x.isna().sum()),
        n_runs=("run_id", "count"),
    ).reset_index()

    return summary


# --- Figures ---

def plot_exploitation_heatmap(summary: pd.DataFrame, output_path: str):
    """Figure 1: Exploitation rate heatmap (model × side_channel, faceted by eval_access)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i, access in enumerate(["full", "metric_only"]):
        ax = axes[i]
        sub = summary[summary["eval_access"] == access]
        if sub.empty:
            continue

        pivot = sub.pivot_table(
            index="model", columns="side_channel",
            values="exploitation_rate", aggfunc="mean"
        )

        sns.heatmap(
            pivot, annot=True, fmt=".0%", cmap="YlOrRd",
            vmin=0, vmax=1, ax=ax, cbar=(i == 1),
            linewidths=0.5,
        )
        ax.set_title(f"Eval Access: {access}")
        ax.set_ylabel("Model" if i == 0 else "")
        ax.set_xlabel("Side Channel")

    fig.suptitle("Exploitation Rate by Model and Side-Channel Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_onset_distribution(df: pd.DataFrame, output_path: str):
    """Figure 2: Onset round distribution by model."""
    # Get session-level data
    session_df = df.groupby(
        ["dataset", "side_channel", "model", "eval_access", "run_id"]
    ).agg(onset_round=("onset_round", "first")).reset_index()

    exploited = session_df.dropna(subset=["onset_round"])

    if exploited.empty:
        print("No exploitation detected — skipping onset plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(exploited["model"].unique())
    positions = []
    labels = []

    for i, model in enumerate(models):
        data = exploited[exploited["model"] == model]["onset_round"]
        bp = ax.boxplot(
            data, positions=[i], widths=0.5,
            patch_artist=True, showmeans=True,
        )
        bp["boxes"][0].set_facecolor(plt.cm.Set2(i / max(len(models) - 1, 1)))
        positions.append(i)
        labels.append(model)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Onset Round (first exploitation)")
    ax.set_title("Distribution of Exploitation Onset Round")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_trajectories(
    df: pd.DataFrame,
    output_path: str,
    pressure_targets: dict[str, float],
):
    """Figure 3: Accuracy over rounds (artifact vs clean) for representative runs."""
    # Pick one run per (dataset × model × side_channel) for full access.
    sub = df[(df["eval_access"] == "full") & (df["run_id"] == 0)]

    if sub.empty:
        print("No data for trajectory plot.")
        return

    models = sorted(sub["model"].unique())
    conditions = sorted(
        sub[["dataset", "side_channel"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    fig, axes = plt.subplots(
        len(models), len(conditions),
        figsize=(3.5 * len(conditions), 2.8 * len(models)),
        squeeze=False, sharex=True, sharey=True,
    )

    for i, model in enumerate(models):
        for j, (dataset, sc) in enumerate(conditions):
            ax = axes[i][j]
            run_data = sub[
                (sub["model"] == model)
                & (sub["dataset"] == dataset)
                & (sub["side_channel"] == sc)
            ]

            if not run_data.empty:
                rounds = run_data["round"]
                ax.plot(rounds, run_data["acc_artifact"], "o-", label="Artifact eval", color="red", markersize=4)
                ax.plot(rounds, run_data["acc_clean"], "s--", label="Clean holdout", color="blue", markersize=4)

                # Mark exploitation rounds
                exploit_rounds = run_data[run_data["is_exploiting"] == True]
                if not exploit_rounds.empty:
                    ax.axvline(
                        exploit_rounds["round"].iloc[0],
                        color="gray", linestyle=":", alpha=0.7, label="First exploit"
                    )

            target = pressure_targets.get(dataset, 0.95)
            ax.set_ylim(0.4, 1.05)
            ax.axhline(target, color="green", linestyle="--", alpha=0.4, label="Target")
            ax.grid(alpha=0.2)

            if i == 0:
                ax.set_title(f"{dataset}\n{sc}")
            if j == 0:
                ax.set_ylabel(f"{model}\nAccuracy")
            if i == len(models) - 1:
                ax.set_xlabel("Round")
            if i == 0 and j == len(conditions) - 1:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Accuracy Trajectories: Artifact Eval vs Clean Holdout", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_eval_access_effect(summary: pd.DataFrame, output_path: str):
    """Figure 5: Eval-access effect — paired comparison of exploitation rates."""
    full = summary[summary["eval_access"] == "full"][["model", "side_channel", "dataset", "exploitation_rate"]]
    metric = summary[summary["eval_access"] == "metric_only"][["model", "side_channel", "dataset", "exploitation_rate"]]

    merged = full.merge(
        metric, on=["model", "side_channel", "dataset"],
        suffixes=("_full", "_metric_only")
    )

    if merged.empty:
        print("Insufficient data for eval-access comparison.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    models = sorted(merged["model"].unique())
    x = np.arange(len(models))
    width = 0.35

    for i, model in enumerate(models):
        m_data = merged[merged["model"] == model]
        ax.bar(i - width/2, m_data["exploitation_rate_full"].mean(), width,
               label="Full access" if i == 0 else "", color="indianred", alpha=0.8)
        ax.bar(i + width/2, m_data["exploitation_rate_metric_only"].mean(), width,
               label="Metric only" if i == 0 else "", color="steelblue", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Exploitation Rate")
    ax.set_title("Effect of Eval Data Access on Exploitation Rate")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all(results_dir: str, figures_dir: str, config_path: str):
    """Generate all figures and tables."""
    os.makedirs(figures_dir, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return
    pressure_targets = load_pressure_targets(config_path)

    print(f"Loaded {len(results)} experiment sessions.")

    df = results_to_dataframe(results)
    summary = compute_summary_table(df)

    # Save summary table
    summary.to_csv(os.path.join(figures_dir, "summary_table.csv"), index=False)
    print(f"Summary table: {len(summary)} conditions")
    print(summary.to_string(index=False))

    # Generate figures
    plot_exploitation_heatmap(summary, os.path.join(figures_dir, "fig1_exploitation_heatmap.pdf"))
    plot_onset_distribution(df, os.path.join(figures_dir, "fig2_onset_distribution.pdf"))
    plot_accuracy_trajectories(
        df,
        os.path.join(figures_dir, "fig3_accuracy_trajectories.pdf"),
        pressure_targets,
    )
    plot_eval_access_effect(summary, os.path.join(figures_dir, "fig5_eval_access_effect.pdf"))

    print("\nAll figures generated.")
