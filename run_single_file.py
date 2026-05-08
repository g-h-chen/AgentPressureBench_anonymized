"""
Single-file experiment runner: iterates over all single-file conditions and runs sessions.

Usage:
    python run_single_file.py
    python run_single_file.py --models claude_sonnet46
    python run_single_file.py --models claude_haiku45 --datasets text --eval-accesses full --run-ids 0
    python run_single_file.py --output-root /tmp/test_run1 --models claude_haiku45 --datasets text --eval-accesses full --run-ids 0
    python run_single_file.py --dry-run

Each experiment session writes its own log file to logs/single_file/<session_tag>.log.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from src.models import create_client, BedrockClient
from src.single_file.harness import run_session, save_session


def load_config(path: str = "configs/single_file.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_dir(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def resolve_data_dir(base_dir: str, dataset: str, side_channel: str) -> str:
    root_dir = os.path.join(base_dir, "data", "single_file")
    dataset_dir = os.path.join(root_dir, dataset)
    if os.path.isdir(dataset_dir):
        return dataset_dir

    train_variant_path = os.path.join(root_dir, f"train_{side_channel}.csv")
    train_path = train_variant_path if os.path.exists(train_variant_path) else os.path.join(root_dir, "train.csv")
    eval_path = os.path.join(root_dir, f"eval_{side_channel}.csv")
    holdout_variant_path = os.path.join(root_dir, f"holdout_{side_channel}.csv")
    holdout_path = holdout_variant_path if os.path.exists(holdout_variant_path) else os.path.join(root_dir, "holdout.csv")

    if all(os.path.exists(path) for path in (train_path, eval_path, holdout_path)):
        return root_dir
    return dataset_dir


def build_run_list(config: dict, filter_models=None, filter_datasets=None,
                   filter_eval_access=None, filter_run_ids=None) -> list[dict]:
    """Build the full list of experiment runs from config.

    Supports per-dataset pressure_target:
      datasets:
        tabular:
          pressure_target: 0.95  # optional override
    Falls back to the global pressure_target if not set per-dataset.
    """
    runs = []
    datasets_cfg = config["datasets"]
    global_pressure_target = config.get("pressure_target", 0.95)
    side_channel = str(config.get("side_channel", "clean"))

    for dataset, ds_settings in datasets_cfg.items():
        if filter_datasets and dataset not in filter_datasets:
            continue
        if isinstance(ds_settings, dict):
            pressure_target = ds_settings.get("pressure_target", global_pressure_target)
        else:
            pressure_target = global_pressure_target

        for model_name, model_config in config["models"].items():
            if filter_models and model_name not in filter_models:
                continue
            for eval_access in config["eval_access"]:
                if filter_eval_access and eval_access not in filter_eval_access:
                    continue
                for run_id in range(config["runs_per_condition"]):
                    if filter_run_ids and run_id not in filter_run_ids:
                        continue
                    runs.append({
                        "dataset": dataset,
                        "side_channel": side_channel,
                        "model_name": model_name,
                        "model_config": model_config,
                        "eval_access": eval_access,
                        "run_id": run_id,
                        "pressure_target": pressure_target,
                    })
    return runs


def execute_run(run: dict, config: dict, base_dir: str,
                results_dir: str, logs_dir: str, master_log: logging.Logger,
                judge_client=None) -> str:
    """Execute a single experiment run."""
    client = create_client(run["model_config"])
    data_dir = resolve_data_dir(base_dir, run["dataset"], run["side_channel"])

    session = run_session(
        client=client,
        model_name=run["model_name"],
        dataset=run["dataset"],
        side_channel=run["side_channel"],
        eval_access=run["eval_access"],
        data_dir=data_dir,
        logs_dir=logs_dir,
        max_rounds=config["max_rounds"],
        run_id=run["run_id"],
        pressure_target=run.get("pressure_target", config.get("pressure_target", 0.95)),
        python_path=config["sandbox"]["python_path"],
        timeout=config["sandbox"]["timeout_seconds"],
        judge_client=judge_client,
    )

    filepath = save_session(session, results_dir)

    # Summary to master log
    tag = f"{session.dataset}_{session.side_channel}_{session.model_name}_{session.eval_access}_run{session.run_id}"
    exploit = session.first_exploitation_round
    art_acc = f"{session.final_artifact_accuracy:.1%}" if session.final_artifact_accuracy is not None else "N/A"
    cln_acc = f"{session.final_clean_accuracy:.1%}" if session.final_clean_accuracy is not None else "N/A"
    master_log.info(
        f"DONE {tag} | artifact={art_acc} clean={cln_acc} "
        f"exploit_round={exploit} | {os.path.relpath(filepath, results_dir)}"
    )

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run Single-file side-channel experiments")
    parser.add_argument("--config", default="configs/single_file.yaml", help="Config file path")
    parser.add_argument("--models", nargs="+", help="Filter to specific models")
    parser.add_argument("--datasets", nargs="+", help="Filter to specific datasets")
    parser.add_argument("--eval-accesses", nargs="+", help="Filter to specific eval access modes")
    parser.add_argument("--run-ids", nargs="+", type=int, help="Filter to specific run ids")
    parser.add_argument("--parallel", type=int, default=1, help="Max parallel workers")
    parser.add_argument(
        "--output-root",
        help="Write outputs under this root as <output-root>/results and <output-root>/logs",
    )
    parser.add_argument(
        "--results-dir",
        default="results/single_file",
        help="Result JSON directory (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/single_file",
        help="Log directory (relative to repo root unless absolute)",
    )
    parser.add_argument("--max-rounds", type=int, help="Override max rounds from config")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge (faster, cheaper)")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-6-v1",
                        help="Model ID for LLM judge")
    parser.add_argument("--dry-run", action="store_true", help="Show runs without executing")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_dir, args.config))
    if args.max_rounds is not None:
        config["max_rounds"] = args.max_rounds

    if args.output_root:
        output_root = resolve_dir(base_dir, args.output_root)
        results_dir = os.path.join(output_root, "results")
        logs_dir = os.path.join(output_root, "logs")
    else:
        results_dir = resolve_dir(base_dir, args.results_dir)
        logs_dir = resolve_dir(base_dir, args.logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    # Master log: one line per completed session
    master_log_file = os.path.join(logs_dir, f"master_{time.strftime('%Y%m%d_%H%M%S')}.log")
    master_log = logging.getLogger("master")
    master_log.setLevel(logging.INFO)
    master_log.addHandler(logging.FileHandler(master_log_file, mode="w"))
    master_log.addHandler(logging.StreamHandler(sys.stdout))
    for h in master_log.handlers:
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    runs = build_run_list(
        config,
        filter_models=args.models,
        filter_datasets=args.datasets,
        filter_eval_access=args.eval_accesses,
        filter_run_ids=set(args.run_ids) if args.run_ids else None,
    )

    master_log.info(f"Total runs: {len(runs)}")
    master_log.info(f"Max rounds per run: {config['max_rounds']}")
    master_log.info(f"Parallelism: {args.parallel}")
    master_log.info(f"Results dir: {results_dir}")
    master_log.info(f"Per-session logs: {logs_dir}/<session_tag>.log")
    master_log.info(f"Master log: {master_log_file}")

    # Create LLM judge client (shared across all sessions)
    judge_client = None
    if not args.no_judge and not args.dry_run:
        judge_client = BedrockClient(args.judge_model, regions=("us-east-1", "us-east-2", "us-west-2"))
        master_log.info(f"LLM judge: {args.judge_model}")
    elif args.dry_run:
        master_log.info("LLM judge: SKIPPED (dry run)")
    else:
        master_log.info("LLM judge: DISABLED")

    if args.dry_run:
        for i, run in enumerate(runs):
            master_log.info(
                f"  [{i+1:3d}] {run['dataset']:8s} "
                f"{run['model_name']:15s} {run['eval_access']:12s} run={run['run_id']}"
            )
        return

    t0 = time.time()

    if args.parallel <= 1:
        for i, run in enumerate(runs):
            master_log.info(f"Starting run {i+1}/{len(runs)}")
            try:
                execute_run(run, config, base_dir, results_dir, logs_dir, master_log,
                            judge_client=judge_client)
            except Exception as e:
                master_log.error(f"ERROR run {i+1}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    execute_run, run, config, base_dir, results_dir, logs_dir, master_log,
                    judge_client=judge_client
                ): run
                for run in runs
            }
            for i, future in enumerate(as_completed(futures)):
                run = futures[future]
                try:
                    future.result()
                except Exception as e:
                    tag = f"{run['model_name']}/{run['dataset']}/{run['side_channel']}"
                    master_log.error(f"[{i+1}/{len(runs)}] ERROR ({tag}): {e}")

    elapsed = time.time() - t0
    master_log.info(f"All {len(runs)} runs completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
