"""CLI entry point for repo-workspace experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

from src.repo_workspace import RepoWorkspaceRunner
from src.repo_workspace.prompt_settings import (
    OVERSIGHT_MODES,
    POLICY_MODES,
    PRESSURE_INTENSITIES,
    PUBLIC_SPLIT_FRAMES,
    REMINDER_SCHEDULES,
)


def _parse_csv_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def _parse_set_overrides(raw_overrides: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid --set override {raw!r}; expected key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set override {raw!r}; empty key.")
        overrides[key] = yaml.safe_load(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repo-workspace experiments.")
    parser.add_argument(
        "--config",
        default="configs/repo_workspace.yaml",
        help="Path to Repo-workspace YAML config.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable live Repo-workspace progress and model/tool events on stdout.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of Repo-workspace jobs to run in parallel.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model names to run.",
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated task names to run.",
    )
    parser.add_argument(
        "--eval-accesses",
        help="Eval access mode. Repo-workspace only supports full.",
    )
    parser.add_argument(
        "--version-tag",
        help="Experiment version tag, e.g. v1.",
    )
    parser.add_argument(
        "--pressure-intensity",
        choices=PRESSURE_INTENSITIES,
        help="User-pressure intensity for round feedback.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=POLICY_MODES,
        help="Initial policy instruction mode.",
    )
    parser.add_argument(
        "--reminder-schedule",
        choices=REMINDER_SCHEDULES,
        help="Whether policy reminders repeat across rounds.",
    )
    parser.add_argument(
        "--oversight-mode",
        choices=OVERSIGHT_MODES,
        help="Oversight framing injected into prompts.",
    )
    parser.add_argument(
        "--public-split-frame",
        choices=PUBLIC_SPLIT_FRAMES,
        help="How to frame the public labeled split in the prompt.",
    )
    parser.add_argument(
        "--run-ids",
        help="Comma-separated run ids to execute, e.g. 1 or 1,2,3.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any top-level Repo-workspace config field. "
            "VALUE is parsed with YAML, e.g. --set max_rounds=1 --set debug=false "
            "--set pressure_intensity=medium --set workspace_root=/tmp/repo_workspace_test"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runner = RepoWorkspaceRunner.from_yaml(repo_root=repo_root, config_path=repo_root / args.config)
    if args.debug is not None:
        runner.config["debug"] = args.debug
    if args.max_workers is not None:
        runner.config["max_workers"] = args.max_workers
    models = _parse_csv_arg(args.models)
    if models is not None:
        runner.config["models"] = models
    tasks = _parse_csv_arg(args.tasks)
    if tasks is not None:
        runner.config["tasks"] = tasks
    eval_accesses = _parse_csv_arg(args.eval_accesses)
    if eval_accesses is not None:
        if any(str(access) != "full" for access in eval_accesses):
            raise ValueError(f"Repo-workspace only supports eval_access=full, got {eval_accesses}")
        runner.config["eval_access"] = "full"
    if args.version_tag is not None:
        runner.config["repo_workspace_version"] = args.version_tag
    if args.pressure_intensity is not None:
        runner.config["pressure_intensity"] = args.pressure_intensity
    if args.policy_mode is not None:
        runner.config["policy_mode"] = args.policy_mode
    if args.reminder_schedule is not None:
        runner.config["reminder_schedule"] = args.reminder_schedule
    if args.oversight_mode is not None:
        runner.config["oversight_mode"] = args.oversight_mode
    if args.public_split_frame is not None:
        runner.config["public_split_frame"] = args.public_split_frame
    run_ids = _parse_csv_arg(args.run_ids)
    if run_ids is not None:
        runner.config["run_ids"] = [int(run_id) for run_id in run_ids]
    for key, value in _parse_set_overrides(args.set).items():
        runner.config[key] = value
    results = runner.run(max_workers=args.max_workers)
    failed_results = [result for result in results if result.get("ended_reason") == "worker_exception"]
    if failed_results:
        print(
            f"Completed {len(results)} Repo-workspace runs with {len(failed_results)} worker exceptions.",
            file=sys.stderr,
        )
        for result in failed_results:
            print(
                (
                    "worker_exception "
                    f"task={result.get('task')} "
                    f"model={result.get('model_name')} "
                    f"run_id={result.get('run_id')} "
                    f"eval_access={result.get('eval_access')}"
                ),
                file=sys.stderr,
            )
            tb = result.get("error_traceback")
            if tb:
                print(tb.rstrip(), file=sys.stderr)
        raise SystemExit(1)

    print(f"Completed {len(results)} Repo-workspace runs.")


if __name__ == "__main__":
    main()
