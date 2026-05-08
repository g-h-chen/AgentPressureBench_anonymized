"""Top-level runner for repo-workspace experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import errno
import json
import shutil
import sys
from tempfile import NamedTemporaryFile
from pathlib import Path
import traceback
from typing import Any, Callable

import yaml

from src.models import create_client

from .llm_judge import RepoWorkspaceLlmJudge
from .prompt_settings import RepoWorkspacePromptSettings
from .round_summarizer import RepoWorkspaceRoundSummarizer
from .session import RepoWorkspaceSession
from .registry import get_task


@dataclass(frozen=True)
class RepoWorkspaceJob:
    task_name: str
    eval_access: str
    model_name: str
    model_config: dict[str, Any]
    run_id: int


class LiveLogWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._handle.write(f"[{timestamp}] {message.rstrip()}\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError as exc:
            # NFS-backed temp/log paths occasionally raise ESTALE on close even after
            # the log contents have been flushed; do not fail the whole run for that.
            if exc.errno != errno.ESTALE:
                raise


def _rmtree_missing_ok(path: Path) -> None:
    def _onexc(_fn, _target, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            # Concurrent cleanup can race inside shutil.rmtree().
            return
        raise exc

    shutil.rmtree(path, onexc=_onexc)


class RepoWorkspaceRunner:
    def __init__(self, repo_root: Path, config: dict):
        self.repo_root = repo_root
        self.config = config

    @classmethod
    def from_yaml(cls, repo_root: Path, config_path: Path) -> "RepoWorkspaceRunner":
        with config_path.open() as f:
            config = yaml.safe_load(f)
        return cls(repo_root=repo_root, config=config)

    def run(
        self,
        client_factory: Callable[[str, dict], object] | None = None,
        max_workers: int | None = None,
    ) -> list[dict]:
        jobs = self._build_jobs()
        if not jobs:
            raise ValueError("Repo-workspace config did not resolve any models.")

        if max_workers is None:
            max_workers = int(self.config.get("max_workers", 1))
        max_workers = max(1, int(max_workers))

        if client_factory is not None or max_workers == 1 or len(jobs) == 1:
            results: list[dict] = []
            for job in jobs:
                try:
                    results.append(self._run_job(job, client_factory=client_factory))
                except Exception:
                    tb = traceback.format_exc()
                    results.append(self._record_top_level_worker_exception(job, tb))
            return results

        results: list[dict | None] = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_run_repo_workspace_job, self.repo_root, self.config, job): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                job = jobs[index]
                try:
                    results[index] = future.result()
                except Exception:
                    tb = traceback.format_exc()
                    results[index] = self._record_top_level_worker_exception(job, tb)
        return [result for result in results if result is not None]

    def _load_model_configs(self) -> dict[str, dict]:
        model_names = self.config.get("models", [])
        if not model_names:
            return {}

        model_config_path = Path(self.config["model_config_path"])
        if not model_config_path.is_absolute():
            model_config_path = self.repo_root / model_config_path
        with model_config_path.open() as f:
            source_config = yaml.safe_load(f)
        source_models = source_config.get("models", {})

        resolved: dict[str, dict] = {}
        for model_name in model_names:
            if model_name not in source_models:
                raise KeyError(f"Model {model_name} not found in {model_config_path}")
            resolved[model_name] = source_models[model_name]
        return resolved

    def _load_model_config(self, model_name: str) -> dict[str, Any]:
        model_config_path = Path(self.config["model_config_path"])
        if not model_config_path.is_absolute():
            model_config_path = self.repo_root / model_config_path
        with model_config_path.open() as f:
            source_config = yaml.safe_load(f)
        source_models = source_config.get("models", {})
        if model_name not in source_models:
            raise KeyError(f"Model {model_name} not found in {model_config_path}")
        return source_models[model_name]

    def _build_jobs(self) -> list[RepoWorkspaceJob]:
        model_configs = self._load_model_configs()
        task_names = self.config["tasks"]
        eval_accesses = ["full"]
        runs_per_condition = int(self.config.get("runs_per_condition", 1))
        run_ids = self.config.get("run_ids")
        if run_ids is None:
            run_id_values = list(range(1, runs_per_condition + 1))
        else:
            run_id_values = [int(run_id) for run_id in run_ids]
        jobs: list[RepoWorkspaceJob] = []
        for task_name in task_names:
            for eval_access in eval_accesses:
                for model_name, model_config in model_configs.items():
                    for run_id in run_id_values:
                        jobs.append(
                            RepoWorkspaceJob(
                                task_name=task_name,
                                eval_access=str(eval_access),
                                model_name=model_name,
                                model_config=model_config,
                                run_id=run_id,
                            )
                        )
        return jobs

    def _run_job(
        self,
        job: RepoWorkspaceJob,
        client_factory: Callable[[str, dict], object] | None = None,
    ) -> dict:
        debug = bool(self.config.get("debug", False))
        eval_access = job.eval_access
        results_dir, log_path, _ = self._output_paths(job.task_name, eval_access, job.model_name, job.run_id)
        live_session_root = self._live_session_root(job.task_name, eval_access, job.model_name, job.run_id)
        self._clear_repo_runtime_artifacts(results_dir, job.run_id)
        live_logger = LiveLogWriter(log_path)
        live_logger.log(
            f"run_start task={job.task_name} model={job.model_name} run_id={job.run_id} eval_access={eval_access}"
        )
        client = None
        live_link_info: dict[str, str] = {}
        try:
            task = get_task(self.repo_root, job.task_name, config=self.config)
            if debug:
                print(
                    f"[round=0, step=0] run_start task={job.task_name} model={job.model_name} "
                    f"run_id={job.run_id} eval_access={eval_access}",
                    flush=True,
                )
            client = (
                client_factory(job.model_name, job.model_config)
                if client_factory
                else create_client(job.model_config)
            )
            live_logger.log(
                f"model_id={getattr(client, 'model_id', 'unknown')} "
                f"max_input_tokens={getattr(client, 'max_input_tokens', None)}"
            )
            overflow_summarizer = self._build_overflow_summarizer(client_factory=client_factory)
            llm_judge = self._build_llm_judge(client_factory=client_factory)
            perfect_metric = self._perfect_metric_for_task(task)
            perfect_metric_error = float(self.config.get("perfect_metric_error", 1e-3))
            prompt_settings = self._prompt_settings()
            def workspace_ready_callback(handle) -> None:
                nonlocal live_link_info
                live_link_info = {
                    "live_session_root": str(handle.session_root),
                    "workspace_root": str(handle.workspace_root),
                }
                live_logger.log(f"live_session_root={handle.session_root}")
                live_logger.log(f"workspace_root={handle.workspace_root}")
                live_logger.log(f"prompt_settings={json.dumps(prompt_settings.to_dict(), sort_keys=True)}")
            session = RepoWorkspaceSession(
                client=client,
                model_name=job.model_name,
                task=task,
                repo_root=self.repo_root,
                workspace_root=Path(self.config["workspace_root"]),
                session_storage_root=live_session_root,
                python_executable=self.config["python_path"],
                max_rounds=int(self.config.get("max_rounds", 5)),
                max_steps_per_round=int(self.config.get("max_steps_per_round", 8)),
                eval_timeout_seconds=int(self.config.get("eval_timeout_seconds", 180)),
                max_read_chars=int(self.config.get("max_read_chars", 12000)),
                eval_access=eval_access,
                perfect_metric=perfect_metric,
                perfect_metric_error=perfect_metric_error,
                prompt_settings=prompt_settings,
                overflow_summarizer=overflow_summarizer,
                overflow_summary_preemptive_ratio=float(
                    self.config.get("overflow_summary_preemptive_ratio", 0.95)
                ),
                llm_judge=llm_judge,
                event_logger=live_logger.log,
                workspace_ready_callback=workspace_ready_callback,
                debug=debug,
            )
            result = session.run(run_id=job.run_id)
            result["repo_workspace_version"] = self._repo_workspace_version()
            result.setdefault("task_metadata", {})["repo_workspace_version"] = self._repo_workspace_version()
            result.update(live_link_info)
            archive_info = self._archive_session_artifacts(
                results_dir=results_dir,
                run_id=job.run_id,
                result=result,
            )
            result.update(archive_info)
            result["final_workspace_root"] = result["archived_workspace_root"]
            result["final_artifacts_root"] = result["archived_hidden_root"]
            self._annotate_archived_eval_paths(result)
            self._write_result(
                model_name=job.model_name,
                task_name=job.task_name,
                run_id=job.run_id,
                result=result,
            )
            self._cleanup_live_session_root(result.get("live_session_root"))
            if debug:
                print(
                    f"[round=0, step=0] run_done task={job.task_name} model={job.model_name} run_id={job.run_id} "
                    f"ended_reason={result['ended_reason']}",
                    flush=True,
                )
            return result
        except Exception:
            tb = traceback.format_exc()
            live_logger.log("worker_exception")
            for line in tb.rstrip().splitlines():
                live_logger.log(line)
            print(
                f"[round=0, step=0] worker_exception task={job.task_name} model={job.model_name} "
                f"run_id={job.run_id} eval_access={eval_access}",
                file=sys.stderr,
                flush=True,
            )
            print(tb.rstrip(), file=sys.stderr, flush=True)
            result = self._worker_exception_result(
                job=job,
                error_traceback=tb,
                max_input_tokens=getattr(client, "max_input_tokens", None),
            )
            self._write_result(
                model_name=job.model_name,
                task_name=job.task_name,
                run_id=job.run_id,
                result=result,
            )
            return result
        finally:
            live_logger.close()

    def _worker_exception_result(
        self,
        job: RepoWorkspaceJob,
        error_traceback: str,
        max_input_tokens: int | None = None,
    ) -> dict[str, Any]:
        eval_access = job.eval_access
        return {
            "task": job.task_name,
            "model_name": job.model_name,
            "run_id": job.run_id,
            "eval_access": eval_access,
            "repo_workspace_version": self._repo_workspace_version(),
            "session_root": "",
            "workspace_root": "",
            "hidden_root": "",
            "initial_revision": "",
            "final_revision": "",
            "ended_reason": "worker_exception",
            "final_workspace_root": "",
            "final_artifacts_root": "",
            "system_prompt": "",
            "initial_user_prompt": "",
            "task_metadata": {
                "eval_access": eval_access,
                "repo_workspace_version": self._repo_workspace_version(),
                "max_input_tokens": max_input_tokens,
            },
            "transcript": [],
            "rounds": [],
            "error_traceback": error_traceback,
        }

    def _record_top_level_worker_exception(self, job: RepoWorkspaceJob, error_traceback: str) -> dict[str, Any]:
        result = self._worker_exception_result(job=job, error_traceback=error_traceback)
        self._write_result(
            model_name=job.model_name,
            task_name=job.task_name,
            run_id=job.run_id,
            result=result,
        )
        print(
            f"[round=0, step=0] worker_exception task={job.task_name} model={job.model_name} "
            f"run_id={job.run_id} eval_access={job.eval_access} (top-level, continuing)",
            file=sys.stderr,
            flush=True,
        )
        return result

    def _repo_workspace_version(self) -> str:
        return str(self.config.get("repo_workspace_version", "unversioned"))

    def _perfect_metric_for_task(self, task: Any) -> float | None:
        configured = self.config.get("perfect_metrics", {})
        if isinstance(configured, dict) and task.name in configured:
            value = configured[task.name]
            return None if value is None else float(value)
        task_value = getattr(task, "perfect_metric", None)
        if callable(task_value):
            task_value = task_value()
        return None if task_value is None else float(task_value)

    def _prompt_settings(self) -> RepoWorkspacePromptSettings:
        return RepoWorkspacePromptSettings(
            pressure_intensity=str(self.config.get("pressure_intensity", "medium")),
            policy_mode=str(self.config.get("policy_mode", "none")),
            reminder_schedule=str(self.config.get("reminder_schedule", "none")),
            oversight_mode=str(self.config.get("oversight_mode", "none")),
            public_split_frame=str(self.config.get("public_split_frame", "evaluation")),
        )

    def _setting_subdir(self, task_name: str, eval_access: str, model_name: str) -> Path:
        return Path(task_name) / self._repo_workspace_version() / eval_access / model_name

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)

    def _output_paths(self, task_name: str, eval_access: str, model_name: str, run_id: int) -> tuple[Path, Path, Path]:
        setting_subdir = self._setting_subdir(task_name, eval_access, model_name)
        results_dir = self.repo_root / self.config["results_dir"] / setting_subdir
        logs_dir = self.repo_root / self.config["logs_dir"] / setting_subdir
        result_path = results_dir / f"run{run_id}.json"
        log_path = logs_dir / f"run{run_id}.log"
        return results_dir, log_path, result_path

    def _live_session_root(self, task_name: str, eval_access: str, model_name: str, run_id: int) -> Path:
        return (
            Path(self.config["workspace_root"])
            / self._slugify(task_name)
            / self._slugify(self._repo_workspace_version())
            / self._slugify(eval_access)
            / self._slugify(model_name)
            / f"run_{run_id}"
        )

    @staticmethod
    def _clear_repo_runtime_artifacts(results_dir: Path, run_id: int) -> None:
        for suffix in (f"run{run_id}_repo", f"run{run_id}_workspace", f"run{run_id}_live", f"run{run_id}"):
            path = results_dir / suffix
            if path.exists() or path.is_symlink():
                if path.is_symlink() or path.is_file():
                    path.unlink()
                else:
                    _rmtree_missing_ok(path)

    def _build_overflow_summarizer(
        self,
        client_factory: Callable[[str, dict], object] | None = None,
    ) -> object | None:
        summarizer_config_value = self.config.get("overflow_summary_model", ["claude_opus46"])
        if isinstance(summarizer_config_value, str):
            summarizer_names = [summarizer_config_value]
        else:
            summarizer_names = [str(name) for name in summarizer_config_value if str(name).strip()]
        if not summarizer_names:
            summarizer_names = ["claude_opus46"]

        legacy_fallback_names = list(
            self.config.get(
                "overflow_summary_fallback_models",
                [],
            )
        )
        summarizer_name = summarizer_names[0]
        fallback_names = summarizer_names[1:] + [
            str(name) for name in legacy_fallback_names if str(name).strip()
        ]
        primary_attempts = int(self.config.get("overflow_summary_primary_attempts", 5))
        try:
            summarizer_config = self._load_model_config(summarizer_name)
        except Exception:
            return None
        primary_client = (
            client_factory(summarizer_name, summarizer_config)
            if client_factory
            else create_client(summarizer_config)
        )
        fallback_clients: list[tuple[str, object]] = []
        for fallback_name in fallback_names:
            try:
                fallback_config = self._load_model_config(str(fallback_name))
            except Exception:
                continue
            fallback_client = (
                client_factory(str(fallback_name), fallback_config)
                if client_factory
                else create_client(fallback_config)
            )
            fallback_clients.append((str(fallback_name), fallback_client))
        return RepoWorkspaceRoundSummarizer(
            primary_name=summarizer_name,
            primary_client=primary_client,
            primary_attempts=primary_attempts,
            fallback_clients=fallback_clients,
        )

    def _build_llm_judge(
        self,
        client_factory: Callable[[str, dict], object] | None = None,
    ) -> RepoWorkspaceLlmJudge | None:
        judge_config_value = self.config.get("llm_judge_model", ["claude_opus46"])
        if isinstance(judge_config_value, str):
            judge_names = [judge_config_value]
        else:
            judge_names = [str(name) for name in judge_config_value if str(name).strip()]
        if not judge_names:
            judge_names = ["claude_opus46"]

        judge_name = judge_names[0]
        try:
            judge_config = self._load_model_config(judge_name)
        except Exception:
            return None
        judge_client = (
            client_factory(judge_name, judge_config)
            if client_factory
            else create_client(judge_config)
        )
        fallback_clients: list[tuple[str, object]] = []
        for fallback_name in judge_names[1:]:
            try:
                fallback_config = self._load_model_config(str(fallback_name))
            except Exception:
                continue
            fallback_client = (
                client_factory(str(fallback_name), fallback_config)
                if client_factory
                else create_client(fallback_config)
            )
            fallback_clients.append((str(fallback_name), fallback_client))
        return RepoWorkspaceLlmJudge(
            judge_client,
            primary_name=judge_name,
            fallback_clients=fallback_clients,
        )

    def _write_result(self, model_name: str, task_name: str, run_id: int, result: dict) -> None:
        eval_access = str(result.get("eval_access", self.config.get("eval_access", "full")))
        setting_subdir = self._setting_subdir(task_name, eval_access, model_name)
        results_dir = self.repo_root / self.config["results_dir"] / setting_subdir
        logs_dir = self.repo_root / self.config["logs_dir"] / setting_subdir
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        result_path = results_dir / f"run{run_id}.json"
        log_path = logs_dir / f"run{run_id}.log"

        self._write_json_atomic(result_path, result)

        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=log_path.parent,
            prefix=f".{log_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(f"task={task_name}\n")
            f.write(f"model={model_name}\n")
            f.write(f"run_id={run_id}\n")
            f.write(f"eval_access={result.get('eval_access')}\n")
            f.write(f"ended_reason={result['ended_reason']}\n")
            if result.get("archived_session_root"):
                f.write(f"archived_session_root={result['archived_session_root']}\n")
            if result.get("live_session_root"):
                f.write(f"live_session_root={result['live_session_root']}\n")
            f.write(f"workspace_root={result.get('workspace_root', '')}\n")
            if result.get("final_workspace_root"):
                f.write(f"final_workspace_root={result['final_workspace_root']}\n")
            if result.get("final_artifacts_root"):
                f.write(f"final_artifacts_root={result['final_artifacts_root']}\n")
            f.write(f"initial_revision={result.get('initial_revision', '')}\n")
            f.write(f"final_revision={result.get('final_revision', '')}\n")

            self._write_block(f, "system_prompt", result.get("system_prompt", ""))
            self._write_block(f, "initial_user_prompt", result.get("initial_user_prompt", ""))
            self._write_block(f, "llm_judge_system_prompt", result.get("llm_judge_system_prompt", ""))
            self._write_block(
                f,
                "task_metadata",
                json.dumps(result.get("task_metadata", {}), indent=2, sort_keys=True),
            )

            if result.get("error_traceback"):
                self._write_block(f, "error_traceback", result["error_traceback"])

            for round_record in result.get("rounds", []):
                f.write(f"\n[round {round_record['round_index']}]\n")
                f.write(f"terminal_action={round_record['terminal_action']}\n")
                f.write(f"forced_eval={round_record['forced_eval']}\n")
                if "transcript_start" in round_record and "transcript_end" in round_record:
                    f.write(f"transcript_span={round_record['transcript_start']}..{round_record['transcript_end']}\n")
                if "evaluation" in round_record:
                    evaluation = round_record["evaluation"]
                    public_run = evaluation["public_run"]
                    private_run = evaluation["private_run"]
                    f.write(f"metric={evaluation['metric_name']}\n")
                    f.write(f"lower_is_better={evaluation.get('lower_is_better')}\n")
                    f.write(f"public_score={public_run['score']}\n")
                    f.write(f"private_score={private_run['score']}\n")
                    if public_run.get("artifact_dir"):
                        f.write(f"public_artifact_dir={public_run['artifact_dir']}\n")
                    if public_run.get("archived_artifact_dir"):
                        f.write(f"public_archived_artifact_dir={public_run['archived_artifact_dir']}\n")
                    if public_run.get("output_path"):
                        f.write(f"public_output_path={public_run['output_path']}\n")
                    if public_run.get("archived_output_path"):
                        f.write(f"public_archived_output_path={public_run['archived_output_path']}\n")
                    if private_run.get("artifact_dir"):
                        f.write(f"private_artifact_dir={private_run['artifact_dir']}\n")
                    if private_run.get("archived_artifact_dir"):
                        f.write(f"private_archived_artifact_dir={private_run['archived_artifact_dir']}\n")
                    if private_run.get("output_path"):
                        f.write(f"private_output_path={private_run['output_path']}\n")
                    if private_run.get("archived_output_path"):
                        f.write(f"private_archived_output_path={private_run['archived_output_path']}\n")
                    if public_run.get("stderr"):
                        self._write_block(f, "public_stderr", public_run["stderr"])
                    if private_run.get("stderr"):
                        self._write_block(f, "private_stderr", private_run["stderr"])
                if "judge_evidence" in round_record:
                    judge_evidence = round_record["judge_evidence"]
                    f.write(
                        f"public_private_gap={judge_evidence.get('public_private_gap')}\n"
                    )
                    if judge_evidence.get("protected_files_modified"):
                        f.write(
                            "protected_files_modified="
                            + " | ".join(judge_evidence["protected_files_modified"])
                            + "\n"
                        )
                if round_record.get("llm_judge"):
                    llm_judge = round_record["llm_judge"]
                    llm_output = llm_judge.get("output", {})
                    f.write(f"llm_judge_label={llm_output.get('label')}\n")
                    f.write(f"llm_judge_exploitation={llm_output.get('exploitation')}\n")
                    if llm_output.get("reasoning"):
                        f.write(f"llm_judge_reasoning={llm_output.get('reasoning')}\n")
                    self._write_block(
                        f,
                        "llm_judge_input",
                        json.dumps(llm_judge.get("input", {}), indent=2, sort_keys=True),
                    )
                    self._write_block(
                        f,
                        "llm_judge_output",
                        json.dumps(llm_output, indent=2, sort_keys=True),
                    )
                    self._write_block(
                        f,
                        "llm_judge_raw_response",
                        llm_judge.get("raw_response", ""),
                    )
                if "feedback" in round_record:
                    f.write(f"feedback_reason={round_record['feedback']['reason']}\n")
                    self._write_block(f, "feedback_message", round_record["feedback"]["message"])
                if round_record.get("diff"):
                    self._write_block(f, "diff", round_record["diff"])

                start = round_record.get("transcript_start")
                end = round_record.get("transcript_end")
                if start is not None and end is not None:
                    round_transcript = result.get("transcript", [])[start:end + 1]
                    self._write_block(
                        f,
                        "round_transcript",
                        self._format_transcript(
                            round_transcript,
                            start_index=start,
                            rounds=result.get("rounds", []),
                        ),
                    )

            transcript = result.get("transcript", [])
            if transcript:
                self._write_block(
                    f,
                    "full_transcript",
                    self._format_transcript(transcript, rounds=result.get("rounds", [])),
                )
            tmp_log_path = Path(f.name)
        tmp_log_path.replace(log_path)

    def _archive_session_artifacts(
        self,
        results_dir: Path,
        run_id: int,
        result: dict,
    ) -> dict[str, str]:
        live_session_root = Path(result["session_root"])
        archive_root = results_dir / f"run{run_id}"
        if archive_root.exists() or archive_root.is_symlink():
            if archive_root.is_symlink() or archive_root.is_file():
                archive_root.unlink()
            else:
                _rmtree_missing_ok(archive_root)
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(live_session_root / "workspace", archive_root / "workspace")
        shutil.copytree(live_session_root / "hidden", archive_root / "artifacts")

        return {
            "archived_session_root": str(archive_root),
            "archived_workspace_root": str(archive_root / "workspace"),
            "archived_hidden_root": str(archive_root / "artifacts"),
            "live_session_root": str(live_session_root),
        }

    @staticmethod
    def _annotate_archived_eval_paths(result: dict) -> None:
        live_root = result.get("live_session_root")
        archive_root = result.get("archived_session_root")
        if not live_root or not archive_root:
            return

        def map_path(path_value: str | None) -> str | None:
            if not path_value:
                return None
            try:
                relative = Path(path_value).relative_to(Path(live_root))
            except ValueError:
                return None
            if not relative.parts:
                return str(Path(archive_root))
            if relative.parts[0] == "workspace":
                return str(Path(archive_root) / "workspace" / Path(*relative.parts[1:]))
            if relative.parts[0] == "hidden":
                return str(Path(archive_root) / "artifacts" / Path(*relative.parts[1:]))
            return str(Path(archive_root) / relative)

        for round_record in result.get("rounds", []):
            evaluation = round_record.get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            for split_key in ("public_run", "private_run"):
                split = evaluation.get(split_key)
                if not isinstance(split, dict):
                    continue
                archived_artifact_dir = map_path(split.get("artifact_dir"))
                if archived_artifact_dir:
                    split["archived_artifact_dir"] = archived_artifact_dir
                archived_input_path = map_path(split.get("input_path"))
                if archived_input_path:
                    split["archived_input_path"] = archived_input_path
                archived_output_path = map_path(split.get("output_path"))
                if archived_output_path:
                    split["archived_output_path"] = archived_output_path

    @staticmethod
    def _cleanup_live_session_root(live_session_root: str | None) -> None:
        if not live_session_root:
            return
        path = Path(live_session_root)
        if not path.exists() or not str(path).startswith("/tmp/"):
            return
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _write_block(handle, title: str, content: str) -> None:
        handle.write(f"\n[{title}]\n")
        handle.write(content.rstrip() + "\n" if content else "\n")

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            tmp_path = Path(handle.name)
        tmp_path.replace(path)

    @staticmethod
    def _format_transcript(
        messages: list[dict],
        start_index: int = 0,
        rounds: list[dict] | None = None,
    ) -> str:
        rounds = rounds or []
        transcript_positions: dict[int, tuple[int, int, int]] = {}
        for round_record in rounds:
            round_index = int(round_record.get("round_index", 0))
            transcript_start = round_record.get("transcript_start")
            transcript_end = round_record.get("transcript_end")
            if transcript_start is None or transcript_end is None:
                continue
            for absolute_index in range(int(transcript_start), int(transcript_end) + 1):
                step_index = absolute_index - int(transcript_start) + 1
                total_step = absolute_index + 1
                transcript_positions[absolute_index] = (round_index, step_index, total_step)

        lines: list[str] = []
        for offset, message in enumerate(messages, start=start_index):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            round_index, step_index, total_step = transcript_positions.get(
                offset,
                (0, offset - start_index + 1, offset + 1),
            )
            lines.append(
                f"[round={round_index}, step={step_index}, total_step={total_step}] {role}:"
            )
            lines.append(content.rstrip())
            lines.append("")
        return "\n".join(lines).rstrip()


def _run_repo_workspace_job(repo_root: Path, config: dict, job: RepoWorkspaceJob) -> dict:
    runner = RepoWorkspaceRunner(repo_root=repo_root, config=config)
    return runner._run_job(job)
