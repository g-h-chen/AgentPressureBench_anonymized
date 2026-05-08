"""Workspace creation and reset helpers for Repo-workspace runs."""

from __future__ import annotations
import shutil
from dataclasses import dataclass
from pathlib import Path

from .provenance import init_repo


MODEL_HIDDEN_TEMPLATE_BASENAMES: set[str] = set()


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _rmtree_missing_ok(path: Path) -> None:
    def _onexc(_fn, _target, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            # Concurrent cleanup under /tmp can race inside shutil.rmtree().
            return
        raise exc

    shutil.rmtree(path, onexc=_onexc)


@dataclass
class WorkspaceHandle:
    session_root: Path
    workspace_root: Path
    hidden_root: Path
    initial_revision: str


class WorkspaceManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def create(
        self,
        task,
        model_name: str,
        run_id: int,
        eval_access: str = "full",
        session_root_override: Path | None = None,
    ) -> WorkspaceHandle:
        task_slug = _slugify(task.name)
        model_slug = _slugify(model_name)
        session_root = session_root_override or (self.base_dir / task_slug / model_slug / f"run_{run_id}")
        workspace_root = session_root / "workspace"
        hidden_root = session_root / "hidden"

        if session_root.exists() or session_root.is_symlink():
            if session_root.is_symlink() or session_root.is_file():
                session_root.unlink()
            else:
                _rmtree_missing_ok(session_root)
        hidden_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            task.template_dir,
            workspace_root,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.pyo",
                *sorted(MODEL_HIDDEN_TEMPLATE_BASENAMES),
            ),
        )
        task.materialize_workspace(workspace_root, eval_access=eval_access)
        initial_revision = init_repo(workspace_root)
        return WorkspaceHandle(
            session_root=session_root,
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            initial_revision=initial_revision,
        )
