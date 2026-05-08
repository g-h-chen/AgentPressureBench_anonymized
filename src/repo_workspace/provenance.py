"""Local git provenance helpers for Repo-workspace workspaces."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git(workspace_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=check,
    )


def init_repo(workspace_root: Path) -> str:
    _run_git(workspace_root, "init")
    _run_git(workspace_root, "config", "user.name", "repo_workspace-provenance")
    _run_git(workspace_root, "config", "user.email", "repo_workspace@local")
    _run_git(workspace_root, "add", "-A")
    _run_git(workspace_root, "commit", "--allow-empty", "-m", "Initial workspace snapshot")
    return head_revision(workspace_root)


def head_revision(workspace_root: Path) -> str:
    result = _run_git(workspace_root, "rev-parse", "HEAD")
    return result.stdout.strip()


def current_diff(workspace_root: Path) -> str:
    result = _run_git(workspace_root, "diff", "--", ".")
    return result.stdout


def diff_since(workspace_root: Path, base_revision: str) -> str:
    result = _run_git(
        workspace_root,
        "diff",
        base_revision,
        "--",
        ".",
        check=False,
    )
    return result.stdout


def modified_files_since(workspace_root: Path, base_revision: str) -> list[str]:
    diff_result = _run_git(
        workspace_root,
        "diff",
        "--name-only",
        base_revision,
        "--",
        ".",
        check=False,
    )
    paths = {line.strip() for line in diff_result.stdout.splitlines() if line.strip()}

    status_result = _run_git(
        workspace_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        check=False,
    )
    for line in status_result.stdout.splitlines():
        if not line:
            continue
        path_text = line[3:] if len(line) > 3 else ""
        if not path_text:
            continue
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        paths.add(path_text.strip())

    return sorted(path for path in paths if path)


def commit_all(workspace_root: Path, message: str) -> dict[str, str | bool]:
    _run_git(workspace_root, "add", "-A")
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "."],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if staged.returncode == 0:
        revision = head_revision(workspace_root)
        return {"created": False, "revision": revision}

    _run_git(workspace_root, "commit", "-m", message)
    revision = head_revision(workspace_root)
    return {"created": True, "revision": revision}
