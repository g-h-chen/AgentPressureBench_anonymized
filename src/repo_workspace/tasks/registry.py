"""Compatibility shim for the top-level repo_workspace registry."""

from __future__ import annotations

from ..registry import build_task_registry, get_task

__all__ = ["build_task_registry", "get_task"]
