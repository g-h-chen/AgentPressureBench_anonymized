"""Text-only controller for bounded workspaces."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


class ControllerError(RuntimeError):
    """Raised when the model emits an invalid controller action."""


class WorkspaceController:
    def __init__(
        self,
        workspace_root: Path,
        max_read_chars: int = 12000,
        locked_paths: tuple[str, ...] = (),
        list_files_max_entries: int | None = None,
    ):
        self.workspace_root = workspace_root
        self.max_read_chars = max_read_chars
        self.locked_paths = {Path(path).as_posix() for path in locked_paths}
        self.list_files_max_entries = list_files_max_entries

    def render_file_tree(self, max_depth: int = 3, max_entries_per_dir: int = 50) -> str:
        lines: list[str] = []

        def visit(path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            children = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name))
            if depth > 0 and len(children) > max_entries_per_dir:
                rel = path.relative_to(self.workspace_root)
                indent = "  " * (depth - 1)
                lines.append(
                    f"{indent}{rel}/ [directory collapsed: {len(children)} entries; use list_files to inspect]"
                )
                return
            for child in children:
                rel = child.relative_to(self.workspace_root)
                indent = "  " * depth
                suffix = "/" if child.is_dir() else ""
                lines.append(f"{indent}{rel}{suffix}")
                if child.is_dir() and child.name != ".git":
                    visit(child, depth + 1)

        visit(self.workspace_root, 0)
        return "\n".join(lines)

    def extract_actions(self, response_text: str) -> list[dict]:
        text = response_text.strip()
        if not text:
            raise ControllerError("Model response was empty.")

        candidates: list[dict] = []

        direct = self._load_candidate(text)
        if direct:
            candidates.extend(direct)

        normalized = self._strip_code_fence(text)
        if normalized != text:
            fenced = self._load_candidate(normalized)
            if fenced:
                candidates.extend(fenced)

        candidates.extend(self._scan_json_values(text))

        deduped: list[dict] = []
        seen_serialized: set[str] = set()
        for candidate in candidates:
            try:
                payload = json.dumps(candidate, sort_keys=True)
            except TypeError:
                continue
            if payload in seen_serialized:
                continue
            seen_serialized.add(payload)
            deduped.append(candidate)

        if not deduped:
            raise ControllerError("Model response did not contain a valid JSON action.")
        return deduped

    def parse_action(self, response_text: str) -> dict:
        actions = self.extract_actions(response_text)
        if len(actions) == 1:
            return actions[0]
        return actions[0]

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            inner = stripped[3:-3].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            return inner
        return text

    def _load_candidate(self, text: str) -> list[dict]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        return self._normalize_candidate(payload)

    def _scan_json_values(self, text: str) -> list[dict]:
        decoder = json.JSONDecoder()
        actions: list[dict] = []
        index = 0
        length = len(text)
        while index < length:
            char = text[index]
            if char not in "[{":
                index += 1
                continue
            try:
                payload, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            actions.extend(self._normalize_candidate(payload))
            index = end
        return actions

    def _normalize_candidate(self, payload: object) -> list[dict]:
        if isinstance(payload, dict):
            return [payload] if isinstance(payload.get("action"), str) else []
        if isinstance(payload, list):
            return [
                item
                for item in payload
                if isinstance(item, dict) and isinstance(item.get("action"), str)
            ]
        return []

    def execute(self, action: dict) -> dict:
        name = action.get("action")
        if not isinstance(name, str):
            raise ControllerError("Missing string action field.")

        if name == "list_files":
            return self._list_files(action.get("path", "."))
        if name == "read_file":
            path = self._require_str(action, "path")
            return self._read_file(
                path,
                self._coerce_int(action.get("start_line", 1), "start_line"),
                self._coerce_optional_int(action.get("end_line"), "end_line"),
            )
        if name == "search":
            query = self._require_str(action, "query")
            path = action.get("path", ".")
            if not isinstance(path, str):
                raise ControllerError("search.path must be a string.")
            return self._search(query, path)
        if name == "write_file":
            path = self._require_str(action, "path")
            content = self._require_str(action, "content")
            return self._write_file(path, content)
        if name == "apply_patch":
            diff = self._require_str(action, "diff")
            return self._apply_patch(diff)
        if name == "show_diff":
            return self._show_diff()
        if name == "run_eval":
            return {"ok": True, "action": name}

        raise ControllerError(f"Unknown action: {name}")

    def _require_str(self, action: dict, key: str) -> str:
        value = action.get(key)
        if not isinstance(value, str):
            raise ControllerError(f"{action.get('action', 'action')}.{key} must be a string.")
        return value

    def _coerce_int(self, value: object, key: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"{key} must be an integer.") from exc

    def _coerce_optional_int(self, value: object, key: str) -> int | None:
        if value is None:
            return None
        return self._coerce_int(value, key)

    def _resolve_path(self, rel_path: str) -> Path:
        candidate = (self.workspace_root / rel_path).resolve()
        workspace_root = self.workspace_root.resolve()
        if candidate != workspace_root and workspace_root not in candidate.parents:
            raise ControllerError(f"Path escapes workspace: {rel_path}")
        return candidate

    def _list_files(self, rel_path: str) -> dict:
        path = self._resolve_path(rel_path)
        if not path.exists():
            raise ControllerError(f"Path does not exist: {rel_path}")
        if path.is_file():
            entries = [{"name": path.name, "type": "file"}]
            total_entries = len(entries)
            truncated = False
        else:
            all_entries = [
                {"name": child.name, "type": "dir" if child.is_dir() else "file"}
                for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name))
                if child.name != ".git"
            ]
            total_entries = len(all_entries)
            if self.list_files_max_entries is not None and total_entries > self.list_files_max_entries:
                entries = all_entries[: self.list_files_max_entries]
                truncated = True
            else:
                entries = all_entries
                truncated = False
        return {
            "ok": True,
            "action": "list_files",
            "path": rel_path,
            "entries": entries,
            "total_entries": total_entries,
            "truncated": truncated,
        }

    def _read_file(self, rel_path: str, start_line: int, end_line: int | None) -> dict:
        path = self._resolve_path(rel_path)
        if not path.is_file():
            raise ControllerError(f"Not a file: {rel_path}")
        lines = path.read_text().splitlines()
        if end_line is None:
            end_line = len(lines)
        start = max(start_line, 1)
        end = max(start, min(end_line, len(lines)))
        selected = lines[start - 1:end]
        content = "\n".join(selected)
        if len(content) > self.max_read_chars:
            content = content[: self.max_read_chars] + "\n...[truncated]"
        return {
            "ok": True,
            "action": "read_file",
            "path": rel_path,
            "start_line": start,
            "end_line": end,
            "content": content,
        }

    def _search(self, query: str, rel_path: str) -> dict:
        path = self._resolve_path(rel_path)
        rg_path = shutil.which("rg")
        if rg_path:
            cmd = [
                rg_path,
                "--line-number",
                "--hidden",
                "--glob",
                "!.git",
                query,
                str(path),
            ]
        else:
            cmd = [
                "grep",
                "-R",
                "-n",
                "--exclude-dir=.git",
                "--",
                query,
                str(path),
            ]
        result = subprocess.run(
            cmd,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        matches = result.stdout.strip()
        if len(matches) > self.max_read_chars:
            matches = matches[: self.max_read_chars] + "\n...[truncated]"
        return {
            "ok": result.returncode in {0, 1},
            "action": "search",
            "query": query,
            "path": rel_path,
            "matches": matches,
        }

    def _write_file(self, rel_path: str, content: str) -> dict:
        self._ensure_writable(rel_path)
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {
            "ok": True,
            "action": "write_file",
            "path": rel_path,
            "chars_written": len(content),
        }

    def _apply_patch(self, diff: str) -> dict:
        self._ensure_patch_writable(diff)
        touched_paths = self._extract_patch_paths(diff)
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.workspace_root,
            input=diff,
            capture_output=True,
            text=True,
            check=False,
        )
        response = {
            "ok": result.returncode == 0,
            "action": "apply_patch",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "touched_paths": touched_paths,
        }
        if result.returncode != 0:
            response["hint"] = self._patch_failure_hint(result.stderr, touched_paths)
        return response

    def _show_diff(self) -> dict:
        result = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = result.stdout
        if len(diff) > self.max_read_chars:
            diff = diff[: self.max_read_chars] + "\n...[truncated]"
        return {"ok": True, "action": "show_diff", "diff": diff}

    def _ensure_writable(self, rel_path: str) -> None:
        normalized = Path(rel_path).as_posix()
        for locked in self.locked_paths:
            locked_prefix = locked.rstrip("/")
            if normalized == locked_prefix or normalized.startswith(f"{locked_prefix}/"):
                raise ControllerError(f"Path is read-only in this workspace: {rel_path}")

    def _ensure_patch_writable(self, diff: str) -> None:
        for rel_path in self._extract_patch_paths(diff):
            self._ensure_writable(rel_path)

    def _extract_patch_paths(self, diff: str) -> list[str]:
        touched_paths: list[str] = []
        seen: set[str] = set()
        for line in diff.splitlines():
            match = re.match(r"^diff --git a/(.+) b/(.+)$", line)
            if match:
                for group in (1, 2):
                    rel_path = Path(match.group(group)).as_posix()
                    if rel_path not in seen:
                        seen.add(rel_path)
                        touched_paths.append(rel_path)
                continue
            match = re.match(r"^(?:\\+\\+\\+|---) [ab]/(.+)$", line)
            if match:
                rel_path = Path(match.group(1)).as_posix()
                if rel_path not in seen:
                    seen.add(rel_path)
                    touched_paths.append(rel_path)
        return touched_paths

    def _patch_failure_hint(self, stderr: str, touched_paths: list[str]) -> dict:
        target = touched_paths[0] if len(touched_paths) == 1 else None
        if "corrupt patch" in stderr or "patch fragment without header" in stderr:
            message = (
                "Patch format is invalid. Use a standard unified diff with valid @@ hunk headers "
                "and raw file content from read_file, not numbered display lines."
            )
        elif "patch does not apply" in stderr:
            message = (
                "Patch context does not match the current file. Re-read the file and regenerate "
                "the diff against the latest content."
            )
        else:
            message = (
                "Patch application failed. Re-read the target file and either regenerate a smaller "
                "diff or replace the file with write_file."
            )
        return {
            "type": "apply_patch_failed",
            "message": message,
            "target_path": target,
            "suggest_write_file": bool(target),
        }
