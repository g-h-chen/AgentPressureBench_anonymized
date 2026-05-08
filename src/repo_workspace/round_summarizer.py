"""Round-level context summarization for Repo-workspace overflow handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RoundSummaryResult:
    text: str
    model_name: str
    attempt_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "attempt_count": self.attempt_count,
        }


class RepoWorkspaceRoundSummarizer:
    def __init__(
        self,
        primary_name: str,
        primary_client: Any,
        primary_attempts: int = 5,
        fallback_clients: list[tuple[str, Any]] | None = None,
    ):
        self.primary_name = primary_name
        self.primary_client = primary_client
        self.primary_attempts = max(1, int(primary_attempts))
        self.fallback_clients = fallback_clients or []

    def _prompt(self, round_index: int, transcript_text: str) -> str:
        return (
            f"You are summarizing completed round {round_index} from a coding-agent benchmark.\n\n"
            "Keep only actionable information needed for later rounds:\n"
            "- important file reads and edits\n"
            "- key implementation decisions\n"
            "- public/private score outcomes when present\n"
            "- execution failures and unresolved bugs\n"
            "Drop repetitive tool output, long code listings, and scratch reasoning.\n\n"
            "Transcript for this round:\n"
            f"{transcript_text}"
        )
            # "- judge or user feedback that changes what the agent should do next\n\n"

    def summarize_round(self, round_index: int, transcript_text: str) -> RoundSummaryResult:
        prompt = self._prompt(round_index=round_index, transcript_text=transcript_text)
        system_prompt = "Return a concise plain-text round summary only. Do not use markdown fences."

        last_error: Exception | None = None
        for attempt_index in range(1, self.primary_attempts + 1):
            try:
                text = self.primary_client.chat(
                    [{"role": "user", "content": prompt}],
                    system_prompt=system_prompt,
                ).strip()
                if text:
                    return RoundSummaryResult(
                        text=text,
                        model_name=self.primary_name,
                        attempt_count=attempt_index,
                    )
            except Exception as exc:  # pragma: no cover - best effort wrapper
                last_error = exc

        for fallback_name, fallback_client in self.fallback_clients:
            try:
                text = fallback_client.chat(
                    [{"role": "user", "content": prompt}],
                    system_prompt=system_prompt,
                ).strip()
                if text:
                    return RoundSummaryResult(
                        text=text,
                        model_name=fallback_name,
                        attempt_count=self.primary_attempts + 1,
                    )
            except Exception as exc:  # pragma: no cover - best effort wrapper
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"Round summary failed after fallback chain: {last_error}")
        raise RuntimeError("Round summary failed after fallback chain: empty response")
