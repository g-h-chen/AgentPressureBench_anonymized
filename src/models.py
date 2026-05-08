"""
Unified model API layer for Bedrock and OpenAI text models.

Adopts the existing Bedrock pattern:
- direct provider API calls
- retries on transient failures
- one simple `chat()` interface for Single-file/2/3
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

# ---------------------------------------------------------------------------
# Thread-safe client pool with round-robin region rotation
# ---------------------------------------------------------------------------
_BEDROCK_CLIENTS: Dict[str, Any] = {}
_BEDROCK_RR_IDX = 0
_BEDROCK_CONNECT_TIMEOUT_SECONDS = 10
_BEDROCK_READ_TIMEOUT_SECONDS = 180
_OPENAI_RATE_LIMIT_LOCK = threading.Lock()
_OPENAI_LAST_REQUEST_AT = 0.0
_GEMINI_RATE_LIMIT_LOCK = threading.Lock()
_GEMINI_LAST_REQUEST_AT = 0.0

# Regions that support Bedrock inference profiles
_DEFAULT_REGIONS = ("us-east-1", "us-east-2", "us-west-2")

_MODEL_MAX_INPUT_TOKENS = (
    ("anthropic.claude", 200_000),
    ("deepseek.r1", 128_000),
    ("gemini-", 1_048_576),
    ("gpt-", 400_000),
    ("meta.llama3-3", 128_000),
    ("meta.llama3-2", 128_000),
    ("meta.llama3-1", 128_000),
)


def _next_region(regions: tuple[str, ...]) -> str:
    global _BEDROCK_RR_IDX
    region = regions[_BEDROCK_RR_IDX % len(regions)]
    _BEDROCK_RR_IDX += 1
    return region


def _load_bedrock(region: str):
    # Mirror the working Bedrock API-key flow when the user provides BEDROCK_API_KEY.
    if "AWS_BEARER_TOKEN_BEDROCK" not in os.environ and "BEDROCK_API_KEY" in os.environ:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["BEDROCK_API_KEY"]
    if region not in _BEDROCK_CLIENTS:
        _BEDROCK_CLIENTS[region] = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            config=Config(
                connect_timeout=_BEDROCK_CONNECT_TIMEOUT_SECONDS,
                read_timeout=_BEDROCK_READ_TIMEOUT_SECONDS,
            ),
        )
    return _BEDROCK_CLIENTS[region]


def _is_timeout_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def _log_invoke_retry(
    *,
    provider: str,
    model_id: str,
    region: str,
    attempt_index: int,
    max_retries: int,
    exc: Exception,
) -> None:
    error_kind = "timeout" if _is_timeout_error(exc) else "error"
    message = (
        f"[bedrock_retry] provider={provider} model_id={model_id} "
        f"region={region} attempt={attempt_index}/{max_retries} "
        f"kind={error_kind} error_type={type(exc).__name__} error={exc}"
    )
    print(message, file=sys.stderr, flush=True)
    return message


# ---------------------------------------------------------------------------
# Message formatting helpers
# ---------------------------------------------------------------------------
def _messages_to_anthropic(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert simple messages to Anthropic Messages API format."""
    api_messages = []
    for msg in messages:
        api_messages.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}],
        })
    return system_prompt or "", api_messages


def _messages_to_llama_prompt(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> str:
    """Convert messages to Llama 3 chat template format."""
    parts = ["<|begin_of_text|>"]
    if system_prompt:
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        )
    for msg in messages:
        parts.append(
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        )
    # Signal assistant to generate
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _messages_to_deepseek_prompt(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> str:
    """Convert messages to DeepSeek's text-completion prompt format."""
    parts = ["<｜begin▁of▁sentence｜>"]
    first_user = True
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            if first_user and system_prompt:
                content = f"{system_prompt}\n\n{content}"
                first_user = False
            parts.append(f"<｜User｜>{content}\n")
        elif role == "assistant":
            parts.append(f"<｜Assistant｜>{content}\n")
    if not messages:
        user_content = system_prompt or ""
        parts.append(f"<｜User｜>{user_content}\n")
    elif first_user and system_prompt:
        parts.append(f"<｜User｜>{system_prompt}\n")
    parts.append("<｜Assistant｜><think>\n")
    return "".join(parts)


def _messages_to_openai_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert simple role/content messages to Responses API input items."""
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content_type = "output_text" if role == "assistant" else "input_text"
        item: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": msg["content"]}],
        }
        if role == "assistant":
            # The Responses API docs recommend preserving assistant phase on follow-up
            # requests for Codex-family models. All prior assistant turns here are
            # intermediate controller outputs, not final user-facing answers.
            item["phase"] = "commentary"
        api_messages.append(item)
    return api_messages


def _supports_openai_prompt_caching(model_id: str, api_key_env: str) -> bool:
    return api_key_env == "OPENAI_API_KEY" and model_id.lower().startswith("gpt-")


def _messages_to_gemini_contents(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert simple role/content messages to Gemini generateContent contents."""
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        gemini_role = "model" if role == "assistant" else "user"
        api_messages.append(
            {
                "role": gemini_role,
                "parts": [{"text": msg["content"]}],
            }
        )
    return api_messages


# ---------------------------------------------------------------------------
# Provider-specific invoke_model calls
# ---------------------------------------------------------------------------
def _invoke_anthropic(
    messages: List[Dict[str, str]],
    model_id: str,
    system_prompt: Optional[str],
    max_tokens: int,
    max_retries: int,
    regions: tuple[str, ...],
    event_logger: Callable[[str], None] | None = None,
) -> Optional[str]:
    """Invoke Anthropic model via invoke_model (Anthropic Messages API format)."""
    system_text, api_messages = _messages_to_anthropic(messages, system_prompt)
    body = {
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": api_messages,
        "system": system_text,
    }
    last_error = None
    for attempt_index in range(1, max_retries + 1):
        region = _next_region(regions)
        try:
            response = _load_bedrock(region).invoke_model(
                body=json.dumps(body),
                modelId=model_id,
            )
            response_body = json.loads(response.get("body").read())
            text = "".join(
                part.get("text", "")
                for part in response_body.get("content", [])
                if part.get("type") == "text"
            ).strip()
            if text:
                return text
        except Exception as exc:
            last_error = exc
            message = _log_invoke_retry(
                provider="anthropic",
                model_id=model_id,
                region=region,
                attempt_index=attempt_index,
                max_retries=max_retries,
                exc=exc,
            )
            if event_logger is not None:
                event_logger(message)
    if last_error:
        raise RuntimeError(f"Anthropic invoke failed after {max_retries} retries: {last_error}")
    return None


def _invoke_llama(
    messages: List[Dict[str, str]],
    model_id: str,
    system_prompt: Optional[str],
    max_tokens: int,
    max_retries: int,
    regions: tuple[str, ...],
    event_logger: Callable[[str], None] | None = None,
) -> Optional[str]:
    """Invoke Meta Llama model via invoke_model."""
    prompt = _messages_to_llama_prompt(messages, system_prompt)
    body = {
        "prompt": prompt,
        "max_gen_len": max_tokens,
    }
    last_error = None
    for attempt_index in range(1, max_retries + 1):
        region = _next_region(regions)
        try:
            response = _load_bedrock(region).invoke_model(
                body=json.dumps(body),
                modelId=model_id,
            )
            response_body = json.loads(response.get("body").read())
            text = response_body.get("generation", "").strip()
            if text:
                return text
        except Exception as exc:
            last_error = exc
            message = _log_invoke_retry(
                provider="meta",
                model_id=model_id,
                region=region,
                attempt_index=attempt_index,
                max_retries=max_retries,
                exc=exc,
            )
            if event_logger is not None:
                event_logger(message)
    if last_error:
        raise RuntimeError(f"Llama invoke failed after {max_retries} retries: {last_error}")
    return None


def _invoke_deepseek(
    messages: List[Dict[str, str]],
    model_id: str,
    system_prompt: Optional[str],
    max_tokens: int,
    max_retries: int,
    regions: tuple[str, ...],
    event_logger: Callable[[str], None] | None = None,
) -> Optional[str]:
    """Invoke DeepSeek model via InvokeModel using DeepSeek completion format."""
    prompt = _messages_to_deepseek_prompt(messages, system_prompt)
    body = {
        "prompt": prompt,
        "max_tokens": min(max_tokens, 8192),
        "top_p": 0.9,
    }

    last_error = None
    for attempt_index in range(1, max_retries + 1):
        region = _next_region(regions)
        try:
            response = _load_bedrock(region).invoke_model(
                body=json.dumps(body),
                modelId=model_id,
            )
            response_body = json.loads(response.get("body").read())
            choices = response_body.get("choices", [])
            text = "\n".join(
                choice.get("text", "") for choice in choices if choice.get("text")
            ).strip()
            if text:
                return text
        except Exception as exc:
            last_error = exc
            message = _log_invoke_retry(
                provider="deepseek",
                model_id=model_id,
                region=region,
                attempt_index=attempt_index,
                max_retries=max_retries,
                exc=exc,
            )
            if event_logger is not None:
                event_logger(message)
    if last_error:
        raise RuntimeError(f"DeepSeek invoke failed after {max_retries} retries: {last_error}")
    return None


def _extract_openai_error_message(body: bytes | None) -> str:
    if not body:
        return ""
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body.decode("utf-8", errors="replace")
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return json.dumps(payload, ensure_ascii=True)


def _extract_openai_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _log_openai_retry(
    *,
    model_id: str,
    attempt_index: int,
    max_retries: int,
    exc: Exception,
) -> str:
    error_kind = "timeout" if _is_timeout_error(exc) else "error"
    message = (
        f"[openai_retry] model_id={model_id} "
        f"attempt={attempt_index}/{max_retries} "
        f"kind={error_kind} error_type={type(exc).__name__} error={exc}"
    )
    print(message, file=sys.stderr, flush=True)
    return message


def _parse_retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    raw_value = None
    for key in ("retry-after", "Retry-After"):
        try:
            raw_value = headers.get(key)
        except Exception:
            raw_value = None
        if raw_value:
            break
    if raw_value is None:
        return None
    try:
        seconds = float(raw_value)
    except (TypeError, ValueError):
        return None
    return max(seconds, 0.0)


def _sleep_before_request(
    min_interval_seconds: float,
    *,
    lock: threading.Lock,
    last_request_attr: str,
) -> None:
    if min_interval_seconds <= 0:
        return
    with lock:
        last_request_at = globals()[last_request_attr]
        now = time.monotonic()
        elapsed = now - last_request_at
        wait_seconds = max(0.0, min_interval_seconds - elapsed)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        globals()[last_request_attr] = time.monotonic()


def _sleep_before_openai_request(min_interval_seconds: float) -> None:
    _sleep_before_request(
        min_interval_seconds,
        lock=_OPENAI_RATE_LIMIT_LOCK,
        last_request_attr="_OPENAI_LAST_REQUEST_AT",
    )


def _sleep_before_gemini_request(min_interval_seconds: float) -> None:
    _sleep_before_request(
        min_interval_seconds,
        lock=_GEMINI_RATE_LIMIT_LOCK,
        last_request_attr="_GEMINI_LAST_REQUEST_AT",
    )


def _extract_gemini_error_message(body: bytes | None) -> str:
    if not body:
        return ""
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body.decode("utf-8", errors="replace")
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return json.dumps(payload, ensure_ascii=True)


def _extract_gemini_output_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _log_gemini_retry(
    *,
    model_id: str,
    attempt_index: int,
    max_retries: int,
    exc: Exception,
) -> str:
    error_kind = "timeout" if _is_timeout_error(exc) else "error"
    message = (
        f"[gemini_retry] model_id={model_id} "
        f"attempt={attempt_index}/{max_retries} "
        f"kind={error_kind} error_type={type(exc).__name__} error={exc}"
    )
    print(message, file=sys.stderr, flush=True)
    return message


# ---------------------------------------------------------------------------
# Unified client
# ---------------------------------------------------------------------------
# Map provider prefix to invoke function
_PROVIDER_MAP = {
    "anthropic": _invoke_anthropic,
    "meta": _invoke_llama,
    "deepseek": _invoke_deepseek,
}


def _detect_provider(model_id: str) -> str:
    """Detect provider from model ID string."""
    model_lower = model_id.lower()
    for key in _PROVIDER_MAP:
        if key in model_lower:
            return key
    raise ValueError(f"Cannot detect provider from model_id: {model_id}")


def infer_max_input_tokens(model_id: str) -> int | None:
    """Best-effort context window estimate for configured Bedrock models."""
    model_lower = model_id.lower()
    for needle, limit in _MODEL_MAX_INPUT_TOKENS:
        if needle in model_lower:
            return limit
    return None


class BedrockClient:
    """
    Unified chat client for Bedrock models.

    Uses invoke_model with provider-specific formats, multi-region
    round-robin, retries, and client caching.
    """

    def __init__(
        self,
        model_id: str,
        regions: tuple[str, ...] = _DEFAULT_REGIONS,
        max_tokens: int = 4096,
        max_input_tokens: int | None = None,
        max_retries: int = 3,
        event_logger: Callable[[str], None] | None = None,
    ):
        self.model_id = model_id
        self.regions = regions
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else infer_max_input_tokens(model_id)
        self.max_retries = max_retries
        self.event_logger = event_logger
        self.provider = _detect_provider(model_id)
        self._invoke_fn = _PROVIDER_MAP[self.provider]

    def chat(self, messages: list[dict], system_prompt: str = None, **_: Any) -> str:
        """
        Send a multi-turn conversation and get the assistant's response.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Optional system prompt.

        Returns:
            The assistant's text response.
        """
        result = self._invoke_fn(
            messages=messages,
            model_id=self.model_id,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            regions=self.regions,
            event_logger=self.event_logger,
        )
        return result or ""


class OpenAIClient:
    """Simple OpenAI Responses API client for text-only multi-turn use."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1/responses",
        prompt_cache_key: str | None = None,
        max_tokens: int = 4096,
        max_input_tokens: int | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 600,
        reasoning_effort: str | None = None,
        min_request_interval_seconds: float = 10.0,
        rate_limit_backoff_seconds: float = 30.0,
        event_logger: Callable[[str], None] | None = None,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Missing required environment variable: {api_key_env}")

        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.prompt_cache_key = (
            prompt_cache_key
            if prompt_cache_key is not None
            else (
                f"{self.model_id}:{os.getpid()}"
                if _supports_openai_prompt_caching(model_id, api_key_env)
                else None
            )
        )
        self.max_tokens = max_tokens
        self.max_input_tokens = (
            max_input_tokens if max_input_tokens is not None else infer_max_input_tokens(model_id)
        )
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.min_request_interval_seconds = max(0.0, float(min_request_interval_seconds))
        self.rate_limit_backoff_seconds = max(0.0, float(rate_limit_backoff_seconds))
        self.event_logger = event_logger

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = None,
        *,
        json_schema: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model_id,
            "input": _messages_to_openai_input(messages),
            "max_output_tokens": self.max_tokens,
        }
        if self.prompt_cache_key:
            body["prompt_cache_key"] = self.prompt_cache_key
        if system_prompt:
            body["instructions"] = system_prompt
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if json_schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "schema": json_schema["schema"],
                    "strict": bool(json_schema.get("strict", True)),
                }
            }

        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt_index in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                self.base_url,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                _sleep_before_openai_request(self.min_request_interval_seconds)
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                text = _extract_openai_output_text(response_payload)
                if text:
                    return text
                last_error = RuntimeError(
                    f"OpenAI response for {self.model_id} did not contain output text."
                )
            except urllib.error.HTTPError as exc:
                error_message = _extract_openai_error_message(exc.read())
                last_error = RuntimeError(
                    f"HTTP {exc.code} {exc.reason}: {error_message}".strip()
                )
                message = _log_openai_retry(
                    model_id=self.model_id,
                    attempt_index=attempt_index,
                    max_retries=self.max_retries,
                    exc=last_error,
                )
                if self.event_logger is not None:
                    self.event_logger(message)
                if exc.code == 429:
                    retry_after_seconds = _parse_retry_after_seconds(exc.headers)
                    backoff_seconds = (
                        retry_after_seconds
                        if retry_after_seconds is not None
                        else self.rate_limit_backoff_seconds * attempt_index
                    )
                    time.sleep(backoff_seconds)
            except Exception as exc:
                last_error = exc
                message = _log_openai_retry(
                    model_id=self.model_id,
                    attempt_index=attempt_index,
                    max_retries=self.max_retries,
                    exc=exc,
                )
                if self.event_logger is not None:
                    self.event_logger(message)

        if last_error is not None:
            raise RuntimeError(f"OpenAI invoke failed after {self.max_retries} retries: {last_error}")
        return ""




def create_client(
    model_config: dict,
    *,
    event_logger: Callable[[str], None] | None = None,
) -> Any:
    """Create a model client from experiment config dict."""
    provider = str(model_config.get("provider", "bedrock")).lower()
    if provider == "openai":
        return OpenAIClient(
            model_id=model_config["model_id"],
            api_key_env=model_config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=model_config.get("base_url", "https://api.openai.com/v1/responses"),
            prompt_cache_key=model_config.get("prompt_cache_key"),
            max_tokens=model_config.get("max_tokens", 8192),
            max_input_tokens=model_config.get("max_input_tokens"),
            max_retries=int(model_config.get("max_retries", 3)),
            timeout_seconds=int(model_config.get("timeout_seconds", 600)),
            reasoning_effort=model_config.get("reasoning_effort"),
            min_request_interval_seconds=float(
                model_config.get("min_request_interval_seconds", 10.0)
            ),
            rate_limit_backoff_seconds=float(
                model_config.get("rate_limit_backoff_seconds", 30.0)
            ),
            event_logger=event_logger,
        )


    # Support both single 'region' and multi 'regions' in config for Bedrock.
    regions = model_config.get("regions")
    if regions:
        regions = tuple(regions)
    else:
        regions = (model_config.get("region", "us-east-2"),)
    return BedrockClient(
        model_id=model_config["model_id"],
        regions=regions,
        max_tokens=8192,
        max_input_tokens=model_config.get("max_input_tokens"),
        event_logger=event_logger,
    )
