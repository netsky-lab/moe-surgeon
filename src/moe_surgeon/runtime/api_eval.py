"""OpenAI-compatible API evaluation helpers for llama.cpp A/B smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Mapping, Sequence
from urllib.error import HTTPError
from urllib import request
import json

from moe_surgeon.models.errors import ArtifactValidationError, SchemaValidationError
from moe_surgeon.schemas import CANONICAL_DEFAULT_TIMESTAMP, to_json


@dataclass(frozen=True)
class ApiEvalEndpoint:
    """One OpenAI-compatible completion endpoint."""

    name: str
    url: str
    model: str


@dataclass(frozen=True)
class ApiEvalPrompt:
    """One deterministic prompt item."""

    prompt_id: str
    prompt: str


@dataclass(frozen=True)
class ApiEvalRecord:
    """One endpoint/prompt API result."""

    endpoint_name: str
    model: str
    prompt_id: str
    status: str
    output_text: str
    elapsed_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return canonical JSON-friendly record payload."""

        return {
            "endpoint_name": self.endpoint_name,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "status": self.status,
            "output_text": self.output_text,
            "elapsed_ms": self.elapsed_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
        }


@dataclass(frozen=True)
class ApiEvalResult:
    """Deterministic A/B API eval payload."""

    created_at: str
    endpoints: tuple[ApiEvalEndpoint, ...]
    prompts: tuple[ApiEvalPrompt, ...]
    max_tokens: int
    temperature: float
    mode: str
    records: tuple[ApiEvalRecord, ...]

    def to_payload(self) -> dict[str, object]:
        """Return canonical JSON-friendly result payload."""

        return {
            "created_at": self.created_at,
            "endpoints": [
                {"name": endpoint.name, "url": endpoint.url, "model": endpoint.model}
                for endpoint in self.endpoints
            ],
            "prompts": [
                {"prompt_id": prompt.prompt_id, "prompt": prompt.prompt}
                for prompt in self.prompts
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "mode": self.mode,
            "records": [record.to_payload() for record in self.records],
            "summary": summarize_api_eval_records(self.records),
        }


def load_api_eval_prompts(path: str | Path) -> tuple[ApiEvalPrompt, ...]:
    """Load prompts from JSONL records or non-empty plain text lines."""

    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise ArtifactValidationError(
            "API eval prompt file does not exist",
            details={"prompt_file": str(prompt_path)},
        )
    prompts: list[ApiEvalPrompt] = []
    for index, raw_line in enumerate(prompt_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        prompt_id = f"prompt-{index:04d}"
        prompt_text = line
        if line.startswith("{"):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ArtifactValidationError("API eval JSONL records must be objects")
            raw_prompt = payload.get("prompt")
            if not isinstance(raw_prompt, str) or not raw_prompt.strip():
                raise ArtifactValidationError("API eval JSONL record missing prompt")
            raw_id = payload.get("id", prompt_id)
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ArtifactValidationError("API eval JSONL id must be non-empty string")
            prompt_id = raw_id
            prompt_text = raw_prompt
        prompts.append(ApiEvalPrompt(prompt_id=prompt_id, prompt=prompt_text))
    if not prompts:
        raise ArtifactValidationError("API eval prompt file contains no prompts")
    return tuple(prompts)


def parse_api_eval_endpoint(value: str) -> ApiEvalEndpoint:
    """Parse ``NAME=MODEL@URL`` endpoint CLI values."""

    if "=" not in value or "@" not in value:
        raise SchemaValidationError("endpoint must use NAME=MODEL@URL format")
    name, rest = value.split("=", 1)
    model, url = rest.rsplit("@", 1)
    if not name.strip() or not model.strip() or not url.strip():
        raise SchemaValidationError("endpoint NAME, MODEL, and URL must be non-empty")
    return ApiEvalEndpoint(name=name.strip(), model=model.strip(), url=url.strip().rstrip("/"))


def run_api_eval(
    *,
    endpoints: Sequence[ApiEvalEndpoint],
    prompts: Sequence[ApiEvalPrompt],
    max_tokens: int = 64,
    temperature: float = 0.0,
    mode: str = "completion",
    timeout_seconds: float = 120.0,
) -> ApiEvalResult:
    """Run completion requests against one or more OpenAI-compatible endpoints."""

    if not endpoints:
        raise SchemaValidationError("api eval requires at least one endpoint")
    if not prompts:
        raise SchemaValidationError("api eval requires at least one prompt")
    if max_tokens <= 0:
        raise SchemaValidationError("max_tokens must be positive")
    if mode not in {"completion", "chat"}:
        raise SchemaValidationError("mode must be completion or chat")
    records: list[ApiEvalRecord] = []
    for endpoint in endpoints:
        for prompt in prompts:
            records.append(
                _run_one_completion(
                    endpoint=endpoint,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    mode=mode,
                    timeout_seconds=timeout_seconds,
                )
            )
    return ApiEvalResult(
        created_at=CANONICAL_DEFAULT_TIMESTAMP,
        endpoints=tuple(endpoints),
        prompts=tuple(prompts),
        max_tokens=max_tokens,
        temperature=temperature,
        mode=mode,
        records=tuple(records),
    )


def write_api_eval_result(path: str | Path, result: ApiEvalResult) -> Path:
    """Write a canonical API eval JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_json(result.to_payload()), encoding="utf-8")
    return output_path


def summarize_api_eval_records(records: Sequence[ApiEvalRecord]) -> dict[str, object]:
    """Return deterministic aggregate metrics grouped by endpoint."""

    by_endpoint: dict[str, list[ApiEvalRecord]] = {}
    for record in records:
        by_endpoint.setdefault(record.endpoint_name, []).append(record)
    summary: dict[str, object] = {}
    for endpoint_name in sorted(by_endpoint):
        items = by_endpoint[endpoint_name]
        ok_items = [item for item in items if item.status == "ok"]
        empty_items = [item for item in ok_items if not item.output_text.strip()]
        elapsed = [item.elapsed_ms for item in items]
        completion_tokens = [
            item.completion_tokens for item in ok_items if item.completion_tokens is not None
        ]
        output_lengths = [len(item.output_text) for item in ok_items]
        summary[endpoint_name] = {
            "record_count": len(items),
            "ok_count": len(ok_items),
            "error_count": len(items) - len(ok_items),
            "empty_output_count": len(empty_items),
            "mean_elapsed_ms": _mean(elapsed),
            "mean_completion_tokens": _mean(completion_tokens),
            "mean_output_chars": _mean(output_lengths),
        }
    return summary


def _run_one_completion(
    *,
    endpoint: ApiEvalEndpoint,
    prompt: ApiEvalPrompt,
    max_tokens: int,
    temperature: float,
    mode: str,
    timeout_seconds: float,
) -> ApiEvalRecord:
    if mode == "chat":
        payload: dict[str, object] = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt.prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{endpoint.url}/v1/chat/completions"
    else:
        payload = {
            "model": endpoint.model,
            "prompt": prompt.prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{endpoint.url}/v1/completions"
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = monotonic()
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        elapsed_ms = round((monotonic() - started) * 1000.0, 3)
        output_text = _completion_text(response_payload)
        usage = response_payload.get("usage", {}) if isinstance(response_payload, Mapping) else {}
        usage_map = usage if isinstance(usage, Mapping) else {}
        return ApiEvalRecord(
            endpoint_name=endpoint.name,
            model=endpoint.model,
            prompt_id=prompt.prompt_id,
            status="ok",
            output_text=output_text,
            elapsed_ms=elapsed_ms,
            prompt_tokens=_optional_int(usage_map.get("prompt_tokens")),
            completion_tokens=_optional_int(usage_map.get("completion_tokens")),
            total_tokens=_optional_int(usage_map.get("total_tokens")),
        )
    except HTTPError as exc:
        elapsed_ms = round((monotonic() - started) * 1000.0, 3)
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        error_message = f"{exc.__class__.__name__}: {exc}"
        if error_body:
            error_message = f"{error_message}; body={error_body}"
        return ApiEvalRecord(
            endpoint_name=endpoint.name,
            model=endpoint.model,
            prompt_id=prompt.prompt_id,
            status="error",
            output_text="",
            elapsed_ms=elapsed_ms,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error=error_message,
        )
    except Exception as exc:
        elapsed_ms = round((monotonic() - started) * 1000.0, 3)
        return ApiEvalRecord(
            endpoint_name=endpoint.name,
            model=endpoint.model,
            prompt_id=prompt.prompt_id,
            status="error",
            output_text="",
            elapsed_ms=elapsed_ms,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def _completion_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text
    message = first.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        content = str(message["content"])
        if content:
            return content
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content
    reasoning_content = first.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content
    return ""


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mean(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values)) / float(len(values)), 3)


__all__ = [
    "ApiEvalEndpoint",
    "ApiEvalPrompt",
    "ApiEvalRecord",
    "ApiEvalResult",
    "load_api_eval_prompts",
    "parse_api_eval_endpoint",
    "run_api_eval",
    "summarize_api_eval_records",
    "write_api_eval_result",
]
