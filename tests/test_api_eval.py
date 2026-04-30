from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from click.testing import CliRunner

from moe_surgeon.cli.main import cli
from moe_surgeon.runtime.api_eval import (
    ApiEvalEndpoint,
    ApiEvalPrompt,
    load_api_eval_prompts,
    parse_api_eval_endpoint,
    run_api_eval,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self._payload = payload or {
            "choices": [{"text": "ok"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_load_api_eval_prompts_accepts_jsonl_and_plain_lines(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"id":"a","prompt":"Hello"}\nPlain prompt\n', encoding="utf-8")

    prompts = load_api_eval_prompts(prompt_file)

    assert prompts == (
        ApiEvalPrompt(prompt_id="a", prompt="Hello"),
        ApiEvalPrompt(prompt_id="prompt-0002", prompt="Plain prompt"),
    )


def test_parse_api_eval_endpoint_uses_name_model_url_contract() -> None:
    endpoint = parse_api_eval_endpoint("p64=gemma-pruned@http://127.0.0.1:8090")

    assert endpoint == ApiEvalEndpoint(
        name="p64",
        model="gemma-pruned",
        url="http://127.0.0.1:8090",
    )


def test_run_api_eval_records_completion_payload(monkeypatch: Any) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        assert timeout == 3.0
        return _FakeResponse()

    monkeypatch.setattr("moe_surgeon.runtime.api_eval.request.urlopen", fake_urlopen)

    result = run_api_eval(
        endpoints=(ApiEvalEndpoint(name="p64", model="m", url="http://127.0.0.1:8090"),),
        prompts=(ApiEvalPrompt(prompt_id="p1", prompt="Hi"),),
        max_tokens=4,
        timeout_seconds=3.0,
    )

    assert len(result.records) == 1
    assert result.records[0].status == "ok"
    assert result.records[0].output_text == "ok"
    assert result.records[0].total_tokens == 3
    assert result.mode == "completion"


def test_run_api_eval_uses_reasoning_content_fallback(monkeypatch: Any) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse({"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]})

    monkeypatch.setattr("moe_surgeon.runtime.api_eval.request.urlopen", fake_urlopen)

    result = run_api_eval(
        endpoints=(ApiEvalEndpoint(name="original", model="m", url="http://127.0.0.1:8080"),),
        prompts=(ApiEvalPrompt(prompt_id="p1", prompt="Hi"),),
        mode="chat",
    )

    assert result.records[0].output_text == "thinking"


def test_run_api_eval_records_http_error_body(monkeypatch: Any) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        raise HTTPError(
            url="http://127.0.0.1:8090/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs=Message(),
            fp=BytesIO(b'{"error":{"message":"parse failed"}}'),
        )

    monkeypatch.setattr("moe_surgeon.runtime.api_eval.request.urlopen", fake_urlopen)

    result = run_api_eval(
        endpoints=(ApiEvalEndpoint(name="p64", model="m", url="http://127.0.0.1:8090"),),
        prompts=(ApiEvalPrompt(prompt_id="p1", prompt="Hi"),),
        mode="chat",
    )

    assert result.records[0].status == "error"
    assert result.records[0].error is not None
    assert "HTTP Error 500" in result.records[0].error
    assert "parse failed" in result.records[0].error


def test_cli_api_eval_writes_result_json(tmp_path: Path, monkeypatch: Any) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr("moe_surgeon.runtime.api_eval.request.urlopen", fake_urlopen)
    prompt_file = tmp_path / "prompts.txt"
    output_path = tmp_path / "eval.json"
    prompt_file.write_text("Hello\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "api-eval",
            "--endpoint",
            "p64=gemma@http://127.0.0.1:8090",
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output_path),
            "--max-tokens",
            "4",
            "--mode",
            "chat",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ok_count=1" in result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "chat"
    assert payload["records"][0]["output_text"] == "ok"
    assert payload["summary"]["p64"]["ok_count"] == 1
    assert payload["summary"]["p64"]["empty_output_count"] == 0
