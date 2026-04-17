from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from moe_surgeon import repo_metrics


def _load_supervisor_verify_config() -> dict[str, str | None]:
    project_json = Path(__file__).resolve().parents[1] / ".supervisor" / "project.json"
    payload = json.loads(project_json.read_text(encoding="utf-8"))
    verify_config = payload.get("verifyConfig")
    assert isinstance(verify_config, dict)
    return verify_config


def _load_repo_metrics_config() -> dict[str, str | None]:
    project_json = Path(__file__).resolve().parents[1] / ".supervisor" / "project.json"
    payload = json.loads(project_json.read_text(encoding="utf-8"))
    metrics_config = payload.get("repoMetricsConfig")
    assert isinstance(metrics_config, dict)
    return metrics_config


def _write_project_json(
    root: Path,
    *,
    include_verify_config: bool = False,
    typecheck_command: str | None = None,
) -> None:
    supervisor_dir = root / ".supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    metrics_config = {
        "lintCommand": f'{sys.executable} -c "print(\'lint-ok\')"',
        "typeCheckCommand": (
            f'{sys.executable} -c "print(\'typecheck-ok\')"'
            if typecheck_command is None
            else typecheck_command
        ),
        "buildCommand": None,
        "testCommand": f'{sys.executable} -c "print(\'tests-ok\')"',
        "coverageCommand": None,
        "browserTestCommand": None,
    }
    payload: dict[str, object] = {"repoMetricsConfig": metrics_config}
    if include_verify_config:
        payload["verifyConfig"] = {
            "lintCommand": f"{sys.executable} -m moe_surgeon.repo_metrics --check lint",
            "typeCheckCommand": f"{sys.executable} -m moe_surgeon.repo_metrics --check typecheck",
            "buildCommand": None,
            "testCommand": f"{sys.executable} -m moe_surgeon.repo_metrics --check tests",
            "coverageCommand": None,
            "browserTestCommand": None,
        }
    (supervisor_dir / "project.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_repo_metrics(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "moe_surgeon.repo_metrics", "--root", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_supervisor_verify_config_uses_repo_metrics_entrypoint() -> None:
    verify_config = _load_supervisor_verify_config()
    metrics_config = _load_repo_metrics_config()

    assert verify_config == {
        "lintCommand": "python -m moe_surgeon.repo_metrics --check lint",
        "typeCheckCommand": "python -m moe_surgeon.repo_metrics --check typecheck",
        "buildCommand": None,
        "testCommand": "python -m moe_surgeon.repo_metrics --check tests",
        "coverageCommand": None,
        "browserTestCommand": None,
    }
    assert metrics_config == {
        "lintCommand": "python -m ruff check src tests",
        "typeCheckCommand": "python -m mypy src",
        "buildCommand": None,
        "testCommand": "python -m pytest",
        "coverageCommand": None,
        "browserTestCommand": None,
    }


def test_ci_workflow_runs_repo_metrics_entrypoint() -> None:
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "metrics.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert 'uses: actions/setup-python@v5' in workflow_text
    assert 'uses: actions/setup-node@v6' in workflow_text
    assert 'run: npm run metrics -- --output .supervisor/logs/ci.metrics.json' in workflow_text


def test_collect_metrics_uses_repo_supervisor_config_and_emits_typecheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    recorded_calls: list[tuple[str, str, str, int]] = []

    def fake_run_check(
        root_path: Path,
        *,
        category: str,
        name: str,
        command: str,
        timeout_seconds: int,
    ) -> repo_metrics.MetricCheck:
        assert root_path == repo_root
        recorded_calls.append((category, name, command, timeout_seconds))
        return repo_metrics.MetricCheck(
            category=category,
            name=name,
            command=command,
            exit_code=0,
            passed=True,
            duration_ms=1,
            output=f"{name} ok",
        )

    monkeypatch.setattr(repo_metrics, "_run_check", fake_run_check)

    report = repo_metrics.collect_metrics(repo_root, timeout_seconds=7)

    assert [check.name for check in report.checks] == ["lint", "typecheck", "tests"]
    assert [check.category for check in report.checks] == ["code_quality", "code_quality", "tests"]
    assert [check.command for check in report.checks] == [
        "python -m ruff check src tests",
        "python -m mypy src",
        "python -m pytest",
    ]
    assert report.summary == repo_metrics.MetricSummary(total=3, passed=3, failed=0)
    assert recorded_calls == [
        ("code_quality", "lint", "python -m ruff check src tests", 7),
        ("code_quality", "typecheck", "python -m mypy src", 7),
        ("tests", "tests", "python -m pytest", 7),
    ]


def test_repo_metrics_resolves_all_named_checks_in_deterministic_order(tmp_path: Path) -> None:
    _write_project_json(tmp_path)

    result = _run_repo_metrics(tmp_path)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [check["name"] for check in payload["checks"]] == ["lint", "typecheck", "tests"]
    assert [check["category"] for check in payload["checks"]] == [
        "code_quality",
        "code_quality",
        "tests",
    ]
    assert [check["output"] for check in payload["checks"]] == ["lint-ok", "typecheck-ok", "tests-ok"]
    assert payload["summary"] == {"total": 3, "passed": 3, "failed": 0}


def test_repo_metrics_collector_emits_named_checks_and_refreshes_task_log(tmp_path: Path) -> None:
    root_path = tmp_path
    logs_dir = root_path / ".supervisor" / "logs"
    logs_dir.mkdir(parents=True)
    task_id = "abc12345-1111-2222-3333-444444444444"
    task_log = logs_dir / "task-abc12345.log"
    task_log.write_text("", encoding="utf-8")
    _write_project_json(root_path, include_verify_config=True)
    output_path = logs_dir / "task-abc12345.metrics.json"

    result = _run_repo_metrics(
        root_path,
        "--task-id",
        task_id,
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr or result.stdout

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = payload["checks"]
    assert [check["name"] for check in checks] == ["lint", "typecheck", "tests"]
    assert [check["category"] for check in checks] == ["code_quality", "code_quality", "tests"]
    assert all(check["passed"] for check in checks)
    assert payload["summary"] == {"failed": 0, "passed": 3, "total": 3}
    assert payload["task_id"] == task_id

    log_lines = [json.loads(line) for line in task_log.read_text(encoding="utf-8").splitlines()]
    assert log_lines[-2]["message"] == "Phase ended: execution (success) — Metrics: 3/3 passed"
    assert log_lines[-1]["message"] == "Repo metrics artifact: .supervisor/logs/task-abc12345.metrics.json"


def test_repo_metrics_collector_single_check_uses_same_entrypoint(tmp_path: Path) -> None:
    _write_project_json(tmp_path, include_verify_config=True)

    result = _run_repo_metrics(tmp_path, "--check", "typecheck")

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert [check["name"] for check in payload["checks"]] == ["typecheck"]
    assert payload["summary"] == {"failed": 0, "passed": 1, "total": 1}


def test_repo_metrics_collector_single_check_fails_when_missing(tmp_path: Path) -> None:
    _write_project_json(tmp_path, include_verify_config=True, typecheck_command=None)
    project_json = tmp_path / ".supervisor" / "project.json"
    payload = json.loads(project_json.read_text(encoding="utf-8"))
    repo_metrics_config = payload["repoMetricsConfig"]
    assert isinstance(repo_metrics_config, dict)
    repo_metrics_config["typeCheckCommand"] = None
    project_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run_repo_metrics(tmp_path, "--check", "typecheck")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Requested check 'typecheck' is not configured" in result.stderr


def test_repo_metrics_runs_configured_lint_check(tmp_path: Path) -> None:
    _write_project_json(tmp_path)

    result = _run_repo_metrics(tmp_path, "--check", "lint")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [check["name"] for check in payload["checks"]] == ["lint"]
    assert payload["checks"][0]["command"] == f'{sys.executable} -c "print(\'lint-ok\')"'
    assert payload["checks"][0]["output"] == "lint-ok"
    assert payload["summary"] == {"total": 1, "passed": 1, "failed": 0}


def test_repo_metrics_runs_configured_typecheck_check(tmp_path: Path) -> None:
    _write_project_json(tmp_path)

    result = _run_repo_metrics(tmp_path, "--check", "typecheck")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [check["name"] for check in payload["checks"]] == ["typecheck"]
    assert payload["checks"][0]["command"] == f'{sys.executable} -c "print(\'typecheck-ok\')"'
    assert payload["checks"][0]["output"] == "typecheck-ok"
    assert payload["summary"] == {"total": 1, "passed": 1, "failed": 0}


def test_repo_metrics_runs_configured_tests_check(tmp_path: Path) -> None:
    _write_project_json(tmp_path)

    result = _run_repo_metrics(tmp_path, "--check", "tests")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [check["name"] for check in payload["checks"]] == ["tests"]
    assert payload["checks"][0]["command"] == f'{sys.executable} -c "print(\'tests-ok\')"'
    assert payload["checks"][0]["output"] == "tests-ok"
    assert payload["summary"] == {"total": 1, "passed": 1, "failed": 0}


def test_actual_supervisor_collector_resolves_repo_config_with_typecheck() -> None:
    collector_path = Path(
        "/home/netsky/dev/labs/codex-supervisor-rails/apps/agent/dist/metrics-collector.js"
    )
    if not collector_path.exists():
        pytest.skip("supervisor collector dist build is not available")

    project_root = Path(__file__).resolve().parents[1]
    probe = f"""
import {{ resolveVerifyConfig }} from {json.dumps(str(collector_path))};

const verifyConfig = resolveVerifyConfig(
  {{
    rootPath: {json.dumps(str(project_root))},
    verifyConfig: null,
  }}
);

console.log(JSON.stringify(verifyConfig));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload == _load_supervisor_verify_config()
    assert payload["typeCheckCommand"] == "python -m moe_surgeon.repo_metrics --check typecheck"
    resolved_checks = [
        name
        for name, command in (
            ("lint", payload["lintCommand"]),
            ("typecheck", payload["typeCheckCommand"]),
            ("test_suite", payload["testCommand"]),
        )
        if command
    ]
    assert resolved_checks == ["lint", "typecheck", "test_suite"]


def test_actual_supervisor_collector_uses_repo_local_repo_metrics_fallback_shape(tmp_path: Path) -> None:
    collector_path = Path(
        "/home/netsky/dev/labs/codex-supervisor-rails/apps/agent/dist/metrics-collector.js"
    )
    if not collector_path.exists():
        pytest.skip("supervisor collector dist build is not available")

    _write_project_json(tmp_path, include_verify_config=True)

    probe = f"""
import {{ collectMetrics }} from {json.dumps(str(collector_path))};

const metrics = await collectMetrics(
  {{
    rootPath: {json.dumps(str(tmp_path))},
    verifyConfig: null
  }},
  "taskid",
  "runid",
);

console.log(JSON.stringify(metrics));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"failed": 0, "passed": 3, "total": 3}
    assert [check["name"] for check in payload["checks"]] == ["lint", "typecheck", "test_suite"]


def test_actual_supervisor_collector_prefers_persisted_verify_config_over_repo_local_file(
    tmp_path: Path,
) -> None:
    collector_path = Path(
        "/home/netsky/dev/labs/codex-supervisor-rails/apps/agent/dist/metrics-collector.js"
    )
    if not collector_path.exists():
        pytest.skip("supervisor collector dist build is not available")

    persisted_lint_command = sys.executable + " -c \"print('lint via persisted state')\""
    persisted_test_command = sys.executable + " -c \"print('tests via persisted state')\""
    repo_lint_command = sys.executable + " -c \"print('lint via repo file')\""
    repo_typecheck_command = sys.executable + " -c \"print('typecheck via repo file')\""
    repo_test_command = sys.executable + " -c \"print('tests via repo file')\""
    supervisor_dir = tmp_path / ".supervisor"
    supervisor_dir.mkdir(parents=True)
    (supervisor_dir / "project.json").write_text(
        json.dumps(
            {
                "verifyConfig": {
                    "lintCommand": repo_lint_command,
                    "typeCheckCommand": repo_typecheck_command,
                    "buildCommand": None,
                    "testCommand": repo_test_command,
                    "coverageCommand": None,
                    "browserTestCommand": None,
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    probe = f"""
import {{ collectMetrics }} from {json.dumps(str(collector_path))};

const metrics = await collectMetrics(
  {{
    rootPath: {json.dumps(str(tmp_path))},
    verifyConfig: {{
      lintCommand: {json.dumps(persisted_lint_command)},
      typeCheckCommand: null,
      buildCommand: null,
      testCommand: {json.dumps(persisted_test_command)},
      coverageCommand: null,
      browserTestCommand: null
    }}
  }},
  "taskid",
  "runid",
);

console.log(JSON.stringify(metrics));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"failed": 0, "passed": 2, "total": 2}
    assert [check["name"] for check in payload["checks"]] == ["lint", "test_suite"]
