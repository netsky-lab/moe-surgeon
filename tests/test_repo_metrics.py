from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import stat
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
    lint_command: str | None = None,
    typecheck_command: str | None = None,
    test_command: str | None = None,
) -> None:
    supervisor_dir = root / ".supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    metrics_config = {
        "lintCommand": (
            f'{sys.executable} -c "print(\'lint-ok\')"'
            if lint_command is None
            else lint_command
        ),
        "typeCheckCommand": (
            f'{sys.executable} -c "print(\'typecheck-ok\')"'
            if typecheck_command is None
            else typecheck_command
        ),
        "buildCommand": None,
        "testCommand": (
            f'{sys.executable} -c "print(\'tests-ok\')"'
            if test_command is None
            else test_command
        ),
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


def _run_repo_metrics(
    root: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "moe_surgeon.repo_metrics", "--root", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _run_repo_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )


def _check_repo_ignore(*paths: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "check-ignore", "-v", "--stdin"],
        input="".join(f"{path}\n" for path in paths),
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )


def _list_tracked_repo_paths(*paths: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return result.stdout.splitlines()


def _write_quality_gate_fixture_repo(
    root: Path,
    *,
    include_supervisor_config: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    package_dir = src_dir / "moe_surgeon"
    package_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "sitecustomize.py").write_text(
        (repo_root / "src" / "sitecustomize.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text(
        (repo_root / "src" / "moe_surgeon" / "__init__.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (package_dir / "test_env.py").write_text(
        (repo_root / "src" / "moe_surgeon" / "test_env.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (package_dir / "repo_metrics.py").write_text(
        (repo_root / "src" / "moe_surgeon" / "repo_metrics.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "AGENTS.md").write_text("# Fixture AGENTS\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """
[tool.mypy]
python_version = "3.11"
incremental = false

[tool.pytest.ini_options]
addopts = ["--disable-plugin-autoload", "--basetemp=.tmp/pytest"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_project_json(root, include_verify_config=include_supervisor_config)
    (root / "sample.py").write_text(
        """
def answer() -> int:
    return 42
""".strip()
        + "\n",
        encoding="utf-8",
    )
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_sample.py").write_text(
        """
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from sample import answer


def test_answer() -> None:
    assert answer() == 42


def test_bootstrap_tempdir() -> None:
    expected = Path(os.environ["TMPDIR"])
    assert Path(tempfile.gettempdir()) == expected
    assert expected.name in {"system", "pytest"}
    assert expected.is_dir()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    if include_supervisor_config:
        _write_project_json(
            root,
            include_verify_config=True,
            lint_command="python -m ruff check sample.py tests",
            typecheck_command="python -m mypy sample.py",
            test_command="python -m pytest tests/test_sample.py -q",
        )


def _make_read_only_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def _quality_gate_env(root: Path) -> dict[str, str]:
    broken_temp = str(root / "missing-temp-root")
    python_path_entries = [str(root / "src")]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(python_path_entries),
        "TMPDIR": broken_temp,
        "TMP": broken_temp,
        "TEMP": broken_temp,
    }


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


def test_repo_pytest_config_disables_plugin_autoload_and_uses_repo_basetemp() -> None:
    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'pytest>=8.4' in pyproject_text
    assert '--disable-plugin-autoload' in pyproject_text
    assert '--basetemp=.tmp/pytest' in pyproject_text
    assert '--strict-markers' in pyproject_text
    assert 'not integration' in pyproject_text
    assert 'incremental = false' in pyproject_text
    assert 'integration: opt-in tests that may access live external services or model artifacts' in (
        pyproject_text
    )


def test_repo_gitignore_quarantines_quality_gate_and_supervisor_artifacts() -> None:
    result = _check_repo_ignore(
        ".tmp/pytest/test.txt",
        ".tmp/system/cache.txt",
        ".tmp/quality-gates-direct/metrics.json",
        ".tmp/quality-gates-metrics/run/stdout.txt",
        ".supervisor/logs/task-0d36df06.log",
        ".supervisor/logs/task-0d36df06.metrics.json",
        ".supervisor/project.json",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    output_lines = result.stdout.splitlines()
    assert any("/.tmp/*" in line and ".tmp/pytest/test.txt" in line for line in output_lines)
    assert any("/.tmp/*" in line and ".tmp/system/cache.txt" in line for line in output_lines)
    assert any(
        "/.tmp/*" in line and ".tmp/quality-gates-direct/metrics.json" in line
        for line in output_lines
    )
    assert any(
        "/.tmp/*" in line and ".tmp/quality-gates-metrics/run/stdout.txt" in line
        for line in output_lines
    )
    assert any(".supervisor/" in line and ".supervisor/logs/task-0d36df06.log" in line for line in output_lines)
    assert any(
        ".supervisor/" in line and ".supervisor/logs/task-0d36df06.metrics.json" in line
        for line in output_lines
    )
    assert any(".supervisor/" in line and ".supervisor/project.json" in line for line in output_lines)

    gitkeep_result = _check_repo_ignore(".tmp/.gitkeep")
    assert gitkeep_result.returncode == 1
    assert gitkeep_result.stdout == ""


def test_repo_index_no_longer_tracks_transient_supervisor_logs() -> None:
    assert _list_tracked_repo_paths(".supervisor/logs", "ci.metrics.json") == []
    assert _list_tracked_repo_paths(".tmp") == [".tmp/.gitkeep"]


def test_default_python_m_pytest_deselects_integration_marker() -> None:
    result = _run_repo_pytest(
        "-q",
        "tests/test_runtime_profiler.py",
        "-k",
        (
            "test_router_activation_profiler_matches_live_gemma4_router_contract "
            "or test_live_gemma4_signature_preserves_pinned_revision"
        ),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    combined_output = result.stdout + result.stderr
    assert "1 passed" in combined_output
    assert "19 deselected" in combined_output


def test_explicit_python_m_pytest_m_integration_selects_live_gemma4_test() -> None:
    result = _run_repo_pytest(
        "--collect-only",
        "-q",
        "-m",
        "integration",
        "tests/test_runtime_profiler.py",
        "-k",
        "test_router_activation_profiler_matches_live_gemma4_router_contract",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    combined_output = result.stdout + result.stderr
    assert "test_router_activation_profiler_matches_live_gemma4_router_contract" in combined_output
    assert "1/20 tests collected (19 deselected)" in combined_output


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


@pytest.mark.parametrize(
    ("check_name", "config_key"),
    [
        ("lint", "lintCommand"),
        ("typecheck", "typeCheckCommand"),
    ],
)
def test_repo_metrics_collector_single_check_fails_when_missing(
    tmp_path: Path,
    check_name: str,
    config_key: str,
) -> None:
    _write_project_json(tmp_path, include_verify_config=True)
    project_json = tmp_path / ".supervisor" / "project.json"
    payload = json.loads(project_json.read_text(encoding="utf-8"))
    repo_metrics_config = payload["repoMetricsConfig"]
    assert isinstance(repo_metrics_config, dict)
    repo_metrics_config[config_key] = None
    project_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run_repo_metrics(tmp_path, "--check", check_name)

    assert result.returncode != 0
    assert result.stdout == ""
    assert f"Requested check '{check_name}' is not configured" in result.stderr


def test_repo_metrics_reports_missing_project_json_cleanly(tmp_path: Path) -> None:
    result = _run_repo_metrics(tmp_path, "--check", "typecheck")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "Missing .supervisor/project.json" in result.stderr
    assert "FileNotFoundError" not in result.stderr


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


def test_repo_metrics_tests_check_uses_isolated_env_and_repo_tempdir(tmp_path: Path) -> None:
    temp_probe = (
        "import json, os, tempfile; "
        "print(json.dumps({"
        "'autoload': os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'), "
        "'tmpdir': tempfile.gettempdir(), "
        "'TMPDIR': os.environ.get('TMPDIR'), "
        "'TMP': os.environ.get('TMP'), "
        "'TEMP': os.environ.get('TEMP')"
        "}, sort_keys=True))"
    )
    _write_project_json(
        tmp_path,
        test_command=f"{shlex.quote(sys.executable)} -c {shlex.quote(temp_probe)}",
    )
    broken_temp = str(tmp_path / "missing-temp-root")
    env = {
        **os.environ,
        "TMPDIR": broken_temp,
        "TMP": broken_temp,
        "TEMP": broken_temp,
    }

    result = _run_repo_metrics(tmp_path, "--check", "tests", env=env)

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    check_payload = json.loads(payload["checks"][0]["output"])
    expected_tempdir = str(tmp_path / ".tmp" / "pytest")

    assert check_payload == {
        "TEMP": expected_tempdir,
        "TMP": expected_tempdir,
        "TMPDIR": expected_tempdir,
        "autoload": "1",
        "tmpdir": expected_tempdir,
    }
    assert payload["summary"] == {"total": 1, "passed": 1, "failed": 0}
    assert (tmp_path / ".tmp" / "pytest").is_dir()


def test_repo_metrics_tests_check_reports_repo_test_failures_not_startup_noise(
    tmp_path: Path,
) -> None:
    failing_probe = (
        "import os, pathlib, sys, tempfile; "
        "expected = pathlib.Path(os.environ['TMPDIR']); "
        "assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'; "
        "assert pathlib.Path(tempfile.gettempdir()) == expected; "
        "assert expected.is_dir(); "
        "sys.stderr.write('repo failure\\n'); "
        "raise SystemExit(5)"
    )
    _write_project_json(
        tmp_path,
        test_command=f"{shlex.quote(sys.executable)} -c {shlex.quote(failing_probe)}",
    )
    broken_temp = str(tmp_path / "missing-temp-root")
    env = {
        **os.environ,
        "TMPDIR": broken_temp,
        "TMP": broken_temp,
        "TEMP": broken_temp,
    }

    result = _run_repo_metrics(tmp_path, "--check", "tests", env=env)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"total": 1, "passed": 0, "failed": 1}
    assert payload["checks"][0]["exit_code"] == 5
    assert payload["checks"][0]["output"] == "repo failure"
    assert "FileNotFoundError" not in payload["checks"][0]["output"]
    assert "plugin" not in payload["checks"][0]["output"].lower()


def test_direct_python_m_pytest_disables_ambient_plugins_and_uses_repo_tempdir() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    broken_temp = str(repo_root / "missing-temp-root")
    env = {
        **os.environ,
        "TMPDIR": broken_temp,
        "TMP": broken_temp,
        "TEMP": broken_temp,
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--trace-config",
            "tests/test_cli.py",
            "-k",
            "version",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    combined_output = result.stdout + result.stderr
    assert "django-" not in combined_output
    assert "langsmith-" not in combined_output
    assert "benchmark-" not in combined_output
    assert (repo_root / ".tmp" / "pytest").is_dir()


def test_direct_quality_gate_commands_are_hermetic_under_hostile_temp_and_cache_env(
    tmp_path: Path,
) -> None:
    _write_quality_gate_fixture_repo(tmp_path)
    _make_read_only_directory(tmp_path / ".ruff_cache")
    _make_read_only_directory(tmp_path / ".mypy_cache")
    env = _quality_gate_env(tmp_path)

    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_sample.py", "-q"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    ruff_result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "sample.py", "tests"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    mypy_result = subprocess.run(
        [sys.executable, "-m", "mypy", "sample.py"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert pytest_result.returncode == 0, pytest_result.stderr or pytest_result.stdout
    assert ruff_result.returncode == 0, ruff_result.stderr or ruff_result.stdout
    assert mypy_result.returncode == 0, mypy_result.stderr or mypy_result.stdout
    assert (tmp_path / ".tmp" / "system").is_dir()
    assert list((tmp_path / ".ruff_cache").iterdir()) == []
    assert list((tmp_path / ".mypy_cache").iterdir()) == []


def test_repo_metrics_quality_commands_are_hermetic_under_hostile_temp_and_cache_env(
    tmp_path: Path,
) -> None:
    _write_quality_gate_fixture_repo(tmp_path, include_supervisor_config=True)
    _make_read_only_directory(tmp_path / ".ruff_cache")
    _make_read_only_directory(tmp_path / ".mypy_cache")
    env = _quality_gate_env(tmp_path)

    result = _run_repo_metrics(tmp_path, env=env)

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert [check["name"] for check in payload["checks"]] == ["lint", "typecheck", "tests"]
    assert payload["summary"] == {"total": 3, "passed": 3, "failed": 0}
    assert (tmp_path / ".tmp" / "system").is_dir()
    assert list((tmp_path / ".ruff_cache").iterdir()) == []
    assert list((tmp_path / ".mypy_cache").iterdir()) == []


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
