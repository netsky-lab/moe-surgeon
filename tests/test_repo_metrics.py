from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_project_json(root: Path) -> None:
    supervisor_dir = root / ".supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    (supervisor_dir / "project.json").write_text(
        json.dumps(
            {
                "repoMetricsConfig": {
                    "lintCommand": (
                        f'{sys.executable} -c "print(\'lint-ok\')"'
                    ),
                    "typeCheckCommand": (
                        f'{sys.executable} -c "print(\'typecheck-ok\')"'
                    ),
                    "testCommand": (
                        f'{sys.executable} -c "print(\'tests-ok\')"'
                    ),
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_repo_metrics(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "moe_surgeon.repo_metrics", "--root", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


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
