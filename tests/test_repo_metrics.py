from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _load_package_scripts() -> dict[str, str]:
    package_json = Path(__file__).resolve().parents[1] / "package.json"
    payload = json.loads(package_json.read_text(encoding="utf-8"))
    scripts = payload.get("scripts")
    assert isinstance(scripts, dict)
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in scripts.items())
    return scripts


def _load_supervisor_verify_config() -> dict[str, str | None]:
    project_json = Path(__file__).resolve().parents[1] / ".supervisor" / "project.json"
    payload = json.loads(project_json.read_text(encoding="utf-8"))
    verify_config = payload.get("verifyConfig")
    assert isinstance(verify_config, dict)
    return verify_config


def test_supervisor_verify_config_collects_lint_typecheck_and_tests() -> None:
    verify_config = _load_supervisor_verify_config()

    assert verify_config == {
        "lintCommand": "npm run lint",
        "typeCheckCommand": "npm run typecheck",
        "buildCommand": None,
        "testCommand": "npm test",
        "coverageCommand": None,
        "browserTestCommand": None,
    }


def test_supervisor_typecheck_command_maps_to_package_script_and_succeeds() -> None:
    scripts = _load_package_scripts()
    verify_config = _load_supervisor_verify_config()

    assert scripts["typecheck"] == "python -m mypy src"
    assert scripts["metrics"] == "python -m moe_surgeon.repo_metrics"
    assert verify_config["typeCheckCommand"] == "npm run typecheck"

    result = subprocess.run(
        ["npm", "run", "typecheck"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined_output
    assert "python -m mypy src" in combined_output
    assert "Success: no issues found" in combined_output


def test_repo_metrics_collector_emits_named_checks_and_refreshes_task_log(tmp_path: Path) -> None:
    root_path = tmp_path
    supervisor_dir = root_path / ".supervisor"
    logs_dir = supervisor_dir / "logs"
    logs_dir.mkdir(parents=True)
    task_id = "abc12345-1111-2222-3333-444444444444"
    task_log = logs_dir / "task-abc12345.log"
    task_log.write_text("", encoding="utf-8")
    project_json = supervisor_dir / "project.json"
    project_json.write_text(
        json.dumps(
            {
                "verifyConfig": {
                    "lintCommand": f"{sys.executable} -c \"print('lint ok')\"",
                    "typeCheckCommand": f"{sys.executable} -c \"print('typecheck ok')\"",
                    "buildCommand": None,
                    "testCommand": f"{sys.executable} -c \"print('tests ok')\"",
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
    output_path = logs_dir / "task-abc12345.metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "moe_surgeon.repo_metrics",
            "--root",
            str(root_path),
            "--task-id",
            task_id,
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
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
