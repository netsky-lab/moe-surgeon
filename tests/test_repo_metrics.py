from __future__ import annotations

import json
from pathlib import Path
import subprocess


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
