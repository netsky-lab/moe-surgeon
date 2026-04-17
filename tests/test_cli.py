from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from moe_surgeon import PACKAGE_DESCRIPTION, PACKAGE_NAME, __version__
from moe_surgeon.__main__ import main as module_main
from moe_surgeon.cli.main import cli

FORBIDDEN_RUNTIME_MODULES = ("torch", "transformers", "safetensors")


def _run_cli_with_import_probe(tmp_path: Path, command: list[str]) -> tuple[subprocess.CompletedProcess[str], set[str]]:
    probe_path = tmp_path / "import-probe"
    probe_path.mkdir()
    output_path = probe_path / "imports.json"
    sitecustomize_path = probe_path / "sitecustomize.py"
    sitecustomize_path.write_text(
        """
from __future__ import annotations

import atexit
import json
import os
import sys


def _dump_imports() -> None:
    output_path = os.environ["MOE_SURGEON_IMPORT_PROBE_OUTPUT"]
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(sys.modules), handle)


atexit.register(_dump_imports)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{probe_path}{os.pathsep}{python_path}" if python_path else str(probe_path)
    )
    env["MOE_SURGEON_IMPORT_PROBE_OUTPUT"] = str(output_path)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    imported_modules = set(json.loads(output_path.read_text(encoding="utf-8")))
    return result, imported_modules


def test_click_group_metadata_is_exposed_without_runtime_imports() -> None:
    assert cli.name == PACKAGE_NAME
    assert PACKAGE_DESCRIPTION == "Python CLI for analyzing and pruning MoE models"
    assert __version__ == "0.1.0"


def test_main_wrapper_runs_click_group_help() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from moe_surgeon.cli.main import main; main()", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "scan" in result.stdout
    assert "bench" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout


def test_module_help_lists_placeholder_subcommands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "moe_surgeon", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage: python -m moe_surgeon" in result.stdout
    assert "scan" in result.stdout
    assert "bench" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout


def test_module_main_wrapper_runs_click_group_help() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from moe_surgeon.__main__ import main; main()", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage: python -m moe_surgeon" in result.stdout
    assert "scan" in result.stdout
    assert "bench" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout


def test_cli_help_does_not_import_heavy_dependencies() -> None:
    probe = """
import sys
from click.testing import CliRunner
from moe_surgeon.cli.main import cli

result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0, result.output
forbidden = [name for name in ("torch", "transformers", "safetensors") if name in sys.modules]
assert not forbidden, forbidden
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_module_main_is_importable() -> None:
    assert callable(module_main)


@pytest.mark.parametrize(
    ("command", "expected_output"),
    [
        ([sys.executable, "-m", "moe_surgeon", "--version"], "python -m moe_surgeon, version 0.1.0"),
        (["moe-surgeon", "--version"], "moe-surgeon, version 0.1.0"),
    ],
)
def test_cli_version_commands_are_lightweight(
    tmp_path: Path, command: list[str], expected_output: str
) -> None:
    if command[0] == "moe-surgeon":
        script_path = shutil.which(command[0])
        assert script_path is not None
        command = [script_path, *command[1:]]

    result, imported_modules = _run_cli_with_import_probe(tmp_path, command)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output
    forbidden = [
        module_name
        for module_name in imported_modules
        if any(
            module_name == forbidden_name or module_name.startswith(f"{forbidden_name}.")
            for forbidden_name in FORBIDDEN_RUNTIME_MODULES
        )
    ]
    assert not forbidden, forbidden


def test_bench_command_accepts_prompt_batching_options() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from moe_surgeon.cli.main import main; main()",
            "bench",
            "--prompt",
            "alpha",
            "--prompt",
            "beta",
            "--batch-size",
            "2",
            "--seed",
            "7",
            "--capture-router-scores",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "prompt_inputs=2" in result.stdout
    assert "prompt_batches=1" in result.stdout
    assert "batch_size=2" in result.stdout
    assert "seed=7" in result.stdout
    assert "capture_router_scores=true" in result.stdout
