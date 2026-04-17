from __future__ import annotations

import subprocess
import sys

from moe_surgeon import PACKAGE_DESCRIPTION, PACKAGE_NAME, __version__
from moe_surgeon.cli.main import cli


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
