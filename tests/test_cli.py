from __future__ import annotations

import subprocess
import sys


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
from moe_surgeon.cli.main import main

result = CliRunner().invoke(main, ["--help"])
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
