"""Repository quality-gate runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricCommand:
    """One repository verification command."""

    name: str
    argv: tuple[str, ...]


_CHECK_NAMES = ("lint", "typecheck", "tests")
_DEFAULT_CHECKS = _CHECK_NAMES


def _command_map() -> Mapping[str, MetricCommand]:
    executable = sys.executable
    return {
        "lint": MetricCommand(name="ruff", argv=(executable, "-m", "ruff", "check", "src", "tests")),
        "typecheck": MetricCommand(name="mypy", argv=(executable, "-m", "mypy", "src")),
        "tests": MetricCommand(name="pytest", argv=(executable, "-m", "pytest", "tests")),
    }


def _resolve_commands(check_name: str | None) -> tuple[MetricCommand, ...]:
    command_map = _command_map()
    if check_name is None:
        return tuple(command_map[name] for name in _DEFAULT_CHECKS)
    return (command_map[check_name],)


def _run_command(command: MetricCommand) -> int:
    print(f"$ {' '.join(command.argv)}")
    completed = subprocess.run(command.argv, check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    """Build the repo-metrics argument parser."""

    parser = argparse.ArgumentParser(prog="python -m moe_surgeon.repo_metrics")
    parser.add_argument(
        "--check",
        choices=_CHECK_NAMES,
        default=None,
        help="run only one named quality gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository quality gates."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    for command in _resolve_commands(args.check):
        return_code = _run_command(command)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
