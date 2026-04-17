"""Repository quality-gate runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MetricCommand:
    """One repository verification command."""

    name: str
    argv: tuple[str, ...]


_DEFAULT_CHECK_TARGET = "tests"


def _commands(*, check_target: str) -> tuple[MetricCommand, ...]:
    return (
        MetricCommand(name="ruff", argv=(sys.executable, "-m", "ruff", "check", "src", check_target)),
        MetricCommand(name="mypy", argv=(sys.executable, "-m", "mypy", "src")),
        MetricCommand(name="pytest", argv=(sys.executable, "-m", "pytest", check_target)),
    )


def _run_command(command: MetricCommand) -> int:
    print(f"$ {' '.join(command.argv)}")
    completed = subprocess.run(command.argv, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository quality gates."""

    parser = argparse.ArgumentParser(prog="python -m moe_surgeon.repo_metrics")
    parser.add_argument("--check", default=_DEFAULT_CHECK_TARGET, help="pytest target path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for command in _commands(check_target=str(args.check)):
        return_code = _run_command(command)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
