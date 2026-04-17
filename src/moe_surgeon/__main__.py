"""Executable module entrypoint for ``python -m moe_surgeon``."""

from __future__ import annotations

from moe_surgeon.cli.main import main


def run() -> None:
    """Invoke the Click application for module execution."""

    main(prog_name="python -m moe_surgeon")


if __name__ == "__main__":
    run()
