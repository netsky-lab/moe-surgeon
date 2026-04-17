"""Executable module entrypoint for ``python -m moe_surgeon``."""

from __future__ import annotations

from moe_surgeon.cli.main import main as cli_main


def main() -> None:
    """Invoke the lightweight Click application for module execution."""

    cli_main(prog_name="python -m moe_surgeon")


if __name__ == "__main__":
    main()
