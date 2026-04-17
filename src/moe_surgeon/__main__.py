"""Package entrypoint for ``python -m moe_surgeon``."""

from __future__ import annotations

from . import __version__


def main() -> int:
    message = (
        "moe-surgeon is installed as a package; module execution is available.\n"
        f"Version: {__version__}\n"
        "Use dedicated CLI command wiring when P2 is enabled."
    )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
