"""Lightweight Click command graph for the moe-surgeon CLI."""

from __future__ import annotations

import click

from moe_surgeon import PACKAGE_NAME, __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
@click.version_option(version=__version__, package_name=PACKAGE_NAME)
def main() -> None:
    """Analyze and prune mixture-of-experts checkpoints safely."""


@main.command()
def scan() -> None:
    """Inspect static router state and emit deterministic artifacts."""

    click.echo("scan is not implemented yet")


@main.command()
def bench() -> None:
    """Profile runtime expert activation without mutating checkpoints."""

    click.echo("bench is not implemented yet")


@main.command()
def prune() -> None:
    """Build deterministic prune plans from analysis artifacts."""

    click.echo("prune is not implemented yet")


@main.command()
def export() -> None:
    """Write validated derived artifacts to a new output directory."""

    click.echo("export is not implemented yet")
