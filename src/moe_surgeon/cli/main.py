"""Lightweight Click command graph for the moe-surgeon CLI."""

from __future__ import annotations

import click

from moe_surgeon import PACKAGE_DESCRIPTION, PACKAGE_NAME, __version__


@click.group(
    name=PACKAGE_NAME,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=PACKAGE_DESCRIPTION,
    no_args_is_help=True,
)
@click.version_option(version=__version__, package_name=PACKAGE_NAME)
def cli() -> None:
    """Top-level command group for moe-surgeon."""


@cli.command()
def scan() -> None:
    """Inspect static router state and emit canonical scan artifacts."""

    click.echo("scan CLI wiring is not implemented yet; canonical scan artifact helpers are available in moe_surgeon.analysis.scan")


@cli.command()
def bench() -> None:
    """Profile runtime expert activation without mutating checkpoints."""

    click.echo("bench is not implemented yet")


@cli.command()
def prune() -> None:
    """Build deterministic prune plans from analysis artifacts."""

    click.echo("prune is not implemented yet")


@cli.command()
def export() -> None:
    """Write validated derived artifacts to a new output directory."""

    click.echo("export is not implemented yet")


def main(*, prog_name: str | None = None) -> None:
    """Invoke the lightweight CLI without importing model backends."""

    cli(prog_name=prog_name)
