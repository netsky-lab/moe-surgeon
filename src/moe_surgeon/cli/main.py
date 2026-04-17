"""Lightweight Click command graph for the moe-surgeon CLI."""

from __future__ import annotations

from pathlib import Path

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
@click.option("--prompt", "prompts", multiple=True, help="Prompt text to profile. Repeat for batches.")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--capture-router-scores/--no-capture-router-scores", default=False, show_default=True)
def bench(
    prompts: tuple[str, ...],
    prompt_file: str | None,
    batch_size: int,
    seed: int,
    capture_router_scores: bool,
) -> None:
    """Profile runtime expert activation without mutating checkpoints."""

    prompt_inputs = list(prompts)
    if prompt_file is not None:
        prompt_inputs.extend(_load_prompt_file(prompt_file))
    prompt_count = len(prompt_inputs)
    batch_count = (prompt_count + batch_size - 1) // batch_size if prompt_count > 0 else 0
    click.echo(
        "bench is not implemented yet "
        f"(prompt_inputs={prompt_count}, prompt_batches={batch_count}, batch_size={batch_size}, seed={seed}, "
        f"capture_router_scores={str(capture_router_scores).lower()})"
    )


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


def _load_prompt_file(prompt_file: str) -> list[str]:
    """Load newline-delimited prompts from disk for lightweight CLI validation."""

    return [line for line in Path(prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
