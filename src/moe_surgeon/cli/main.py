"""Top-level Click command graph for moe-surgeon workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import click

from moe_surgeon import PACKAGE_DESCRIPTION, PACKAGE_NAME, __version__

F = TypeVar("F", bound=Callable[..., object])


@dataclass(frozen=True)
class CliContext:
    """Shared root CLI options propagated to all subcommands."""

    model_id: str | None
    source_path: Path | None
    revision: str | None
    backend_name: str | None
    dtype: str | None
    seed: int | None
    artifact_root: Path | None


@dataclass(frozen=True)
class ScanCommandRequest:
    """Parsed options for the ``scan`` command."""

    shared: CliContext
    output_path: Path | None


@dataclass(frozen=True)
class BenchCommandRequest:
    """Parsed options for the ``bench`` command."""

    shared: CliContext
    prompts: tuple[str, ...]
    batch_size: int
    capture_router_scores: bool
    output_path: Path | None


@dataclass(frozen=True)
class PruneCommandRequest:
    """Parsed options for the ``prune`` command."""

    shared: CliContext
    scan_artifact: Path | None
    bench_artifact: Path | None
    strategy: str
    output_dir: Path | None


@dataclass(frozen=True)
class ExportCommandRequest:
    """Parsed options for the ``export`` command."""

    shared: CliContext
    apply_artifact_dir: Path | None
    output_dir: Path | None


def _path_text(path: Path | None) -> str:
    return "-" if path is None else str(path)


def _root_option(*param_decls: str, **kwargs: Any) -> Callable[[F], F]:
    return click.option(*param_decls, show_default=True, **kwargs)


def _shared_root_options(command: F) -> F:
    command = _root_option(
        "--artifact-root",
        type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=False),
        default=None,
        help="Root directory for workflow artifacts.",
    )(command)
    command = _root_option(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed propagated into command workflows.",
    )(command)
    command = _root_option(
        "--dtype",
        type=str,
        default=None,
        help="Explicit model dtype override.",
    )(command)
    command = _root_option(
        "--backend",
        "backend_name",
        type=str,
        default=None,
        help="Explicit backend override.",
    )(command)
    command = _root_option(
        "--revision",
        type=str,
        default=None,
        help="Model revision or checkpoint snapshot identifier.",
    )(command)
    command = _root_option(
        "--source-path",
        type=click.Path(path_type=Path, file_okay=True, dir_okay=True, resolve_path=False),
        default=None,
        help="Local checkpoint path.",
    )(command)
    command = _root_option(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier for backend resolution.",
    )(command)
    return command


def _shared_command_options(command: F) -> F:
    return _shared_root_options(command)


def _resolve_shared_context(
    parent: CliContext | None,
    *,
    model_id: str | None,
    source_path: Path | None,
    revision: str | None,
    backend_name: str | None,
    dtype: str | None,
    seed: int | None,
    artifact_root: Path | None,
) -> CliContext:
    base = parent or CliContext(
        model_id=None,
        source_path=None,
        revision=None,
        backend_name=None,
        dtype=None,
        seed=None,
        artifact_root=None,
    )
    return CliContext(
        model_id=base.model_id if model_id is None else model_id,
        source_path=base.source_path if source_path is None else source_path,
        revision=base.revision if revision is None else revision,
        backend_name=base.backend_name if backend_name is None else backend_name,
        dtype=base.dtype if dtype is None else dtype,
        seed=base.seed if seed is None else seed,
        artifact_root=base.artifact_root if artifact_root is None else artifact_root,
    )


def _run_scan(request: ScanCommandRequest) -> int:
    click.echo("command=scan")
    click.echo(f"model_id={request.shared.model_id or '-'}")
    click.echo(f"source_path={_path_text(request.shared.source_path)}")
    click.echo(f"output_path={_path_text(request.output_path)}")
    return 0


def _run_bench(request: BenchCommandRequest) -> int:
    prompt_batches = 0
    if request.prompts:
        prompt_batches = (len(request.prompts) + request.batch_size - 1) // request.batch_size
    click.echo("command=bench")
    click.echo(f"prompt_inputs={len(request.prompts)}")
    click.echo(f"prompt_batches={prompt_batches}")
    click.echo(f"batch_size={request.batch_size}")
    click.echo(f"seed={request.shared.seed if request.shared.seed is not None else '-'}")
    click.echo(
        "capture_router_scores="
        f"{'true' if request.capture_router_scores else 'false'}"
    )
    click.echo(f"output_path={_path_text(request.output_path)}")
    return 0


def _run_prune(request: PruneCommandRequest) -> int:
    click.echo("command=prune")
    click.echo(f"scan_artifact={_path_text(request.scan_artifact)}")
    click.echo(f"bench_artifact={_path_text(request.bench_artifact)}")
    click.echo(f"strategy={request.strategy}")
    click.echo(f"output_dir={_path_text(request.output_dir)}")
    return 0


def _run_export(request: ExportCommandRequest) -> int:
    click.echo("command=export")
    click.echo(f"apply_artifact_dir={_path_text(request.apply_artifact_dir)}")
    click.echo(f"output_dir={_path_text(request.output_dir)}")
    return 0


@click.group(name=PACKAGE_NAME, help=PACKAGE_DESCRIPTION)
@click.version_option(version=__version__)
@_shared_root_options
@click.pass_context
def cli(
    ctx: click.Context,
    model_id: str | None,
    source_path: Path | None,
    revision: str | None,
    backend_name: str | None,
    dtype: str | None,
    seed: int | None,
    artifact_root: Path | None,
) -> None:
    """Root command group for deterministic MoE workflows."""

    ctx.obj = CliContext(
        model_id=model_id,
        source_path=source_path,
        revision=revision,
        backend_name=backend_name,
        dtype=dtype,
        seed=seed,
        artifact_root=artifact_root,
    )


@cli.command("scan")
@_shared_command_options
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False, resolve_path=False),
    default=None,
    help="Output path for the scan artifact JSON.",
)
@click.pass_context
def scan_command(
    ctx: click.Context,
    model_id: str | None,
    source_path: Path | None,
    revision: str | None,
    backend_name: str | None,
    dtype: str | None,
    seed: int | None,
    artifact_root: Path | None,
    output_path: Path | None,
) -> None:
    """Run static router analysis."""

    shared = _resolve_shared_context(
        ctx.find_object(CliContext),
        model_id=model_id,
        source_path=source_path,
        revision=revision,
        backend_name=backend_name,
        dtype=dtype,
        seed=seed,
        artifact_root=artifact_root,
    )
    _run_scan(ScanCommandRequest(shared=shared, output_path=output_path))


@cli.command("bench")
@_shared_command_options
@click.option(
    "--prompt",
    "prompts",
    type=str,
    multiple=True,
    help="Prompt text to profile; repeatable.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="Prompt batch size for profiling input construction.",
)
@click.option(
    "--capture-router-scores/--no-capture-router-scores",
    default=False,
    help="Capture router score tensors in addition to top-k routes.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False, resolve_path=False),
    default=None,
    help="Output path for the benchmark artifact JSON.",
)
@click.pass_context
def bench_command(
    ctx: click.Context,
    model_id: str | None,
    source_path: Path | None,
    revision: str | None,
    backend_name: str | None,
    dtype: str | None,
    seed: int | None,
    artifact_root: Path | None,
    prompts: tuple[str, ...],
    batch_size: int,
    capture_router_scores: bool,
    output_path: Path | None,
) -> None:
    """Run runtime routing benchmarks."""

    shared = _resolve_shared_context(
        ctx.find_object(CliContext),
        model_id=model_id,
        source_path=source_path,
        revision=revision,
        backend_name=backend_name,
        dtype=dtype,
        seed=seed,
        artifact_root=artifact_root,
    )
    _run_bench(
        BenchCommandRequest(
            shared=shared,
            prompts=prompts,
            batch_size=batch_size,
            capture_router_scores=capture_router_scores,
            output_path=output_path,
        )
    )


@cli.command("prune")
@_shared_command_options
@click.option(
    "--scan-artifact",
    type=click.Path(path_type=Path, dir_okay=False, exists=False, resolve_path=False),
    default=None,
    help="Scan artifact consumed by prune planning.",
)
@click.option(
    "--bench-artifact",
    type=click.Path(path_type=Path, dir_okay=False, exists=False, resolve_path=False),
    default=None,
    help="Benchmark artifact consumed by prune planning.",
)
@click.option(
    "--strategy",
    type=click.Choice(("frequency", "router_mass", "combined"), case_sensitive=False),
    default="combined",
    help="Deterministic prune strategy name.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=False),
    default=None,
    help="Destination directory for prune artifacts.",
)
@click.pass_context
def prune_command(
    ctx: click.Context,
    model_id: str | None,
    source_path: Path | None,
    revision: str | None,
    backend_name: str | None,
    dtype: str | None,
    seed: int | None,
    artifact_root: Path | None,
    scan_artifact: Path | None,
    bench_artifact: Path | None,
    strategy: str,
    output_dir: Path | None,
) -> None:
    """Run prune planning and apply orchestration."""

    shared = _resolve_shared_context(
        ctx.find_object(CliContext),
        model_id=model_id,
        source_path=source_path,
        revision=revision,
        backend_name=backend_name,
        dtype=dtype,
        seed=seed,
        artifact_root=artifact_root,
    )
    _run_prune(
        PruneCommandRequest(
            shared=shared,
            scan_artifact=scan_artifact,
            bench_artifact=bench_artifact,
            strategy=strategy.lower(),
            output_dir=output_dir,
        )
    )


@cli.command("export")
@_shared_command_options
@click.option(
    "--apply-artifact-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=False),
    default=None,
    help="Prune-apply artifact directory to export.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=False),
    default=None,
    help="Destination directory for exported checkpoint artifacts.",
)
@click.pass_context
def export_command(
    ctx: click.Context,
    model_id: str | None,
    source_path: Path | None,
    revision: str | None,
    backend_name: str | None,
    dtype: str | None,
    seed: int | None,
    artifact_root: Path | None,
    apply_artifact_dir: Path | None,
    output_dir: Path | None,
) -> None:
    """Run deterministic export from apply artifacts."""

    shared = _resolve_shared_context(
        ctx.find_object(CliContext),
        model_id=model_id,
        source_path=source_path,
        revision=revision,
        backend_name=backend_name,
        dtype=dtype,
        seed=seed,
        artifact_root=artifact_root,
    )
    _run_export(
        ExportCommandRequest(
            shared=shared,
            apply_artifact_dir=apply_artifact_dir,
            output_dir=output_dir,
        )
    )


def main(args: Sequence[str] | None = None, *, prog_name: str | None = None) -> int | None:
    """Execute the CLI entrypoint."""

    return cli.main(args=list(args) if args is not None else None, prog_name=prog_name)


__all__ = [
    "BenchCommandRequest",
    "CliContext",
    "ExportCommandRequest",
    "PruneCommandRequest",
    "ScanCommandRequest",
    "cli",
    "main",
]
