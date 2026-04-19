"""Top-level Click command graph for moe-surgeon workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, cast

import click

from moe_surgeon import PACKAGE_DESCRIPTION, PACKAGE_NAME, __version__
from moe_surgeon.models.errors import (
    ArtifactValidationError,
    BackendMismatchError,
    ModelError,
    SchemaValidationError,
    ShapeInvariantViolationError,
    TopologyMismatchError,
    UnsupportedModelError,
)

if TYPE_CHECKING:
    from moe_surgeon.analysis.scan import StaticScanResult
    from moe_surgeon.models.backend import LoadedBackendBundle, ModelBackend
    from moe_surgeon.runtime.profiler import BenchmarkResult
    from moe_surgeon.schemas import LayerTopology, ModelHandle, RunArtifactManifest

F = TypeVar("F", bound=Callable[..., object])

_COMMAND_EXIT_CODES: dict[type[BaseException], int] = {
    ArtifactValidationError: 24,
    UnsupportedModelError: 20,
    BackendMismatchError: 21,
    TopologyMismatchError: 22,
    ShapeInvariantViolationError: 23,
    SchemaValidationError: 25,
}


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
    scan_artifact: Path | None
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
    target_experts: int | None
    min_experts_per_layer: int
    output_dir: Path | None


@dataclass(frozen=True)
class ExportCommandRequest:
    """Parsed options for the ``export`` command."""

    shared: CliContext
    apply_artifact_dir: Path | None
    output_dir: Path | None


def _path_text(path: Path | None) -> str:
    return "-" if path is None else str(path)


def _resolve_command_error_code(exc: BaseException) -> int:
    for error_type, exit_code in _COMMAND_EXIT_CODES.items():
        if isinstance(exc, error_type):
            return exit_code
    return 1


def _raise_command_failure(exc: BaseException) -> None:
    exit_code = _resolve_command_error_code(exc)
    error_label = getattr(exc, "error_code", "unhandled")
    click.echo(f"error[{exit_code}:{error_label}]: {exc}", err=True)
    raise click.exceptions.Exit(exit_code)


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


def _default_artifact_root(shared: CliContext) -> Path:
    return shared.artifact_root if shared.artifact_root is not None else Path("artifacts")


def _default_scan_output_path(shared: CliContext) -> Path:
    return _default_artifact_root(shared) / "scan" / "scan.json"


def _default_bench_output_path(shared: CliContext) -> Path:
    return _default_artifact_root(shared) / "bench" / "bench.json"


def _default_prune_output_dir(shared: CliContext) -> Path:
    return _default_artifact_root(shared) / "prune"


def _default_export_output_dir(shared: CliContext) -> Path:
    return _default_artifact_root(shared) / "export"


def _run_manifest_sidecar_path(artifact_path: Path, *, command: str) -> Path:
    if artifact_path.suffix:
        return artifact_path.with_name(f"{artifact_path.stem}.run-manifest.json")
    return artifact_path / f"{command}-run-manifest.json"


def _ensure_local_checkpoint_source(shared: CliContext, *, command: str) -> Path:
    if shared.source_path is None:
        raise TopologyMismatchError(
            f"{command} requires --source-path pointing to a local safetensors checkpoint",
            details={"command": command},
        )
    return Path(shared.source_path)


def _ensure_output_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_output_file_path(
    path: Path,
    *,
    command: str,
    model_id: str | None = None,
) -> Path:
    target = path.expanduser()
    if target.exists():
        raise ArtifactValidationError(
            f"{command} output_path must not already exist",
            model_id=model_id,
            details={"output_path": str(target)},
        )
    if target.parent.exists() and not target.parent.is_dir():
        raise ArtifactValidationError(
            f"{command} output_path parent must be a directory",
            model_id=model_id,
            details={"output_parent": str(target.parent)},
        )
    _ensure_output_parent(target)
    return target


def _prepare_output_dir_path(
    path: Path,
    *,
    command: str,
    model_id: str | None = None,
    input_dirs: Sequence[Path] = (),
) -> Path:
    target = path.expanduser()
    normalized_inputs = {candidate.expanduser().resolve() for candidate in input_dirs}
    try:
        target_resolved = target.resolve(strict=False)
    except OSError:
        target_resolved = target
    if target_resolved in normalized_inputs:
        raise ArtifactValidationError(
            f"{command} output_dir must differ from input artifact directories",
            model_id=model_id,
            details={"output_dir": str(target)},
        )
    if target.exists():
        if not target.is_dir():
            raise ArtifactValidationError(
                f"{command} output_dir must be a directory path",
                model_id=model_id,
                details={"output_dir": str(target)},
            )
        if any(target.iterdir()):
            raise ArtifactValidationError(
                f"{command} output_dir must be empty",
                model_id=model_id,
                details={"output_dir": str(target)},
            )
    else:
        target.mkdir(parents=True, exist_ok=False)
    return target


def _write_manifest_sidecar(path: Path, manifest: "RunArtifactManifest") -> Path:
    from moe_surgeon.schemas import to_json

    final_manifest = manifest.finalized()
    _ensure_output_parent(path)
    path.write_text(to_json(final_manifest), encoding="utf-8")
    return path


def _topology_signature(layer: "LayerTopology") -> tuple[int, int, int, int]:
    return (
        layer.layer_index,
        layer.expert_count,
        layer.top_k,
        layer.hidden_size,
    )


def _validate_context_model_handle(shared: CliContext, model_handle: "ModelHandle") -> None:
    if shared.model_id is not None and shared.model_id != model_handle.model_id:
        raise TopologyMismatchError(
            "CLI model_id does not match artifact model_id",
            model_id=model_handle.model_id,
            details={"cli_model_id": shared.model_id, "artifact_model_id": model_handle.model_id},
        )
    if shared.revision is not None and shared.revision != model_handle.revision:
        raise TopologyMismatchError(
            "CLI revision does not match artifact revision",
            model_id=model_handle.model_id,
            details={"cli_revision": shared.revision, "artifact_revision": model_handle.revision or "none"},
        )
    if shared.backend_name is not None and model_handle.backend_name not in (None, shared.backend_name):
        raise BackendMismatchError(
            "CLI backend does not match artifact backend",
            model_id=model_handle.model_id,
            backend_name=model_handle.backend_name,
            details={"cli_backend_name": shared.backend_name},
        )


def _validate_model_handle_compatibility(
    expected: "ModelHandle",
    actual: "ModelHandle",
    *,
    message: str,
) -> None:
    details: dict[str, object] = {}
    if expected.model_id != actual.model_id:
        details["expected_model_id"] = expected.model_id
        details["actual_model_id"] = actual.model_id
    if expected.revision != actual.revision:
        details["expected_revision"] = expected.revision or "none"
        details["actual_revision"] = actual.revision or "none"
    if expected.backend_name not in (None, actual.backend_name):
        details["expected_backend_name"] = expected.backend_name
        details["actual_backend_name"] = actual.backend_name or "none"
    if expected.dtype not in (None, actual.dtype):
        details["expected_dtype"] = expected.dtype
        details["actual_dtype"] = actual.dtype or "none"
    if details:
        raise BackendMismatchError(
            message,
            model_id=expected.model_id,
            backend_name=actual.backend_name,
            details=details,
        )


def _resolve_workflow_seed(
    *,
    command: str,
    shared_seed: int | None,
    manifests: Sequence["RunArtifactManifest"] = (),
    model_handles: Sequence["ModelHandle"] = (),
    model_id: str | None = None,
) -> int:
    from moe_surgeon.schemas import resolve_deterministic_seed

    candidates: list[tuple[str, int | None]] = []
    candidates.append(("cli_seed", shared_seed))
    for index, manifest in enumerate(manifests):
        candidates.append((f"manifest_{index}", manifest.seed))
        if manifest.model_handle is not None:
            candidates.append((f"manifest_{index}_model_handle", manifest.model_handle.seed))
    for index, handle in enumerate(model_handles):
        candidates.append((f"model_handle_{index}", handle.seed))
    try:
        return resolve_deterministic_seed(*(value for _, value in candidates), name=f"{command}_seed")
    except SchemaValidationError as exc:
        raise ArtifactValidationError(
            f"{command} seed must be deterministic across CLI context and artifacts",
            model_id=model_id,
            details={label: str(value) for label, value in candidates if value is not None},
        ) from exc


def _validate_topology_compatibility(
    expected_layers: Sequence["LayerTopology"],
    actual_layers: Sequence["LayerTopology"],
    *,
    model_id: str,
    message: str,
) -> None:
    expected_signature = tuple(_topology_signature(layer) for layer in expected_layers)
    actual_signature = tuple(_topology_signature(layer) for layer in actual_layers)
    if expected_signature != actual_signature:
        raise TopologyMismatchError(
            message,
            model_id=model_id,
            details={
                "expected_layers": str(expected_signature),
                "actual_layers": str(actual_signature),
            },
        )


def _validate_scan_and_bench_compatibility(
    scan_result: "StaticScanResult",
    bench_result: "BenchmarkResult",
) -> None:
    from moe_surgeon.prune.planner import validate_planner_inputs

    scan_handle = scan_result.manifest.model_handle
    bench_handle = bench_result.manifest.model_handle
    if scan_handle is None or bench_handle is None:
        raise TopologyMismatchError("scan and bench artifacts must include model_handle")
    _validate_model_handle_compatibility(
        scan_handle,
        bench_handle,
        message="scan and bench artifacts target different model/backend identities",
    )
    _resolve_workflow_seed(
        command="prune",
        shared_seed=None,
        manifests=(scan_result.manifest, bench_result.manifest),
        model_id=scan_handle.model_id,
    )
    _validate_topology_compatibility(
        scan_result.layers,
        bench_result.topology,
        model_id=scan_handle.model_id,
        message="scan and bench topology snapshots do not match",
    )
    validate_planner_inputs(
        scan_result.layers,
        expert_stats=scan_result.expert_stats,
        activation_stats=bench_result.activation_stats,
    )


def _load_runtime_bundle(shared: CliContext) -> tuple["ModelBackend", "LoadedBackendBundle"]:
    from moe_surgeon.models.backend import BackendSignature, resolve_backend
    from moe_surgeon.models.checkpoints import open_local_safetensors_checkpoint
    from moe_surgeon.models.gemma4 import Gemma4Backend

    seed = 0 if shared.seed is None else shared.seed
    if shared.source_path is not None:
        checkpoint = open_local_safetensors_checkpoint(shared.source_path)
        signature = checkpoint.to_backend_signature()
        backend = resolve_backend(signature)
        if shared.backend_name is not None and getattr(backend, "name", None) != shared.backend_name:
            raise BackendMismatchError(
                "CLI backend override does not match resolved checkpoint backend",
                model_id=checkpoint.model_id,
                backend_name=getattr(backend, "name", None),
                details={"cli_backend_name": shared.backend_name},
            )
        loaded_bundle = backend.load(signature, dtype=shared.dtype, seed=seed)
        return backend, loaded_bundle

    if shared.model_id is None:
        raise TopologyMismatchError("runtime workflow requires --model-id or --source-path")

    if shared.backend_name not in (None, "gemma4"):
        raise BackendMismatchError(
            "unsupported explicit backend override",
            model_id=shared.model_id,
            backend_name=shared.backend_name,
        )
    backend = Gemma4Backend()
    signature = BackendSignature.from_mapping(
        {
            "_name_or_path": shared.model_id,
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            **({} if shared.revision is None else {"revision": shared.revision}),
        },
        model_id=shared.model_id,
        source_path=shared.source_path,
    )
    loaded_bundle = backend.load(signature, dtype=shared.dtype, seed=seed)
    return backend, loaded_bundle


def _run_scan(request: ScanCommandRequest) -> int:
    from moe_surgeon.analysis.scan import (
        load_local_scan_bundle,
        scan_model,
        write_scan_artifact,
    )

    source_path = _ensure_local_checkpoint_source(request.shared, command="scan")
    bundle, backend = load_local_scan_bundle(source_path)
    _validate_context_model_handle(request.shared, bundle.model_handle)
    resolved_seed = _resolve_workflow_seed(
        command="scan",
        shared_seed=request.shared.seed,
        model_handles=(bundle.model_handle,),
        model_id=bundle.model_handle.model_id,
    )
    output_path = _prepare_output_file_path(
        request.output_path or _default_scan_output_path(request.shared),
        command="scan",
        model_id=bundle.model_handle.model_id,
    )
    manifest_path = _prepare_output_file_path(
        _run_manifest_sidecar_path(output_path, command="scan"),
        command="scan",
        model_id=bundle.model_handle.model_id,
    )
    result = scan_model(bundle, backend=backend)
    if result.manifest.model_handle is None:
        raise ArtifactValidationError("scan result must include model_handle", model_id=bundle.model_handle.model_id)
    manifest_model_handle = replace(result.manifest.model_handle, seed=resolved_seed)
    artifact_manifest = replace(
        result.manifest,
        model_handle=manifest_model_handle,
        seed=resolved_seed,
        output_paths={"scan_artifact": str(output_path)},
        metadata={
            **dict(result.manifest.metadata),
            "run_manifest_path": str(manifest_path),
        },
    )
    result = replace(result, manifest=artifact_manifest)
    written_artifact = write_scan_artifact(output_path, result)
    written_manifest = _write_manifest_sidecar(manifest_path, artifact_manifest)
    click.echo("command=scan")
    click.echo(f"source_path={source_path}")
    click.echo(f"output_path={written_artifact}")
    click.echo(f"run_manifest={written_manifest}")
    return 0


def _run_bench(request: BenchCommandRequest) -> int:
    import importlib

    from moe_surgeon.analysis.scan import load_scan_artifact, validate_scan_artifact
    from moe_surgeon.runtime.bench import validate_benchmark_artifact, write_benchmark_artifact
    from moe_surgeon.runtime.profiler import (
        RouterActivationProfiler,
        benchmark,
        iter_prompt_batches,
    )

    if request.scan_artifact is None:
        raise TopologyMismatchError("bench requires --scan-artifact")
    if not request.prompts:
        raise TopologyMismatchError("bench requires at least one --prompt")

    scan_result = validate_scan_artifact(load_scan_artifact(request.scan_artifact))
    scan_handle = scan_result.manifest.model_handle
    if scan_handle is None:
        raise TopologyMismatchError("scan artifact must include model_handle")
    _validate_context_model_handle(request.shared, scan_handle)
    resolved_seed = _resolve_workflow_seed(
        command="bench",
        shared_seed=request.shared.seed,
        manifests=(scan_result.manifest,),
        model_id=scan_handle.model_id,
    )

    backend, bundle = _load_runtime_bundle(request.shared)
    _validate_model_handle_compatibility(
        scan_handle,
        bundle.model_handle,
        message="scan artifact does not match runtime-loaded model/backend identity",
    )
    output_path = _prepare_output_file_path(
        request.output_path or _default_bench_output_path(request.shared),
        command="bench",
        model_id=bundle.model_handle.model_id,
    )
    manifest_path = _prepare_output_file_path(
        _run_manifest_sidecar_path(output_path, command="bench"),
        command="bench",
        model_id=bundle.model_handle.model_id,
    )
    bundle_topology = tuple(backend.extract_topology(bundle))
    _validate_topology_compatibility(
        scan_result.layers,
        bundle_topology,
        model_id=bundle.model_handle.model_id,
        message="scan artifact topology does not match runtime-loaded model topology",
    )

    tokenizer = bundle.tokenizer
    if tokenizer is None:
        raise TopologyMismatchError(
            "runtime benchmark requires a tokenizer-enabled backend bundle",
            model_id=bundle.model_handle.model_id,
        )
    tokenizer_callable = cast(Callable[..., object], tokenizer)
    model = bundle.model
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()

    torch = importlib.import_module("torch")
    with RouterActivationProfiler(
        backend=backend,
        bundle=bundle,
        topology=bundle_topology,
        include_router_scores=request.capture_router_scores,
    ) as profiler:
        with torch.no_grad():
            for batch in iter_prompt_batches(
                tokenizer=tokenizer_callable,
                prompts=request.prompts,
                batch_size=request.batch_size,
            ):
                inputs = dict(batch.encoded_inputs)
                inputs.setdefault("use_cache", False)
                model_call = getattr(model, "__call__", None)
                if not callable(model_call):
                    raise TopologyMismatchError(
                        "runtime benchmark requires a callable model object",
                        model_id=bundle.model_handle.model_id,
                    )
                model_call(**inputs)
                profiler.accumulate(attention_mask=batch.attention_mask)
        result = validate_benchmark_artifact(
            benchmark(
                profiler=profiler,
                prompts=request.prompts,
                profiler_config={
                    "batch_size": request.batch_size,
                    "capture_router_scores": request.capture_router_scores,
                },
                parent_artifacts=(str(request.scan_artifact),),
                output_paths={
                    "benchmark_artifact": str(output_path)
                },
                seed=resolved_seed,
            )
        )

    if result.manifest.model_handle is None:
        raise ArtifactValidationError("benchmark result must include model_handle", model_id=bundle.model_handle.model_id)
    manifest_model_handle = replace(result.manifest.model_handle, seed=resolved_seed)
    artifact_manifest = replace(
        result.manifest,
        model_handle=manifest_model_handle,
        seed=resolved_seed,
        output_paths={"benchmark_artifact": str(output_path)},
        parent_artifacts=(str(request.scan_artifact),),
        metadata={
            **dict(result.manifest.metadata),
            "run_manifest_path": str(manifest_path),
        },
    )
    result = replace(result, manifest=artifact_manifest)
    written_artifact = write_benchmark_artifact(output_path, result)
    written_manifest = _write_manifest_sidecar(manifest_path, artifact_manifest)

    prompt_batches = (len(request.prompts) + request.batch_size - 1) // request.batch_size
    click.echo("command=bench")
    click.echo(f"scan_artifact={request.scan_artifact}")
    click.echo(f"prompt_inputs={len(request.prompts)}")
    click.echo(f"prompt_batches={prompt_batches}")
    click.echo(f"batch_size={request.batch_size}")
    click.echo(f"output_path={written_artifact}")
    click.echo(f"run_manifest={written_manifest}")
    return 0


def _run_prune(request: PruneCommandRequest) -> int:
    from moe_surgeon.analysis.scan import load_scan_artifact, validate_scan_artifact
    from moe_surgeon.prune.apply import apply_prune_plan
    from moe_surgeon.prune.planner import (
        PlannerConstraints,
        build_prune_plan,
        write_prune_plan,
    )
    from moe_surgeon.runtime.bench import load_benchmark_artifact, validate_benchmark_artifact
    from moe_surgeon.schemas import RunArtifactManifest

    source_path = _ensure_local_checkpoint_source(request.shared, command="prune")
    if request.scan_artifact is None or request.bench_artifact is None:
        raise TopologyMismatchError("prune requires both --scan-artifact and --bench-artifact")

    scan_result = validate_scan_artifact(load_scan_artifact(request.scan_artifact))
    bench_result = validate_benchmark_artifact(load_benchmark_artifact(request.bench_artifact))
    _validate_scan_and_bench_compatibility(scan_result, bench_result)

    scan_handle = scan_result.manifest.model_handle
    if scan_handle is None:
        raise TopologyMismatchError("scan artifact must include model_handle")
    _validate_context_model_handle(request.shared, scan_handle)

    resolved_seed = _resolve_workflow_seed(
        command="prune",
        shared_seed=request.shared.seed,
        manifests=(scan_result.manifest, bench_result.manifest),
        model_id=scan_handle.model_id,
    )
    output_root = _prepare_output_dir_path(
        request.output_dir or _default_prune_output_dir(request.shared),
        command="prune",
        model_id=scan_handle.model_id,
        input_dirs=(source_path,),
    )

    constraints = PlannerConstraints(
        global_target_experts=request.target_experts,
        min_experts_per_layer=request.min_experts_per_layer,
    )
    plan = build_prune_plan(
        scan_result.layers,
        strategy=request.strategy,
        expert_stats=scan_result.expert_stats,
        activation_stats=bench_result.activation_stats,
        constraints=constraints,
        model_handle=scan_handle,
        source_run_id=bench_result.manifest.run_id,
    )
    plan_path = write_prune_plan(output_root / "prune-plan.json", plan)
    apply_artifact_dir = output_root / "applied-checkpoint"
    apply_result = apply_prune_plan(source_path, plan=plan, dry_run=False, output_dir=apply_artifact_dir)

    prune_manifest = RunArtifactManifest(
        run_id=f"prune-{plan.plan_id}",
        command="prune",
        model_handle=scan_handle,
        top_k=scan_result.manifest.top_k,
        seed=resolved_seed,
        input_checksums={
            "scan_manifest_digest": scan_result.manifest.canonical_digest,
            "bench_manifest_digest": bench_result.manifest.canonical_digest,
            "plan_manifest_id": plan.versioned_manifest_id,
        },
        output_paths={
            "prune_plan": str(plan_path),
            "apply_artifact_dir": str(apply_artifact_dir),
        },
        parent_artifacts=(str(request.scan_artifact), str(request.bench_artifact)),
        run_plan=plan,
        metadata={
            "apply_id": apply_result.apply_id,
            "apply_artifact_dir": str(apply_artifact_dir),
            "export_manifest_path": str(apply_artifact_dir / "run-manifest.json"),
        },
    )
    prune_manifest_path = _write_manifest_sidecar(output_root / "prune-run-manifest.json", prune_manifest)

    click.echo("command=prune")
    click.echo(f"scan_artifact={request.scan_artifact}")
    click.echo(f"bench_artifact={request.bench_artifact}")
    click.echo(f"strategy={request.strategy}")
    click.echo(f"plan_path={plan_path}")
    click.echo(f"apply_artifact_dir={apply_artifact_dir}")
    click.echo(f"run_manifest={prune_manifest_path}")
    return 0


def _run_export(request: ExportCommandRequest) -> int:
    from moe_surgeon.export.runner import run_export_from_apply_artifact
    from moe_surgeon.prune.apply import load_apply_result, validate_apply_result

    if request.apply_artifact_dir is None:
        raise TopologyMismatchError("export requires --apply-artifact-dir")
    apply_result = validate_apply_result(load_apply_result(request.apply_artifact_dir), require_materialized=True)
    _validate_context_model_handle(request.shared, apply_result.model_handle)
    resolved_seed = _resolve_workflow_seed(
        command="export",
        shared_seed=request.shared.seed,
        model_handles=(apply_result.model_handle,),
        model_id=apply_result.model_handle.model_id,
    )
    output_dir = _prepare_output_dir_path(
        request.output_dir or _default_export_output_dir(request.shared),
        command="export",
        model_id=apply_result.model_handle.model_id,
        input_dirs=(Path(request.apply_artifact_dir),),
    )
    export_result = run_export_from_apply_artifact(request.apply_artifact_dir, output_dir=output_dir)
    click.echo("command=export")
    click.echo(f"apply_artifact_dir={request.apply_artifact_dir}")
    click.echo(f"output_dir={output_dir}")
    click.echo(f"export_id={export_result.export_id}")
    click.echo(f"seed={resolved_seed}")
    click.echo(f"run_manifest={Path(output_dir) / 'run-manifest.json'}")
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
    try:
        _run_scan(ScanCommandRequest(shared=shared, output_path=output_path))
    except ModelError as exc:
        _raise_command_failure(exc)


@cli.command("bench")
@_shared_command_options
@click.option(
    "--scan-artifact",
    type=click.Path(path_type=Path, dir_okay=False, exists=False, resolve_path=False),
    default=None,
    help="Static scan artifact consumed by bench preflight validation.",
)
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
    scan_artifact: Path | None,
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
    try:
        _run_bench(
            BenchCommandRequest(
                shared=shared,
                scan_artifact=scan_artifact,
                prompts=prompts,
                batch_size=batch_size,
                capture_router_scores=capture_router_scores,
                output_path=output_path,
            )
        )
    except ModelError as exc:
        _raise_command_failure(exc)


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
    "--target-experts",
    type=click.IntRange(min=1),
    default=None,
    help="Optional global target number of surviving experts across all MoE layers.",
)
@click.option(
    "--min-experts-per-layer",
    type=click.IntRange(min=1),
    default=1,
    help="Minimum number of experts to preserve per MoE layer.",
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
    target_experts: int | None,
    min_experts_per_layer: int,
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
    try:
        _run_prune(
            PruneCommandRequest(
                shared=shared,
                scan_artifact=scan_artifact,
                bench_artifact=bench_artifact,
                strategy=strategy.lower(),
                target_experts=target_experts,
                min_experts_per_layer=min_experts_per_layer,
                output_dir=output_dir,
            )
        )
    except ModelError as exc:
        _raise_command_failure(exc)


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
    try:
        _run_export(
            ExportCommandRequest(
                shared=shared,
                apply_artifact_dir=apply_artifact_dir,
                output_dir=output_dir,
            )
        )
    except ModelError as exc:
        _raise_command_failure(exc)


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
