"""Compatibility wrapper for runtime benchmark profiling."""

from __future__ import annotations

import json
from pathlib import Path

from moe_surgeon.runtime.profiler import (
    benchmark,
    BenchmarkResult,
    iter_prompt_batches,
    PromptBatch,
    RouterActivationProfiler,
    RouterActivationRecord,
    RouterCaptureCollector,
)
from moe_surgeon.models.errors import ArtifactValidationError, TopologyMismatchError
from moe_surgeon.schemas import ActivationStats, LayerTopology, RunArtifactManifest, from_json

__all__ = [
    "benchmark",
    "BenchmarkResult",
    "iter_prompt_batches",
    "PromptBatch",
    "RouterActivationProfiler",
    "RouterActivationRecord",
    "RouterCaptureCollector",
    "load_benchmark_artifact",
    "validate_benchmark_artifact",
    "write_benchmark_artifact",
]


def write_benchmark_artifact(path: str | Path, result: BenchmarkResult) -> Path:
    """Write the canonical benchmark artifact to disk."""

    return result.write_json(path)


def _load_artifact_payload(path: str | Path, *, artifact_name: str) -> dict[str, object]:
    target = Path(path)
    if not target.is_file():
        raise ArtifactValidationError(
            f"{artifact_name} artifact does not exist",
            details={"artifact_path": str(target)},
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(
            f"{artifact_name} artifact must contain valid JSON",
            details={"artifact_path": str(target)},
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(
            f"{artifact_name} artifact payload must be a JSON object",
            details={"artifact_path": str(target)},
        )
    return payload


def load_benchmark_artifact(path: str | Path) -> BenchmarkResult:
    """Load a persisted benchmark artifact from disk."""

    payload = _load_artifact_payload(path, artifact_name="benchmark")
    try:
        manifest_payload = payload["manifest"]
        if not isinstance(manifest_payload, dict):
            raise ArtifactValidationError("benchmark artifact manifest payload must be a JSON object")
        manifest = from_json(manifest_payload)
        if not isinstance(manifest, RunArtifactManifest):
            raise ArtifactValidationError("benchmark artifact manifest must be RunArtifactManifest")

        topology_payload = payload["topology"]
        activation_stats_payload = payload["activation_stats"]
        profiler_config = payload.get("profiler_config", {})
        input_payload_hash = payload.get("input_payload_hash")
        if not isinstance(topology_payload, list):
            raise ArtifactValidationError("benchmark artifact topology must be a JSON array")
        if not isinstance(activation_stats_payload, list):
            raise ArtifactValidationError("benchmark artifact activation_stats must be a JSON array")
        if not isinstance(profiler_config, dict):
            raise ArtifactValidationError("benchmark artifact profiler_config must be a JSON object")

        topology = tuple(from_json(layer_payload) for layer_payload in topology_payload)
        if not all(isinstance(item, LayerTopology) for item in topology):
            raise ArtifactValidationError("benchmark artifact topology contains malformed entries")
        activation_stats = tuple(from_json(stat_payload) for stat_payload in activation_stats_payload)
        if not all(isinstance(item, ActivationStats) for item in activation_stats):
            raise ArtifactValidationError("benchmark artifact activation_stats contain malformed entries")
    except ArtifactValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "benchmark artifact payload is malformed",
            details={"artifact_path": str(Path(path))},
        ) from exc
    result = BenchmarkResult(
        manifest=manifest,
        topology=topology,
        activation_stats=activation_stats,
        profiler_config=profiler_config,
        input_payload_hash=input_payload_hash if isinstance(input_payload_hash, str) else None,
    )
    return validate_benchmark_artifact(result)


def validate_benchmark_artifact(result: BenchmarkResult) -> BenchmarkResult:
    """Validate a benchmark artifact before planner or CLI chaining."""

    from moe_surgeon.analysis.scan import align_activation_stats, build_layer_topology_index

    if result.manifest.command != "bench":
        raise TopologyMismatchError(
            "benchmark artifact manifest command must be bench",
            details={"command": result.manifest.command},
        )
    if result.manifest.model_handle is None:
        raise TopologyMismatchError("benchmark artifact manifest must include model_handle")
    build_layer_topology_index(result.topology)
    align_activation_stats(layers=result.topology, stats=result.activation_stats)
    return result
