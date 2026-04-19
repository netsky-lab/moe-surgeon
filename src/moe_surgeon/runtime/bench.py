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
from moe_surgeon.models.errors import TopologyMismatchError
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


def load_benchmark_artifact(path: str | Path) -> BenchmarkResult:
    """Load a persisted benchmark artifact from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("benchmark artifact payload must be a JSON object")
    manifest = from_json(payload["manifest"])
    if not isinstance(manifest, RunArtifactManifest):
        raise TypeError("benchmark artifact manifest must be RunArtifactManifest")
    topology = tuple(
        item for item in (from_json(layer_payload) for layer_payload in payload["topology"]) if isinstance(item, LayerTopology)
    )
    activation_stats = tuple(
        item
        for item in (from_json(stat_payload) for stat_payload in payload["activation_stats"])
        if isinstance(item, ActivationStats)
    )
    profiler_config = payload.get("profiler_config", {})
    input_payload_hash = payload.get("input_payload_hash")
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
