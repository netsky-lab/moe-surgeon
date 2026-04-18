"""Static router scan built on backend-exposed topology and router metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import fsum
from pathlib import Path
from typing import Mapping, Sequence

import torch

from moe_surgeon.analysis.metrics import RouterMetricSummary, build_expert_stats
from moe_surgeon.models.backend import LoadedBackendBundle, ModelBackend, resolve_backend
from moe_surgeon.models.checkpoints import (
    LocalSafetensorsCheckpoint,
    open_local_safetensors_checkpoint,
)
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.schemas import (
    ActivationStats,
    ExpertStats,
    LayerTopology,
    RouterState,
    RunArtifactManifest,
    sort_activation_stats,
    to_dict,
    to_json,
    sort_topology,
)


@dataclass(frozen=True)
class StaticScanAggregateSummary:
    """Deterministic aggregate scan summary across all MoE router layers."""

    layer_count: int
    expert_stat_count: int
    total_static_gate_mass: float
    total_top_k_mass_proxy: float
    mean_normalized_entropy: float


@dataclass(frozen=True)
class StaticScanResult:
    """Deterministic static scan payload for one loaded MoE model bundle."""

    layers: tuple[LayerTopology, ...]
    router_states: tuple[RouterState, ...]
    expert_stats: tuple[ExpertStats, ...]
    layer_summaries: tuple[RouterMetricSummary, ...]
    aggregate_summary: StaticScanAggregateSummary
    manifest: RunArtifactManifest


def _sorted_router_states(states: tuple[RouterState, ...]) -> tuple[RouterState, ...]:
    return tuple(sorted(states, key=lambda state: state.layer_index))


def _sorted_expert_stats(stats: tuple[ExpertStats, ...]) -> tuple[ExpertStats, ...]:
    return tuple(
        sorted(
            stats,
            key=lambda stat: (
                stat.layer_index,
                stat.static_rank if stat.static_rank is not None else stat.expert_index,
                stat.expert_index,
            ),
        )
    )


def _resolve_scan_backend(
    bundle: LoadedBackendBundle,
    *,
    backend: ModelBackend | None,
) -> ModelBackend:
    if backend is not None:
        return backend
    return resolve_backend(bundle.config, model_id=bundle.model_handle.model_id)


def _bundle_with_local_checkpoint_state(
    bundle: LoadedBackendBundle,
    checkpoint: LocalSafetensorsCheckpoint,
) -> LoadedBackendBundle:
    metadata = dict(bundle.metadata)
    metadata["state_keys"] = checkpoint.state_keys()
    metadata["state_dict"] = {
        item.tensor_key: item for item in checkpoint.tensor_metadata()
    }
    return LoadedBackendBundle(
        backend_name=bundle.backend_name,
        model_handle=bundle.model_handle,
        model=bundle.model,
        config=bundle.config,
        tokenizer=bundle.tokenizer,
        metadata=metadata,
    )


def _resolve_local_checkpoint(bundle: LoadedBackendBundle) -> LocalSafetensorsCheckpoint | None:
    source_path = bundle.model_handle.source_path
    if source_path is None:
        return None
    root = Path(source_path).expanduser()
    if not root.is_dir():
        return None
    return open_local_safetensors_checkpoint(root)


def _scan_state(
    bundle: LoadedBackendBundle,
) -> tuple[LoadedBackendBundle, Mapping[str, object] | None, LocalSafetensorsCheckpoint | None]:
    metadata_state = bundle.metadata.get("state_dict")
    if isinstance(metadata_state, Mapping):
        return bundle, metadata_state, None

    checkpoint = _resolve_local_checkpoint(bundle)
    if checkpoint is not None:
        prepared_bundle = _bundle_with_local_checkpoint_state(bundle, checkpoint)
        return prepared_bundle, None, checkpoint

    state_dict = getattr(bundle.model, "state_dict", None)
    if callable(state_dict):
        loaded_state = state_dict()
        if isinstance(loaded_state, Mapping):
            return bundle, loaded_state, None

    if "state_keys" in bundle.metadata or bundle.model_handle.source_path is not None:
        raise TopologyMismatchError(
            "static scan requires materialized numeric tensors or a readable local safetensors checkpoint",
            model_id=bundle.model_handle.model_id,
            details={"source_path": bundle.model_handle.source_path or "unknown"},
        )

    raise TopologyMismatchError(
        "static scan requires materialized state_dict tensor values",
        model_id=bundle.model_handle.model_id,
    )


def _require_tensor(
    state_dict: Mapping[str, object],
    *,
    bundle: LoadedBackendBundle,
    layer: LayerTopology,
    tensor_role: str,
) -> torch.Tensor:
    tensor_key = layer.module_paths.get(tensor_role)
    if tensor_key is None:
        raise TopologyMismatchError(
            "layer module_paths missing required router tensor",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            details={"tensor_role": tensor_role},
        )
    value = state_dict.get(tensor_key)
    if value is None:
        raise TopologyMismatchError(
            "static scan requires materialized numeric tensor values",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            tensor_key=tensor_key,
            details={"tensor_role": tensor_role},
        )
    if not isinstance(value, torch.Tensor):
        raise ShapeInvariantViolationError(
            "scan router tensor must be torch.Tensor",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            tensor_key=tensor_key,
            details={"tensor_role": tensor_role, "value_type": type(value).__name__},
        )
    return value


def _enrich_router_state(
    router_state: RouterState,
    *,
    route_scale: torch.Tensor,
    per_expert_scale: torch.Tensor,
) -> RouterState:
    metadata = dict(router_state.metadata)
    if route_scale.numel() == 1:
        metadata["route_scale_value"] = float(route_scale.detach().to(dtype=torch.float64).item())
    metadata["per_expert_scale_mean_abs"] = float(
        per_expert_scale.detach().to(dtype=torch.float64).abs().mean().item()
    )
    return RouterState(
        layer_index=router_state.layer_index,
        num_experts=router_state.num_experts,
        top_k=router_state.top_k,
        logits_shape=router_state.logits_shape,
        top_k_indices_shape=router_state.top_k_indices_shape,
        top_k_weights_shape=router_state.top_k_weights_shape,
        projection_shape=router_state.projection_shape,
        per_expert_scale_shape=router_state.per_expert_scale_shape,
        has_router_probabilities=router_state.has_router_probabilities,
        has_raw_logits_capture=router_state.has_raw_logits_capture,
        route_scale_present=router_state.route_scale_present,
        metadata=metadata,
    )


def _metric_tensors_for_layer(
    *,
    bundle: LoadedBackendBundle,
    layer: LayerTopology,
    state_dict: Mapping[str, object] | None,
    checkpoint: LocalSafetensorsCheckpoint | None,
) -> Mapping[str, object]:
    if checkpoint is None:
        if state_dict is None:
            raise TopologyMismatchError(
                "static scan requires materialized state_dict tensor values",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        return state_dict

    tensor_names = [
        layer.module_paths["router_proj"],
        layer.module_paths["router_scale"],
        layer.module_paths["router_per_expert_scale"],
    ]
    return checkpoint.load_tensors(tensor_names)


def _validate_metric_tensor(
    tensor: torch.Tensor,
    *,
    bundle: LoadedBackendBundle,
    layer: LayerTopology,
    tensor_role: str,
    expected_rank: int,
) -> None:
    tensor_key = layer.module_paths.get(tensor_role)
    if tensor_key is None:
        raise TopologyMismatchError(
            "layer module_paths missing required router tensor",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            details={"tensor_role": tensor_role},
        )
    actual_shape = tuple(int(dimension) for dimension in tensor.shape)
    if tensor.ndim != expected_rank:
        raise ShapeInvariantViolationError(
            "scan metric tensor rank mismatch",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            tensor_key=tensor_key,
            actual_shape=actual_shape,
            details={"tensor_role": tensor_role, "expected_rank": expected_rank, "actual_rank": tensor.ndim},
        )
    if not torch.isfinite(tensor).all():
        raise ShapeInvariantViolationError(
            "scan metric tensor must be finite",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            tensor_key=tensor_key,
            actual_shape=actual_shape,
            details={"tensor_role": tensor_role},
        )


def _aggregate_summary(layer_summaries: list[RouterMetricSummary], expert_stats: list[ExpertStats]) -> StaticScanAggregateSummary:
    """Reduce per-layer metrics into a deterministic scan-level summary."""

    layer_count = len(layer_summaries)
    normalized_entropy = 0.0
    if layer_count > 0:
        normalized_entropy = fsum(summary.normalized_entropy for summary in layer_summaries) / float(layer_count)
    return StaticScanAggregateSummary(
        layer_count=layer_count,
        expert_stat_count=len(expert_stats),
        total_static_gate_mass=fsum(summary.total_static_gate_mass for summary in layer_summaries),
        total_top_k_mass_proxy=fsum(summary.total_top_k_mass_proxy for summary in layer_summaries),
        mean_normalized_entropy=normalized_entropy,
    )


def _manifest_with_scan_digests(result: StaticScanResult) -> RunArtifactManifest:
    metadata = dict(result.manifest.metadata)
    metadata["model_fingerprint"] = result.manifest.model_handle.model_fingerprint if result.manifest.model_handle else None
    base_manifest = RunArtifactManifest(
        run_id=result.manifest.run_id,
        command=result.manifest.command,
        model_handle=result.manifest.model_handle,
        top_k=result.manifest.top_k,
        prompt_count=result.manifest.prompt_count,
        seed=result.manifest.seed,
        prompt_set_hash=result.manifest.prompt_set_hash,
        started_at=result.manifest.started_at,
        finished_at=result.manifest.finished_at,
        input_checksums=dict(result.manifest.input_checksums),
        output_paths=dict(result.manifest.output_paths),
        parent_artifacts=tuple(result.manifest.parent_artifacts),
        run_plan=result.manifest.run_plan,
        metadata=metadata,
    )
    manifest_digest = base_manifest.canonical_digest
    metadata["canonical_manifest_digest"] = manifest_digest
    artifact_digest = sha256(
        to_json(_scan_artifact_payload(result, manifest=base_manifest, include_artifact_digest=False)).encode("utf-8")
    ).hexdigest()
    metadata["canonical_artifact_digest"] = artifact_digest
    return RunArtifactManifest(
        run_id=base_manifest.run_id,
        command=base_manifest.command,
        model_handle=base_manifest.model_handle,
        top_k=base_manifest.top_k,
        prompt_count=base_manifest.prompt_count,
        seed=base_manifest.seed,
        prompt_set_hash=base_manifest.prompt_set_hash,
        started_at=base_manifest.started_at,
        finished_at=base_manifest.finished_at,
        input_checksums=dict(base_manifest.input_checksums),
        output_paths=dict(base_manifest.output_paths),
        parent_artifacts=tuple(base_manifest.parent_artifacts),
        run_plan=base_manifest.run_plan,
        metadata=metadata,
    )


def _scan_artifact_payload(
    result: StaticScanResult,
    *,
    manifest: RunArtifactManifest | None = None,
    include_artifact_digest: bool = True,
) -> dict[str, object]:
    active_manifest = result.manifest if manifest is None else manifest
    metadata = dict(active_manifest.metadata)
    if not include_artifact_digest:
        metadata.pop("canonical_artifact_digest", None)
    if active_manifest.model_handle is not None:
        metadata["model_fingerprint"] = active_manifest.model_handle.model_fingerprint
    manifest_payload = to_dict(active_manifest)
    manifest_payload["metadata"] = metadata
    return {
        "aggregate_summary": to_dict(asdict(result.aggregate_summary)),
        "expert_stats": [to_dict(item) for item in _sorted_expert_stats(result.expert_stats)],
        "layer_summaries": [
            to_dict(asdict(item))
            for item in sorted(result.layer_summaries, key=lambda item: item.layer_index)
        ],
        "layers": [to_dict(item) for item in sort_topology(result.layers)],
        "manifest": manifest_payload,
        "model_handle": to_dict(active_manifest.model_handle),
        "router_states": [to_dict(item) for item in _sorted_router_states(result.router_states)],
        "scan_type": "static_router_scan",
        "schema_version": active_manifest.schema_version,
    }


def scan_result_payload(result: StaticScanResult) -> dict[str, object]:
    """Return the canonical JSON-ready payload for a static scan result."""

    return _scan_artifact_payload(result)


def scan_result_json(result: StaticScanResult, *, compact: bool = True) -> str:
    """Serialize a static scan result with canonical ordering."""

    return to_json(scan_result_payload(result), compact=compact)


def write_scan_artifact(path: str | Path, result: StaticScanResult, *, compact: bool = True) -> Path:
    """Write a canonical static scan artifact and return the output path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(scan_result_json(result, compact=compact), encoding="utf-8")
    return target


def scan_model(
    bundle: LoadedBackendBundle,
    *,
    backend: ModelBackend | None = None,
) -> StaticScanResult:
    """Scan static router weights for all backend-validated MoE layers."""

    prepared_bundle, state_dict, checkpoint = _scan_state(bundle)
    active_backend = _resolve_scan_backend(prepared_bundle, backend=backend)
    layers = sort_topology(active_backend.extract_topology(prepared_bundle))

    router_states: list[RouterState] = []
    expert_stats: list[ExpertStats] = []
    layer_summaries: list[RouterMetricSummary] = []
    for layer in layers:
        router_state = active_backend.extract_router_state(prepared_bundle, layer=layer)
        active_backend.validate_layer(prepared_bundle, layer=layer, router_state=router_state)
        layer_metric_tensors = _metric_tensors_for_layer(
            bundle=prepared_bundle,
            layer=layer,
            state_dict=state_dict,
            checkpoint=checkpoint,
        )

        router_proj_weight = _require_tensor(
            layer_metric_tensors,
            bundle=prepared_bundle,
            layer=layer,
            tensor_role="router_proj",
        )
        route_scale = _require_tensor(
            layer_metric_tensors,
            bundle=prepared_bundle,
            layer=layer,
            tensor_role="router_scale",
        )
        per_expert_scale = _require_tensor(
            layer_metric_tensors,
            bundle=prepared_bundle,
            layer=layer,
            tensor_role="router_per_expert_scale",
        )
        router_state = _enrich_router_state(
            router_state,
            route_scale=route_scale,
            per_expert_scale=per_expert_scale,
        )
        _validate_metric_tensor(
            router_proj_weight,
            bundle=prepared_bundle,
            layer=layer,
            tensor_role="router_proj",
            expected_rank=2,
        )
        _validate_metric_tensor(
            per_expert_scale,
            bundle=prepared_bundle,
            layer=layer,
            tensor_role="router_per_expert_scale",
            expected_rank=1,
        )

        layer_stats, layer_summary = build_expert_stats(
            layer_index=layer.layer_index,
            router_proj_weight=router_proj_weight,
            top_k=layer.top_k,
            per_expert_scale=per_expert_scale,
        )
        router_states.append(router_state)
        expert_stats.extend(layer_stats)
        layer_summaries.append(layer_summary)

    aggregate_summary = _aggregate_summary(layer_summaries, expert_stats)
    manifest = RunArtifactManifest(
        run_id=f"scan-static-{prepared_bundle.model_handle.model_fingerprint[:12]}",
        command="scan",
        model_handle=prepared_bundle.model_handle,
        top_k=layers[0].top_k if layers else 1,
        prompt_count=0,
        seed=prepared_bundle.model_handle.seed,
        metadata={
            "moe_layer_count": len(layers),
            "expert_stat_count": len(expert_stats),
            "backend_name": prepared_bundle.backend_name,
            "total_static_gate_mass": aggregate_summary.total_static_gate_mass,
            "total_top_k_mass_proxy": aggregate_summary.total_top_k_mass_proxy,
            "mean_normalized_entropy": aggregate_summary.mean_normalized_entropy,
        },
    )
    result = StaticScanResult(
        layers=layers,
        router_states=_sorted_router_states(tuple(router_states)),
        expert_stats=_sorted_expert_stats(tuple(expert_stats)),
        layer_summaries=tuple(layer_summaries),
        aggregate_summary=aggregate_summary,
        manifest=manifest,
    )
    return StaticScanResult(
        layers=result.layers,
        router_states=result.router_states,
        expert_stats=result.expert_stats,
        layer_summaries=result.layer_summaries,
        aggregate_summary=result.aggregate_summary,
        manifest=_manifest_with_scan_digests(result),
    )


def build_layer_topology_index(layers: Sequence[LayerTopology]) -> dict[int, LayerTopology]:
    """Build a deterministic layer-index lookup with duplicate protection."""

    ordered = sort_topology(layers)
    index: dict[int, LayerTopology] = {}
    for layer in ordered:
        if layer.layer_index in index:
            raise TopologyMismatchError(
                "duplicate layer_index in topology",
                layer_index=layer.layer_index,
            )
        index[layer.layer_index] = layer
    return index


def align_activation_stats(
    *,
    layers: Sequence[LayerTopology],
    stats: Sequence[ActivationStats],
) -> tuple[ActivationStats, ...]:
    """Validate activation stats against topology and return canonical ordering."""

    topology_index = build_layer_topology_index(layers)
    layer_token_totals: dict[int, int] = {}
    layer_weighted_totals: dict[int, float | None] = {}
    for item in stats:
        layer = topology_index.get(item.layer_index)
        if layer is None:
            raise TopologyMismatchError(
                "activation stats reference unknown layer",
                layer_index=item.layer_index,
            )
        if item.expert_index >= layer.expert_count:
            raise TopologyMismatchError(
                "activation stats expert index exceeds layer topology",
                layer_index=item.layer_index,
                details={"expert_index": item.expert_index, "expert_count": layer.expert_count},
            )
        existing_n_tokens = layer_token_totals.setdefault(item.layer_index, item.n_tokens)
        if existing_n_tokens != item.n_tokens:
            raise TopologyMismatchError(
                "activation stats layer token totals are inconsistent",
                layer_index=item.layer_index,
                details={"expected_n_tokens": existing_n_tokens, "actual_n_tokens": item.n_tokens},
            )
        existing_weighted = layer_weighted_totals.setdefault(item.layer_index, item.weighted_n_tokens)
        if existing_weighted is None:
            if item.weighted_n_tokens is not None:
                raise TopologyMismatchError(
                    "activation stats layer weighted token totals are inconsistent",
                    layer_index=item.layer_index,
                    details={"expected_weighted_n_tokens": None, "actual_weighted_n_tokens": item.weighted_n_tokens},
                )
        elif item.weighted_n_tokens is None or abs(existing_weighted - item.weighted_n_tokens) > 1e-12:
            raise TopologyMismatchError(
                "activation stats layer weighted token totals are inconsistent",
                layer_index=item.layer_index,
                details={
                    "expected_weighted_n_tokens": existing_weighted,
                    "actual_weighted_n_tokens": item.weighted_n_tokens,
                },
            )
    return sort_activation_stats(stats)

__all__ = [
    "StaticScanAggregateSummary",
    "StaticScanResult",
    "scan_model",
    "scan_result_json",
    "scan_result_payload",
    "write_scan_artifact",
    "build_layer_topology_index",
    "align_activation_stats",
]
