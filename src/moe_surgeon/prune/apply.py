"""Deterministic prune/apply engine for remapping Gemma4 MoE expert tensors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

from moe_surgeon.models.backend import LoadedBackendBundle, resolve_backend
from moe_surgeon.models.checkpoints import LocalSafetensorsCheckpoint, open_local_safetensors_checkpoint
from moe_surgeon.models.errors import BackendMismatchError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.schemas import (
    CANONICAL_DEFAULT_TIMESTAMP,
    LayerTopology,
    ModelHandle,
    PrunePlan,
    PrunePlanItem,
    SchemaKey,
    sort_plan_items,
    sort_topology,
    to_json,
)

import torch


@dataclass(frozen=True)
class ApplyTensorDelta:
    """Deterministic metadata for one rewritten or validated tensor."""

    tensor_key: str
    tensor_role: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    rewritten: bool


@dataclass(frozen=True)
class ApplyLayerReport:
    """Per-layer prune/apply diagnostics."""

    layer_index: int
    layer_name: str
    source_expert_count: int
    target_expert_count: int
    keep_indices: tuple[int, ...]
    drop_indices: tuple[int, ...]
    old_to_new_index: tuple[tuple[int, int], ...]
    tensor_keys_to_rewrite: tuple[str, ...]
    tensor_deltas: tuple[ApplyTensorDelta, ...]


@dataclass(frozen=True)
class ApplyResult:
    """Structured prune/apply result for dry-run and materialized paths."""

    apply_id: str
    plan_id: str
    source_metadata_digest: str
    model_handle: ModelHandle
    source_checkpoint_dir: str
    source_checkpoint_fingerprint: str
    dry_run: bool
    created_at: str
    layer_reports: tuple[ApplyLayerReport, ...]
    rewritten_tensor_keys: tuple[str, ...]
    passthrough_tensor_keys: tuple[str, ...]
    derived_state_dict: Mapping[str, torch.Tensor] | None
    metadata: Mapping[str, SchemaKey]

    def manifest_payload(self) -> dict[str, object]:
        """Return the canonical JSON-friendly manifest payload."""

        return {
            "apply_id": self.apply_id,
            "plan_id": self.plan_id,
            "source_metadata_digest": self.source_metadata_digest,
            "source_checkpoint_dir": self.source_checkpoint_dir,
            "source_checkpoint_fingerprint": self.source_checkpoint_fingerprint,
            "dry_run": self.dry_run,
            "created_at": self.created_at,
            "rewritten_tensor_keys": list(self.rewritten_tensor_keys),
            "passthrough_tensor_keys": list(self.passthrough_tensor_keys),
            "model_handle": {
                "model_id": self.model_handle.model_id,
                "revision": self.model_handle.revision,
                "backend_name": self.model_handle.backend_name,
                "dtype": self.model_handle.dtype,
                "seed": self.model_handle.seed,
                "metadata": dict(self.model_handle.metadata),
            },
            "layer_reports": [
                {
                    "layer_index": report.layer_index,
                    "layer_name": report.layer_name,
                    "source_expert_count": report.source_expert_count,
                    "target_expert_count": report.target_expert_count,
                    "keep_indices": list(report.keep_indices),
                    "drop_indices": list(report.drop_indices),
                    "old_to_new_index": [[old, new] for old, new in report.old_to_new_index],
                    "tensor_keys_to_rewrite": list(report.tensor_keys_to_rewrite),
                    "tensor_deltas": [
                        {
                            "tensor_key": delta.tensor_key,
                            "tensor_role": delta.tensor_role,
                            "source_shape": list(delta.source_shape),
                            "target_shape": list(delta.target_shape),
                            "rewritten": delta.rewritten,
                        }
                        for delta in report.tensor_deltas
                    ],
                }
                for report in self.layer_reports
            ],
            "metadata": dict(self.metadata),
        }

    def manifest_json(self) -> str:
        """Return the canonical manifest JSON."""

        return to_json(self.manifest_payload())

    def audit_payload(self) -> dict[str, object]:
        """Return the deterministic audit payload used for apply IDs."""

        return {
            "plan_id": self.plan_id,
            "source_metadata_digest": self.source_metadata_digest,
            "layer_reports": [
                {
                    "layer_index": report.layer_index,
                    "source_expert_count": report.source_expert_count,
                    "target_expert_count": report.target_expert_count,
                    "keep_indices": list(report.keep_indices),
                    "drop_indices": list(report.drop_indices),
                    "old_to_new_index": [[old, new] for old, new in report.old_to_new_index],
                    "tensor_deltas": [
                        {
                            "tensor_role": delta.tensor_role,
                            "tensor_key": delta.tensor_key,
                            "source_shape": list(delta.source_shape),
                            "target_shape": list(delta.target_shape),
                            "rewritten": delta.rewritten,
                        }
                        for delta in report.tensor_deltas
                    ],
                }
                for report in self.layer_reports
            ],
            "rewritten_tensor_keys": list(self.rewritten_tensor_keys),
            "passthrough_tensor_keys": list(self.passthrough_tensor_keys),
            "metadata": dict(self.metadata),
        }

    def audit_json(self) -> str:
        """Return the canonical audit JSON."""

        return to_json(self.audit_payload())


@dataclass(frozen=True)
class _TensorShapeView:
    shape: tuple[int, ...]
    dtype: str | None = None


def apply_prune_plan(
    checkpoint_dir: str | Path,
    *,
    plan: PrunePlan,
    dry_run: bool = False,
) -> ApplyResult:
    """Apply a deterministic prune plan to a local checkpoint in memory."""

    checkpoint = open_local_safetensors_checkpoint(checkpoint_dir)
    backend = _resolve_apply_backend(checkpoint)
    metadata_bundle = _metadata_bundle_from_checkpoint(checkpoint, backend=backend)
    topology = sort_topology(backend.extract_topology(metadata_bundle))
    plan_items = _validate_plan_against_topology(plan, topology=topology, model_id=checkpoint.model_id)
    layer_reports = _build_layer_reports(
        backend=backend,
        bundle=metadata_bundle,
        topology=topology,
        plan_items=plan_items,
    )

    rewrite_roles = (
        "experts_down_proj",
        "experts_gate_up_proj",
        "router_per_expert_scale",
        "router_proj",
    )
    rewritten_tensor_keys = tuple(
        sorted(
            {
                delta.tensor_key
                for report in layer_reports
                for delta in report.tensor_deltas
                if delta.tensor_role in rewrite_roles
            }
        )
    )
    passthrough_tensor_keys = tuple(
        sorted(key for key in checkpoint.state_keys() if key not in rewritten_tensor_keys)
    )

    source_metadata_digest = _source_metadata_digest(checkpoint)
    model_handle = ModelHandle(
        model_id=checkpoint.model_id,
        revision=checkpoint.revision,
        backend_name=backend.name,
        source_path=str(checkpoint.checkpoint_dir),
        metadata={
            "checkpoint_fingerprint": source_metadata_digest,
            "checkpoint_tensor_count": len(checkpoint.state_keys()),
        },
    )

    if dry_run:
        derived_state_dict: Mapping[str, torch.Tensor] | None = None
        validation_state = _build_validation_state_from_metadata(
            checkpoint=checkpoint,
            layer_reports=layer_reports,
        )
    else:
        source_state = checkpoint.load_tensors(checkpoint.state_keys())
        derived_state = {key: tensor for key, tensor in source_state.items()}
        for report in layer_reports:
            layer = next(item for item in topology if item.layer_index == report.layer_index)
            keep_indices = report.keep_indices
            tensor_keys = backend.resolve_prune_tensor_keys(metadata_bundle, layer_index=layer.layer_index)
            for tensor_role in rewrite_roles:
                tensor_key = tensor_keys[tensor_role]
                derived_state[tensor_key] = _remap_expert_tensor(
                    source_state[tensor_key],
                    keep_indices=keep_indices,
                )
        derived_state_dict = derived_state
        validation_state = derived_state

    validation_bundle = _bundle_with_state(checkpoint=checkpoint, backend=backend, state=validation_state)
    reports_by_layer = {report.layer_index: report for report in layer_reports}
    for layer in topology:
        report = reports_by_layer[layer.layer_index]
        derived_layer = _derived_target_layer(layer, target_expert_count=report.target_expert_count)
        tensor_keys = backend.resolve_prune_tensor_keys(metadata_bundle, layer_index=layer.layer_index)
        for tensor_role, tensor_key in tensor_keys.items():
            backend.validate_prune_tensor(
                validation_bundle,
                layer=derived_layer,
                tensor_role=tensor_role,
                tensor_key=tensor_key,
                tensor_value=validation_state[tensor_key],
                target_expert_count=report.target_expert_count,
            )

    metadata = {
        "checkpoint_tensor_count": len(checkpoint.state_keys()),
        "layer_count": len(topology),
        "rewritten_tensor_count": len(rewritten_tensor_keys),
        "passthrough_tensor_count": len(passthrough_tensor_keys),
    }
    apply_seed = {
        "plan_id": plan.plan_id,
        "source_metadata_digest": source_metadata_digest,
        "layer_reports": [
            {
                "layer_index": report.layer_index,
                "keep_indices": list(report.keep_indices),
                "drop_indices": list(report.drop_indices),
                "old_to_new_index": [[old, new] for old, new in report.old_to_new_index],
                "tensor_deltas": [
                    {
                        "tensor_key": delta.tensor_key,
                        "tensor_role": delta.tensor_role,
                        "source_shape": list(delta.source_shape),
                        "target_shape": list(delta.target_shape),
                        "rewritten": delta.rewritten,
                    }
                    for delta in report.tensor_deltas
                ],
            }
            for report in layer_reports
        ],
        "rewritten_tensor_keys": list(rewritten_tensor_keys),
        "metadata": metadata,
    }
    apply_id = f"apply-{sha256(to_json(apply_seed).encode('utf-8')).hexdigest()[:16]}"

    return ApplyResult(
        apply_id=apply_id,
        plan_id=plan.plan_id,
        source_metadata_digest=source_metadata_digest,
        model_handle=model_handle,
        source_checkpoint_dir=str(checkpoint.checkpoint_dir),
        source_checkpoint_fingerprint=source_metadata_digest,
        dry_run=dry_run,
        created_at=CANONICAL_DEFAULT_TIMESTAMP,
        layer_reports=layer_reports,
        rewritten_tensor_keys=rewritten_tensor_keys,
        passthrough_tensor_keys=passthrough_tensor_keys,
        derived_state_dict=derived_state_dict,
        metadata=metadata,
    )


def _resolve_apply_backend(checkpoint: LocalSafetensorsCheckpoint) -> Gemma4Backend:
    backend = resolve_backend(checkpoint.to_backend_signature())
    if not isinstance(backend, Gemma4Backend):
        raise BackendMismatchError(
            "prune apply requires a Gemma4 backend with prune tensor helpers",
            model_id=checkpoint.model_id,
            backend_name=getattr(backend, "name", None),
        )
    return backend


def _metadata_bundle_from_checkpoint(
    checkpoint: LocalSafetensorsCheckpoint,
    *,
    backend: Gemma4Backend,
) -> LoadedBackendBundle:
    metadata_state = {item.tensor_key: item for item in checkpoint.tensor_metadata()}
    return LoadedBackendBundle(
        backend_name=backend.name,
        model_handle=ModelHandle(
            model_id=checkpoint.model_id,
            revision=checkpoint.revision,
            backend_name=backend.name,
            source_path=str(checkpoint.checkpoint_dir),
        ),
        model=object(),
        config=checkpoint.config,
        metadata={
            "state_dict": metadata_state,
            "backend_version": backend.backend_version,
        },
    )


def _bundle_with_state(
    checkpoint: LocalSafetensorsCheckpoint,
    *,
    backend: Gemma4Backend,
    state: Mapping[str, object],
) -> LoadedBackendBundle:
    return LoadedBackendBundle(
        backend_name=backend.name,
        model_handle=ModelHandle(
            model_id=checkpoint.model_id,
            revision=checkpoint.revision,
            backend_name=backend.name,
            source_path=str(checkpoint.checkpoint_dir),
        ),
        model=object(),
        config=checkpoint.config,
        metadata={
            "state_dict": state,
            "backend_version": backend.backend_version,
        },
    )


def _validate_plan_against_topology(
    plan: PrunePlan,
    *,
    topology: Sequence[LayerTopology],
    model_id: str,
) -> Mapping[int, PrunePlanItem]:
    expected_layers = {layer.layer_index for layer in topology}
    plan_items = {item.layer_index: item for item in sort_plan_items(plan.per_layer_plans)}
    missing_layers = tuple(sorted(expected_layers.difference(plan_items)))
    unknown_layers = tuple(sorted(set(plan_items).difference(expected_layers)))
    if unknown_layers:
        raise TopologyMismatchError(
            "prune plan contains unknown MoE layer coverage",
            model_id=model_id,
            layer_index=unknown_layers[0],
            details={"unknown_layer_indices": ",".join(str(index) for index in unknown_layers)},
        )
    if missing_layers:
        raise TopologyMismatchError(
            "prune plan is missing MoE layer coverage",
            model_id=model_id,
            layer_index=missing_layers[0],
            details={"missing_layer_indices": ",".join(str(index) for index in missing_layers)},
        )
    for layer in topology:
        item = plan_items[layer.layer_index]
        if len(item.keep_indices) + len(item.drop_indices) != layer.expert_count:
            raise TopologyMismatchError(
                "prune plan expert count does not match layer topology",
                model_id=model_id,
                layer_index=layer.layer_index,
                details={
                    "plan_expert_count": len(item.keep_indices) + len(item.drop_indices),
                    "layer_expert_count": layer.expert_count,
                },
            )
        if len(item.keep_indices) < layer.top_k:
            raise TopologyMismatchError(
                "prune plan target expert count cannot be below layer top_k",
                model_id=model_id,
                layer_index=layer.layer_index,
                details={
                    "target_expert_count": len(item.keep_indices),
                    "layer_top_k": layer.top_k,
                },
            )
    return plan_items


def _build_layer_reports(
    *,
    backend: Gemma4Backend,
    bundle: LoadedBackendBundle,
    topology: Sequence[LayerTopology],
    plan_items: Mapping[int, PrunePlanItem],
) -> tuple[ApplyLayerReport, ...]:
    reports: list[ApplyLayerReport] = []
    state_index = bundle.metadata.get("state_dict")
    if not isinstance(state_index, Mapping):
        raise TopologyMismatchError(
            "prune apply requires metadata state_dict for source validation",
            model_id=bundle.model_handle.model_id,
        )
    for layer in topology:
        item = plan_items[layer.layer_index]
        tensor_keys = backend.resolve_prune_tensor_keys(bundle, layer_index=layer.layer_index)
        for tensor_role, tensor_key in tensor_keys.items():
            backend.validate_prune_tensor(
                bundle,
                layer=layer,
                tensor_role=tensor_role,
                tensor_key=tensor_key,
                tensor_value=state_index[tensor_key],
                target_expert_count=layer.expert_count,
            )
        expected_source_shapes = backend.expected_prune_tensor_shapes(
            layer=layer,
            target_expert_count=layer.expert_count,
        )
        expected_target_shapes = backend.expected_prune_tensor_shapes(
            layer=layer,
            target_expert_count=len(item.keep_indices),
        )

        deltas = [
            ApplyTensorDelta(
                tensor_key=tensor_keys["experts_down_proj"],
                tensor_role="experts_down_proj",
                source_shape=expected_source_shapes["experts_down_proj"],
                target_shape=expected_target_shapes["experts_down_proj"],
                rewritten=True,
            ),
            ApplyTensorDelta(
                tensor_key=tensor_keys["experts_gate_up_proj"],
                tensor_role="experts_gate_up_proj",
                source_shape=expected_source_shapes["experts_gate_up_proj"],
                target_shape=expected_target_shapes["experts_gate_up_proj"],
                rewritten=True,
            ),
            ApplyTensorDelta(
                tensor_key=tensor_keys["router_per_expert_scale"],
                tensor_role="router_per_expert_scale",
                source_shape=expected_source_shapes["router_per_expert_scale"],
                target_shape=expected_target_shapes["router_per_expert_scale"],
                rewritten=True,
            ),
            ApplyTensorDelta(
                tensor_key=tensor_keys["router_proj"],
                tensor_role="router_proj",
                source_shape=expected_source_shapes["router_proj"],
                target_shape=expected_target_shapes["router_proj"],
                rewritten=True,
            ),
            ApplyTensorDelta(
                tensor_key=tensor_keys["router_scale"],
                tensor_role="router_scale",
                source_shape=(),
                target_shape=(),
                rewritten=False,
            ),
        ]
        reports.append(
            ApplyLayerReport(
                layer_index=layer.layer_index,
                layer_name=layer.layer_name,
                source_expert_count=layer.expert_count,
                target_expert_count=len(item.keep_indices),
                keep_indices=item.ordered_keep_indices,
                drop_indices=item.ordered_drop_indices,
                old_to_new_index=tuple((old_index, new_index) for new_index, old_index in enumerate(item.ordered_keep_indices)),
                tensor_keys_to_rewrite=tuple(
                    sorted(
                        delta.tensor_key
                        for delta in deltas
                        if delta.rewritten
                    )
                ),
                tensor_deltas=tuple(deltas),
            )
        )
    return tuple(reports)


def _source_metadata_digest(checkpoint: LocalSafetensorsCheckpoint) -> str:
    payload = {
        "config": checkpoint.config,
        "model_id": checkpoint.model_id,
        "revision": checkpoint.revision,
        "tensors": [
            {
                "tensor_key": item.tensor_key,
                "shape": list(item.shape),
                "dtype": item.dtype,
                "shard_filename": item.shard_filename,
            }
            for item in checkpoint.tensor_metadata()
        ],
    }
    return sha256(to_json(payload).encode("utf-8")).hexdigest()


def _build_validation_state_from_metadata(
    *,
    checkpoint: LocalSafetensorsCheckpoint,
    layer_reports: Sequence[ApplyLayerReport],
) -> Mapping[str, object]:
    metadata_by_key = {item.tensor_key: item for item in checkpoint.tensor_metadata()}
    overridden: dict[str, object] = {}
    for report in layer_reports:
        for delta in report.tensor_deltas:
            if delta.rewritten:
                source_metadata = metadata_by_key[delta.tensor_key]
                overridden[delta.tensor_key] = _TensorShapeView(
                    shape=delta.target_shape,
                    dtype=source_metadata.dtype,
                )
    return {
        key: overridden.get(key, metadata_by_key[key])
        for key in checkpoint.state_keys()
    }


def _remap_expert_tensor(
    tensor: torch.Tensor,
    *,
    keep_indices: Sequence[int],
) -> torch.Tensor:
    index = torch.tensor(tuple(keep_indices), dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, 0, index).clone()


def _derived_target_layer(layer: LayerTopology, *, target_expert_count: int) -> LayerTopology:
    return replace(
        layer,
        expert_count=target_expert_count,
        metadata={
            **dict(layer.metadata),
            "source_expert_count": layer.expert_count,
        },
    )


__all__ = [
    "ApplyLayerReport",
    "ApplyResult",
    "ApplyTensorDelta",
    "apply_prune_plan",
]
