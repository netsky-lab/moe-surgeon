"""Offline-safe runtime router hook instrumentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import log
from pathlib import Path
from types import TracebackType
from typing import Callable, Iterable, Iterator, Literal, Mapping, MutableSequence, Protocol, Sequence, cast

from moe_surgeon.models.backend import LoadedBackendBundle, ModelBackend
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.schemas import (
    CANONICAL_DEFAULT_TIMESTAMP,
    ActivationStats,
    LayerTopology,
    RouterState,
    RunArtifactManifest,
    to_json,
    to_json_file,
)


class RemovableHandle(Protocol):
    """Minimal removable hook handle protocol."""

    def remove(self) -> None:
        """Remove an attached hook."""


@dataclass(frozen=True)
class RouterActivationRecord:
    """Normalized router forward output captured for one layer invocation."""

    layer_index: int
    top_k_indices: object
    top_k_weights: object
    router_scores: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class RouterCaptureCollector:
    """Append-only collector for router activation records."""

    records: MutableSequence[RouterActivationRecord] = field(default_factory=list)

    def append(self, record: RouterActivationRecord) -> None:
        """Store one normalized router output."""

        self.records.append(record)

    def clear(self) -> None:
        """Drop all currently collected records."""

        self.records.clear()


@dataclass
class _ExpertAccumulator:
    token_count: int = 0
    weighted_token_count: float = 0.0
    mass_sum: float = 0.0
    entropy_sum: float = 0.0
    top1_mass: float = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    """Canonical benchmark artifact payload."""

    manifest: RunArtifactManifest
    topology: tuple[LayerTopology, ...]
    activation_stats: tuple[ActivationStats, ...]
    profiler_config: Mapping[str, object] = field(default_factory=dict)
    input_payload_hash: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-ready payload for deterministic artifact writing."""

        return {
            "manifest": self.manifest.to_dict(),
            "profiler_config": _canonical_json_mapping(self.profiler_config),
            "topology": [layer.to_dict() for layer in self.topology],
            "activation_stats": [item.to_dict() for item in self.activation_stats],
            "input_payload_hash": self.input_payload_hash,
        }

    def to_json(self, *, compact: bool = True) -> str:
        """Return the canonical benchmark artifact JSON payload."""

        return to_json(self.to_payload(), compact=compact)

    def write_json(self, path: str | Path) -> Path:
        """Write the canonical benchmark artifact payload to disk."""

        return to_json_file(path, self.to_payload())


@dataclass(frozen=True)
class PromptBatch:
    """Tokenizer-ready prompt batch used by the offline-safe bench flow."""

    prompts: tuple[str, ...]
    prompt_indices: tuple[int, ...]
    encoded_inputs: Mapping[str, object]
    attention_mask: object
    active_token_count: int

    def prompt_payload(self) -> dict[str, object]:
        """Return a canonical JSON-ready batch summary."""

        return {
            "prompts": list(self.prompts),
            "prompt_indices": list(self.prompt_indices),
            "active_token_count": self.active_token_count,
        }


class RouterActivationProfiler:
    """Context-managed router hook profiler with deterministic layer ordering."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        bundle: LoadedBackendBundle,
        topology: Sequence[LayerTopology],
        include_router_scores: bool = False,
        collector: RouterCaptureCollector | None = None,
    ) -> None:
        self._backend = backend
        self._bundle = bundle
        self._topology = tuple(sorted(topology, key=lambda layer: layer.layer_index))
        self._include_router_scores = include_router_scores
        self._collector = collector or RouterCaptureCollector()
        self._handles: list[RemovableHandle] = []
        self._attached = False
        self._router_states: dict[int, RouterState] = {}
        self._aggregates: dict[int, dict[int, _ExpertAccumulator]] = {}
        self._layer_token_totals: dict[int, int] = {}
        self._layer_weight_totals: dict[int, float] = {}
        self._layer_topology = {layer.layer_index: layer for layer in self._topology}

    @property
    def collector(self) -> RouterCaptureCollector:
        """Return the mutable forward-output collector."""

        return self._collector

    @property
    def topology(self) -> tuple[LayerTopology, ...]:
        """Return the deterministic topology order for this profiler."""

        return self._topology

    @property
    def records(self) -> Sequence[RouterActivationRecord]:
        """Return the currently collected router records."""

        return tuple(self._collector.records)

    @property
    def attached(self) -> bool:
        """Return whether hooks are currently installed."""

        return self._attached

    def attach(self) -> "RouterActivationProfiler":
        """Attach forward hooks to all configured router modules."""

        if self._attached:
            return self

        try:
            for layer in self._topology:
                router_state = self._backend.extract_router_state(self._bundle, layer=layer)
                self._backend.validate_layer(self._bundle, layer=layer, router_state=router_state)
                module = self._backend.resolve_router_module(self._bundle, layer=layer)
                register_forward_hook = getattr(module, "register_forward_hook", None)
                if not callable(register_forward_hook):
                    raise TopologyMismatchError(
                        "router module does not support register_forward_hook",
                        model_id=self._bundle.model_handle.model_id,
                        layer_index=layer.layer_index,
                        details={"module_type": type(module).__name__},
                    )
                handle = register_forward_hook(self._build_hook(layer=layer, router_state=router_state))
                if not hasattr(handle, "remove"):
                    raise TopologyMismatchError(
                        "router hook registration did not return removable handle",
                        model_id=self._bundle.model_handle.model_id,
                        layer_index=layer.layer_index,
                        details={"module_type": type(module).__name__},
                    )
                self._router_states[layer.layer_index] = router_state
                self._handles.append(handle)
        except Exception:
            self.detach()
            raise

        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all installed hooks, even after partial attachment."""

        while self._handles:
            handle = self._handles.pop()
            handle.remove()
        self._router_states.clear()
        self._attached = False

    def clear_records(self) -> None:
        """Clear captured forward outputs without touching hook state."""

        self._collector.clear()

    def reset_aggregation(self) -> None:
        """Reset accumulated activation totals."""

        self._aggregates.clear()
        self._layer_token_totals.clear()
        self._layer_weight_totals.clear()

    def accumulate(
        self,
        *,
        attention_mask: object,
        position_mask: object | None = None,
        clear_records: bool = True,
    ) -> tuple[ActivationStats, ...]:
        """Aggregate buffered captures into deterministic activation stats."""

        for record in self.records:
            layer = self._layer_topology[record.layer_index]
            router_state = self._router_states.get(record.layer_index)
            if router_state is None:
                router_state = self._backend.extract_router_state(self._bundle, layer=layer)
                self._router_states[record.layer_index] = router_state
            output_shape = self._shape_tuple(record.top_k_indices, allow_scalar=False)[:-1]
            active_mask = self._resolve_active_mask(
                attention_mask=attention_mask,
                position_mask=position_mask,
                output_shape=output_shape,
                layer=layer,
            )
            indices_data = self._to_python_nested(record.top_k_indices)
            weights_data = self._to_python_nested(record.top_k_weights)
            active_positions = 0
            active_weight_total = 0.0
            for rank_indices, rank_weights in self._iter_active_positions(
                indices_data=indices_data,
                weights_data=weights_data,
                active_mask=active_mask,
                layer=layer,
                router_state=router_state,
            ):
                active_positions += 1
                position_weight_total = 0.0
                layer_aggregate = self._aggregates.setdefault(layer.layer_index, {})
                for rank, (raw_expert_index, raw_weight) in enumerate(zip(rank_indices, rank_weights)):
                    expert_index = self._coerce_expert_index(raw_expert_index, layer=layer)
                    weight = self._coerce_weight(raw_weight, layer=layer)
                    position_weight_total += weight
                    aggregate = layer_aggregate.setdefault(expert_index, _ExpertAccumulator())
                    aggregate.token_count += 1
                    aggregate.weighted_token_count += weight
                    aggregate.mass_sum += weight
                    if weight > 0:
                        aggregate.entropy_sum += -weight * log(weight)
                    if rank == 0:
                        aggregate.top1_mass += weight
                active_weight_total += position_weight_total
            self._layer_token_totals[layer.layer_index] = self._layer_token_totals.get(layer.layer_index, 0) + active_positions
            self._layer_weight_totals[layer.layer_index] = self._layer_weight_totals.get(layer.layer_index, 0.0) + active_weight_total
        stats = self.activation_stats()
        if clear_records:
            self.clear_records()
        return stats

    def activation_stats(self) -> tuple[ActivationStats, ...]:
        """Return per-expert activation totals for each configured layer."""

        from moe_surgeon.schemas import sort_activation_stats

        stats: list[ActivationStats] = []
        for layer in self._topology:
            n_tokens = self._layer_token_totals.get(layer.layer_index, 0)
            weighted_n_tokens = self._layer_weight_totals.get(layer.layer_index, 0.0)
            layer_aggregate = self._aggregates.get(layer.layer_index, {})
            for expert_index in range(layer.expert_count):
                aggregate = layer_aggregate.get(expert_index, _ExpertAccumulator())
                mean_weight = (
                    aggregate.weighted_token_count / aggregate.token_count if aggregate.token_count > 0 else 0.0
                )
                entropy = aggregate.entropy_sum / aggregate.token_count if aggregate.token_count > 0 else 0.0
                density = float(aggregate.token_count) / float(n_tokens) if n_tokens > 0 else 0.0
                stats.append(
                    ActivationStats(
                        layer_index=layer.layer_index,
                        expert_index=expert_index,
                        token_count=aggregate.token_count,
                        weighted_token_count=aggregate.weighted_token_count,
                        mass_sum=aggregate.mass_sum,
                        mean_weight=mean_weight,
                        entropy=entropy,
                        n_tokens=n_tokens,
                        weighted_n_tokens=weighted_n_tokens,
                        top1_mass=aggregate.top1_mass,
                        density=density,
                    )
                )
        return sort_activation_stats(stats)

    def __enter__(self) -> "RouterActivationProfiler":
        return self.attach()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.detach()
        return False

    def _build_hook(
        self,
        *,
        layer: LayerTopology,
        router_state: RouterState,
    ) -> Callable[[object, tuple[object, ...], object], object | None]:
        def _hook(module: object, args: tuple[object, ...], output: object) -> None:
            record = self._normalize_record(layer=layer, router_state=router_state, output=output)
            self._collector.append(record)
            return None

        return _hook

    def _normalize_record(
        self,
        *,
        layer: LayerTopology,
        router_state: RouterState,
        output: object,
    ) -> RouterActivationRecord:
        values = self._coerce_output_mapping(output)
        top_k_indices = self._require_output_value(
            values,
            field_name="top_k_indices",
            aliases=("top_k_indices", "topk_indices", "expert_indices", "indices"),
            layer=layer,
        )
        top_k_weights = self._require_output_value(
            values,
            field_name="top_k_weights",
            aliases=("top_k_weights", "topk_weights", "router_probs", "weights"),
            layer=layer,
        )
        router_scores = self._optional_output_value(
            values,
            aliases=("router_scores", "scores", "logits", "router_logits"),
        )
        self._validate_capture_shape(
            value=top_k_indices,
            expected_last_dim=router_state.top_k,
            field_name="top_k_indices",
            layer=layer,
            allow_scalar=False,
        )
        self._validate_capture_shape(
            value=top_k_weights,
            expected_last_dim=router_state.top_k,
            field_name="top_k_weights",
            layer=layer,
            allow_scalar=False,
        )
        if self._include_router_scores and router_scores is None:
            raise ShapeInvariantViolationError(
                "router scores capture requested but not present in router output",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        if router_scores is not None:
            self._validate_capture_shape(
                value=router_scores,
                expected_last_dim=router_state.num_experts,
                field_name="router_scores",
                layer=layer,
                allow_scalar=False,
            )
        elif self._include_router_scores:
            raise ShapeInvariantViolationError(
                "router scores capture requested but not present in router output",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )

        return RouterActivationRecord(
            layer_index=layer.layer_index,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            router_scores=router_scores if self._include_router_scores else None,
            metadata={"layer_name": layer.layer_name},
        )

    def _coerce_output_mapping(self, output: object) -> Mapping[str, object]:
        if isinstance(output, Mapping):
            return {str(key): value for key, value in output.items()}
        if hasattr(output, "__dict__"):
            return {
                str(key): value
                for key, value in vars(output).items()
                if not str(key).startswith("_")
            }
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            sequence_output = list(output)
            if len(sequence_output) < 2 or len(sequence_output) > 3:
                raise ShapeInvariantViolationError("router forward output sequence must contain 2 or 3 items")
            mapping: dict[str, object] = {
                "top_k_indices": sequence_output[0],
                "top_k_weights": sequence_output[1],
            }
            if len(sequence_output) == 3:
                mapping["router_scores"] = sequence_output[2]
            return mapping
        raise ShapeInvariantViolationError(
            "router forward output must be mapping, object, or sequence",
        )

    def _require_output_value(
        self,
        values: Mapping[str, object],
        *,
        field_name: str,
        aliases: Sequence[str],
        layer: LayerTopology,
    ) -> object:
        resolved = self._optional_output_value(values, aliases=aliases)
        if resolved is None:
            raise ShapeInvariantViolationError(
                f"router forward output missing {field_name}",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                details={"available_fields": ",".join(sorted(values))},
            )
        return resolved

    def _optional_output_value(self, values: Mapping[str, object], *, aliases: Sequence[str]) -> object | None:
        for alias in aliases:
            if alias in values:
                return values[alias]
        return None

    def _validate_capture_shape(
        self,
        *,
        value: object,
        expected_last_dim: int,
        field_name: str,
        layer: LayerTopology,
        allow_scalar: bool,
    ) -> None:
        shape = self._shape_tuple(value, allow_scalar=allow_scalar)
        if not shape:
            raise ShapeInvariantViolationError(
                f"{field_name} shape cannot be empty",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        if shape[-1] != expected_last_dim:
            raise ShapeInvariantViolationError(
                f"{field_name} last dimension mismatch",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                expected_shape=(expected_last_dim,),
                actual_shape=shape,
            )

    def _shape_tuple(self, value: object, *, allow_scalar: bool) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if shape is None:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                normalized = self._infer_nested_shape(self._to_python_nested(value))
            else:
                raise ShapeInvariantViolationError("captured router tensor must expose shape")
        elif isinstance(shape, tuple):
            normalized = tuple(int(item) for item in shape)
        elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
            normalized = tuple(int(item) for item in shape)
        else:
            raise ShapeInvariantViolationError("captured router tensor shape must be sequence")
        if not allow_scalar and len(normalized) == 0:
            raise ShapeInvariantViolationError("captured router tensor shape cannot be scalar")
        if any(item < 0 for item in normalized):
            raise ShapeInvariantViolationError("captured router tensor shape dimensions must be non-negative")
        return normalized

    def _resolve_active_mask(
        self,
        *,
        attention_mask: object,
        position_mask: object | None,
        output_shape: tuple[int, ...],
        layer: LayerTopology,
    ) -> object:
        mask = self._align_mask(mask=attention_mask, output_shape=output_shape, layer=layer, field_name="attention_mask")
        if position_mask is None:
            return mask
        return self._combine_masks(
            left=mask,
            right=self._align_mask(mask=position_mask, output_shape=output_shape, layer=layer, field_name="position_mask"),
        )

    def _align_mask(
        self,
        *,
        mask: object,
        output_shape: tuple[int, ...],
        layer: LayerTopology,
        field_name: str,
    ) -> object:
        mask_data = self._to_python_nested(mask)
        mask_shape = self._infer_nested_shape(mask_data)
        if mask_shape == output_shape:
            return mask_data
        if len(output_shape) == 2 and len(mask_shape) == 2 and mask_shape[0] == output_shape[0] and mask_shape[1] >= output_shape[1]:
            rows = cast(list[object], mask_data)
            return [cast(list[object], row)[-output_shape[1] :] for row in rows]
        if len(output_shape) == 1 and len(mask_shape) == 1 and mask_shape[0] >= output_shape[0]:
            entries = cast(list[object], mask_data)
            return entries[-output_shape[0] :]
        raise ShapeInvariantViolationError(
            f"{field_name} shape is incompatible with captured router outputs",
            model_id=self._bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            expected_shape=output_shape,
            actual_shape=mask_shape,
        )

    def _combine_masks(self, *, left: object, right: object) -> object:
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise ShapeInvariantViolationError("mask operands must have matching shapes")
            return [self._combine_masks(left=item_left, right=item_right) for item_left, item_right in zip(left, right)]
        return bool(left) and bool(right)

    def _iter_active_positions(
        self,
        *,
        indices_data: object,
        weights_data: object,
        active_mask: object,
        layer: LayerTopology,
        router_state: RouterState,
    ) -> Iterable[tuple[list[object], list[object]]]:
        if isinstance(active_mask, list):
            if not isinstance(indices_data, list) or not isinstance(weights_data, list):
                raise ShapeInvariantViolationError(
                    "captured router outputs do not match active mask rank",
                    model_id=self._bundle.model_handle.model_id,
                    layer_index=layer.layer_index,
                )
            if len(indices_data) != len(weights_data) or len(indices_data) != len(active_mask):
                raise ShapeInvariantViolationError(
                    "captured router outputs do not match active mask shape",
                    model_id=self._bundle.model_handle.model_id,
                    layer_index=layer.layer_index,
                )
            for child_indices, child_weights, child_mask in zip(indices_data, weights_data, active_mask):
                yield from self._iter_active_positions(
                    indices_data=child_indices,
                    weights_data=child_weights,
                    active_mask=child_mask,
                    layer=layer,
                    router_state=router_state,
                )
            return
        if not bool(active_mask):
            return
        if not isinstance(indices_data, list) or not isinstance(weights_data, list):
            raise ShapeInvariantViolationError(
                "captured router outputs must expose per-position top-k lists",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        if len(indices_data) != router_state.top_k or len(weights_data) != router_state.top_k:
            raise ShapeInvariantViolationError(
                "captured router outputs do not match configured top_k",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                expected_shape=(router_state.top_k,),
                actual_shape=(len(indices_data),),
            )
        yield list(indices_data), list(weights_data)

    def _coerce_expert_index(self, value: object, *, layer: LayerTopology) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShapeInvariantViolationError(
                "captured expert index must be numeric",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        expert_index = int(value)
        if expert_index < 0 or expert_index >= layer.expert_count:
            raise TopologyMismatchError(
                "captured expert index is out of range",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                details={"expert_index": expert_index, "expert_count": layer.expert_count},
            )
        return expert_index

    def _coerce_weight(self, value: object, *, layer: LayerTopology) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShapeInvariantViolationError(
                "captured router weight must be numeric",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        weight = float(value)
        if weight < 0:
            raise ShapeInvariantViolationError(
                "captured router weight must be non-negative",
                model_id=self._bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        return weight

    def _to_python_nested(self, value: object) -> object:
        candidate = value
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        tolist = getattr(candidate, "tolist", None)
        if callable(tolist):
            candidate = tolist()
        return candidate

    def _infer_nested_shape(self, value: object) -> tuple[int, ...]:
        if isinstance(value, list):
            if not value:
                return (0,)
            child_shape = self._infer_nested_shape(value[0])
            for child in value[1:]:
                if self._infer_nested_shape(child) != child_shape:
                    raise ShapeInvariantViolationError("nested tensor data must be rectangular")
            return (len(value), *child_shape)
        return ()


def benchmark(
    *,
    profiler: RouterActivationProfiler,
    prompts: Sequence[str] = (),
    input_payloads: Sequence[Mapping[str, object]] = (),
    profiler_config: Mapping[str, object] | None = None,
    parent_artifacts: Sequence[str] = (),
    output_paths: Mapping[str, str] | None = None,
    seed: int | None = None,
    started_at: str = CANONICAL_DEFAULT_TIMESTAMP,
    finished_at: str | None = None,
) -> BenchmarkResult:
    """Build a deterministic benchmark artifact from accumulated activation stats."""

    from moe_surgeon.analysis.scan import align_activation_stats

    ordered_topology = profiler.topology
    activation_stats = align_activation_stats(layers=ordered_topology, stats=profiler.activation_stats())
    canonical_profiler_config = _canonical_json_mapping(profiler_config or {})
    canonical_input_payloads = tuple(_canonical_json_mapping(payload) for payload in input_payloads)
    prompt_payload = {"prompts": list(prompts)}
    prompt_hash = sha256(to_json(prompt_payload).encode("utf-8")).hexdigest()
    input_payload_hash = None
    if canonical_input_payloads:
        input_payload_hash = sha256(
            to_json({"input_payloads": list(canonical_input_payloads)}).encode("utf-8")
        ).hexdigest()
    prompt_set_hash = input_payload_hash if not prompts and input_payload_hash is not None else prompt_hash
    config_hash = sha256(to_json(canonical_profiler_config).encode("utf-8")).hexdigest()
    model_fingerprint = profiler._bundle.model_handle.model_fingerprint
    run_seed = profiler._bundle.model_handle.seed if seed is None else seed
    run_identity = {
        "backend_name": profiler._bundle.backend_name,
        "model_fingerprint": model_fingerprint,
        "prompt_set_hash": prompt_set_hash,
        "input_payload_hash": input_payload_hash,
        "profiler_config_hash": config_hash,
        "top_k": ordered_topology[0].top_k if ordered_topology else 1,
        "seed": run_seed,
        "layer_indices": [layer.layer_index for layer in ordered_topology],
    }
    run_digest = sha256(to_json(run_identity).encode("utf-8")).hexdigest()
    manifest = RunArtifactManifest(
        run_id=f"bench-{run_digest[:16]}",
        command="bench",
        model_handle=profiler._bundle.model_handle,
        top_k=ordered_topology[0].top_k if ordered_topology else 1,
        prompt_count=len(prompts) if prompts else len(input_payloads),
        seed=run_seed,
        prompt_set_hash=prompt_set_hash,
        started_at=started_at,
        finished_at=finished_at,
        input_checksums={
            "prompt_set": prompt_set_hash,
            "profiler_config": config_hash,
            **({} if input_payload_hash is None else {"input_payloads": input_payload_hash}),
        },
        output_paths={} if output_paths is None else dict(sorted(output_paths.items())),
        parent_artifacts=tuple(sorted(str(item) for item in parent_artifacts)),
        metadata={
            "model_fingerprint": model_fingerprint,
            "profiler_config_hash": config_hash,
            "result_digest": run_digest,
            **({} if input_payload_hash is None else {"input_payload_hash": input_payload_hash}),
            **{f"profiler_config.{key}": _metadata_scalar(value) for key, value in canonical_profiler_config.items()},
        },
    )
    return BenchmarkResult(
        manifest=manifest,
        topology=ordered_topology,
        activation_stats=activation_stats,
        profiler_config=canonical_profiler_config,
        input_payload_hash=input_payload_hash,
    )


def iter_prompt_batches(
    *,
    tokenizer: Callable[..., object],
    prompts: Sequence[str],
    batch_size: int,
    tokenizer_kwargs: Mapping[str, object] | None = None,
) -> Iterator[PromptBatch]:
    """Yield deterministic prompt batches with explicit attention-mask totals."""

    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    kwargs: dict[str, object] = {"padding": True, "return_attention_mask": True}
    if tokenizer_kwargs is not None:
        kwargs.update(dict(tokenizer_kwargs))

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = tuple(prompts[batch_start : batch_start + batch_size])
        batch_indices = tuple(range(batch_start, batch_start + len(batch_prompts)))
        encoded = _normalize_tokenizer_output(tokenizer(batch_prompts, **kwargs))
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            raise ShapeInvariantViolationError("tokenizer output must include attention_mask")
        active_token_count = _count_mask_tokens(attention_mask)
        yield PromptBatch(
            prompts=batch_prompts,
            prompt_indices=batch_indices,
            encoded_inputs=encoded,
            attention_mask=attention_mask,
            active_token_count=active_token_count,
        )


def _canonical_json_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in sorted(payload.items()):
        if isinstance(value, Mapping):
            normalized[str(key)] = _canonical_json_mapping(cast(Mapping[str, object], value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized[str(key)] = [_canonical_json_value(item) for item in value]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized[str(key)] = value
        else:
            normalized[str(key)] = str(value)
    return normalized


def _canonical_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _canonical_json_mapping(cast(Mapping[str, object], value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metadata_scalar(value: object) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_tokenizer_output(output: object) -> dict[str, object]:
    if isinstance(output, Mapping):
        return {str(key): value for key, value in output.items()}
    if hasattr(output, "__dict__"):
        return {
            str(key): value
            for key, value in vars(output).items()
            if not str(key).startswith("_")
        }
    raise ShapeInvariantViolationError("tokenizer output must be mapping or object")


def _count_mask_tokens(mask: object) -> int:
    candidate = mask
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()
    mask = candidate
    if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes)):
        return sum(_count_mask_tokens(item) for item in mask)
    return 1 if bool(mask) else 0


__all__ = [
    "benchmark",
    "BenchmarkResult",
    "iter_prompt_batches",
    "PromptBatch",
    "RouterActivationProfiler",
    "RouterActivationRecord",
    "RouterCaptureCollector",
]
