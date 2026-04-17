"""Canonical schema contracts for deterministic MoE analysis and pruning.

All objects in this module are pure data containers with explicit validation,
deterministic ordering helpers, and reversible JSON serialization.
The module is intentionally free from runtime heavy dependencies such as
``torch`` and ``transformers``.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import floor, isfinite
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_type_hints
import json
import re

CANONICAL_SCHEMA_VERSION = "1.0.0"
CANONICAL_FLOAT_EPSILON = 1e-12
_LAYER_REF_PATTERNS = (re.compile(r"^layer_(\d+)$"), re.compile(r"^module_(\d+)$"))


class SchemaValidationError(ValueError):
    """Base error for schema and contract violations."""


class ShapeInvariantViolationError(SchemaValidationError):
    """Raised when tensor-like metadata is malformed."""


class TopologyMismatchError(SchemaValidationError):
    """Raised when topology-level invariants cannot be satisfied."""


class LayerReferenceError(SchemaValidationError):
    """Raised when a layer reference string is invalid."""


SchemaKey = Union[str, int, float, bool, None]
ShapeTuple = Tuple[int, ...]


SchemaType = TypeVar("SchemaType", bound="_SchemaBase")


def _utcnow_iso() -> str:
    """Current UTC timestamp in stable ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{name} must be bool")
    return value


def _ensure_non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{name} must be int")
    if value < 0:
        raise SchemaValidationError(f"{name} must be >= 0")
    return value


def _ensure_positive_int(value: Any, *, name: str) -> int:
    value = _ensure_non_negative_int(value, name=name)
    if value <= 0:
        raise SchemaValidationError(f"{name} must be > 0")
    return value


def _ensure_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    value_f = float(value)
    if not isfinite(value_f):
        raise SchemaValidationError(f"{name} must be finite")
    return value_f


def _ensure_non_empty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be non-empty string")
    return value


def _ensure_optional_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _ensure_non_negative_int(value, name=name)


def _ensure_optional_positive_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _ensure_positive_int(value, name=name)


def _ensure_shape_tuple(value: Any, *, name: str, allow_empty: bool = False) -> Optional[ShapeTuple]:
    if value is None:
        return None
    if isinstance(value, tuple):
        raw = value
    elif isinstance(value, list):
        raw = tuple(value)
    else:
        raise ShapeInvariantViolationError(f"{name} must be list or tuple")

    if not allow_empty and len(raw) == 0:
        raise ShapeInvariantViolationError(f"{name} must be non-empty")

    out: list[int] = []
    for idx, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ShapeInvariantViolationError(f"{name}[{idx}] must be non-negative int")
        out.append(item)
    return tuple(out)


def _canonicalize_metadata(value: Mapping[str, Any] | None) -> Dict[str, SchemaKey]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaValidationError("metadata must be mapping")
    out: Dict[str, SchemaKey] = {}
    for key, val in value.items():
        if not isinstance(key, str):
            raise SchemaValidationError("metadata keys must be strings")
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[key] = val
        else:
            raise SchemaValidationError("metadata values must be JSON scalar")
    return out


def _canonicalize_int_tuple(values: Sequence[int] | None, *, name: str) -> Tuple[int, ...]:
    if values is None:
        return ()
    out = []
    for i, item in enumerate(values):
        try:
            out.append(_ensure_non_negative_int(item, name=f"{name}[{i}]"))
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"{name} contains invalid entry: {exc}") from exc
    return tuple(out)


def _canonicalize_str_tuple(values: Sequence[str] | None, *, name: str) -> Tuple[str, ...]:
    if values is None:
        return ()
    out: list[str] = []
    for i, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise SchemaValidationError(f"{name}[{i}] must be non-empty string")
        out.append(item)
    return tuple(out)


def _parse_layer_ref(value: str) -> int:
    for pattern in _LAYER_REF_PATTERNS:
        match = pattern.match(value)
        if match is not None:
            return int(match.group(1))
    raise LayerReferenceError("layer_ref must match 'layer_<n>' or 'module_<n>'")


def _sort_bucket(value: float, *, epsilon: float) -> int:
    eps = max(float(epsilon), CANONICAL_FLOAT_EPSILON)
    scaled = value / eps
    if scaled >= 0:
        return floor(scaled + 0.5)
    return floor(scaled - 0.5)


def _expert_sort_tuple(
    *,
    score: float,
    secondary: float,
    expert_index: int,
    layer_index: int,
    rank_position: int,
    score_epsilon: float,
    secondary_epsilon: float,
) -> Tuple[int, int, int, int, int]:
    return (
        -_sort_bucket(score, epsilon=score_epsilon),
        -_sort_bucket(secondary, epsilon=secondary_epsilon),
        int(layer_index),
        int(expert_index),
        int(rank_position),
    )


def _as_schema_data(value: Any) -> Any:
    if is_dataclass(value) and hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[union-attr]
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted((_as_schema_data(v) for v in value), key=repr)
    if isinstance(value, tuple):
        return [_as_schema_data(v) for v in value]
    if isinstance(value, list):
        return [_as_schema_data(v) for v in list(value)]
    if isinstance(value, dict):
        return {str(k): _as_schema_data(v) for k, v in value.items()}
    return value


def _coerce_mapping_for_payload(payload: Mapping[str, Any], target: Type["_SchemaBase"]) -> Dict[str, Any]:
    hints = get_type_hints(target)
    kwargs: Dict[str, Any] = {}
    for f in fields(target):
        if f.name.startswith("_"):
            continue
        if f.name in payload:
            value = payload[f.name]
        elif f.default is not MISSING:
            continue
        elif f.default_factory is not MISSING:
            continue
        else:
            raise SchemaValidationError(f"Missing required field '{f.name}' for {target.__name__}")

        annotation = hints.get(f.name)
        if annotation is tuple[int, ...]:
            if value is not None:
                value = _canonicalize_int_tuple(value, name=f.name)
            else:
                value = ()
        elif annotation is Optional[Tuple[int, ...]]:
            if value is not None:
                value = _ensure_shape_tuple(value, name=f.name, allow_empty=True)
        elif annotation is Tuple[str, ...]:
            value = _canonicalize_str_tuple(value, name=f.name) if value is not None else ()
        elif f.name == "per_layer_plans" and target.__name__ == "PrunePlan":
            converted: list[PrunePlanItem] = []
            for item in value or ():
                if isinstance(item, Mapping):
                    converted.append(PrunePlanItem.from_dict(item))
                elif isinstance(item, PrunePlanItem):
                    converted.append(item)
                else:
                    raise SchemaValidationError("per_layer_plans entries must be mappings")
            value = tuple(converted)
        elif f.name == "run_plan" and target.__name__ == "RunArtifactManifest":
            if value is None:
                pass
            elif isinstance(value, Mapping):
                value = PrunePlan.from_dict(value)
            elif not isinstance(value, PrunePlan):
                raise SchemaValidationError("run_plan must be a PrunePlan")
        elif f.name == "model_handle" and target.__name__ == "RunArtifactManifest":
            if value is None:
                pass
            elif isinstance(value, Mapping):
                value = ModelHandle.from_dict(value)
            elif not isinstance(value, ModelHandle):
                raise SchemaValidationError("model_handle must be a ModelHandle")

        kwargs[f.name] = value
    return kwargs


@dataclass
class _SchemaBase:
    """Base container with deterministic encoding/decoding helpers."""

    _schema_type: str = field(init=False, default="_SchemaBase", repr=False)

    @property
    def schema_version(self) -> str:
        return CANONICAL_SCHEMA_VERSION

    def _validate(self) -> None:
        return

    def __post_init__(self) -> None:
        self._validate()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["__schema_type"] = self.__class__.__name__
        data["__schema_version"] = self.schema_version
        return data

    @classmethod
    def from_dict(cls: Type[SchemaType], payload: Mapping[str, Any]) -> SchemaType:
        kwargs = _coerce_mapping_for_payload(payload, cls)
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class ModelHandle(_SchemaBase):
    """Canonical, lightweight handle for a model artifact.

    Metric units are metadata-only and have no tensor coupling.
    """

    model_id: str
    revision: Optional[str] = None
    backend_name: Optional[str] = None
    source_path: Optional[str] = None
    tokenizer_id: Optional[str] = None
    framework_version: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[str] = None
    seed: int = 0
    created_at: str = field(default_factory=_utcnow_iso)
    git_hash: Optional[str] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="ModelHandle", repr=False)

    def _validate(self) -> None:
        _ensure_non_empty_str(self.model_id, name="model_id")
        _ensure_non_empty_str(self.device, name="device")
        _ensure_non_negative_int(self.seed, name="seed")
        if self.revision is not None:
            _ensure_non_empty_str(self.revision, name="revision")
        if self.backend_name is not None:
            _ensure_non_empty_str(self.backend_name, name="backend_name")
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def layer_id(self) -> str:
        return f"{self.model_id}:layer"

    def layer_key(self, layer_index: int) -> str:
        _ensure_non_negative_int(layer_index, name="layer_index")
        return f"layer_{layer_index:04d}"

    @property
    def model_fingerprint(self) -> str:
        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "backend_name": self.backend_name,
            "source_path": self.source_path,
            "tokenizer_id": self.tokenizer_id,
            "framework_version": self.framework_version,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "git_hash": self.git_hash,
            "metadata": self.metadata,
        }
        return sha256(to_json(payload).encode("utf-8")).hexdigest()


@dataclass
class LayerTopology(_SchemaBase):
    """Static MoE layer topology metadata."""

    layer_index: int
    layer_name: str
    layer_type: str
    expert_count: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: Optional[int] = None
    expert_dim: Optional[int] = None
    ffn_in_features: Optional[int] = None
    ffn_out_features: Optional[int] = None
    layer_ref: Optional[str] = None
    module_paths: Dict[str, str] = field(default_factory=dict)
    is_moe: bool = True
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="LayerTopology", repr=False)

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_non_empty_str(self.layer_name, name="layer_name")
        _ensure_non_empty_str(self.layer_type, name="layer_type")
        _ensure_positive_int(self.expert_count, name="expert_count")
        _ensure_positive_int(self.top_k, name="top_k")
        _ensure_positive_int(self.hidden_size, name="hidden_size")
        _ensure_optional_positive_int(self.moe_intermediate_size, name="moe_intermediate_size")
        _ensure_optional_positive_int(self.expert_dim, name="expert_dim")
        _ensure_optional_positive_int(self.ffn_in_features, name="ffn_in_features")
        _ensure_optional_positive_int(self.ffn_out_features, name="ffn_out_features")
        if self.top_k > self.expert_count:
            raise TopologyMismatchError("top_k cannot exceed expert_count")
        if self.layer_ref is not None:
            _parse_layer_ref(self.layer_ref)
        self.module_paths = {str(k): str(v) for k, v in self.module_paths.items()}
        self.metadata = _canonicalize_metadata(self.metadata)
        self.is_moe = _ensure_bool(self.is_moe, name="is_moe")

    @property
    def layer_id(self) -> str:
        return f"layer_{self.layer_index:04d}"

    @property
    def expert_key(self) -> str:
        return f"{self.layer_id}:expert_{self.expert_count}"


@dataclass
class RouterState(_SchemaBase):
    """Router metadata captured for deterministic diagnostics."""

    layer_index: int
    num_experts: int
    top_k: int
    logits_shape: ShapeTuple
    top_k_indices_shape: Optional[ShapeTuple] = None
    top_k_weights_shape: Optional[ShapeTuple] = None
    projection_shape: Optional[ShapeTuple] = None
    per_expert_scale_shape: Optional[ShapeTuple] = None
    has_router_probabilities: bool = False
    has_raw_logits_capture: bool = False
    route_scale_present: bool = False
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="RouterState", repr=False)

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_positive_int(self.num_experts, name="num_experts")
        _ensure_positive_int(self.top_k, name="top_k")
        if self.top_k > self.num_experts:
            raise TopologyMismatchError("top_k cannot exceed num_experts")

        self.logits_shape = _ensure_shape_tuple(self.logits_shape, name="logits_shape", allow_empty=False)
        self.top_k_indices_shape = _ensure_shape_tuple(
            self.top_k_indices_shape,
            name="top_k_indices_shape",
            allow_empty=False,
        )
        self.top_k_weights_shape = _ensure_shape_tuple(
            self.top_k_weights_shape,
            name="top_k_weights_shape",
            allow_empty=False,
        )
        self.projection_shape = _ensure_shape_tuple(self.projection_shape, name="projection_shape", allow_empty=True)
        self.per_expert_scale_shape = _ensure_shape_tuple(
            self.per_expert_scale_shape,
            name="per_expert_scale_shape",
            allow_empty=True,
        )

        if self.top_k_indices_shape is not None and self.top_k_weights_shape is not None:
            if len(self.top_k_indices_shape) != len(self.top_k_weights_shape):
                raise ShapeInvariantViolationError(
                    "top_k_indices_shape and top_k_weights_shape must have same rank"
                )

        self.has_router_probabilities = _ensure_bool(self.has_router_probabilities, name="has_router_probabilities")
        self.has_raw_logits_capture = _ensure_bool(self.has_raw_logits_capture, name="has_raw_logits_capture")
        self.route_scale_present = _ensure_bool(self.route_scale_present, name="route_scale_present")
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def layer_id(self) -> str:
        return f"layer_{self.layer_index:04d}"

    @property
    def expert_key(self) -> str:
        return f"{self.layer_id}:expert_{self.num_experts}"


@dataclass
class ExpertStats(_SchemaBase):
    """Static expert utility scores.

    ``static_gate_mass`` and ``static_gate_entropy`` are non-negative floats.
    """

    layer_index: int
    expert_index: int
    static_gate_mass: float
    static_gate_entropy: float
    static_gate_entropy_norm: Optional[float] = None
    router_bias_norm: Optional[float] = None
    static_rank: Optional[int] = None
    ffn_param_count: Optional[int] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="ExpertStats", repr=False)

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_non_negative_int(self.expert_index, name="expert_index")
        self.static_gate_mass = _ensure_float(self.static_gate_mass, name="static_gate_mass")
        self.static_gate_entropy = _ensure_float(self.static_gate_entropy, name="static_gate_entropy")
        if self.static_gate_mass < 0:
            raise SchemaValidationError("static_gate_mass must be >= 0")
        if self.static_gate_entropy < 0:
            raise SchemaValidationError("static_gate_entropy must be >= 0")
        if self.static_gate_entropy_norm is not None:
            self.static_gate_entropy_norm = _ensure_float(self.static_gate_entropy_norm, name="static_gate_entropy_norm")
        if self.router_bias_norm is not None:
            self.router_bias_norm = _ensure_float(self.router_bias_norm, name="router_bias_norm")
        if self.static_rank is not None:
            self.static_rank = _ensure_non_negative_int(self.static_rank, name="static_rank")
        if self.ffn_param_count is not None:
            self.ffn_param_count = _ensure_non_negative_int(self.ffn_param_count, name="ffn_param_count")
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def expert_key(self) -> str:
        return f"layer_{self.layer_index:04d}:expert_{self.expert_index:04d}"

    def sort_key(
        self,
        *,
        score_epsilon: float = CANONICAL_FLOAT_EPSILON,
        secondary_epsilon: float = CANONICAL_FLOAT_EPSILON,
        rank_position: int = 0,
    ) -> Tuple[int, int, int, int, int]:
        secondary = self.router_bias_norm if self.router_bias_norm is not None else 0.0
        return _expert_sort_tuple(
            score=self.static_gate_mass,
            secondary=self.static_gate_entropy + secondary,
            expert_index=self.expert_index,
            layer_index=self.layer_index,
            rank_position=rank_position,
            score_epsilon=score_epsilon,
            secondary_epsilon=secondary_epsilon,
        )


@dataclass
class ActivationStats(_SchemaBase):
    """Runtime per-expert activation counters.

    ``token_count`` and ``n_tokens`` are raw counts; ``mass_sum`` and
    ``weighted_token_count`` are additive float totals.
    """

    layer_index: int
    expert_index: int
    token_count: int
    weighted_token_count: float
    mass_sum: float
    mean_weight: float
    entropy: float
    n_tokens: int
    timestamp_span: Optional[str] = None
    top1_mass: Optional[float] = None
    density: Optional[float] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="ActivationStats", repr=False)

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_non_negative_int(self.expert_index, name="expert_index")
        _ensure_non_negative_int(self.token_count, name="token_count")
        _ensure_non_negative_int(self.n_tokens, name="n_tokens")
        self.weighted_token_count = _ensure_float(self.weighted_token_count, name="weighted_token_count")
        self.mass_sum = _ensure_float(self.mass_sum, name="mass_sum")
        self.mean_weight = _ensure_float(self.mean_weight, name="mean_weight")
        self.entropy = _ensure_float(self.entropy, name="entropy")
        if self.weighted_token_count < 0:
            raise SchemaValidationError("weighted_token_count must be >= 0")
        if self.mass_sum < 0:
            raise SchemaValidationError("mass_sum must be >= 0")
        if self.mean_weight < 0:
            raise SchemaValidationError("mean_weight must be >= 0")
        if self.entropy < 0:
            raise SchemaValidationError("entropy must be >= 0")
        if self.top1_mass is not None:
            self.top1_mass = _ensure_float(self.top1_mass, name="top1_mass")
        if self.density is not None:
            self.density = _ensure_float(self.density, name="density")
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def occupancy(self) -> float:
        if self.n_tokens == 0:
            return 0.0
        return float(self.token_count) / float(self.n_tokens)


@dataclass
class PruneCandidate(_SchemaBase):
    """Strategy-neutral ranking candidate for a single expert."""

    layer_index: int
    expert_index: int
    score: float
    secondary_score: float = 0.0
    strategy_name: str = "frequency"
    score_components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="PruneCandidate", repr=False)

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_non_negative_int(self.expert_index, name="expert_index")
        self.score = _ensure_float(self.score, name="score")
        self.secondary_score = _ensure_float(self.secondary_score, name="secondary_score")
        _ensure_non_empty_str(self.strategy_name, name="strategy_name")
        for key, val in self.score_components.items():
            if not isinstance(key, str):
                raise SchemaValidationError("score_components keys must be strings")
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise SchemaValidationError("score_components values must be numeric")
        self.metadata = _canonicalize_metadata(self.metadata)

    def sort_key(
        self,
        *,
        score_epsilon: float = CANONICAL_FLOAT_EPSILON,
        secondary_epsilon: float = CANONICAL_FLOAT_EPSILON,
        rank_position: int = 0,
    ) -> Tuple[int, int, int, int, int]:
        secondary = self.secondary_score
        mass = self.score_components.get("mass")
        if isinstance(mass, (int, float)):
            secondary = float(mass)
        return _expert_sort_tuple(
            score=self.score,
            secondary=secondary,
            expert_index=self.expert_index,
            layer_index=self.layer_index,
            rank_position=rank_position,
            score_epsilon=score_epsilon,
            secondary_epsilon=secondary_epsilon,
        )


@dataclass
class PrunePlanItem(_SchemaBase):
    """Deterministic per-layer keep/drop plan."""

    layer_index: int
    keep_indices: Tuple[int, ...]
    drop_indices: Tuple[int, ...]
    source_expert_count: Optional[int] = None
    target_expert_count: Optional[int] = None
    expected_expert_count: Optional[int] = None
    rationale: Optional[str] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="PrunePlanItem", repr=False)

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        self.keep_indices = _canonicalize_int_tuple(self.keep_indices, name="keep_indices")
        self.drop_indices = _canonicalize_int_tuple(self.drop_indices, name="drop_indices")

        keep = list(self.keep_indices)
        drop = list(self.drop_indices)
        if len(set(keep)) != len(keep):
            raise TopologyMismatchError("keep_indices must be unique")
        if len(set(drop)) != len(drop):
            raise TopologyMismatchError("drop_indices must be unique")
        if set(keep) & set(drop):
            raise TopologyMismatchError("keep_indices and drop_indices must be disjoint")

        inferred_source = len(keep) + len(drop)
        inferred_target = len(keep)

        if self.source_expert_count is not None:
            source = _ensure_positive_int(self.source_expert_count, name="source_expert_count")
            if source != inferred_source:
                raise TopologyMismatchError(
                    f"source_expert_count mismatch: expected {inferred_source}, got {source}"
                )

        if self.target_expert_count is not None:
            target = _ensure_non_negative_int(self.target_expert_count, name="target_expert_count")
            if target != inferred_target:
                raise TopologyMismatchError(
                    f"target_expert_count mismatch: expected {inferred_target}, got {target}"
                )

        if self.expected_expert_count is not None:
            expected = _ensure_non_negative_int(self.expected_expert_count, name="expected_expert_count")
            if expected != inferred_source:
                raise TopologyMismatchError(
                    f"expected_expert_count mismatch: expected {inferred_source}, got {expected}"
                )

        if self.rationale is not None:
            _ensure_non_empty_str(self.rationale, name="rationale")
        self.metadata = _canonicalize_metadata(self.metadata)

        self.keep_indices = tuple(sorted(self.keep_indices))
        self.drop_indices = tuple(sorted(self.drop_indices))

    @property
    def ordered_keep_indices(self) -> Tuple[int, ...]:
        return tuple(sorted(self.keep_indices))

    @property
    def ordered_drop_indices(self) -> Tuple[int, ...]:
        return tuple(sorted(self.drop_indices))

    @property
    def total_dropped(self) -> int:
        return len(self.drop_indices)

    @property
    def prune_ratio(self) -> float:
        source = len(self.keep_indices) + len(self.drop_indices)
        if source == 0:
            return 0.0
        return float(self.total_dropped) / float(source)


@dataclass
class PrunePlan(_SchemaBase):
    """Canonical multi-layer pruning plan with deterministic ordering."""

    plan_id: str = "plan-0000"
    model_signature: str = "unknown"
    strategy_name: str = "frequency"
    strategy_version: str = "0"
    per_layer_plans: Tuple[PrunePlanItem, ...] = field(default_factory=tuple)
    global_target_experts: Optional[int] = None
    model_handle: Optional[ModelHandle] = None
    created_by: str = "system"
    created_at: str = field(default_factory=_utcnow_iso)
    source_run_id: Optional[str] = None
    constraints: Dict[str, SchemaKey] = field(default_factory=dict)
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="PrunePlan", repr=False)

    def _validate(self) -> None:
        _ensure_non_empty_str(self.plan_id, name="plan_id")
        _ensure_non_empty_str(self.model_signature, name="model_signature")
        _ensure_non_empty_str(self.strategy_name, name="strategy_name")
        _ensure_non_empty_str(self.strategy_version, name="strategy_version")
        _ensure_non_empty_str(self.created_by, name="created_by")
        self.per_layer_plans = sort_plan_items(self.per_layer_plans)

        seen: set[int] = set()
        for item in self.per_layer_plans:
            if item.layer_index in seen:
                raise TopologyMismatchError("duplicate layer_index in per_layer_plans")
            seen.add(item.layer_index)

        if self.global_target_experts is not None:
            _ensure_positive_int(self.global_target_experts, name="global_target_experts")
        if self.source_run_id is not None:
            _ensure_non_empty_str(self.source_run_id, name="source_run_id")
        if self.model_handle is not None and not isinstance(self.model_handle, ModelHandle):
            raise SchemaValidationError("model_handle must be ModelHandle")

        self.constraints = _canonicalize_metadata(self.constraints)
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def total_source_experts(self) -> int:
        total = 0
        for item in self.per_layer_plans:
            total += len(item.keep_indices) + len(item.drop_indices)
        return total

    @property
    def total_target_experts(self) -> int:
        return sum(len(item.keep_indices) for item in self.per_layer_plans)

    @property
    def versioned_manifest_id(self) -> str:
        payload = self.to_json(compact=True)
        digest = sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.schema_version}:{self.plan_id}:{digest[:16]}"

    def to_json(self, compact: bool = True) -> str:
        return to_json(self, compact=compact)


@dataclass
class RunArtifactManifest(_SchemaBase):
    """Run-level manifest carrying execution metadata."""

    run_id: str
    command: str
    model_handle: Optional[ModelHandle] = None
    top_k: int = 1
    prompt_count: int = 0
    seed: int = 0
    prompt_set_hash: Optional[str] = None
    started_at: str = field(default_factory=_utcnow_iso)
    finished_at: Optional[str] = None
    input_checksums: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    parent_artifacts: Tuple[str, ...] = field(default_factory=tuple)
    run_plan: Optional[PrunePlan] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: str = field(init=False, default="RunArtifactManifest", repr=False)

    def _validate(self) -> None:
        _ensure_non_empty_str(self.run_id, name="run_id")
        _ensure_non_empty_str(self.command, name="command")
        _ensure_positive_int(self.top_k, name="top_k")
        _ensure_non_negative_int(self.prompt_count, name="prompt_count")
        _ensure_non_negative_int(self.seed, name="seed")
        if self.prompt_set_hash is not None:
            _ensure_non_empty_str(self.prompt_set_hash, name="prompt_set_hash")
        self.parent_artifacts = _canonicalize_str_tuple(self.parent_artifacts, name="parent_artifacts")
        self.input_checksums = {str(k): str(v) for k, v in self.input_checksums.items()}
        self.output_paths = {str(k): str(v) for k, v in self.output_paths.items()}
        if self.model_handle is not None and not isinstance(self.model_handle, ModelHandle):
            raise SchemaValidationError("model_handle must be ModelHandle")
        if self.run_plan is not None and not isinstance(self.run_plan, PrunePlan):
            raise SchemaValidationError("run_plan must be PrunePlan")
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def versioned_manifest_id(self) -> str:
        key = f"{self.schema_version}:{self.run_id}:{self.command}:{self.started_at}"
        return sha256(key.encode("utf-8")).hexdigest()


SchemaType = Union[
    ModelHandle,
    LayerTopology,
    RouterState,
    ExpertStats,
    ActivationStats,
    PruneCandidate,
    PrunePlanItem,
    PrunePlan,
    RunArtifactManifest,
]


def sort_experts(
    items: Iterable[
        Union[
            PruneCandidate,
            ExpertStats,
            Tuple[int, float, float, int],
            Tuple[int, int, float, float, int],
        ]
    ],
    *,
    score_epsilon: float = CANONICAL_FLOAT_EPSILON,
    secondary_epsilon: float = CANONICAL_FLOAT_EPSILON,
) -> list[Union[PruneCandidate, ExpertStats, Tuple[int, float, float, int], Tuple[int, int, float, float, int]]]:
    """Stable expert ordering with explicit tie-breaks.

    Primary key: ``-score`` (epsilon-safe), ``-secondary``, ``layer_index``,
    ``expert_index``, and stable input position as final tiebreak.
    """

    normalized: list[tuple[tuple[int, int, int, int, int], Any]] = []
    for position, item in enumerate(items):
        if isinstance(item, PruneCandidate):
            normalized.append((item.sort_key(score_epsilon=score_epsilon, secondary_epsilon=secondary_epsilon, rank_position=position), item))
            continue
        if isinstance(item, ExpertStats):
            normalized.append((item.sort_key(score_epsilon=score_epsilon, secondary_epsilon=secondary_epsilon, rank_position=position), item))
            continue

        if not isinstance(item, tuple):
            raise SchemaValidationError(f"Unsupported expert record type: {type(item).__name__}")

        if len(item) == 4:
            layer_index, score, secondary, expert_index = item
        elif len(item) == 5:
            layer_index, score, secondary, expert_index, _position = item
            # external position can be re-used only as a stable fallback
            position = int(_position)
        else:
            raise SchemaValidationError("Tuple expert records must be (layer, score, secondary, expert) or plus position")

        layer_index_i = _ensure_non_negative_int(layer_index, name="tuple layer_index")
        expert_index_i = _ensure_non_negative_int(expert_index, name="tuple expert_index")
        score_f = _ensure_float(score, name="tuple score")
        secondary_f = _ensure_float(secondary, name="tuple secondary")

        key = _expert_sort_tuple(
            score=score_f,
            secondary=secondary_f,
            expert_index=expert_index_i,
            layer_index=layer_index_i,
            rank_position=position,
            score_epsilon=score_epsilon,
            secondary_epsilon=secondary_epsilon,
        )
        normalized.append((key, item))

    normalized.sort(key=lambda entry: entry[0])
    return [item for _, item in normalized]


def sort_plan_items(items: Iterable[PrunePlanItem]) -> Tuple[PrunePlanItem, ...]:
    """Sort plans deterministically for reproducible manifests."""
    return tuple(
        sorted(
            items,
            key=lambda it: (
                it.layer_index,
                len(it.keep_indices),
                len(it.drop_indices),
                tuple(it.ordered_keep_indices),
                tuple(it.ordered_drop_indices),
            ),
        )
    )


def sort_topology(layers: Iterable[LayerTopology], *, with_ref_fallback: bool = True) -> Tuple[LayerTopology, ...]:
    """Sort topologies by layer index with deterministic reference fallback."""
    def _key(layer: LayerTopology) -> tuple[int, int, str, str]:
        ref_value = layer.layer_ref or layer.layer_name or ""
        ref_rank = -_parse_layer_ref(layer.layer_id)
        if with_ref_fallback and layer.layer_ref is not None:
            ref_rank = _parse_layer_ref(layer.layer_ref)
        return (layer.layer_index, ref_rank, len(layer.module_paths), ref_value)

    return tuple(sorted(layers, key=_key))


def to_dict(obj: SchemaType | dict[str, Any] | list[Any] | tuple[Any, ...] | None) -> Any:
    """Serialize schema objects to JSON-ready primitive structures."""
    return _as_schema_data(obj)


def to_json(
    obj: SchemaType | dict[str, Any] | list[Any] | tuple[Any, ...] | None,
    *,
    compact: bool = True,
) -> str:
    payload = to_dict(obj)
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps(payload, sort_keys=True, indent=2)


def _schema_name_to_ctor() -> MutableMapping[str, Type[_SchemaBase]]:
    return {
        ModelHandle.__name__: ModelHandle,
        LayerTopology.__name__: LayerTopology,
        RouterState.__name__: RouterState,
        ExpertStats.__name__: ExpertStats,
        ActivationStats.__name__: ActivationStats,
        PruneCandidate.__name__: PruneCandidate,
        PrunePlanItem.__name__: PrunePlanItem,
        PrunePlan.__name__: PrunePlan,
        RunArtifactManifest.__name__: RunArtifactManifest,
    }


def _construct_from_payload(payload: Mapping[str, Any]) -> SchemaType:
    schema_type = payload.get("__schema_type")
    if not isinstance(schema_type, str):
        raise SchemaValidationError("Missing __schema_type")
    mapping = dict(payload)
    mapping.pop("__schema_type", None)
    mapping.pop("__schema_version", None)
    constructors = _schema_name_to_ctor()
    ctor = constructors.get(schema_type)
    if ctor is None:
        raise SchemaValidationError(f"Unsupported __schema_type '{schema_type}'")
    return ctor.from_dict(mapping)  # type: ignore[return-value]


def from_json(payload: Union[str, Mapping[str, Any], list[Any]]) -> Any:
    """Deserialize canonical JSON back into schema objects."""
    data = json.loads(payload) if isinstance(payload, str) else payload

    if isinstance(data, Mapping):
        if "__schema_type" in data:
            return _construct_from_payload(data)
        if "type" in data and isinstance(data.get("type"), str):
            payload_ = dict(data)
            payload_["__schema_type"] = payload_.pop("type")
            return _construct_from_payload(payload_)
        if "kind" in data and isinstance(data.get("kind"), str):
            payload_ = dict(data)
            payload_["__schema_type"] = payload_.pop("kind")
            return _construct_from_payload(payload_)
        raise SchemaValidationError("Unsupported mapping payload")

    if isinstance(data, list):
        return [from_json(item) if isinstance(item, (str, dict, list)) else item for item in data]

    raise SchemaValidationError("Unsupported payload type")


def to_json_file(path: str | Path, obj: SchemaType | dict[str, Any] | list[Any] | tuple[Any, ...]) -> Path:
    target = Path(path)
    target.write_text(to_json(obj), encoding="utf-8")
    return target


def from_json_file(path: str | Path) -> Any:
    return from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CANONICAL_FLOAT_EPSILON",
    "SchemaType",
    "ModelHandle",
    "LayerTopology",
    "RouterState",
    "ExpertStats",
    "ActivationStats",
    "PruneCandidate",
    "PrunePlanItem",
    "PrunePlan",
    "RunArtifactManifest",
    "SchemaValidationError",
    "ShapeInvariantViolationError",
    "TopologyMismatchError",
    "LayerReferenceError",
    "sort_experts",
    "sort_plan_items",
    "sort_topology",
    "to_dict",
    "to_json",
    "from_json",
    "to_json_file",
    "from_json_file",
]
