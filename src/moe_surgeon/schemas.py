"""Canonical schema contracts for MoE analysis and pruning.

This module defines lightweight, dependency-free dataclasses used by every
pipeline stage before any tensor mutation. Every object is deterministic,
serializable, and validates structural invariants on construction.
"""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence, cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import get_type_hints
from typing import TypeVar
from math import isfinite, floor
from pathlib import Path
from dataclasses import MISSING
import json
import re


CANONICAL_SCHEMA_VERSION = "1.0.0"
CANONICAL_FLOAT_EPSILON = 1e-12
INT_SEQUENCE_ID_PATTERN = re.compile(r"^(?:layer|module)_([0-9]+)$")


class SchemaValidationError(ValueError):
    """Base validation error for schema and invariant failures."""


class ShapeInvariantViolationError(SchemaValidationError):
    """Raised when tensor-shape-like metadata does not satisfy invariants."""


class TopologyMismatchError(SchemaValidationError):
    """Raised when topology or expert-count invariants are inconsistent."""


class LayerReferenceError(SchemaValidationError):
    """Raised when a layer reference string does not match canonical form."""


SchemaKey = Union[str, int, float, bool, None]
ShapeTuple = Tuple[int, ...]


def _utcnow_iso() -> str:
    """Return an RFC3339-style UTC timestamp in seconds precision."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_bool(value: Any, *, name: str) -> bool:
    """Validate that a field is exactly a boolean."""

    if not isinstance(value, bool):
        raise SchemaValidationError(f"{name} must be bool, got {type(value).__name__}")
    return value


def _ensure_non_negative_int(value: Any, *, name: str) -> int:
    """Validate integer >= 0."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaValidationError(f"{name} must be int, got {type(value).__name__}")
    if value < 0:
        raise SchemaValidationError(f"{name} must be non-negative, got {value}")
    return value


def _ensure_positive_int(value: Any, *, name: str) -> int:
    """Validate integer > 0."""

    value = _ensure_non_negative_int(value, name=name)
    if value == 0:
        raise SchemaValidationError(f"{name} must be > 0, got {value}")
    return int(value)


def _ensure_non_empty_str(value: Any, *, name: str) -> str:
    """Validate non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value


def _ensure_optional_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _ensure_non_negative_int(value, name=name)


def _ensure_optional_positive_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _ensure_positive_int(value, name=name)


def _ensure_float(value: Any, *, name: str, finite: bool = True) -> float:
    """Validate numeric float-like value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    if finite:
        if not isfinite(float(value)):
            raise SchemaValidationError(f"{name} must be finite, got {value}")
    return float(value)


def _ensure_shape_tuple(value: Any, *, name: str, allow_empty: bool = False) -> Optional[ShapeTuple]:
    """Validate tensor-like shape metadata as tuple of positive ints."""

    if value is None:
        return None
    if isinstance(value, tuple):
        values: Sequence[Any] = value
    elif isinstance(value, list):
        values = tuple(value)
    else:
        raise ShapeInvariantViolationError(
            f"{name} must be list/tuple of ints or None, got {type(value).__name__}"
        )
    if not allow_empty and len(values) == 0:
        raise ShapeInvariantViolationError(f"{name} must be non-empty")
    out: List[int] = []
    for idx, dim in enumerate(values):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ShapeInvariantViolationError(
                f"{name}[{idx}] must be non-negative int, got {dim!r}"
            )
        out.append(dim)
    return tuple(out)


def _parse_layer_reference(value: Any) -> int:
    """Parse canonical layer references and return the numeric layer index."""

    if not isinstance(value, str):
        raise LayerReferenceError("layer_ref must be a string")
    match = INT_SEQUENCE_ID_PATTERN.match(value)
    if not match:
        raise LayerReferenceError(
            f"layer_ref '{value}' is invalid. Expected 'layer_<index>' or 'module_<index>'"
        )
    return int(match.group(1))


def _float_sort_bucket(value: float, *, epsilon: float) -> float:
    """Bucket float values into deterministic bins before sorting ties."""

    epsilon = max(float(epsilon), CANONICAL_FLOAT_EPSILON)
    scale = 1.0 / epsilon
    return floor(value * scale + (0.5 if value >= 0 else -0.5)) / scale


def _expert_sort_terms(
    *,
    score: float,
    secondary: float,
    expert_index: int,
    layer_index: int,
    rank_position: int,
    score_epsilon: float,
    secondary_epsilon: float,
) -> Tuple[float, float, int, int, int]:
    """Build a deterministic comparator key for expert-level ranking."""

    q_score = _float_sort_bucket(score, epsilon=score_epsilon)
    q_secondary = _float_sort_bucket(secondary, epsilon=secondary_epsilon)
    return (
        -q_score,
        -q_secondary,
        int(expert_index),
        int(layer_index),
        int(rank_position),
    )


def _canonicalize_metadata(value: Mapping[str, Any] | None) -> Dict[str, SchemaKey]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaValidationError("metadata must be a mapping")
    out: Dict[str, SchemaKey] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise SchemaValidationError("metadata keys must be strings")
        out[k] = cast(SchemaKey, v)
    return out


def _canonicalize_int_list(values: Sequence[int] | None, *, name: str) -> Tuple[int, ...]:
    if values is None:
        return ()
    out: List[int] = []
    for value in values:
        out.append(_ensure_non_negative_int(value, name=f"{name} index"))
    return tuple(out)


def _sort_shape(shape: ShapeTuple) -> ShapeTuple:
    return tuple(shape)


def _as_schema_dict(value: Any) -> Any:
    """Convert schema values to JSON-safe primitives recursively."""

    if is_dataclass(value) and hasattr(value, "to_dict"):
        return cast(Dict[str, Any], value.to_dict())
    if is_dataclass(value):
        return cast(Dict[str, Any], asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted((_as_schema_dict(v) for v in value), key=repr)
    if isinstance(value, tuple):
        return [_as_schema_dict(v) for v in value]
    if isinstance(value, list):
        return [_as_schema_dict(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_schema_dict(v) for k, v in value.items()}
    return value


def _coerce_schema_kwargs(data: Mapping[str, Any], target: Type["_SchemaBase"]) -> Dict[str, Any]:
    """Build ctor kwargs by keeping only declared fields and preserving defaults."""

    hints = get_type_hints(target)
    kwargs: Dict[str, Any] = {}
    for f in fields(target):
        key = f.name
        if key in data:
            value = data[key]
        elif f.default is not MISSING:
            continue
        elif f.default_factory is not MISSING:
            continue
        else:
            raise SchemaValidationError(f"Missing required schema field '{key}' for {target.__name__}")
        # Preserve raw shape/list conversions for tuple typed fields.
        annotation = hints.get(key)
        if annotation in (Tuple[int, ...], tuple[int, ...], Tuple[int, int], tuple[int, int]):
            if value is not None:
                value = _ensure_shape_tuple(value, name=key)
        if target.__name__ == "PrunePlan" and key == "per_layer_plans":
            if value is not None:
                converted = []
                for item in value:
                    if isinstance(item, Mapping):
                        converted.append(PrunePlanItem.from_dict(item))
                    else:
                        converted.append(item)
                value = tuple(converted)
        elif target.__name__ == "RunArtifactManifest" and key == "run_plan":
            if isinstance(value, Mapping):
                value = PrunePlan.from_dict(value)
        elif target.__name__ == "RunArtifactManifest" and key == "model_handle":
            if isinstance(value, Mapping):
                value = ModelHandle.from_dict(value)
        kwargs[key] = value
    return kwargs

SchemaClass = TypeVar("SchemaClass", bound="_SchemaBase")


@dataclass
class _SchemaBase:
    """Base class implementing canonical serialisation helpers."""

    _schema_type: ClassVar[str] = "_SchemaBase"

    @property
    def schema_version(self) -> str:
        return CANONICAL_SCHEMA_VERSION

    def _validate(self) -> None:
        return

    def __post_init__(self) -> None:
        self._validate()

    def to_dict(self) -> Dict[str, Any]:
        data = cast(Dict[str, Any], asdict(self))
        data["__schema_version"] = self.schema_version
        data["__schema_type"] = self.__class__.__name__
        return data

    @classmethod
    def from_dict(cls: Type[SchemaClass], payload: Mapping[str, Any]) -> SchemaClass:
        kwargs = _coerce_schema_kwargs(payload, cls)
        return cast(SchemaClass, cls(**kwargs))


@dataclass
class ModelHandle(_SchemaBase):
    """Lightweight canonical handle for a model artifact.

    All fields are metadata only and safe to persist.
    """

    model_id: str = "unknown"
    revision: Optional[str] = None
    tokenizer_id: Optional[str] = None
    backend_name: Optional[str] = None
    source_path: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[str] = None
    seed: int = 0
    framework_version: Optional[str] = None
    created_at: str = field(default_factory=_utcnow_iso)
    git_hash: Optional[str] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: ClassVar[str] = "ModelHandle"

    def _validate(self) -> None:
        _ensure_non_empty_str(self.model_id, name="model_id")
        _ensure_non_empty_str(self.device, name="device")
        _ensure_non_negative_int(self.seed, name="seed")
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def canonical_id(self) -> str:
        """Deterministic identifier used in logs and derived filenames."""

        return f"{self.model_id}:{self.backend_name or 'backend-unknown'}:{self.seed}"

    def layer_id(self, layer_index: int) -> str:
        """Canonical layer id in zero-padded form for stable comparisons."""

        _ensure_non_negative_int(layer_index, name="layer_index")
        return f"layer_{layer_index:04d}"

    @property
    def model_fingerprint(self) -> str:
        """Hash-like fingerprint stable across JSON-equivalent field orderings."""

        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "backend_name": self.backend_name,
            "source_path": self.source_path,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "framework_version": self.framework_version,
            "git_hash": self.git_hash,
            "metadata": self.metadata,
        }
        canonical = json.dumps(_as_schema_dict(payload), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class LayerTopology(_SchemaBase):
    """Static topology metadata for one MoE layer.

    Units:
    - expert_count: count of routed experts in the layer.
    - top_k: number of experts selected per token.
    - hidden_size: feed-forward hidden width.
    """

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
    _schema_type: ClassVar[str] = "LayerTopology"

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
            _parse_layer_reference(self.layer_ref)
        self.module_paths = {str(k): str(v) for k, v in self.module_paths.items()}
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def layer_id(self) -> str:
        """Canonical layer identifier used across logs and manifests."""

        return f"layer_{self.layer_index:04d}"

    @property
    def expert_ref(self) -> str:
        """Reference key for expert collections at this layer."""

        return f"{self.layer_id}:expert_count_{self.expert_count}"


@dataclass
class RouterState(_SchemaBase):
    """Snapshot of router metadata used for reproducible diagnostics.

    Shapes are stored as plain integer tuples for JSON portability.

    Attributes:
        logits_shape: router logits tensor shape.
        top_k_indices_shape: indices tensor shape.
        top_k_weights_shape: top-k weight tensor shape.
        projection_shape: projection weight tensor shape.
        per_expert_scale_shape: per-expert scale tensor shape.
    """

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
    _schema_type: ClassVar[str] = "RouterState"

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_positive_int(self.num_experts, name="num_experts")
        _ensure_positive_int(self.top_k, name="top_k")
        _ensure_shape_tuple(self.logits_shape, name="logits_shape", allow_empty=False)
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
        if self.top_k > self.num_experts:
            raise TopologyMismatchError("top_k cannot exceed num_experts")
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
    """Static expert statistics collected from offline router inspection.

    Metrics:
    - static_gate_mass (dimensionless probability mass) in [0, ∞).
    - static_gate_entropy (nats): entropy proxy over the routing distribution.
    - router_bias_norm (L2 norm): magnitude of router prior before softmax.
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
    _schema_type: ClassVar[str] = "ExpertStats"

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
            _ensure_float(self.static_gate_entropy_norm, name="static_gate_entropy_norm")
        if self.router_bias_norm is not None:
            _ensure_float(self.router_bias_norm, name="router_bias_norm")
        if self.static_rank is not None:
            _ensure_non_negative_int(self.static_rank, name="static_rank")
        if self.ffn_param_count is not None:
            _ensure_non_negative_int(self.ffn_param_count, name="ffn_param_count")
        self.metadata = _canonicalize_metadata(self.metadata)

    def expert_key(self) -> str:
        """Canonical key that uniquely identifies this expert in a layer."""

        return f"layer_{self.layer_index:04d}:expert_{self.expert_index:04d}"

    def sort_key(
        self,
        *,
        secondary: Optional[float] = None,
        score_epsilon: float = CANONICAL_FLOAT_EPSILON,
        secondary_epsilon: float = CANONICAL_FLOAT_EPSILON,
    ) -> Tuple[float, float, int, int, int]:
        """Deterministic sort key for expert ranking."""

        score = self.static_gate_mass
        mass = self.static_gate_entropy
        if secondary is None:
            secondary = self.router_bias_norm or 0.0
        return _expert_sort_terms(
            score=score,
            secondary=mass + float(secondary or 0.0),
            expert_index=self.expert_index,
            layer_index=self.layer_index,
            rank_position=0,
            score_epsilon=score_epsilon,
            secondary_epsilon=secondary_epsilon,
        )


@dataclass
class ActivationStats(_SchemaBase):
    """Runtime activation statistics per expert.

    Metrics:
    - token_count: number of tokens routed to this expert.
    - weighted_token_count: token_count scaled by routing confidence.
    - mass_sum: sum of routing weights assigned to this expert.
    - mean_weight: mean routing weight conditioned on routed tokens.
    - entropy: entropy (nats) of per-token routing mass for this expert.
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
    _schema_type: ClassVar[str] = "ActivationStats"

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
        """Observed occupancy ratio for this expert in [0,1]."""

        if self.n_tokens == 0:
            return 0.0
        return float(self.token_count) / float(self.n_tokens)


@dataclass
class PruneCandidate(_SchemaBase):
    """Candidate signal used by pruning strategies.

    score is a canonical ranking value where larger values are preferred.
    secondary_score is an optional tie-break value (same orientation).
    """

    layer_index: int
    expert_index: int
    score: float
    secondary_score: float = 0.0
    strategy_name: str = "frequency"
    score_components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: ClassVar[str] = "PruneCandidate"

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        _ensure_non_negative_int(self.expert_index, name="expert_index")
        self.score = _ensure_float(self.score, name="score")
        self.secondary_score = _ensure_float(self.secondary_score, name="secondary_score")
        _ensure_non_empty_str(self.strategy_name, name="strategy_name")
        for key, value in self.score_components.items():
            if not isinstance(key, str):
                raise SchemaValidationError("score_components keys must be strings")
            _ensure_float(value, name=f"score_components[{key}]")
        self.metadata = _canonicalize_metadata(self.metadata)

    def sort_key(
        self,
        *,
        score_epsilon: float = CANONICAL_FLOAT_EPSILON,
        secondary_epsilon: float = CANONICAL_FLOAT_EPSILON,
        rank_position: int = 0,
    ) -> Tuple[float, float, int, int, int]:
        """Deterministic sort key with explicit tie-breakers."""

        secondary = self.secondary_score
        if "mass" in self.score_components:
            secondary = self.score_components["mass"]
        return _expert_sort_terms(
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
    """Per-layer pruning decision with deterministic keep/drop sets."""

    layer_index: int
    keep_indices: Tuple[int, ...]
    drop_indices: Tuple[int, ...]
    source_expert_count: Optional[int] = None
    target_expert_count: Optional[int] = None
    expected_expert_count: Optional[int] = None
    rationale: Optional[str] = None
    metadata: Dict[str, SchemaKey] = field(default_factory=dict)
    _schema_type: ClassVar[str] = "PrunePlanItem"

    def _validate(self) -> None:
        _ensure_non_negative_int(self.layer_index, name="layer_index")
        self.keep_indices = _canonicalize_int_list(self.keep_indices, name="keep_indices")
        self.drop_indices = _canonicalize_int_list(self.drop_indices, name="drop_indices")
        if any(v >= 0 for v in self.keep_indices) is False and len(self.keep_indices) > 0:
            pass
        keep = set(self.keep_indices)
        drop = set(self.drop_indices)
        if len(keep) != len(self.keep_indices):
            raise TopologyMismatchError("keep_indices must be unique")
        if len(drop) != len(self.drop_indices):
            raise TopologyMismatchError("drop_indices must be unique")
        if keep & drop:
            raise TopologyMismatchError("keep_indices and drop_indices must be disjoint")
        actual_source = len(keep) + len(drop)
        if self.source_expert_count is not None:
            self.source_expert_count = _ensure_positive_int(self.source_expert_count, name="source_expert_count")
            if self.source_expert_count != actual_source:
                raise TopologyMismatchError(
                    "source_expert_count must equal len(keep_indices)+len(drop_indices), "
                    f"got {self.source_expert_count} vs {actual_source}"
                )
        if self.target_expert_count is not None:
            self.target_expert_count = _ensure_non_negative_int(self.target_expert_count, name="target_expert_count")
            if self.target_expert_count != len(keep):
                raise TopologyMismatchError(
                    "target_expert_count must equal len(keep_indices), "
                    f"got {self.target_expert_count} vs {len(keep)}"
                )
        if self.expected_expert_count is not None:
            self.expected_expert_count = _ensure_positive_int(self.expected_expert_count, name="expected_expert_count")
            if self.expected_expert_count < actual_source:
                raise TopologyMismatchError(
                    "expected_expert_count must be >= existing source experts count"
                )
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def ordered_keep_indices(self) -> Tuple[int, ...]:
        """Sorted keep indices for deterministic serialization."""

        return tuple(sorted(self.keep_indices))

    @property
    def ordered_drop_indices(self) -> Tuple[int, ...]:
        """Sorted drop indices for deterministic serialization."""

        return tuple(sorted(self.drop_indices))

    @property
    def total_dropped(self) -> int:
        return len(self.drop_indices)

    @property
    def prune_ratio(self) -> float:
        source = self.source_expert_count if self.source_expert_count is not None else 0
        if source == 0:
            return 0.0
        return len(self.drop_indices) / float(source)


@dataclass
class PrunePlan(_SchemaBase):
    """Canonical deterministic pruning plan across all MoE layers."""

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
    _schema_type: ClassVar[str] = "PrunePlan"

    def _validate(self) -> None:
        _ensure_non_empty_str(self.plan_id, name="plan_id")
        _ensure_non_empty_str(self.model_signature, name="model_signature")
        _ensure_non_empty_str(self.strategy_name, name="strategy_name")
        _ensure_non_empty_str(self.strategy_version, name="strategy_version")
        _ensure_non_empty_str(self.created_by, name="created_by")
        for item in self.per_layer_plans:
            _ = item
        self.per_layer_plans = tuple(sort_plan_items(self.per_layer_plans))
        if self.global_target_experts is not None:
            self.global_target_experts = _ensure_positive_int(self.global_target_experts, name="global_target_experts")
        if self.source_run_id is not None:
            _ensure_non_empty_str(self.source_run_id, name="source_run_id")
        self.constraints = _canonicalize_metadata(self.constraints)
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def total_source_experts(self) -> int:
        """Sum of source expert counts across layers."""

        total = 0
        for item in self.per_layer_plans:
            if item.source_expert_count is not None:
                total += item.source_expert_count
        return total

    @property
    def total_target_experts(self) -> int:
        """Sum of target expert counts across layers."""

        total = 0
        for item in self.per_layer_plans:
            if item.source_expert_count is not None:
                total += item.source_expert_count - item.total_dropped
            else:
                total += len(item.keep_indices)
        return total

    @property
    def versioned_manifest_id(self) -> str:
        """Canonical identifier tying schema version and plan identity."""

        seed = self.to_compact_json()
        digest = sha256(seed.encode("utf-8")).hexdigest()
        return f"{self.schema_version}:{self.plan_id}:{digest[:16]}"

    def to_compact_json(self) -> str:
        """Compact, stable JSON representation for a pruning plan."""

        return to_json(self, compact=True)


@dataclass
class RunArtifactManifest(_SchemaBase):
    """Execution manifest shared by scan/bench/prune/export steps."""

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
    _schema_type: ClassVar[str] = "RunArtifactManifest"

    def _validate(self) -> None:
        _ensure_non_empty_str(self.run_id, name="run_id")
        _ensure_non_empty_str(self.command, name="command")
        _ensure_positive_int(self.top_k, name="top_k")
        _ensure_non_negative_int(self.prompt_count, name="prompt_count")
        _ensure_non_negative_int(self.seed, name="seed")
        _ensure_non_negative_int(len(self.parent_artifacts), name="parent_artifacts")
        self.input_checksums = {str(k): str(v) for k, v in self.input_checksums.items()}
        self.output_paths = {str(k): str(v) for k, v in self.output_paths.items()}
        self.parent_artifacts = tuple(self.parent_artifacts)
        self.metadata = _canonicalize_metadata(self.metadata)

    @property
    def versioned_manifest_id(self) -> str:
        """Versioned manifest id used for deterministic artifact naming."""

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
    items: Iterable[Union[PruneCandidate, ExpertStats, Tuple[float, float, int], Tuple[int, float, float, int]]],
    *,
    score_epsilon: float = CANONICAL_FLOAT_EPSILON,
    secondary_epsilon: float = CANONICAL_FLOAT_EPSILON,
) -> List[Union[PruneCandidate, ExpertStats, Tuple[float, float, int], Tuple[int, float, float, int]]]:
    """Sort experts deterministically.

    Supports:
    - ExpertStats
    - PruneCandidate
    - tuples (layer_index, score, secondary, expert_index)

    Sort policy: (-score, -secondary, expert_index, layer_index, input_position)
    with epsilon-safe bucketing.
    """

    normalized: List[Tuple[Tuple[float, float, int, int, int], Any]] = []
    for position, item in enumerate(items):
        if isinstance(item, PruneCandidate):
            normalized.append((item.sort_key(score_epsilon=score_epsilon, secondary_epsilon=secondary_epsilon, rank_position=position), item))
            continue
        if isinstance(item, ExpertStats):
            normalized.append((item.sort_key(score_epsilon=score_epsilon, secondary_epsilon=secondary_epsilon), item))
            continue
        if isinstance(item, tuple):
            if len(item) < 4:
                raise SchemaValidationError("tuple expert records must have at least 4 elements")
            layer_index, score, secondary, expert_index = item[:4]
            layer_index_i = _ensure_non_negative_int(layer_index, name="tuple layer_index")
            score_f = _ensure_float(score, name="tuple score")
            secondary_f = _ensure_float(secondary, name="tuple secondary")
            expert_index_i = _ensure_non_negative_int(expert_index, name="tuple expert_index")
            normalized.append(
                (
                    _expert_sort_terms(
                        score=score_f,
                        secondary=secondary_f,
                        expert_index=expert_index_i,
                        layer_index=layer_index_i,
                        rank_position=position,
                        score_epsilon=score_epsilon,
                        secondary_epsilon=secondary_epsilon,
                    ),
                    item,
                )
            )
            continue
        raise SchemaValidationError(f"Unsupported expert item type: {type(item).__name__}")
    normalized.sort(key=lambda x: x[0])
    return [item for _, item in normalized]


def sort_plan_items(items: Iterable[PrunePlanItem]) -> Tuple[PrunePlanItem, ...]:
    """Sort prune-plan items by layer and deterministic per-layer keys."""

    return tuple(
        sorted(
            items,
            key=lambda item: (item.layer_index, item.rationale or "", item.source_expert_count or 0, item.rationale or ""),
        )
    )


def sort_topology(layers: Iterable[LayerTopology], *, with_ref_fallback: bool = True) -> Tuple[LayerTopology, ...]:
    """Sort topologies by numeric layer index and stable references."""

    def _key(item: LayerTopology) -> Tuple[int, str, int]:
        ref = item.layer_ref if item.layer_ref is not None else ""
        if with_ref_fallback:
            return (item.layer_index, ref, len(item.layer_name))
        return (item.layer_index, "", len(item.layer_name))

    return tuple(sorted(layers, key=_key))


def to_dict(obj: SchemaType | dict[str, Any] | list[Any] | tuple[Any, ...]) -> Any:
    """Serialize supported schema objects to a JSON-ready Python structure."""

    return _as_schema_dict(obj)


def to_json(obj: SchemaType | dict[str, Any] | list[Any] | tuple[Any, ...], compact: bool = True) -> str:
    """Serialize schema objects to canonical JSON bytes."""

    if compact:
        return json.dumps(to_dict(obj), sort_keys=True, separators=(",", ":"))
    return json.dumps(to_dict(obj), sort_keys=True, indent=2)


def _construct_object(payload: Mapping[str, Any]) -> SchemaType:
    schema_type = payload.get("__schema_type")
    if not isinstance(schema_type, str):
        raise SchemaValidationError("JSON payload missing __schema_type")
    constructors = {
        cls.__name__: cls
        for cls in [
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
    }
    if schema_type not in constructors:
        raise SchemaValidationError(f"Unsupported __schema_type: {schema_type}")
    constructor = constructors[schema_type]
    core = dict(payload)
    core.pop("__schema_type", None)
    core.pop("__schema_version", None)
    return constructor.from_dict(core)  # type: ignore[return-value]


def from_json(payload: Union[str, Mapping[str, Any]]) -> Any:
    """Parse JSON text or mapping back into canonical schema objects.

    Supported payload formats:
    - serialized canonical dict produced by `to_dict`/`to_json`
    - mapping with root key `type`/`kind` and object body
    - raw list of schema dicts
    """

    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, Mapping):
        if "__schema_type" in data:
            return _construct_object(data)
        if "type" in data and data.get("type") in {
            "ModelHandle",
            "LayerTopology",
            "RouterState",
            "ExpertStats",
            "ActivationStats",
            "PruneCandidate",
            "PrunePlanItem",
            "PrunePlan",
            "RunArtifactManifest",
        }:
            payload_dict = dict(data)
            payload_dict["__schema_type"] = payload_dict.pop("type")
            return _construct_object(payload_dict)
        if "kind" in data and data.get("kind") in {
            "ModelHandle",
            "LayerTopology",
            "RouterState",
            "ExpertStats",
            "ActivationStats",
            "PruneCandidate",
            "PrunePlanItem",
            "PrunePlan",
            "RunArtifactManifest",
        }:
            payload_dict = dict(data)
            payload_dict["__schema_type"] = payload_dict.pop("kind")
            return _construct_object(payload_dict)
        raise SchemaValidationError("Unsupported JSON payload; missing __schema_type")
    if isinstance(data, list):
        return [from_json(entry) for entry in data]
    raise SchemaValidationError("Unsupported JSON payload type")


def to_json_file(path: Union[str, Path], obj: SchemaType | dict[str, Any] | list[Any] | tuple[Any, ...]) -> Path:
    """Write canonical json to disk with stable encoding."""

    target = Path(path)
    target.write_text(to_json(obj), encoding="utf-8")
    return target


def from_json_file(path: Union[str, Path]) -> Any:
    """Read canonical json from disk."""

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
