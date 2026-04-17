"""Canonical datamodels used across analysis, runtime, pruning, and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

CANONICAL_SCHEMA_VERSION = "1.0.0"
SchemaSortType = Tuple[float, float, int]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_canonical(obj: Any) -> Any:
    if is_dataclass(obj):
        return {key: _to_canonical(value) for key, value in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_canonical(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_canonical(v) for v in obj]
    if isinstance(obj, list):
        return [_to_canonical(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_canonical(v) for v in obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def dumps_canonical(obj: Any) -> str:
    return json.dumps(_to_canonical(obj), sort_keys=True, separators=(",", ":"))


def file_sha256(path: str | Path) -> str:
    hasher = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prune_sort_key(*, score: float, secondary: float = 0.0, expert_index: int = 0) -> SchemaSortType:
    """Canonical tie-breaker helper.

    Higher primary/secondary scores first; expert index ties are deterministic.
    """
    return (-float(score), -float(secondary), int(expert_index))


@dataclass(frozen=True)
class ModelHandle:
    model_id: str
    revision: Optional[str] = None
    tokenizer_id: Optional[str] = None
    backend_name: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[str] = None
    seed: int = 0
    source_path: Optional[str] = None
    created_at: str = field(default_factory=_utcnow_iso)
    git_hash: Optional[str] = None


@dataclass(frozen=True)
class LayerTopology:
    layer_index: int
    layer_name: str
    layer_type: str
    expert_count: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: Optional[int] = None
    module_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.layer_type.lower() == "moe"


@dataclass(frozen=True)
class RouterState:
    layer_index: int
    num_experts: int
    top_k: int
    logits_shape: Tuple[int, ...]
    top_k_indices_shape: Tuple[int, ...]
    top_k_weights_shape: Tuple[int, ...]
    projection_shape: Optional[Tuple[int, ...]] = None
    per_expert_scale_shape: Optional[Tuple[int, ...]] = None
    has_router_probabilities: bool = False
    has_raw_logits_capture: bool = False
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertStats:
    layer_index: int
    expert_index: int
    static_gate_mass: float
    static_gate_entropy: float
    router_bias_norm: Optional[float] = None
    static_rank: Optional[int] = None
    ffn_param_count: Optional[int] = None

    def sort_key(self) -> SchemaSortType:
        return prune_sort_key(
            score=self.static_gate_mass,
            secondary=self.router_bias_norm or 0.0,
            expert_index=self.expert_index,
        )


@dataclass(frozen=True)
class ActivationStats:
    layer_index: int
    expert_index: int
    token_count: int
    weighted_token_count: float
    mass_sum: float
    mean_weight: float
    entropy: float
    n_tokens: int
    timestamp_span: Optional[str] = None

    @property
    def occupancy(self) -> float:
        if self.n_tokens <= 0:
            return 0.0
        return float(self.token_count) / float(self.n_tokens)


@dataclass(frozen=True)
class PruneCandidate:
    layer_index: int
    expert_index: int
    score: float
    score_components: Dict[str, float] = field(default_factory=dict)
    strategy_name: str = "frequency"

    def sort_key(self) -> SchemaSortType:
        return prune_sort_key(
            score=self.score,
            secondary=self.score_components.get("secondary", 0.0),
            expert_index=self.expert_index,
        )


@dataclass(frozen=True)
class PrunePlanItem:
    layer_index: int
    keep_indices: Tuple[int, ...]
    drop_indices: Tuple[int, ...]
    expected_param_delta: int
    source_expert_count: int
    target_expert_count: int
    rationale: Optional[str] = None

    @property
    def normalized_keep_indices(self) -> Tuple[int, ...]:
        return tuple(sorted(self.keep_indices))

    @property
    def normalized_drop_indices(self) -> Tuple[int, ...]:
        return tuple(sorted(self.drop_indices))


@dataclass(frozen=True)
class PrunePlan:
    model_signature: str
    global_target_experts: Optional[int]
    per_layer_plans: Tuple[PrunePlanItem, ...]
    strategy_version: str
    created_by: str
    created_at: str = field(default_factory=_utcnow_iso)
    run_id: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_source_experts(self) -> int:
        return sum(item.source_expert_count for item in self.per_layer_plans)

    @property
    def total_target_experts(self) -> int:
        return sum(item.target_expert_count for item in self.per_layer_plans)

    @property
    def total_dropped(self) -> int:
        return self.total_source_experts - self.total_target_experts


@dataclass(frozen=True)
class RunArtifactManifest:
    run_id: str
    command: str
    seed: int
    top_k: int
    prompt_count: int
    prompt_set_hash: Optional[str]
    schema_version: str = CANONICAL_SCHEMA_VERSION
    model_handle: Optional[ModelHandle] = None
    input_checksums: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_utcnow_iso)
    finished_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def canonical_json(self) -> str:
        return dumps_canonical(self)
