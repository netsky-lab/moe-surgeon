"""Pruning strategy protocol, metadata, and built-in deterministic scorers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence, cast

from moe_surgeon.models.errors import SchemaValidationError, TopologyMismatchError
from moe_surgeon.schemas import (
    CANONICAL_EXPERT_TIE_BREAK_POLICY,
    ActivationStats,
    ExpertStats,
    LayerTopology,
    PruneCandidate,
    sort_experts,
    sort_topology,
)

FREQUENCY_STRATEGY_VERSION = "1"
ROUTER_MASS_STRATEGY_VERSION = "1"
COMBINED_STRATEGY_VERSION = "1"
COMBINED_FREQUENCY_WEIGHT = 0.7
COMBINED_ROUTER_MASS_WEIGHT = 0.3


class StrategyName(str, Enum):
    """Canonical built-in pruning strategy names."""

    FREQUENCY = "frequency"
    ROUTER_MASS = "router_mass"
    COMBINED = "combined"


@dataclass(frozen=True)
class StrategyMetadata:
    """Static strategy configuration for planning and auditability."""

    name: str
    version: str
    score_columns: tuple[str, ...]
    normalization_behavior: str
    tie_break_policy: str = CANONICAL_EXPERT_TIE_BREAK_POLICY
    requires_static_stats: bool = False
    requires_activation_stats: bool = False


class PruneStrategy(Protocol):
    """Protocol for pluggable expert ranking strategies."""

    @property
    def metadata(self) -> StrategyMetadata:
        """Return immutable strategy metadata."""

    def build_candidates(
        self,
        topology: Sequence[LayerTopology],
        *,
        expert_stats: Sequence[ExpertStats] | None = None,
        activation_stats: Sequence[ActivationStats] | None = None,
    ) -> tuple[PruneCandidate, ...]:
        """Produce one deterministic candidate per topology expert."""


@dataclass(frozen=True)
class _StaticStrategy:
    metadata: StrategyMetadata
    _builder: "_StrategyBuilder"

    def build_candidates(
        self,
        topology: Sequence[LayerTopology],
        *,
        expert_stats: Sequence[ExpertStats] | None = None,
        activation_stats: Sequence[ActivationStats] | None = None,
    ) -> tuple[PruneCandidate, ...]:
        return self._builder(
            self.metadata,
            topology,
            expert_stats=expert_stats,
            activation_stats=activation_stats,
        )


_StrategyBuilder = Callable[..., tuple[PruneCandidate, ...]]


def _sorted_unique_topology(topology: Sequence[LayerTopology]) -> tuple[LayerTopology, ...]:
    ordered = sort_topology(topology)
    seen: set[int] = set()
    for layer in ordered:
        if layer.layer_index in seen:
            raise TopologyMismatchError(
                "duplicate layer_index in topology",
                layer_index=layer.layer_index,
            )
        seen.add(layer.layer_index)
    return ordered


def _validate_required_inputs(
    metadata: StrategyMetadata,
    *,
    expert_stats: Sequence[ExpertStats] | None,
    activation_stats: Sequence[ActivationStats] | None,
) -> None:
    if metadata.requires_activation_stats and activation_stats is None:
        raise SchemaValidationError(f"{metadata.name} strategy requires runtime activation_stats")
    if metadata.requires_static_stats and expert_stats is None:
        raise SchemaValidationError(f"{metadata.name} strategy requires static expert_stats")


def _index_static_stats(
    topology: Sequence[LayerTopology],
    stats: Sequence[ExpertStats] | None,
    *,
    strategy_name: str,
) -> dict[tuple[int, int], ExpertStats]:
    if stats is None:
        return {}
    expected = {
        (layer.layer_index, expert_index)
        for layer in topology
        for expert_index in range(layer.expert_count)
    }
    indexed: dict[tuple[int, int], ExpertStats] = {}
    for stat in stats:
        key = (stat.layer_index, stat.expert_index)
        if key not in expected:
            raise TopologyMismatchError(
                "expert_stats contains entry outside topology coverage",
                layer_index=stat.layer_index,
                details={"expert_index": stat.expert_index, "strategy_name": strategy_name},
            )
        if key in indexed:
            raise TopologyMismatchError(
                "duplicate expert_stats entry",
                layer_index=stat.layer_index,
                details={"expert_index": stat.expert_index, "strategy_name": strategy_name},
            )
        indexed[key] = stat
    if len(indexed) != len(expected):
        missing = sorted(expected.difference(indexed))
        layer_index, expert_index = missing[0]
        raise TopologyMismatchError(
            "expert_stats coverage does not match topology",
            layer_index=layer_index,
            details={"expert_index": expert_index, "strategy_name": strategy_name},
        )
    return indexed


def _index_activation_stats(
    topology: Sequence[LayerTopology],
    stats: Sequence[ActivationStats] | None,
    *,
    strategy_name: str,
) -> dict[tuple[int, int], ActivationStats]:
    if stats is None:
        return {}
    expected = {
        (layer.layer_index, expert_index)
        for layer in topology
        for expert_index in range(layer.expert_count)
    }
    indexed: dict[tuple[int, int], ActivationStats] = {}
    for stat in stats:
        key = (stat.layer_index, stat.expert_index)
        if key not in expected:
            raise TopologyMismatchError(
                "activation_stats contains entry outside topology coverage",
                layer_index=stat.layer_index,
                details={"expert_index": stat.expert_index, "strategy_name": strategy_name},
            )
        if key in indexed:
            raise TopologyMismatchError(
                "duplicate activation_stats entry",
                layer_index=stat.layer_index,
                details={"expert_index": stat.expert_index, "strategy_name": strategy_name},
            )
        indexed[key] = stat
    if len(indexed) != len(expected):
        missing = sorted(expected.difference(indexed))
        layer_index, expert_index = missing[0]
        raise TopologyMismatchError(
            "activation_stats coverage does not match topology",
            layer_index=layer_index,
            details={"expert_index": expert_index, "strategy_name": strategy_name},
        )
    return indexed


def _layer_runtime_totals(
    topology: Sequence[LayerTopology],
    activation_index: Mapping[tuple[int, int], ActivationStats],
) -> dict[int, tuple[float, float]]:
    totals: dict[int, tuple[float, float]] = {}
    for layer in topology:
        token_total = 0.0
        mass_total = 0.0
        for expert_index in range(layer.expert_count):
            stat = activation_index[(layer.layer_index, expert_index)]
            token_total += float(stat.token_count)
            mass_total += float(stat.mass_sum)
        totals[layer.layer_index] = (token_total, mass_total)
    return totals


def _layer_static_totals(
    topology: Sequence[LayerTopology],
    expert_index: Mapping[tuple[int, int], ExpertStats],
) -> dict[int, float]:
    totals: dict[int, float] = {}
    for layer in topology:
        mass_total = 0.0
        for expert_id in range(layer.expert_count):
            mass_total += float(expert_index[(layer.layer_index, expert_id)].static_gate_mass)
        totals[layer.layer_index] = mass_total
    return totals


def _candidate(
    *,
    metadata: StrategyMetadata,
    layer_index: int,
    expert_index: int,
    score: float,
    secondary_score: float,
    score_components: Mapping[str, float],
) -> PruneCandidate:
    return PruneCandidate(
        layer_index=layer_index,
        expert_index=expert_index,
        score=score,
        secondary_score=secondary_score,
        strategy_name=metadata.name,
        score_components=dict(score_components),
        metadata={
            "strategy_version": metadata.version,
            "normalization_behavior": metadata.normalization_behavior,
            "tie_break_policy": metadata.tie_break_policy,
        },
    )


def _sorted_candidates(candidates: Sequence[PruneCandidate]) -> tuple[PruneCandidate, ...]:
    return tuple(cast(PruneCandidate, candidate) for candidate in sort_experts(candidates))


def _build_frequency_candidates(
    metadata: StrategyMetadata,
    topology: Sequence[LayerTopology],
    *,
    expert_stats: Sequence[ExpertStats] | None = None,
    activation_stats: Sequence[ActivationStats] | None = None,
) -> tuple[PruneCandidate, ...]:
    del expert_stats
    ordered_topology = _sorted_unique_topology(topology)
    _validate_required_inputs(metadata, expert_stats=None, activation_stats=activation_stats)
    activation_index = _index_activation_stats(
        ordered_topology,
        activation_stats,
        strategy_name=metadata.name,
    )
    runtime_totals = _layer_runtime_totals(ordered_topology, activation_index)
    candidates: list[PruneCandidate] = []
    for layer in ordered_topology:
        token_total, mass_total = runtime_totals[layer.layer_index]
        token_denom = token_total if token_total > 0 else 1.0
        mass_denom = mass_total if mass_total > 0 else 1.0
        for expert_id in range(layer.expert_count):
            stat = activation_index[(layer.layer_index, expert_id)]
            token_share = float(stat.token_count) / token_denom
            mass_share = float(stat.mass_sum) / mass_denom
            candidates.append(
                _candidate(
                    metadata=metadata,
                    layer_index=layer.layer_index,
                    expert_index=expert_id,
                    score=token_share,
                    secondary_score=mass_share,
                    score_components={
                        "token_count": float(stat.token_count),
                        "token_share": token_share,
                        "mass_sum": float(stat.mass_sum),
                        "mass_share": mass_share,
                    },
                )
            )
    return _sorted_candidates(candidates)


def _build_router_mass_candidates(
    metadata: StrategyMetadata,
    topology: Sequence[LayerTopology],
    *,
    expert_stats: Sequence[ExpertStats] | None = None,
    activation_stats: Sequence[ActivationStats] | None = None,
) -> tuple[PruneCandidate, ...]:
    del activation_stats
    ordered_topology = _sorted_unique_topology(topology)
    _validate_required_inputs(metadata, expert_stats=expert_stats, activation_stats=None)
    expert_index = _index_static_stats(
        ordered_topology,
        expert_stats,
        strategy_name=metadata.name,
    )
    static_totals = _layer_static_totals(ordered_topology, expert_index)
    candidates: list[PruneCandidate] = []
    for layer in ordered_topology:
        layer_total = static_totals[layer.layer_index]
        layer_denom = layer_total if layer_total > 0 else 1.0
        for expert_id in range(layer.expert_count):
            stat = expert_index[(layer.layer_index, expert_id)]
            gate_share = float(stat.static_gate_mass) / layer_denom
            entropy_norm = (
                float(stat.static_gate_entropy_norm)
                if stat.static_gate_entropy_norm is not None
                else float(stat.static_gate_entropy)
            )
            candidates.append(
                _candidate(
                    metadata=metadata,
                    layer_index=layer.layer_index,
                    expert_index=expert_id,
                    score=gate_share,
                    secondary_score=-entropy_norm,
                    score_components={
                        "static_gate_mass": float(stat.static_gate_mass),
                        "static_gate_share": gate_share,
                        "static_gate_entropy": float(stat.static_gate_entropy),
                        "static_gate_entropy_norm": entropy_norm,
                    },
                )
            )
    return _sorted_candidates(candidates)


def _build_combined_candidates(
    metadata: StrategyMetadata,
    topology: Sequence[LayerTopology],
    *,
    expert_stats: Sequence[ExpertStats] | None = None,
    activation_stats: Sequence[ActivationStats] | None = None,
) -> tuple[PruneCandidate, ...]:
    ordered_topology = _sorted_unique_topology(topology)
    _validate_required_inputs(
        metadata,
        expert_stats=expert_stats,
        activation_stats=activation_stats,
    )
    expert_index = _index_static_stats(
        ordered_topology,
        expert_stats,
        strategy_name=metadata.name,
    )
    activation_index = _index_activation_stats(
        ordered_topology,
        activation_stats,
        strategy_name=metadata.name,
    )
    static_totals = _layer_static_totals(ordered_topology, expert_index)
    runtime_totals = _layer_runtime_totals(ordered_topology, activation_index)
    candidates: list[PruneCandidate] = []
    for layer in ordered_topology:
        token_total, mass_total = runtime_totals[layer.layer_index]
        static_total = static_totals[layer.layer_index]
        token_denom = token_total if token_total > 0 else 1.0
        runtime_mass_denom = mass_total if mass_total > 0 else 1.0
        static_mass_denom = static_total if static_total > 0 else 1.0
        for expert_id in range(layer.expert_count):
            runtime_stat = activation_index[(layer.layer_index, expert_id)]
            static_stat = expert_index[(layer.layer_index, expert_id)]
            token_share = float(runtime_stat.token_count) / token_denom
            runtime_mass_share = float(runtime_stat.mass_sum) / runtime_mass_denom
            static_mass_share = float(static_stat.static_gate_mass) / static_mass_denom
            combined_score = (
                COMBINED_FREQUENCY_WEIGHT * token_share
                + COMBINED_ROUTER_MASS_WEIGHT * static_mass_share
            )
            combined_secondary = (
                COMBINED_FREQUENCY_WEIGHT * runtime_mass_share
                + COMBINED_ROUTER_MASS_WEIGHT * static_mass_share
            )
            candidates.append(
                _candidate(
                    metadata=metadata,
                    layer_index=layer.layer_index,
                    expert_index=expert_id,
                    score=combined_score,
                    secondary_score=combined_secondary,
                    score_components={
                        "combined_score": combined_score,
                        "combined_secondary": combined_secondary,
                        "frequency_weight": COMBINED_FREQUENCY_WEIGHT,
                        "router_mass_weight": COMBINED_ROUTER_MASS_WEIGHT,
                        "token_share": token_share,
                        "runtime_mass_share": runtime_mass_share,
                        "static_gate_share": static_mass_share,
                    },
                )
            )
    return _sorted_candidates(candidates)


class StrategyRegistry:
    """Named strategy registry with deterministic lookup ordering."""

    def __init__(self, strategies: Sequence[PruneStrategy] | None = None) -> None:
        self._strategies: dict[str, PruneStrategy] = {}
        if strategies is not None:
            for strategy in strategies:
                self.register(strategy)

    def register(self, strategy: PruneStrategy) -> None:
        name = strategy.metadata.name
        if name in self._strategies:
            raise ValueError(f"duplicate strategy registration: {name}")
        self._strategies[name] = strategy

    def get(self, name: str | StrategyName) -> PruneStrategy:
        lookup = name.value if isinstance(name, StrategyName) else name
        strategy = self._strategies.get(lookup)
        if strategy is None:
            available = ", ".join(sorted(self._strategies))
            raise ValueError(f"unknown strategy '{lookup}' (available: {available})")
        return strategy

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def items(self) -> tuple[tuple[str, PruneStrategy], ...]:
        return tuple((name, self._strategies[name]) for name in sorted(self._strategies))


def create_strategy(name: str | StrategyName, *, registry: StrategyRegistry | None = None) -> PruneStrategy:
    """Resolve a strategy instance from the registry."""

    active_registry = strategy_registry if registry is None else registry
    return active_registry.get(name)


def _built_in_strategies() -> tuple[PruneStrategy, ...]:
    return (
        _StaticStrategy(
            metadata=StrategyMetadata(
                name=StrategyName.FREQUENCY.value,
                version=FREQUENCY_STRATEGY_VERSION,
                score_columns=("token_count", "token_share", "mass_sum", "mass_share"),
                normalization_behavior="per_layer_fraction",
                requires_activation_stats=True,
            ),
            _builder=_build_frequency_candidates,
        ),
        _StaticStrategy(
            metadata=StrategyMetadata(
                name=StrategyName.ROUTER_MASS.value,
                version=ROUTER_MASS_STRATEGY_VERSION,
                score_columns=(
                    "static_gate_mass",
                    "static_gate_share",
                    "static_gate_entropy",
                    "static_gate_entropy_norm",
                ),
                normalization_behavior="per_layer_fraction",
                requires_static_stats=True,
            ),
            _builder=_build_router_mass_candidates,
        ),
        _StaticStrategy(
            metadata=StrategyMetadata(
                name=StrategyName.COMBINED.value,
                version=COMBINED_STRATEGY_VERSION,
                score_columns=(
                    "combined_score",
                    "token_share",
                    "runtime_mass_share",
                    "static_gate_share",
                ),
                normalization_behavior="per_layer_fraction_weighted_sum",
                requires_static_stats=True,
                requires_activation_stats=True,
            ),
            _builder=_build_combined_candidates,
        ),
    )


strategy_registry = StrategyRegistry(_built_in_strategies())


__all__ = [
    "COMBINED_FREQUENCY_WEIGHT",
    "COMBINED_ROUTER_MASS_WEIGHT",
    "COMBINED_STRATEGY_VERSION",
    "FREQUENCY_STRATEGY_VERSION",
    "PruneStrategy",
    "ROUTER_MASS_STRATEGY_VERSION",
    "StrategyMetadata",
    "StrategyName",
    "StrategyRegistry",
    "create_strategy",
    "strategy_registry",
]
