"""Deterministic pruning plan generation over pluggable strategy outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Sequence, cast

from moe_surgeon.models.errors import TopologyMismatchError
from moe_surgeon.prune.strategies import (
    PruneStrategy,
    StrategyName,
    StrategyRegistry,
    create_strategy,
)
from moe_surgeon.schemas import (
    CANONICAL_DEFAULT_TIMESTAMP,
    ActivationStats,
    ExpertStats,
    LayerTopology,
    ModelHandle,
    PruneCandidate,
    PrunePlan,
    PrunePlanItem,
    SchemaKey,
    sort_experts,
    sort_plan_items,
    sort_topology,
    to_json,
)


@dataclass(frozen=True)
class LayerConstraintOverride:
    """Optional exact or bounded keep-count overrides for one layer."""

    target_experts: int | None = None
    min_experts: int | None = None
    max_experts: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("target_experts", "min_experts", "max_experts"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{field_name} must be an integer when provided")
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            if value == 0:
                raise ValueError(f"{field_name} must be >= 1 when provided")
        if (
            self.min_experts is not None
            and self.max_experts is not None
            and self.min_experts > self.max_experts
        ):
            raise ValueError("min_experts cannot exceed max_experts")
        if self.target_experts is not None and self.min_experts is not None:
            if self.target_experts < self.min_experts:
                raise ValueError("target_experts cannot be below min_experts")
        if self.target_experts is not None and self.max_experts is not None:
            if self.target_experts > self.max_experts:
                raise ValueError("target_experts cannot exceed max_experts")


@dataclass(frozen=True)
class PlannerConstraints:
    """Global and per-layer pruning bounds."""

    global_target_experts: int | None = None
    min_experts_per_layer: int = 1
    max_experts_per_layer: int | None = None
    layer_overrides: Mapping[int, LayerConstraintOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.global_target_experts is not None and (
            not isinstance(self.global_target_experts, int) or isinstance(self.global_target_experts, bool)
        ):
            raise ValueError("global_target_experts must be an integer when provided")
        if not isinstance(self.min_experts_per_layer, int) or isinstance(self.min_experts_per_layer, bool):
            raise ValueError("min_experts_per_layer must be an integer")
        if self.max_experts_per_layer is not None and (
            not isinstance(self.max_experts_per_layer, int) or isinstance(self.max_experts_per_layer, bool)
        ):
            raise ValueError("max_experts_per_layer must be an integer when provided")
        if self.global_target_experts is not None and self.global_target_experts <= 0:
            raise ValueError("global_target_experts must be >= 1 when provided")
        if self.min_experts_per_layer <= 0:
            raise ValueError("min_experts_per_layer must be >= 1")
        if self.max_experts_per_layer is not None and self.max_experts_per_layer <= 0:
            raise ValueError("max_experts_per_layer must be >= 1 when provided")
        if (
            self.max_experts_per_layer is not None
            and self.min_experts_per_layer > self.max_experts_per_layer
        ):
            raise ValueError("min_experts_per_layer cannot exceed max_experts_per_layer")
        normalized_items: list[tuple[int, LayerConstraintOverride]] = []
        for layer, override in self.layer_overrides.items():
            if not isinstance(layer, int) or isinstance(layer, bool):
                raise ValueError("layer_overrides keys must be integers")
            if layer < 0:
                raise ValueError("layer_overrides keys must be >= 0")
            if not isinstance(override, LayerConstraintOverride):
                raise ValueError(
                    "layer_overrides values must be LayerConstraintOverride instances"
                )
            normalized_items.append((layer, override))
        normalized = dict(sorted(normalized_items))
        for layer_index in normalized:
            if layer_index < 0:
                raise ValueError("layer_overrides keys must be >= 0")
        object.__setattr__(self, "layer_overrides", normalized)

    def canonical_payload(self) -> dict[str, object]:
        """Return a structured deterministic payload for hashing and audits."""

        overrides: dict[str, dict[str, int]] = {}
        for layer_index, override in self.layer_overrides.items():
            layer_payload: dict[str, int] = {}
            if override.target_experts is not None:
                layer_payload["target_experts"] = override.target_experts
            if override.min_experts is not None:
                layer_payload["min_experts"] = override.min_experts
            if override.max_experts is not None:
                layer_payload["max_experts"] = override.max_experts
            overrides[str(layer_index)] = layer_payload
        payload: dict[str, object] = {
            "min_experts_per_layer": self.min_experts_per_layer,
            "layer_overrides": overrides,
        }
        if self.global_target_experts is not None:
            payload["global_target_experts"] = self.global_target_experts
        if self.max_experts_per_layer is not None:
            payload["max_experts_per_layer"] = self.max_experts_per_layer
        return payload

    def plan_constraints(
        self,
        budgets: Sequence["_LayerBudget"] | None = None,
        *,
        global_target_experts: int | None = None,
    ) -> dict[str, SchemaKey]:
        """Return resolved scalar constraints for PrunePlan serialization."""

        flattened: dict[str, SchemaKey] = {
            "min_experts_per_layer": self.min_experts_per_layer,
        }
        resolved_global_target = (
            self.global_target_experts if global_target_experts is None else global_target_experts
        )
        if resolved_global_target is not None:
            flattened["global_target_experts"] = resolved_global_target
        if budgets is None:
            if self.max_experts_per_layer is not None:
                flattened["max_experts_per_layer"] = self.max_experts_per_layer
            for layer_index, layer_override in self.layer_overrides.items():
                if layer_override.target_experts is not None:
                    flattened[f"layer_{layer_index}_target_experts"] = layer_override.target_experts
                if layer_override.min_experts is not None:
                    flattened[f"layer_{layer_index}_min_experts"] = layer_override.min_experts
                if layer_override.max_experts is not None:
                    flattened[f"layer_{layer_index}_max_experts"] = layer_override.max_experts
            return flattened

        uniform_global_max: int | None = None
        if self.max_experts_per_layer is not None:
            resolved_global_maxima = {
                min(self.max_experts_per_layer, budget.layer.expert_count) for budget in budgets
            }
            if len(resolved_global_maxima) == 1:
                uniform_global_max = resolved_global_maxima.pop()
                flattened["max_experts_per_layer"] = uniform_global_max

        for budget in budgets:
            layer_index = budget.layer.layer_index
            resolved_override: LayerConstraintOverride | None = self.layer_overrides.get(layer_index)
            if resolved_override is not None and resolved_override.target_experts is not None:
                flattened[f"layer_{layer_index}_target_experts"] = budget.minimum_keep
                continue
            if budget.minimum_keep != self.min_experts_per_layer:
                flattened[f"layer_{layer_index}_min_experts"] = budget.minimum_keep
            if uniform_global_max is not None:
                if budget.maximum_keep != uniform_global_max:
                    flattened[f"layer_{layer_index}_max_experts"] = budget.maximum_keep
                continue
            if self.max_experts_per_layer is not None or budget.maximum_keep != budget.layer.expert_count:
                flattened[f"layer_{layer_index}_max_experts"] = budget.maximum_keep
        return flattened


@dataclass(frozen=True)
class _LayerBudget:
    layer: LayerTopology
    minimum_keep: int
    maximum_keep: int


def _canonical_constraints_json(constraints: PlannerConstraints) -> str:
    """Return canonical JSON for deterministic traceability fields."""

    return to_json(constraints.canonical_payload())


def _ordered_unique_topology(topology: Sequence[LayerTopology]) -> tuple[LayerTopology, ...]:
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


def _resolve_strategy(
    strategy: str | StrategyName | PruneStrategy,
    *,
    registry: StrategyRegistry | None,
) -> PruneStrategy:
    if isinstance(strategy, str):
        return create_strategy(strategy, registry=registry)
    if isinstance(strategy, StrategyName):
        return create_strategy(strategy, registry=registry)
    return strategy


def _validate_candidate_coverage(
    topology: Sequence[LayerTopology],
    candidates: Sequence[PruneCandidate],
    *,
    strategy_name: str,
) -> dict[int, tuple[PruneCandidate, ...]]:
    expected = {
        (layer.layer_index, expert_index)
        for layer in topology
        for expert_index in range(layer.expert_count)
    }
    indexed: dict[tuple[int, int], PruneCandidate] = {}
    for candidate in candidates:
        key = (candidate.layer_index, candidate.expert_index)
        if key not in expected:
            raise TopologyMismatchError(
                "prune candidates contain entry outside topology coverage",
                layer_index=candidate.layer_index,
                details={"expert_index": candidate.expert_index, "strategy_name": strategy_name},
            )
        if key in indexed:
            raise TopologyMismatchError(
                "duplicate prune candidate entry",
                layer_index=candidate.layer_index,
                details={"expert_index": candidate.expert_index, "strategy_name": strategy_name},
            )
        indexed[key] = candidate
    if len(indexed) != len(expected):
        missing = sorted(expected.difference(indexed))
        layer_index, expert_index = missing[0]
        raise TopologyMismatchError(
            "prune candidate coverage does not match topology",
            layer_index=layer_index,
            details={"expert_index": expert_index, "strategy_name": strategy_name},
        )

    grouped: dict[int, list[PruneCandidate]] = {layer.layer_index: [] for layer in topology}
    for layer in topology:
        for expert_index in range(layer.expert_count):
            grouped[layer.layer_index].append(indexed[(layer.layer_index, expert_index)])
    return {
        layer_index: tuple(
            cast(PruneCandidate, candidate) for candidate in sort_experts(layer_candidates)
        )
        for layer_index, layer_candidates in grouped.items()
    }


def _layer_budgets(
    topology: Sequence[LayerTopology],
    constraints: PlannerConstraints,
) -> tuple[_LayerBudget, ...]:
    layer_indices = {layer.layer_index for layer in topology}
    unknown_override_layers = sorted(set(constraints.layer_overrides).difference(layer_indices))
    if unknown_override_layers:
        raise TopologyMismatchError(
            "layer override references unknown topology layer",
            layer_index=unknown_override_layers[0],
        )

    budgets: list[_LayerBudget] = []
    for layer in topology:
        override = constraints.layer_overrides.get(layer.layer_index)
        minimum_keep = constraints.min_experts_per_layer
        maximum_keep = (
            layer.expert_count
            if constraints.max_experts_per_layer is None
            else min(constraints.max_experts_per_layer, layer.expert_count)
        )
        if override is not None:
            if override.min_experts is not None:
                minimum_keep = max(minimum_keep, override.min_experts)
            if override.max_experts is not None:
                maximum_keep = min(maximum_keep, override.max_experts)
            if override.target_experts is not None:
                if override.target_experts < constraints.min_experts_per_layer:
                    raise ValueError(
                        "layer target_experts cannot be below min_experts_per_layer"
                    )
                if override.target_experts < minimum_keep:
                    raise ValueError(
                        f"layer {layer.layer_index} target_experts is below the resolved minimum_keep "
                        f"{minimum_keep}"
                    )
                if override.target_experts > maximum_keep:
                    raise ValueError(
                        f"layer {layer.layer_index} target_experts exceeds the resolved maximum_keep "
                        f"{maximum_keep}"
                    )
                minimum_keep = override.target_experts
                maximum_keep = override.target_experts
        if minimum_keep > layer.expert_count:
            raise TopologyMismatchError(
                "minimum experts exceeds layer expert_count",
                layer_index=layer.layer_index,
                details={"minimum_keep": minimum_keep, "expert_count": layer.expert_count},
            )
        if maximum_keep > layer.expert_count:
            raise ValueError(
                f"maximum experts exceeds layer expert_count for layer {layer.layer_index}: "
                f"{maximum_keep} > {layer.expert_count}"
            )
        if maximum_keep < minimum_keep:
            raise ValueError(
                f"layer {layer.layer_index} keep bounds are infeasible: {minimum_keep} > {maximum_keep}"
            )
        budgets.append(
            _LayerBudget(
                layer=layer,
                minimum_keep=minimum_keep,
                maximum_keep=maximum_keep,
            )
        )
    return tuple(budgets)


def _requested_global_target(
    budgets: Sequence[_LayerBudget],
    constraints: PlannerConstraints,
) -> int:
    minimum_total = sum(budget.minimum_keep for budget in budgets)
    maximum_total = sum(budget.maximum_keep for budget in budgets)
    requested = maximum_total if constraints.global_target_experts is None else constraints.global_target_experts
    if requested < minimum_total:
        raise ValueError(
            f"global_target_experts={requested} is below required minimum {minimum_total}"
        )
    if requested > maximum_total:
        raise ValueError(
            f"global_target_experts={requested} exceeds allowed maximum {maximum_total}"
        )
    return requested


def _selected_keep_indices(
    budgets: Sequence[_LayerBudget],
    ranked_candidates: Mapping[int, tuple[PruneCandidate, ...]],
    *,
    global_target_experts: int,
) -> dict[int, tuple[int, ...]]:
    selected: dict[int, list[int]] = {}
    optional_pool: list[PruneCandidate] = []
    for budget in budgets:
        ranked = ranked_candidates[budget.layer.layer_index]
        selected[budget.layer.layer_index] = [
            candidate.expert_index for candidate in ranked[: budget.minimum_keep]
        ]
        optional_pool.extend(ranked[budget.minimum_keep : budget.maximum_keep])
    remaining = global_target_experts - sum(len(indices) for indices in selected.values())
    if remaining < 0:
        raise ValueError("global target allocation underflow")
    for candidate in (
        cast(PruneCandidate, ranked_candidate) for ranked_candidate in sort_experts(optional_pool)[:remaining]
    ):
        selected[candidate.layer_index].append(candidate.expert_index)
    return {layer_index: tuple(sorted(indices)) for layer_index, indices in selected.items()}


def _plan_items(
    budgets: Sequence[_LayerBudget],
    keep_indices: Mapping[int, tuple[int, ...]],
    *,
    strategy_name: str,
) -> tuple[PrunePlanItem, ...]:
    items: list[PrunePlanItem] = []
    for budget in budgets:
        layer = budget.layer
        kept = keep_indices[layer.layer_index]
        keep_set = set(kept)
        dropped = tuple(expert_index for expert_index in range(layer.expert_count) if expert_index not in keep_set)
        items.append(
            PrunePlanItem(
                layer_index=layer.layer_index,
                keep_indices=kept,
                drop_indices=dropped,
                source_expert_count=layer.expert_count,
                target_expert_count=len(kept),
                expected_expert_count=layer.expert_count,
                rationale=f"keep top {len(kept)} experts by {strategy_name}",
                metadata={
                    "minimum_keep": budget.minimum_keep,
                    "maximum_keep": budget.maximum_keep,
                    "layer_name": layer.layer_name,
                },
            )
        )
    return sort_plan_items(items)


def _plan_id(
    *,
    model_signature: str,
    strategy_name: str,
    strategy_version: str,
    source_run_id: str | None,
    constraints: PlannerConstraints,
    topology: Sequence[LayerTopology],
    candidate_digest: str,
    plan_items: Sequence[PrunePlanItem],
) -> str:
    payload = {
        "model_signature": model_signature,
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "source_run_id": source_run_id,
        "constraints": constraints.canonical_payload(),
        "topology": [layer.to_dict() for layer in topology],
        "candidate_digest": candidate_digest,
        "per_layer_plans": [item.to_dict() for item in plan_items],
    }
    return f"plan-{sha256(to_json(payload).encode('utf-8')).hexdigest()[:16]}"


def _budget_metadata(
    budgets: Sequence[_LayerBudget],
    *,
    requested_global_target: int,
    candidate_digest: str,
    constraints_json: str,
) -> dict[str, SchemaKey]:
    """Flatten resolved budget diagnostics into scalar metadata fields."""

    metadata: dict[str, SchemaKey] = {
        "budget_min_total": sum(budget.minimum_keep for budget in budgets),
        "budget_max_total": sum(budget.maximum_keep for budget in budgets),
        "candidate_digest": candidate_digest,
        "constraints_json": constraints_json,
        "layer_count": len(budgets),
        "total_dropped_experts": sum(budget.layer.expert_count for budget in budgets)
        - requested_global_target,
        "total_source_experts": sum(budget.layer.expert_count for budget in budgets),
        "total_target_experts": requested_global_target,
    }
    for budget in budgets:
        metadata[f"layer_{budget.layer.layer_index}_minimum_keep"] = budget.minimum_keep
        metadata[f"layer_{budget.layer.layer_index}_maximum_keep"] = budget.maximum_keep
    return metadata


def build_prune_plan(
    topology: Sequence[LayerTopology],
    *,
    strategy: str | StrategyName | PruneStrategy = StrategyName.FREQUENCY,
    expert_stats: Sequence[ExpertStats] | None = None,
    activation_stats: Sequence[ActivationStats] | None = None,
    constraints: PlannerConstraints | None = None,
    model_handle: ModelHandle | None = None,
    model_signature: str | None = None,
    source_run_id: str | None = None,
    registry: StrategyRegistry | None = None,
) -> PrunePlan:
    """Build a deterministic prune plan from strategy-produced expert scores."""

    ordered_topology = _ordered_unique_topology(topology)
    active_constraints = PlannerConstraints() if constraints is None else constraints
    active_strategy = _resolve_strategy(strategy, registry=registry)
    ranked_candidates = _validate_candidate_coverage(
        ordered_topology,
        active_strategy.build_candidates(
            ordered_topology,
            expert_stats=expert_stats,
            activation_stats=activation_stats,
        ),
        strategy_name=active_strategy.metadata.name,
    )
    budgets = _layer_budgets(ordered_topology, active_constraints)
    requested_global_target = _requested_global_target(budgets, active_constraints)
    keep_indices = _selected_keep_indices(
        budgets,
        ranked_candidates,
        global_target_experts=requested_global_target,
    )
    plan_items = _plan_items(
        budgets,
        keep_indices,
        strategy_name=active_strategy.metadata.name,
    )
    resolved_model_signature = (
        model_signature
        if model_signature is not None
        else (
            _default_model_signature(model_handle)
            if model_handle is not None
            else "unknown"
        )
    )
    constraints_json = _canonical_constraints_json(active_constraints)
    constraint_digest = sha256(constraints_json.encode("utf-8")).hexdigest()
    candidate_digest = sha256(
        to_json([candidate.to_dict() for candidates in ranked_candidates.values() for candidate in candidates]).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = _budget_metadata(
        budgets,
        requested_global_target=requested_global_target,
        candidate_digest=candidate_digest,
        constraints_json=constraints_json,
    )
    metadata["constraint_digest"] = constraint_digest
    return PrunePlan(
        plan_id=_plan_id(
            model_signature=resolved_model_signature,
            strategy_name=active_strategy.metadata.name,
            strategy_version=active_strategy.metadata.version,
            source_run_id=source_run_id,
            constraints=active_constraints,
            topology=ordered_topology,
            candidate_digest=candidate_digest,
            plan_items=plan_items,
        ),
        model_signature=resolved_model_signature,
        strategy_name=active_strategy.metadata.name,
        strategy_version=active_strategy.metadata.version,
        per_layer_plans=plan_items,
        global_target_experts=requested_global_target,
        model_handle=model_handle,
        created_by="planner",
        created_at=CANONICAL_DEFAULT_TIMESTAMP,
        source_run_id=source_run_id,
        constraints=active_constraints.plan_constraints(
            budgets,
            global_target_experts=requested_global_target,
        ),
        metadata=metadata,
    )


def _default_model_signature(model_handle: ModelHandle) -> str:
    revision = model_handle.revision or "none"
    return f"{model_handle.model_id}:{revision}"


__all__ = [
    "LayerConstraintOverride",
    "PlannerConstraints",
    "build_prune_plan",
]
