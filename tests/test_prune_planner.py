from dataclasses import dataclass

import pytest

from moe_surgeon.models.errors import SchemaValidationError, TopologyMismatchError
from moe_surgeon.prune.planner import (
    LayerConstraintOverride,
    PlannerConstraints,
    build_prune_plan,
)
from moe_surgeon.prune.strategies import StrategyMetadata, strategy_registry
from moe_surgeon.schemas import (
    CANONICAL_EXPERT_TIE_BREAK_POLICY,
    ActivationStats,
    ExpertStats,
    LayerTopology,
    ModelHandle,
    PruneCandidate,
    to_json,
)


def _topology() -> tuple[LayerTopology, ...]:
    return (
        LayerTopology(
            layer_index=0,
            layer_name="layer0",
            layer_type="moe",
            expert_count=3,
            top_k=2,
            hidden_size=64,
        ),
        LayerTopology(
            layer_index=1,
            layer_name="layer1",
            layer_type="moe",
            expert_count=3,
            top_k=2,
            hidden_size=64,
        ),
    )


def _activation_stats() -> tuple[ActivationStats, ...]:
    return (
        ActivationStats(
            layer_index=0,
            expert_index=0,
            token_count=10,
            weighted_token_count=8.0,
            mass_sum=8.0,
            mean_weight=0.8,
            entropy=0.1,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=0,
            expert_index=1,
            token_count=6,
            weighted_token_count=5.0,
            mass_sum=5.0,
            mean_weight=0.83,
            entropy=0.2,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=0,
            expert_index=2,
            token_count=4,
            weighted_token_count=3.0,
            mass_sum=3.0,
            mean_weight=0.75,
            entropy=0.3,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=1,
            expert_index=0,
            token_count=9,
            weighted_token_count=7.0,
            mass_sum=7.0,
            mean_weight=0.78,
            entropy=0.15,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=1,
            expert_index=1,
            token_count=6,
            weighted_token_count=4.0,
            mass_sum=4.0,
            mean_weight=0.66,
            entropy=0.25,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=1,
            expert_index=2,
            token_count=5,
            weighted_token_count=3.0,
            mass_sum=3.0,
            mean_weight=0.6,
            entropy=0.35,
            n_tokens=20,
        ),
    )


def _expert_stats() -> tuple[ExpertStats, ...]:
    return (
        ExpertStats(
            layer_index=0,
            expert_index=0,
            static_gate_mass=0.65,
            static_gate_entropy=0.1,
            static_gate_entropy_norm=0.1,
        ),
        ExpertStats(
            layer_index=0,
            expert_index=1,
            static_gate_mass=0.20,
            static_gate_entropy=0.2,
            static_gate_entropy_norm=0.2,
        ),
        ExpertStats(
            layer_index=0,
            expert_index=2,
            static_gate_mass=0.15,
            static_gate_entropy=0.3,
            static_gate_entropy_norm=0.3,
        ),
        ExpertStats(
            layer_index=1,
            expert_index=0,
            static_gate_mass=0.20,
            static_gate_entropy=0.4,
            static_gate_entropy_norm=0.4,
        ),
        ExpertStats(
            layer_index=1,
            expert_index=1,
            static_gate_mass=0.55,
            static_gate_entropy=0.2,
            static_gate_entropy_norm=0.2,
        ),
        ExpertStats(
            layer_index=1,
            expert_index=2,
            static_gate_mass=0.25,
            static_gate_entropy=0.1,
            static_gate_entropy_norm=0.1,
        ),
    )


def test_frequency_strategy_ranking_is_deterministic_with_tie_breaks() -> None:
    tied_stats = tuple(
        ActivationStats(
            layer_index=0,
            expert_index=index,
            token_count=5,
            weighted_token_count=2.0,
            mass_sum=2.0,
            mean_weight=0.4,
            entropy=0.1,
            n_tokens=10,
        )
        for index in range(3)
    )
    candidates = strategy_registry.get("frequency").build_candidates(_topology()[:1], activation_stats=tied_stats)

    assert [(candidate.layer_index, candidate.expert_index) for candidate in candidates] == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]
    assert all(
        candidate.metadata["tie_break_policy"] == CANONICAL_EXPERT_TIE_BREAK_POLICY
        for candidate in candidates
    )


def test_strategy_registry_exposes_built_in_metadata() -> None:
    combined = strategy_registry.get("combined")

    assert combined.metadata.score_columns == (
        "combined_score",
        "token_share",
        "runtime_mass_share",
        "static_gate_share",
    )
    assert combined.metadata.normalization_behavior == "per_layer_fraction_weighted_sum"
    assert combined.metadata.tie_break_policy == CANONICAL_EXPERT_TIE_BREAK_POLICY


def test_built_in_strategies_validate_missing_inputs_and_partial_coverage() -> None:
    with pytest.raises(SchemaValidationError):
        strategy_registry.get("frequency").build_candidates(_topology(), activation_stats=None)
    with pytest.raises(SchemaValidationError):
        strategy_registry.get("router_mass").build_candidates(_topology(), expert_stats=None)
    with pytest.raises(TopologyMismatchError):
        strategy_registry.get("combined").build_candidates(
            _topology(),
            expert_stats=_expert_stats()[:-1],
            activation_stats=_activation_stats(),
        )


def test_strategy_registry_can_inject_custom_strategy_without_planner_changes() -> None:
    @dataclass(frozen=True)
    class CustomStrategy:
        metadata: StrategyMetadata = StrategyMetadata(
            name="custom",
            version="1",
            score_columns=("score",),
            normalization_behavior="none",
        )

        def build_candidates(
            self,
            topology: tuple[LayerTopology, ...],
            *,
            expert_stats: tuple[ExpertStats, ...] | None = None,
            activation_stats: tuple[ActivationStats, ...] | None = None,
        ) -> tuple[PruneCandidate, ...]:
            del expert_stats, activation_stats
            return tuple(
                PruneCandidate(
                    layer_index=layer.layer_index,
                    expert_index=expert_index,
                    score=float(layer.expert_count - expert_index),
                    strategy_name=self.metadata.name,
                )
                for layer in topology
                for expert_index in range(layer.expert_count)
            )

    plan = build_prune_plan(
        _topology(),
        strategy=CustomStrategy(),
        constraints=PlannerConstraints(global_target_experts=2, min_experts_per_layer=1),
        model_signature="custom-model",
    )

    assert plan.strategy_name == "custom"
    assert [item.keep_indices for item in plan.per_layer_plans] == [(0,), (0,)]


def test_planner_enforces_minimum_survivors_and_global_budget() -> None:
    plan = build_prune_plan(
        _topology(),
        strategy="frequency",
        activation_stats=_activation_stats(),
        constraints=PlannerConstraints(global_target_experts=3, min_experts_per_layer=1),
        model_signature="model-a",
    )

    assert [item.keep_indices for item in plan.per_layer_plans] == [(0, 1), (0,)]
    assert plan.global_target_experts == 3
    assert plan.total_target_experts == 3


def test_planner_rejects_invalid_global_budget_and_unknown_override_layer() -> None:
    with pytest.raises(ValueError, match="must be >= 1 when provided"):
        PlannerConstraints(global_target_experts=0)

    with pytest.raises(ValueError, match="below required minimum"):
        build_prune_plan(
            _topology(),
            strategy="frequency",
            activation_stats=_activation_stats(),
            constraints=PlannerConstraints(global_target_experts=1, min_experts_per_layer=1),
            model_signature="model-a",
        )

    with pytest.raises(TopologyMismatchError, match="unknown topology layer"):
        build_prune_plan(
            _topology(),
            strategy="frequency",
            activation_stats=_activation_stats(),
            constraints=PlannerConstraints(
                global_target_experts=4,
                layer_overrides={9: LayerConstraintOverride(target_experts=1)},
            ),
            model_signature="model-a",
        )


def test_planner_rejects_zero_survivor_constraints_early() -> None:
    with pytest.raises(ValueError, match="min_experts_per_layer must be >= 1"):
        PlannerConstraints(min_experts_per_layer=0)

    with pytest.raises(ValueError, match="max_experts_per_layer must be >= 1 when provided"):
        PlannerConstraints(max_experts_per_layer=0)

    with pytest.raises(ValueError, match="target_experts must be >= 1 when provided"):
        LayerConstraintOverride(target_experts=0)

    with pytest.raises(ValueError, match="min_experts must be >= 1 when provided"):
        LayerConstraintOverride(min_experts=0)

    with pytest.raises(ValueError, match="max_experts must be >= 1 when provided"):
        LayerConstraintOverride(max_experts=0)


def test_planner_rejects_infeasible_layer_bounds() -> None:
    with pytest.raises(TopologyMismatchError, match="minimum experts exceeds layer expert_count"):
        build_prune_plan(
            _topology(),
            strategy="frequency",
            activation_stats=_activation_stats(),
            constraints=PlannerConstraints(min_experts_per_layer=4),
            model_signature="model-a",
        )

    with pytest.raises(ValueError, match="keep bounds are infeasible"):
        build_prune_plan(
            _topology(),
            strategy="frequency",
            activation_stats=_activation_stats(),
            constraints=PlannerConstraints(
                min_experts_per_layer=1,
                max_experts_per_layer=1,
                layer_overrides={0: LayerConstraintOverride(min_experts=2)},
            ),
            model_signature="model-a",
        )


def test_planner_honors_exact_per_layer_override() -> None:
    plan = build_prune_plan(
        _topology(),
        strategy="frequency",
        activation_stats=_activation_stats(),
        constraints=PlannerConstraints(
            global_target_experts=4,
            min_experts_per_layer=1,
            layer_overrides={1: LayerConstraintOverride(target_experts=2)},
        ),
        model_signature="model-a",
    )

    assert [item.keep_indices for item in plan.per_layer_plans] == [(0, 1), (0, 1)]
    assert plan.constraints["layer_1_target_experts"] == 2


def test_combined_strategy_fuses_static_and_runtime_signals() -> None:
    plan = build_prune_plan(
        _topology(),
        strategy="combined",
        activation_stats=_activation_stats(),
        expert_stats=_expert_stats(),
        constraints=PlannerConstraints(global_target_experts=3, min_experts_per_layer=1),
        model_signature="model-b",
    )

    assert [item.keep_indices for item in plan.per_layer_plans] == [(0,), (0, 1)]


def test_planner_uses_stable_cross_layer_tie_breaking_for_global_budget() -> None:
    tied_activation_stats = (
        ActivationStats(
            layer_index=0,
            expert_index=0,
            token_count=10,
            weighted_token_count=10.0,
            mass_sum=10.0,
            mean_weight=1.0,
            entropy=0.1,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=0,
            expert_index=1,
            token_count=5,
            weighted_token_count=5.0,
            mass_sum=5.0,
            mean_weight=1.0,
            entropy=0.1,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=0,
            expert_index=2,
            token_count=5,
            weighted_token_count=5.0,
            mass_sum=5.0,
            mean_weight=1.0,
            entropy=0.1,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=1,
            expert_index=0,
            token_count=10,
            weighted_token_count=10.0,
            mass_sum=10.0,
            mean_weight=1.0,
            entropy=0.1,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=1,
            expert_index=1,
            token_count=5,
            weighted_token_count=5.0,
            mass_sum=5.0,
            mean_weight=1.0,
            entropy=0.1,
            n_tokens=20,
        ),
        ActivationStats(
            layer_index=1,
            expert_index=2,
            token_count=5,
            weighted_token_count=5.0,
            mass_sum=5.0,
            mean_weight=1.0,
            entropy=0.1,
            n_tokens=20,
        ),
    )

    plan = build_prune_plan(
        _topology(),
        strategy="frequency",
        activation_stats=tied_activation_stats,
        constraints=PlannerConstraints(global_target_experts=3, min_experts_per_layer=1),
        model_signature="model-tied",
    )

    assert [item.keep_indices for item in plan.per_layer_plans] == [(0, 1), (0,)]


def test_plan_metadata_traceability_and_repeated_output_are_stable() -> None:
    constraints = PlannerConstraints(
        global_target_experts=4,
        min_experts_per_layer=1,
        layer_overrides={0: LayerConstraintOverride(max_experts=2)},
    )
    model_handle = ModelHandle(model_id="gemma-test", backend_name="gemma4")

    first = build_prune_plan(
        _topology(),
        strategy="router_mass",
        expert_stats=_expert_stats(),
        constraints=constraints,
        model_handle=model_handle,
        source_run_id="run-001",
    )
    second = build_prune_plan(
        _topology(),
        strategy="router_mass",
        expert_stats=_expert_stats(),
        constraints=constraints,
        model_handle=model_handle,
        source_run_id="run-001",
    )

    assert to_json(first) == to_json(second)
    assert first.plan_id == second.plan_id
    assert first.created_at == second.created_at == "1970-01-01T00:00:00+00:00"
    assert first.strategy_version == strategy_registry.get("router_mass").metadata.version
    assert first.source_run_id == "run-001"
    assert first.constraints["layer_0_max_experts"] == 2
    assert first.metadata["total_target_experts"] == 4
    assert first.metadata["budget_min_total"] == 2
    assert first.metadata["budget_max_total"] == 5
    assert first.metadata["layer_0_minimum_keep"] == 1
    assert first.metadata["layer_0_maximum_keep"] == 2
    assert first.metadata["constraints_json"] == (
        '{"global_target_experts":4,"layer_overrides":{"0":{"max_experts":2}},'
        '"min_experts_per_layer":1}'
    )
    assert isinstance(first.metadata["constraint_digest"], str)
    assert isinstance(first.metadata["candidate_digest"], str)
