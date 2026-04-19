from __future__ import annotations

import pytest

from moe_surgeon.models.errors import TopologyMismatchError
from moe_surgeon.prune.planner import (
    LayerConstraintOverride,
    PlannerConstraints,
    _layer_budgets,
    build_prune_plan,
    validate_planner_inputs,
)
from moe_surgeon.prune.strategy import create_strategy
from moe_surgeon.schemas import to_json

from tests.fixtures.tiny_gemma_like import (
    tiny_activation_stats,
    tiny_bundle,
    tiny_expert_stats,
    tiny_topology,
)


def test_frequency_strategy_is_repeatable_for_shared_fixture_payloads() -> None:
    topology = tiny_topology()
    activation_stats = tiny_activation_stats()

    strategy = create_strategy("frequency")
    first = strategy.build_candidates(topology, activation_stats=activation_stats)
    second = strategy.build_candidates(topology, activation_stats=activation_stats)

    assert first == second
    assert [(candidate.layer_index, candidate.expert_index) for candidate in first[:4]] == [
        (0, 0),
        (1, 1),
        (0, 1),
        (1, 0),
    ]


def test_combined_strategy_and_plan_json_are_deterministic_with_fixed_fixture_seed() -> None:
    topology = tiny_topology()
    expert_stats = tiny_expert_stats()
    activation_stats = tiny_activation_stats()
    model_handle = tiny_bundle().model_handle
    constraints = PlannerConstraints(global_target_experts=3, min_experts_per_layer=1)

    first = build_prune_plan(
        topology,
        strategy="combined",
        expert_stats=expert_stats,
        activation_stats=activation_stats,
        constraints=constraints,
        model_handle=model_handle,
        source_run_id="bench-tiny",
    )
    second = build_prune_plan(
        topology,
        strategy="combined",
        expert_stats=expert_stats,
        activation_stats=activation_stats,
        constraints=constraints,
        model_handle=model_handle,
        source_run_id="bench-tiny",
    )

    assert to_json(first) == to_json(second)
    assert first.model_handle is not None
    assert first.model_handle.seed == 7
    assert first.global_target_experts == 3
    assert [item.keep_indices for item in first.per_layer_plans] == [(0, 1), (1,)]


def test_router_mass_strategy_is_deterministic_for_reordered_inputs() -> None:
    strategy = create_strategy("router_mass")
    topology = tuple(reversed(tiny_topology()))
    expert_stats = tuple(reversed(tiny_expert_stats()))

    first = strategy.build_candidates(topology, expert_stats=expert_stats)
    second = strategy.build_candidates(topology, expert_stats=expert_stats)

    assert first == second
    assert [(candidate.layer_index, candidate.expert_index) for candidate in first[:4]] == [
        (0, 0),
        (1, 1),
        (1, 0),
        (1, 2),
    ]
    assert first[0].score_components["static_gate_share"] == pytest.approx(0.60)
    assert first[1].score_components["static_gate_share"] == pytest.approx(0.40)


def test_validate_planner_inputs_rejects_layer_override_outside_topology() -> None:
    with pytest.raises(TopologyMismatchError, match="unknown topology layer"):
        build_prune_plan(
            tiny_topology(),
            strategy="router_mass",
            expert_stats=tiny_expert_stats(),
            constraints=PlannerConstraints(
                global_target_experts=4,
                min_experts_per_layer=1,
                layer_overrides={99: LayerConstraintOverride(target_experts=1)},
            ),
        )


def test_validate_planner_inputs_accepts_shared_fixture_coverage() -> None:
    validate_planner_inputs(
        tiny_topology(),
        expert_stats=tiny_expert_stats(),
        activation_stats=tiny_activation_stats(),
    )


def test_planner_constraints_and_plan_output_are_stable_across_override_mapping_order() -> None:
    topology = tiny_topology()
    constraints_left = PlannerConstraints(
        global_target_experts=4,
        min_experts_per_layer=1,
        layer_overrides={
            1: LayerConstraintOverride(target_experts=1),
            0: LayerConstraintOverride(max_experts=3),
        },
    )
    constraints_right = PlannerConstraints(
        global_target_experts=4,
        min_experts_per_layer=1,
        layer_overrides={
            0: LayerConstraintOverride(max_experts=3),
            1: LayerConstraintOverride(target_experts=1),
        },
    )

    assert constraints_left.canonical_payload() == constraints_right.canonical_payload()

    left = build_prune_plan(
        topology,
        strategy="combined",
        expert_stats=tiny_expert_stats(),
        activation_stats=tiny_activation_stats(),
        constraints=constraints_left,
        model_handle=tiny_bundle().model_handle,
        source_run_id="bench-tiny",
    )
    right = build_prune_plan(
        topology,
        strategy="combined",
        expert_stats=tiny_expert_stats(),
        activation_stats=tiny_activation_stats(),
        constraints=constraints_right,
        model_handle=tiny_bundle().model_handle,
        source_run_id="bench-tiny",
    )

    assert to_json(left) == to_json(right)
    assert left.constraints == right.constraints


def test_planner_constraints_resolve_exact_layer_budgets_deterministically() -> None:
    topology = tiny_topology()
    constraints = PlannerConstraints(
        global_target_experts=4,
        min_experts_per_layer=1,
        max_experts_per_layer=3,
        layer_overrides={
            1: LayerConstraintOverride(target_experts=1),
            0: LayerConstraintOverride(min_experts=2),
        },
    )

    budgets = _layer_budgets(topology, constraints)
    flattened = constraints.plan_constraints(budgets, global_target_experts=4)

    assert [(budget.layer.layer_index, budget.minimum_keep, budget.maximum_keep) for budget in budgets] == [
        (0, 2, 3),
        (1, 1, 1),
    ]
    assert flattened == {
        "min_experts_per_layer": 1,
        "global_target_experts": 4,
        "max_experts_per_layer": 3,
        "layer_0_min_experts": 2,
        "layer_1_target_experts": 1,
    }


def test_build_prune_plan_rejects_global_target_below_resolved_minimum() -> None:
    with pytest.raises(ValueError, match="below required minimum 4"):
        build_prune_plan(
            tiny_topology(),
            strategy="combined",
            expert_stats=tiny_expert_stats(),
            activation_stats=tiny_activation_stats(),
            constraints=PlannerConstraints(
                global_target_experts=3,
                min_experts_per_layer=1,
                layer_overrides={
                    0: LayerConstraintOverride(min_experts=2),
                    1: LayerConstraintOverride(min_experts=2),
                },
            ),
        )
