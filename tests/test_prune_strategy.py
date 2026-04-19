from __future__ import annotations

import pytest

from moe_surgeon.models.errors import TopologyMismatchError
from moe_surgeon.prune.planner import (
    LayerConstraintOverride,
    PlannerConstraints,
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
