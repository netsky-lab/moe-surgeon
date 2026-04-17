import pytest

from moe_surgeon.schemas import (
    LayerTopology,
    PrunePlan,
    PrunePlanItem,
    PruneCandidate,
    ExpertStats,
    ModelHandle,
    RouterState,
    ActivationStats,
    RunArtifactManifest,
    TopologyMismatchError,
    ShapeInvariantViolationError,
    LayerReferenceError,
    to_json,
    from_json,
    sort_experts,
    validate_shape_tuple,
    validate_layer_ref,
)


def test_core_types_exist() -> None:
    assert callable(PrunePlan)
    assert callable(ExpertStats)
    assert callable(LayerTopology)


def test_sort_experts_is_deterministic_with_tie_fallback() -> None:
    ordered = sort_experts([(0, 0.2, 0.5, 1), (1, 0.2, 0.5, 1), (0, 0.2, 0.5, 0)])
    assert ordered == [(0, 0.2, 0.5, 0), (0, 0.2, 0.5, 1), (1, 0.2, 0.5, 1)]


def test_sort_experts_uses_explicit_tiebreaks() -> None:
    ordered = sort_experts([(0, 0.1, 0.2, 3), (1, 0.1, 0.2, 1), (0, 0.1, 0.2, 1)])
    assert ordered == [(0, 0.1, 0.2, 1), (0, 0.1, 0.2, 3), (1, 0.1, 0.2, 1)]


def test_sort_experts_stable_for_equal_records() -> None:
    first = (0, 0.2, 0.4, 1)
    second = (0, 0.2, 0.4, 1)
    ordered = sort_experts([second, first])
    assert ordered == [second, first]


def test_json_round_trip_preserves_plan() -> None:
    plan = PrunePlan(
        model_signature="m",
        per_layer_plans=(
            PrunePlanItem(
                layer_index=0,
                keep_indices=(0, 1),
                drop_indices=(2,),
                source_expert_count=3,
            ),
        ),
    )
    payload = to_json(plan)
    restored = from_json(payload)
    assert restored == plan
    assert type(restored).__name__ == "PrunePlan"


def test_json_round_trip_preserves_manifest_with_nested_plan() -> None:
    plan = PrunePlan(
        model_signature="m",
        per_layer_plans=(
            PrunePlanItem(
                layer_index=1,
                keep_indices=(0,),
                drop_indices=(),
                source_expert_count=1,
            ),
        ),
    )
    manifest = RunArtifactManifest(
        run_id="run-123",
        command="scan",
        top_k=8,
        prompt_count=3,
        run_plan=plan,
    )
    payload = to_json(manifest)
    restored = from_json(payload)
    assert restored == manifest
    assert restored.versioned_manifest_id and isinstance(restored.versioned_manifest_id, str)


def test_shape_validation_router_state() -> None:
    with pytest.raises(ShapeInvariantViolationError):
        RouterState(
            layer_index=0,
            num_experts=4,
            top_k=2,
            logits_shape=(1, 2),
            top_k_indices_shape=(1,),
            top_k_weights_shape=(1, 2),
        )


def test_topology_validator() -> None:
    with pytest.raises(TopologyMismatchError):
        LayerTopology(
            layer_index=0,
            layer_name="x",
            layer_type="moe",
            expert_count=4,
            top_k=8,
            hidden_size=64,
        )


def test_shape_tuple_helper_accepts_lists_and_raises_on_invalid() -> None:
    assert validate_shape_tuple([4, 8], name="shape") == (4, 8)
    with pytest.raises(ShapeInvariantViolationError):
        validate_shape_tuple([-1, 8], name="shape")
    with pytest.raises(ShapeInvariantViolationError):
        validate_shape_tuple([4, "8"], name="shape")  # type: ignore[list-item]


def test_model_and_layer_ref_validation() -> None:
    assert validate_layer_ref("layer_0032") == 32
    with pytest.raises(LayerReferenceError):
        LayerTopology(
            layer_index=0,
            layer_name="x",
            layer_type="moe",
            expert_count=4,
            top_k=2,
            hidden_size=64,
            layer_ref="bad_ref",
        )


def test_model_and_activation_objects_validate_bounds() -> None:
    handle = ModelHandle(model_id="test", seed=3, metadata={"k": "v"})
    assert handle.model_fingerprint

    stats = ActivationStats(
        layer_index=0,
        expert_index=0,
        token_count=10,
        weighted_token_count=3.2,
        mass_sum=3.2,
        mean_weight=0.32,
        entropy=0.1,
        n_tokens=30,
    )
    assert stats.occupancy == pytest.approx(1 / 3)

    candidate = PruneCandidate(
        layer_index=0,
        expert_index=1,
        score=0.3,
        secondary_score=0.5,
        strategy_name="combined",
    )
    sorted_candidates = sort_experts([candidate, candidate])
    assert len(sorted_candidates) == 2
