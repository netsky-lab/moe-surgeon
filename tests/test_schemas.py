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
    SchemaValidationError,
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
        model_handle=ModelHandle(model_id="abc"),
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


def test_json_round_trip_preserves_plan_model_handle() -> None:
    plan = PrunePlan(
        model_signature="m",
        model_handle=ModelHandle(model_id="abc"),
        per_layer_plans=(
            PrunePlanItem(
                layer_index=0,
                keep_indices=(0,),
                drop_indices=(1,),
                source_expert_count=2,
            ),
        ),
    )

    restored = from_json(to_json(plan))

    assert restored == plan
    assert isinstance(restored.model_handle, ModelHandle)


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


def test_deterministic_default_json_for_default_plan_instances() -> None:
    first = PrunePlan()
    second = PrunePlan()
    assert to_json(first) == to_json(second)


def test_deterministic_default_json_for_default_manifest_instances() -> None:
    first = RunArtifactManifest(run_id="run-001", command="scan")
    second = RunArtifactManifest(run_id="run-001", command="scan")
    assert to_json(first) == to_json(second)


def test_manifest_canonical_digest_ignores_runtime_timestamps() -> None:
    first = RunArtifactManifest(
        run_id="run-001",
        command="scan",
        started_at="2026-04-17T12:00:00+00:00",
        finished_at="2026-04-17T12:05:00+00:00",
        metadata={"artifact_kind": "scan"},
    )
    second = RunArtifactManifest(
        run_id="run-001",
        command="scan",
        started_at="2026-04-18T12:00:00+00:00",
        finished_at="2026-04-18T12:05:00+00:00",
        metadata={"artifact_kind": "scan"},
    )

    assert first.canonical_digest == second.canonical_digest
    assert first.versioned_manifest_id != second.versioned_manifest_id


def test_from_dict_coerces_list_payloads_for_plan_components() -> None:
    plan = PrunePlan.from_dict(
        {
            "plan_id": "p",
            "model_signature": "test",
            "per_layer_plans": [
                {
                    "layer_index": 0,
                    "keep_indices": [0, 1],
                    "drop_indices": [],
                    "source_expert_count": 2,
                }
            ],
        }
    )
    assert isinstance(plan.per_layer_plans, tuple)
    assert plan.per_layer_plans[0].keep_indices == (0, 1)


def test_from_dict_coerces_prune_plan_model_handle_payload() -> None:
    plan = PrunePlan.from_dict(
        {
            "model_signature": "m",
            "model_handle": {"model_id": "abc"},
            "per_layer_plans": [
                {
                    "layer_index": 0,
                    "keep_indices": [0],
                    "drop_indices": [1],
                    "source_expert_count": 2,
                }
            ],
        }
    )
    assert isinstance(plan.model_handle, ModelHandle)
    assert plan.model_handle.model_id == "abc"


def test_from_json_rejects_unknown_schema_type() -> None:
    with pytest.raises(SchemaValidationError, match="Unsupported __schema_type"):
        from_json('{"__schema_type":"UnknownSchema","value":1}')


def test_from_json_rejects_mapping_without_schema_type() -> None:
    with pytest.raises(SchemaValidationError, match="Unsupported mapping payload"):
        from_json({"plan_id": "plan-001"})


def test_prune_plan_from_dict_rejects_non_mapping_plan_items() -> None:
    with pytest.raises(SchemaValidationError, match="per_layer_plans entries must be mappings"):
        PrunePlan.from_dict(
            {
                "plan_id": "plan-001",
                "model_signature": "sig",
                "per_layer_plans": ["bad-item"],
            }
        )


def test_prune_plan_from_dict_rejects_invalid_tuple_payloads() -> None:
    with pytest.raises(SchemaValidationError, match="keep_indices contains invalid entry"):
        PrunePlan.from_dict(
            {
                "plan_id": "plan-001",
                "model_signature": "sig",
                "per_layer_plans": [
                    {
                        "layer_index": 0,
                        "keep_indices": [0, "1"],
                        "drop_indices": [],
                    }
                ],
            }
        )


def test_manifest_from_dict_rejects_invalid_run_plan_payload() -> None:
    with pytest.raises(SchemaValidationError, match="per_layer_plans entries must be mappings"):
        RunArtifactManifest.from_dict(
            {
                "run_id": "run-001",
                "command": "scan",
                "run_plan": {
                    "plan_id": "plan-001",
                    "per_layer_plans": ["bad-item"],
                },
            }
        )


def test_manifest_from_dict_rejects_invalid_model_handle_payload() -> None:
    with pytest.raises(SchemaValidationError, match="Missing required field 'model_id' for ModelHandle"):
        RunArtifactManifest.from_dict(
            {
                "run_id": "run-001",
                "command": "scan",
                "model_handle": {"seed": 7},
            }
        )


def test_prune_plan_item_rejects_out_of_range_expert_indices() -> None:
    with pytest.raises(
        TopologyMismatchError,
        match=r"keep_indices and drop_indices must cover contiguous expert indices 0\.\.1",
    ):
        PrunePlanItem(
            layer_index=0,
            keep_indices=(0, 99),
            drop_indices=(),
            source_expert_count=2,
        )


def test_prune_plan_item_rejects_non_covering_expert_indices() -> None:
    with pytest.raises(
        TopologyMismatchError,
        match=r"keep_indices and drop_indices must cover contiguous expert indices 0\.\.1",
    ):
        PrunePlanItem(
            layer_index=0,
            keep_indices=(0, 2),
            drop_indices=(),
        )
