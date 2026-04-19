from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
import torch

from moe_surgeon.models.checkpoints import open_local_safetensors_checkpoint
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.prune import apply_prune_plan
from moe_surgeon.prune.planner import PlannerConstraints, build_prune_plan
from moe_surgeon.prune.strategies import StrategyMetadata
from moe_surgeon.schemas import LayerTopology, ModelHandle, PruneCandidate, PrunePlan, PrunePlanItem


def _gemma4_config() -> dict[str, object]:
    return {
        "_name_or_path": "google/gemma-4-27b",
        "_commit_hash": "rev-123",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 3,
            "enable_moe_block": True,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 2,
            "moe_layer_indices": [0],
        },
    }


def _write_config(root: Path) -> None:
    (root / "config.json").write_text(json.dumps(_gemma4_config()), encoding="utf-8")


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "model.language_model.layers.0.router.scale": torch.tensor(1.25, dtype=torch.float32),
        "model.language_model.layers.0.router.per_expert_scale": torch.tensor(
            [0.2, 0.4, 0.1, 0.3],
            dtype=torch.float32,
        ),
        "model.language_model.layers.0.router.proj.weight": torch.arange(
            12, dtype=torch.float32
        ).reshape(4, 3),
        "model.language_model.layers.0.experts.gate_up_proj": torch.arange(
            4 * 4 * 3, dtype=torch.float32
        ).reshape(4, 4, 3),
        "model.language_model.layers.0.experts.down_proj": torch.arange(
            4 * 3 * 2, dtype=torch.float32
        ).reshape(4, 3, 2),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(
            18, dtype=torch.float32
        ).reshape(3, 6),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(
            18, dtype=torch.float32
        ).reshape(6, 3),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(
            18, dtype=torch.float32
        ).reshape(6, 3),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.ones(
            (3,), dtype=torch.float32
        ),
        "model.language_model.layers.0.pre_feedforward_layernorm_2.weight": torch.full(
            (3,), 2.0, dtype=torch.float32
        ),
        "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.full(
            (3,), 3.0, dtype=torch.float32
        ),
        "model.language_model.layers.0.post_feedforward_layernorm_1.weight": torch.full(
            (3,), 4.0, dtype=torch.float32
        ),
        "model.language_model.layers.0.post_feedforward_layernorm_2.weight": torch.full(
            (3,), 5.0, dtype=torch.float32
        ),
    }


def _write_checkpoint(root: Path, *, state_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    tensors = _state_dict() if state_dict is None else state_dict
    _write_config(root)
    save_file(tensors, str(root / "model.safetensors"))
    return tensors


def _write_sharded_checkpoint(root: Path, *, state_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    tensors = _state_dict() if state_dict is None else state_dict
    _write_config(root)
    first_shard_keys = tuple(sorted(key for key in tensors if "router" in key or "embed_tokens" in key))
    second_shard_keys = tuple(sorted(key for key in tensors if key not in first_shard_keys))
    first_shard = {key: tensors[key] for key in first_shard_keys}
    second_shard = {key: tensors[key] for key in second_shard_keys}
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    save_file(first_shard, str(root / first_name))
    save_file(second_shard, str(root / second_name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    **{key: first_name for key in first_shard_keys},
                    **{key: second_name for key in second_shard_keys},
                },
            }
        ),
        encoding="utf-8",
    )
    return tensors


def _plan(*, layer_index: int = 0) -> PrunePlan:
    return PrunePlan(
        plan_id="plan-prune-0001",
        model_signature="google/gemma-4-27b:rev-123",
        strategy_name="frequency",
        strategy_version="1",
        per_layer_plans=(
            PrunePlanItem(
                layer_index=layer_index,
                keep_indices=(3, 1),
                drop_indices=(0, 2),
                source_expert_count=4,
                target_expert_count=2,
                expected_expert_count=4,
            ),
        ),
        model_handle=ModelHandle(model_id="google/gemma-4-27b", revision="rev-123", backend_name="gemma4"),
    )


def _identity_plan(*, layer_index: int = 0) -> PrunePlan:
    return PrunePlan(
        plan_id="plan-prune-identity",
        model_signature="google/gemma-4-27b:rev-123",
        strategy_name="frequency",
        strategy_version="1",
        per_layer_plans=(
            PrunePlanItem(
                layer_index=layer_index,
                keep_indices=(0, 1, 2, 3),
                drop_indices=(),
                source_expert_count=4,
                target_expert_count=4,
                expected_expert_count=4,
            ),
        ),
        model_handle=ModelHandle(model_id="google/gemma-4-27b", revision="rev-123", backend_name="gemma4"),
    )


def _planner_topology() -> tuple[LayerTopology, ...]:
    return (
        LayerTopology(
            layer_index=0,
            layer_name="model.language_model.layers.0",
            layer_type="moe",
            expert_count=4,
            top_k=2,
            hidden_size=3,
            moe_intermediate_size=2,
        ),
    )


def test_public_apply_module_path_exists() -> None:
    assert importlib.util.find_spec("moe_surgeon.prune.apply") is not None

    module = importlib.import_module("moe_surgeon.prune.apply")

    assert module.apply_prune_plan is apply_prune_plan


def test_apply_prune_plan_dry_run_is_deterministic_and_reports_remap(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()

    first = apply_prune_plan(tmp_path, plan=plan, dry_run=True)
    second = apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert first.apply_id == second.apply_id
    assert first.manifest_json() == second.manifest_json()
    assert first.audit_json() == second.audit_json()
    assert first.derived_state_dict is None
    assert first.layer_reports[0].keep_indices == (3, 1)
    assert first.layer_reports[0].drop_indices == (0, 2)
    assert first.layer_reports[0].old_to_new_index == ((3, 0), (1, 1))
    assert first.rewritten_tensor_keys == (
        "model.language_model.layers.0.experts.down_proj",
        "model.language_model.layers.0.experts.gate_up_proj",
        "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.router.proj.weight",
    )
    assert first.rewritten_tensor_mapping == {
        "model.language_model.layers.0.experts.down_proj": "model.language_model.layers.0.experts.down_proj",
        "model.language_model.layers.0.experts.gate_up_proj": "model.language_model.layers.0.experts.gate_up_proj",
        "model.language_model.layers.0.router.per_expert_scale": "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.router.proj.weight": "model.language_model.layers.0.router.proj.weight",
    }
    assert "model.language_model.layers.0.router.scale" in first.passthrough_tensor_keys
    assert "model.embed_tokens.weight" in first.passthrough_tensor_keys
    assert first.passthrough_tensor_mapping["model.language_model.layers.0.router.scale"] == (
        "model.language_model.layers.0.router.scale"
    )
    assert first.passthrough_tensor_mapping["model.embed_tokens.weight"] == "model.embed_tokens.weight"


def test_apply_prune_plan_accepts_planner_produced_default_model_signature(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    class PlannerStrategy:
        metadata = StrategyMetadata(
            name="planner-test",
            version="1",
            score_columns=("score",),
            normalization_behavior="none",
        )

        def build_candidates(
            self,
            topology: tuple[LayerTopology, ...],
            *,
            expert_stats: tuple[object, ...] | None = None,
            activation_stats: tuple[object, ...] | None = None,
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
        _planner_topology(),
        strategy=PlannerStrategy(),
        constraints=PlannerConstraints(global_target_experts=2, min_experts_per_layer=2),
        model_handle=ModelHandle(
            model_id="google/gemma-4-27b",
            revision="rev-123",
            backend_name="gemma4",
        ),
    )

    result = apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert plan.model_signature == "google/gemma-4-27b:rev-123"
    assert result.layer_reports[0].keep_indices == (0, 1)


def test_apply_prune_plan_reports_vector_router_scale_shape_in_audit(tmp_path: Path) -> None:
    state_dict = _state_dict()
    state_dict["model.language_model.layers.0.router.scale"] = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
    )
    _write_checkpoint(tmp_path, state_dict=state_dict)

    result = apply_prune_plan(tmp_path, plan=_plan(), dry_run=True)

    router_scale_delta = next(
        delta
        for delta in result.layer_reports[0].tensor_deltas
        if delta.tensor_role == "router_scale"
    )
    assert router_scale_delta.source_shape == (3,)
    assert router_scale_delta.target_shape == (3,)
    assert router_scale_delta.rewritten is False


def test_apply_prune_plan_materializes_remapped_tensors_and_preserves_passthrough(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path)
    plan = _plan()
    output_dir = tmp_path / "derived-checkpoint"

    result = apply_prune_plan(tmp_path, plan=plan, dry_run=False, output_dir=output_dir)

    assert result.derived_state_dict is not None
    assert result.output_checkpoint_dir == str(output_dir.resolve())
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "model.safetensors").is_file()
    assert (output_dir / "apply-manifest.json").is_file()
    assert (output_dir / "apply-audit.json").is_file()
    assert (output_dir / "run-manifest.json").is_file()
    assert (output_dir / "SHA256SUMS").is_file()
    derived = result.derived_state_dict
    written = load_file(str(output_dir / "model.safetensors"))
    keep = torch.tensor([3, 1], dtype=torch.long)

    assert torch.equal(
        derived["model.language_model.layers.0.router.proj.weight"],
        torch.index_select(source["model.language_model.layers.0.router.proj.weight"], 0, keep),
    )
    assert torch.equal(
        derived["model.language_model.layers.0.router.per_expert_scale"],
        torch.index_select(source["model.language_model.layers.0.router.per_expert_scale"], 0, keep),
    )
    assert torch.equal(
        derived["model.language_model.layers.0.experts.gate_up_proj"],
        torch.index_select(source["model.language_model.layers.0.experts.gate_up_proj"], 0, keep),
    )
    assert torch.equal(
        derived["model.language_model.layers.0.experts.down_proj"],
        torch.index_select(source["model.language_model.layers.0.experts.down_proj"], 0, keep),
    )
    assert torch.equal(
        derived["model.language_model.layers.0.router.scale"],
        source["model.language_model.layers.0.router.scale"],
    )
    assert torch.equal(
        derived["model.language_model.layers.0.mlp.down_proj.weight"],
        source["model.language_model.layers.0.mlp.down_proj.weight"],
    )
    assert torch.equal(
        derived["model.embed_tokens.weight"],
        source["model.embed_tokens.weight"],
    )
    assert torch.equal(
        written["model.language_model.layers.0.router.proj.weight"],
        torch.index_select(source["model.language_model.layers.0.router.proj.weight"], 0, keep),
    )
    assert torch.equal(
        written["model.embed_tokens.weight"],
        source["model.embed_tokens.weight"],
    )
    reloaded_source = load_file(str(tmp_path / "model.safetensors"))
    assert torch.equal(
        reloaded_source["model.language_model.layers.0.router.proj.weight"],
        source["model.language_model.layers.0.router.proj.weight"],
    )

    reopened = open_local_safetensors_checkpoint(output_dir)

    assert reopened.model_id == "google/gemma-4-27b"
    assert reopened.revision == "rev-123"
    assert reopened.state_keys() == tuple(sorted(derived))
    reopened_metadata = {item.tensor_key: item for item in reopened.tensor_metadata()}
    assert reopened_metadata["model.language_model.layers.0.router.proj.weight"].shape == (2, 3)
    assert reopened_metadata["model.language_model.layers.0.router.per_expert_scale"].shape == (2,)
    assert reopened_metadata["model.language_model.layers.0.experts.gate_up_proj"].shape == (2, 4, 3)
    assert reopened_metadata["model.language_model.layers.0.experts.down_proj"].shape == (2, 3, 2)
    assert reopened_metadata["model.language_model.layers.0.router.scale"].shape == ()
    assert reopened_metadata["model.embed_tokens.weight"].shape == (4, 3)


def test_apply_prune_plan_identity_plan_preserves_rewritten_and_passthrough_tensors(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path)
    output_dir = tmp_path / "derived-checkpoint"

    result = apply_prune_plan(tmp_path, plan=_identity_plan(), dry_run=False, output_dir=output_dir)

    assert result.derived_state_dict is not None
    assert result.layer_reports[0].keep_indices == (0, 1, 2, 3)
    assert result.layer_reports[0].drop_indices == ()
    assert result.layer_reports[0].old_to_new_index == ((0, 0), (1, 1), (2, 2), (3, 3))

    written = load_file(str(output_dir / "model.safetensors"))
    identity_keys = (
        "model.language_model.layers.0.router.proj.weight",
        "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.experts.gate_up_proj",
        "model.language_model.layers.0.experts.down_proj",
        "model.language_model.layers.0.router.scale",
        "model.embed_tokens.weight",
    )
    for tensor_key in identity_keys:
        assert torch.equal(written[tensor_key], source[tensor_key])
        assert torch.equal(result.derived_state_dict[tensor_key], source[tensor_key])

    reopened_source = open_local_safetensors_checkpoint(tmp_path)
    reopened_output = open_local_safetensors_checkpoint(output_dir)

    assert reopened_output.state_keys() == reopened_source.state_keys()
    assert reopened_output.weight_map == reopened_source.weight_map
    assert tuple(
        (item.tensor_key, item.shape, item.dtype, item.shard_filename)
        for item in reopened_output.tensor_metadata()
    ) == tuple(
        (item.tensor_key, item.shape, item.dtype, item.shard_filename)
        for item in reopened_source.tensor_metadata()
    )


def test_apply_prune_plan_materializes_from_sharded_checkpoint(tmp_path: Path) -> None:
    source = _write_sharded_checkpoint(tmp_path)
    output_dir = tmp_path / "derived-checkpoint"

    result = apply_prune_plan(tmp_path, plan=_plan(), dry_run=False, output_dir=output_dir)

    assert result.output_checkpoint_dir == str(output_dir.resolve())
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "model.safetensors.index.json").is_file()
    assert (output_dir / "model-00001-of-00002.safetensors").is_file()
    assert (output_dir / "model-00002-of-00002.safetensors").is_file()
    reopened = open_local_safetensors_checkpoint(output_dir)
    written = reopened.load_tensors(
        (
            "model.embed_tokens.weight",
            "model.language_model.layers.0.router.per_expert_scale",
            "model.language_model.layers.0.experts.down_proj",
        )
    )
    keep = torch.tensor([3, 1], dtype=torch.long)

    assert torch.equal(
        written["model.language_model.layers.0.router.per_expert_scale"],
        torch.index_select(source["model.language_model.layers.0.router.per_expert_scale"], 0, keep),
    )
    assert torch.equal(
        written["model.language_model.layers.0.experts.down_proj"],
        torch.index_select(source["model.language_model.layers.0.experts.down_proj"], 0, keep),
    )
    assert torch.equal(
        written["model.embed_tokens.weight"],
        source["model.embed_tokens.weight"],
    )


def test_apply_prune_plan_writes_stable_plan_identity_to_sidecars(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    output_dir = tmp_path / "derived-checkpoint"

    result = apply_prune_plan(tmp_path, plan=_plan(), dry_run=False, output_dir=output_dir)

    manifest_payload = json.loads((output_dir / "apply-manifest.json").read_text(encoding="utf-8"))
    audit_payload = json.loads((output_dir / "apply-audit.json").read_text(encoding="utf-8"))

    assert manifest_payload["metadata"]["plan_versioned_manifest_id"] == result.metadata["plan_versioned_manifest_id"]
    assert manifest_payload["metadata"]["plan_canonical_digest"] == result.metadata["plan_canonical_digest"]
    assert audit_payload["metadata"]["plan_versioned_manifest_id"] == result.metadata["plan_versioned_manifest_id"]
    assert "output_checkpoint_dir" not in manifest_payload


def test_apply_prune_plan_does_not_write_output_tree_when_post_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checkpoint(tmp_path)
    output_dir = tmp_path / "derived-checkpoint"
    original_validate = Gemma4Backend.validate_prune_tensor

    def fail_after_remap_validation(
        self: Gemma4Backend,
        bundle: object,
        *,
        layer: object,
        tensor_role: str,
        tensor_key: str,
        tensor_value: object,
        target_expert_count: int,
    ) -> None:
        if target_expert_count == 2 and tensor_role == "router_proj":
            raise ShapeInvariantViolationError(
                "forced post-remap validation failure",
                model_id="google/gemma-4-27b",
            )
        original_validate(
            self,
            bundle,
            layer=layer,
            tensor_role=tensor_role,
            tensor_key=tensor_key,
            tensor_value=tensor_value,
            target_expert_count=target_expert_count,
        )

    monkeypatch.setattr(Gemma4Backend, "validate_prune_tensor", fail_after_remap_validation)

    with pytest.raises(ShapeInvariantViolationError, match="forced post-remap validation failure"):
        apply_prune_plan(tmp_path, plan=_plan(), dry_run=False, output_dir=output_dir)

    assert not output_dir.exists()


def test_apply_prune_plan_rejects_plan_checkpoint_identity_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    plan.model_signature = "some-other-model:wrong-revision"
    plan.model_handle = ModelHandle(
        model_id="some-other-model",
        revision="wrong-revision",
        backend_name="gemma4",
    )

    with pytest.raises(TopologyMismatchError, match="model signature does not match checkpoint") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "plan_model_signature=some-other-model:wrong-revision" in str(exc_info.value)


def test_apply_prune_plan_rejects_unknown_plan_layer(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    with pytest.raises(TopologyMismatchError, match="unknown MoE layer coverage") as exc_info:
        apply_prune_plan(tmp_path, plan=_plan(layer_index=1), dry_run=True)

    assert "unknown_layer_indices=1" in str(exc_info.value)


def test_apply_prune_plan_rejects_missing_plan_layer_coverage(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    plan.per_layer_plans = ()

    with pytest.raises(TopologyMismatchError, match="missing MoE layer coverage") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "missing_layer_indices=0" in str(exc_info.value)


def test_apply_prune_plan_rejects_duplicate_plan_layer_coverage(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    plan.per_layer_plans = (
        *plan.per_layer_plans,
        PrunePlanItem(
            layer_index=0,
            keep_indices=(0, 1),
            drop_indices=(2, 3),
            source_expert_count=4,
            target_expert_count=2,
            expected_expert_count=4,
        ),
    )

    with pytest.raises(TopologyMismatchError, match="duplicate MoE layer coverage") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "duplicate_layer_indices=0" in str(exc_info.value)


def test_apply_prune_plan_rejects_plan_expert_count_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    item = plan.per_layer_plans[0]
    item.drop_indices = (0,)

    with pytest.raises(TopologyMismatchError, match="expert count does not match layer topology") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "plan_expert_count=3" in str(exc_info.value)


def test_apply_prune_plan_rechecks_mutated_source_and_expected_expert_counts(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    item = plan.per_layer_plans[0]
    item.source_expert_count = 999
    item.expected_expert_count = 999

    with pytest.raises(TopologyMismatchError, match="source_expert_count does not match layer topology") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "plan_source_expert_count=999" in str(exc_info.value)


def test_apply_prune_plan_rejects_out_of_range_keep_indices_with_domain_error(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    item = plan.per_layer_plans[0]
    item.keep_indices = (3, 99)
    item.drop_indices = (0, 1)

    with pytest.raises(TopologyMismatchError, match="expert indices must cover contiguous layer expert indices") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "plan_indices=0,1,3,99" in str(exc_info.value)


def test_apply_prune_plan_rejects_target_expert_count_below_top_k(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    plan = _plan()
    item = plan.per_layer_plans[0]
    item.keep_indices = (3,)
    item.drop_indices = (0, 1, 2)

    with pytest.raises(TopologyMismatchError, match="cannot be below layer top_k") as exc_info:
        apply_prune_plan(tmp_path, plan=plan, dry_run=True)

    assert "target_expert_count=1" in str(exc_info.value)
    assert "layer_top_k=2" in str(exc_info.value)


def test_apply_prune_plan_rejects_missing_required_prune_tensor(tmp_path: Path) -> None:
    state_dict = _state_dict()
    del state_dict["model.language_model.layers.0.experts.down_proj"]
    _write_checkpoint(tmp_path, state_dict=state_dict)

    with pytest.raises(TopologyMismatchError, match="missing Gemma4 hybrid layer tensor keys") as exc_info:
        apply_prune_plan(tmp_path, plan=_plan(), dry_run=True)

    assert "experts.down_proj" in str(exc_info.value)


def test_apply_prune_plan_rejects_invalid_source_expert_shape(tmp_path: Path) -> None:
    state_dict = _state_dict()
    state_dict["model.language_model.layers.0.experts.down_proj"] = torch.arange(
        4 * 3 * 3, dtype=torch.float32
    ).reshape(4, 3, 3)
    _write_checkpoint(tmp_path, state_dict=state_dict)

    with pytest.raises(ShapeInvariantViolationError, match="experts.down_proj shape mismatch") as exc_info:
        apply_prune_plan(tmp_path, plan=_plan(), dry_run=True)

    assert "expected_shape=4x3x2" in str(exc_info.value)
    assert "actual_shape=4x3x3" in str(exc_info.value)
