from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.prune import apply_prune_plan
from moe_surgeon.schemas import ModelHandle, PrunePlan, PrunePlanItem


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
    assert first.layer_reports[0].keep_indices == (1, 3)
    assert first.layer_reports[0].drop_indices == (0, 2)
    assert first.layer_reports[0].old_to_new_index == ((1, 0), (3, 1))
    assert first.rewritten_tensor_keys == (
        "model.language_model.layers.0.experts.down_proj",
        "model.language_model.layers.0.experts.gate_up_proj",
        "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.router.proj.weight",
    )
    assert "model.language_model.layers.0.router.scale" in first.passthrough_tensor_keys
    assert "model.embed_tokens.weight" in first.passthrough_tensor_keys


def test_apply_prune_plan_materializes_remapped_tensors_and_preserves_passthrough(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path)
    plan = _plan()

    result = apply_prune_plan(tmp_path, plan=plan, dry_run=False)

    assert result.derived_state_dict is not None
    derived = result.derived_state_dict
    keep = torch.tensor([1, 3], dtype=torch.long)

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


def test_apply_prune_plan_rejects_unknown_plan_layer(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    with pytest.raises(TopologyMismatchError, match="unknown MoE layer coverage") as exc_info:
        apply_prune_plan(tmp_path, plan=_plan(layer_index=1), dry_run=True)

    assert "unknown_layer_indices=1" in str(exc_info.value)


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
