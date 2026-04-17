from __future__ import annotations

import math

import pytest
import torch

from moe_surgeon.analysis.metrics import build_expert_stats, static_expert_distribution
from moe_surgeon.analysis.scan import scan_model
from moe_surgeon.models.backend import LoadedBackendBundle
from moe_surgeon.models.errors import TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.schemas import ModelHandle


def _gemma4_config(*, moe_layer_indices: list[int]) -> dict[str, object]:
    return {
        "_name_or_path": "google/gemma-4-27b",
        "_commit_hash": "rev-123",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "num_hidden_layers": 4,
            "hidden_size": 3,
            "enable_moe_block": True,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 2,
            "moe_layer_indices": moe_layer_indices,
        },
    }


def _layer_state(layer_index: int, *, shift: float) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer_index}"
    router_proj = torch.tensor(
        [
            [3.0 + shift, 0.0, -2.0],
            [1.0, 2.0 + shift, 0.0],
            [-2.0, 1.0, 3.0 + shift],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    per_expert_scale = torch.tensor([0.2, 0.4 + shift, 0.1, 0.3], dtype=torch.float32)
    return {
        f"{prefix}.router.proj.weight": router_proj,
        f"{prefix}.router.scale": torch.tensor(1.0 + shift, dtype=torch.float32),
        f"{prefix}.router.per_expert_scale": per_expert_scale,
        f"{prefix}.experts.gate_up_proj": torch.arange(48, dtype=torch.float32).reshape(4, 4, 3),
        f"{prefix}.experts.down_proj": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        f"{prefix}.mlp.down_proj.weight": torch.ones((3, 6), dtype=torch.float32),
        f"{prefix}.mlp.gate_proj.weight": torch.ones((6, 3), dtype=torch.float32),
        f"{prefix}.mlp.up_proj.weight": torch.ones((6, 3), dtype=torch.float32),
        f"{prefix}.pre_feedforward_layernorm.weight": torch.ones((3,), dtype=torch.float32),
        f"{prefix}.pre_feedforward_layernorm_2.weight": torch.ones((3,), dtype=torch.float32),
        f"{prefix}.post_feedforward_layernorm.weight": torch.ones((3,), dtype=torch.float32),
        f"{prefix}.post_feedforward_layernorm_1.weight": torch.ones((3,), dtype=torch.float32),
        f"{prefix}.post_feedforward_layernorm_2.weight": torch.ones((3,), dtype=torch.float32),
    }


def _bundle(
    *,
    config: dict[str, object],
    state_dict: dict[str, torch.Tensor] | None = None,
    state_keys: tuple[str, ...] | None = None,
) -> LoadedBackendBundle:
    metadata: dict[str, object] = {"backend_version": "1.0.0"}
    if state_dict is not None:
        metadata["state_dict"] = state_dict
    if state_keys is not None:
        metadata["state_keys"] = state_keys
    return LoadedBackendBundle(
        backend_name="gemma4",
        model_handle=ModelHandle(model_id="google/gemma-4-27b", revision="rev-123", backend_name="gemma4"),
        model=object(),
        config=config,
        metadata=metadata,
    )


def test_static_expert_distribution_is_normalized() -> None:
    router_proj = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]],
        dtype=torch.float32,
    )

    distribution = static_expert_distribution(router_proj)

    assert distribution.shape == (3,)
    assert float(distribution.sum().item()) == pytest.approx(1.0)
    assert torch.all(distribution >= 0)


def test_build_expert_stats_returns_finite_ranked_metrics() -> None:
    router_proj = torch.tensor(
        [
            [3.0, 0.0, -2.0],
            [1.0, 2.0, 0.0],
            [-2.0, 1.0, 3.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    per_expert_scale = torch.tensor([0.2, 0.4, 0.1, 0.3], dtype=torch.float32)

    stats, summary = build_expert_stats(
        layer_index=7,
        router_proj_weight=router_proj,
        top_k=2,
        per_expert_scale=per_expert_scale,
    )

    assert len(stats) == 4
    assert summary.layer_index == 7
    assert summary.total_static_gate_mass == pytest.approx(1.0)
    assert summary.total_top_k_mass_proxy <= 1.0
    assert 0.0 <= summary.normalized_entropy <= 1.0
    assert [stat.static_rank for stat in stats] == [0, 1, 2, 3]
    for stat in stats:
        assert stat.layer_index == 7
        assert stat.static_gate_mass >= 0.0
        assert stat.static_gate_entropy >= 0.0
        assert stat.static_gate_entropy_norm is not None
        assert 0.0 <= stat.static_gate_entropy_norm <= 1.0
        assert stat.router_bias_norm is not None
        assert stat.router_bias_norm >= 0.0
        assert math.isfinite(stat.metadata["top_k_mass_proxy"])
        assert math.isfinite(stat.metadata["bias_adjusted_score"])
        assert stat.metadata["feature_count_proxy"] >= 0


def test_build_expert_stats_breaks_ties_deterministically() -> None:
    router_proj = torch.zeros((4, 3), dtype=torch.float32)

    stats, summary = build_expert_stats(
        layer_index=2,
        router_proj_weight=router_proj,
        top_k=2,
    )

    assert [stat.expert_index for stat in stats] == [0, 1, 2, 3]
    assert [stat.static_rank for stat in stats] == [0, 1, 2, 3]
    assert all(stat.static_gate_mass == pytest.approx(0.25) for stat in stats)
    assert all(stat.metadata["top_k_mass_proxy"] == pytest.approx(0.25) for stat in stats[:2])
    assert all(stat.metadata["top_k_mass_proxy"] == pytest.approx(0.0) for stat in stats[2:])
    assert summary.total_static_gate_mass == pytest.approx(1.0)
    assert summary.total_top_k_mass_proxy == pytest.approx(0.5)


def test_scan_model_returns_one_router_state_per_moe_layer() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[3, 1])
    state_dict: dict[str, torch.Tensor] = {}
    state_dict.update(_layer_state(1, shift=0.0))
    state_dict.update(_layer_state(3, shift=0.5))
    bundle = _bundle(config=config, state_dict=state_dict)

    result = scan_model(bundle, backend=backend)

    assert [layer.layer_index for layer in result.layers] == [1, 3]
    assert [router.layer_index for router in result.router_states] == [1, 3]
    assert len(result.router_states) == len(result.layers) == 2
    assert len(result.layer_summaries) == 2
    assert result.aggregate_summary.layer_count == 2
    assert result.aggregate_summary.expert_stat_count == 8
    assert result.manifest.command == "scan"
    assert result.manifest.metadata["moe_layer_count"] == 2
    assert result.manifest.metadata["total_static_gate_mass"] == pytest.approx(2.0)

    for layer, summary in zip(result.layers, result.layer_summaries):
        per_layer = [stat for stat in result.expert_stats if stat.layer_index == layer.layer_index]
        assert len(per_layer) == layer.expert_count
        assert {stat.expert_index for stat in per_layer} == set(range(layer.expert_count))
        assert [stat.static_rank for stat in per_layer] == list(range(layer.expert_count))
        assert sum(stat.static_gate_mass for stat in per_layer) == pytest.approx(1.0)
        assert summary.total_static_gate_mass == pytest.approx(1.0)
        assert 0.0 <= summary.normalized_entropy <= 1.0

    assert result.aggregate_summary.total_static_gate_mass == pytest.approx(2.0)
    assert 0.0 <= result.aggregate_summary.mean_normalized_entropy <= 1.0

    repeated = scan_model(bundle, backend=backend)
    assert repeated == result


def test_scan_model_rejects_topology_only_state_keys_metadata() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_keys = tuple(_layer_state(0, shift=0.0))
    bundle = _bundle(config=config, state_keys=state_keys)

    with pytest.raises(
        TopologyMismatchError,
        match="static scan requires materialized numeric tensors, not topology-only state_keys metadata",
    ):
        scan_model(bundle, backend=backend)
