from __future__ import annotations

import math

import pytest
import torch

from moe_surgeon.analysis.metrics import build_expert_stats, static_expert_distribution
from moe_surgeon.analysis.scan import (
    _require_tensor,
    align_activation_stats,
    build_layer_topology_index,
    scan_model,
    scan_result_json,
    write_scan_artifact,
)
from moe_surgeon.models.backend import LoadedBackendBundle, TensorMetadata
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.schemas import ActivationStats, LayerTopology, ModelHandle, RouterState


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


class _PermissiveMetricBackend(Gemma4Backend):
    def extract_router_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> RouterState:
        state_dict = bundle.metadata["state_dict"]
        router_proj = state_dict[layer.module_paths["router_proj"]]
        per_expert_scale = state_dict[layer.module_paths["router_per_expert_scale"]]
        return RouterState(
            layer_index=layer.layer_index,
            num_experts=layer.expert_count,
            top_k=layer.top_k,
            logits_shape=(0, layer.expert_count),
            projection_shape=tuple(int(dimension) for dimension in router_proj.shape),
            per_expert_scale_shape=tuple(int(dimension) for dimension in per_expert_scale.shape),
            route_scale_present=True,
            metadata={
                "router_proj_key": layer.module_paths["router_proj"],
                "router_scale_key": layer.module_paths["router_scale"],
                "router_per_expert_scale_key": layer.module_paths["router_per_expert_scale"],
            },
        )

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        del bundle, layer, router_state

    def extract_expert_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> dict[str, TensorMetadata]:
        del bundle, layer
        return {}


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
    assert result.manifest.metadata["model_fingerprint"] == result.manifest.model_handle.model_fingerprint
    assert result.manifest.metadata["canonical_manifest_digest"] == result.manifest.canonical_digest
    assert isinstance(result.manifest.metadata["canonical_artifact_digest"], str)

    repeated = scan_model(bundle, backend=backend)
    assert repeated == result


def test_scan_writer_emits_byte_identical_canonical_json(tmp_path) -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0, 2])
    state_dict: dict[str, torch.Tensor] = {}
    state_dict.update(_layer_state(0, shift=0.0))
    state_dict.update(_layer_state(2, shift=0.25))
    bundle = _bundle(config=config, state_dict=state_dict)

    first = scan_model(bundle, backend=backend)
    second = scan_model(bundle, backend=backend)

    first_json = scan_result_json(first)
    second_json = scan_result_json(second)
    assert first_json == second_json

    first_path = write_scan_artifact(tmp_path / "scan-first.json", first)
    second_path = write_scan_artifact(tmp_path / "scan-second.json", second)

    assert first_path.read_bytes() == second_path.read_bytes()


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


def test_require_tensor_rejects_missing_materialized_router_tensor_payload() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0, shift=0.0)
    bundle = _bundle(config=config, state_dict=state_dict)
    layer = backend.extract_topology(bundle)[0]

    materialized_state = dict(state_dict)
    del materialized_state["model.language_model.layers.0.router.scale"]

    with pytest.raises(
        TopologyMismatchError,
        match="static scan requires materialized numeric tensor values",
    ) as exc_info:
        _require_tensor(
            materialized_state,
            bundle=bundle,
            layer=layer,
            tensor_role="router_scale",
        )

    message = str(exc_info.value)
    assert "layer_index=0" in message
    assert "tensor_key=model.language_model.layers.0.router.scale" in message
    assert "tensor_role=router_scale" in message


def test_require_tensor_rejects_non_tensor_router_payload() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0, shift=0.0)
    bundle = _bundle(config=config, state_dict=state_dict)
    layer = backend.extract_topology(bundle)[0]

    materialized_state: dict[str, object] = dict(state_dict)
    materialized_state["model.language_model.layers.0.router.per_expert_scale"] = [0.2, 0.4, 0.1, 0.3]

    with pytest.raises(
        ShapeInvariantViolationError,
        match="scan router tensor must be torch.Tensor",
    ) as exc_info:
        _require_tensor(
            materialized_state,
            bundle=bundle,
            layer=layer,
            tensor_role="router_per_expert_scale",
        )

    message = str(exc_info.value)
    assert "layer_index=0" in message
    assert "tensor_key=model.language_model.layers.0.router.per_expert_scale" in message
    assert "tensor_role=router_per_expert_scale" in message
    assert "value_type=list" in message


def test_scan_model_rejects_router_tensor_shape_mismatch() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0, shift=0.0)
    state_dict["model.language_model.layers.0.router.per_expert_scale"] = torch.ones(
        (3,),
        dtype=torch.float32,
    )
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(TopologyMismatchError, match="Gemma4 per-expert scale shape mismatch") as exc_info:
        scan_model(bundle, backend=backend)

    message = str(exc_info.value)
    assert "layer_index=0" in message
    assert "tensor_key=model.language_model.layers.0.router.per_expert_scale" in message
    assert "expected_shape=4" in message
    assert "actual_shape=3" in message


def test_scan_model_rejects_non_finite_router_metric_tensor() -> None:
    backend = _PermissiveMetricBackend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0, shift=0.0)
    state_dict["model.language_model.layers.0.router.proj.weight"][1, 1] = float("nan")
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(ShapeInvariantViolationError, match="scan metric tensor must be finite") as exc_info:
        scan_model(bundle, backend=backend)

    message = str(exc_info.value)
    assert "layer_index=0" in message
    assert "tensor_key=model.language_model.layers.0.router.proj.weight" in message
    assert "tensor_role=router_proj" in message


def test_scan_model_rejects_invalid_per_expert_scale_rank_in_metrics_path() -> None:
    backend = _PermissiveMetricBackend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0, shift=0.0)
    state_dict["model.language_model.layers.0.router.per_expert_scale"] = torch.ones(
        (4, 1),
        dtype=torch.float32,
    )
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(ShapeInvariantViolationError, match="scan metric tensor rank mismatch") as exc_info:
        scan_model(bundle, backend=backend)

    message = str(exc_info.value)
    assert "layer_index=0" in message
    assert "tensor_key=model.language_model.layers.0.router.per_expert_scale" in message
    assert "tensor_role=router_per_expert_scale" in message
    assert "expected_rank=1" in message
    assert "actual_rank=2" in message


def _layer(layer_index: int, *, expert_count: int = 4) -> LayerTopology:
    return LayerTopology(
        layer_index=layer_index,
        layer_name=f"layer-{layer_index}",
        layer_type="fake_moe",
        expert_count=expert_count,
        top_k=2,
        hidden_size=16,
        layer_ref=f"layer_{layer_index}",
    )


def test_build_layer_topology_index_is_ordered_by_layer_index() -> None:
    index = build_layer_topology_index([_layer(3), _layer(1)])

    assert list(index) == [1, 3]


def test_align_activation_stats_rejects_unknown_layer() -> None:
    with pytest.raises(TopologyMismatchError, match="activation stats reference unknown layer"):
        align_activation_stats(
            layers=[_layer(0)],
            stats=[
                ActivationStats(
                    layer_index=1,
                    expert_index=0,
                    token_count=0,
                    weighted_token_count=0.0,
                    mass_sum=0.0,
                    mean_weight=0.0,
                    entropy=0.0,
                    n_tokens=0,
                    weighted_n_tokens=0.0,
                )
            ],
        )


def test_align_activation_stats_rejects_out_of_range_expert() -> None:
    with pytest.raises(TopologyMismatchError, match="activation stats expert index exceeds layer topology"):
        align_activation_stats(
            layers=[_layer(0, expert_count=2)],
            stats=[
                ActivationStats(
                    layer_index=0,
                    expert_index=2,
                    token_count=0,
                    weighted_token_count=0.0,
                    mass_sum=0.0,
                    mean_weight=0.0,
                    entropy=0.0,
                    n_tokens=0,
                    weighted_n_tokens=0.0,
                )
            ],
        )


def test_align_activation_stats_rejects_inconsistent_layer_weighted_totals() -> None:
    with pytest.raises(TopologyMismatchError, match="weighted token totals are inconsistent"):
        align_activation_stats(
            layers=[_layer(0)],
            stats=[
                ActivationStats(
                    layer_index=0,
                    expert_index=0,
                    token_count=1,
                    weighted_token_count=0.7,
                    mass_sum=0.7,
                    mean_weight=0.7,
                    entropy=0.0,
                    n_tokens=1,
                    weighted_n_tokens=1.0,
                ),
                ActivationStats(
                    layer_index=0,
                    expert_index=1,
                    token_count=1,
                    weighted_token_count=0.3,
                    mass_sum=0.3,
                    mean_weight=0.3,
                    entropy=0.0,
                    n_tokens=1,
                    weighted_n_tokens=0.5,
                ),
            ],
        )
