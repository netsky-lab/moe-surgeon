from __future__ import annotations

import pytest

from moe_surgeon.analysis.scan import align_activation_stats, build_layer_topology_index
from moe_surgeon.models.errors import TopologyMismatchError
from moe_surgeon.schemas import ActivationStats, LayerTopology


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
