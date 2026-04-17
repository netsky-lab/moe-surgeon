"""Deterministic topology and activation ordering helpers."""

from __future__ import annotations

from typing import Sequence

from moe_surgeon.models.errors import TopologyMismatchError
from moe_surgeon.schemas import ActivationStats, LayerTopology, sort_activation_stats, sort_topology


def build_layer_topology_index(layers: Sequence[LayerTopology]) -> dict[int, LayerTopology]:
    """Build a deterministic layer-index lookup with duplicate protection."""

    ordered = sort_topology(layers)
    index: dict[int, LayerTopology] = {}
    for layer in ordered:
        if layer.layer_index in index:
            raise TopologyMismatchError(
                "duplicate layer_index in topology",
                layer_index=layer.layer_index,
            )
        index[layer.layer_index] = layer
    return index


def align_activation_stats(
    *,
    layers: Sequence[LayerTopology],
    stats: Sequence[ActivationStats],
) -> tuple[ActivationStats, ...]:
    """Validate activation stats against topology and return canonical ordering."""

    topology_index = build_layer_topology_index(layers)
    layer_token_totals: dict[int, int] = {}
    layer_weighted_totals: dict[int, float | None] = {}
    for item in stats:
        layer = topology_index.get(item.layer_index)
        if layer is None:
            raise TopologyMismatchError(
                "activation stats reference unknown layer",
                layer_index=item.layer_index,
            )
        if item.expert_index >= layer.expert_count:
            raise TopologyMismatchError(
                "activation stats expert index exceeds layer topology",
                layer_index=item.layer_index,
                details={"expert_index": item.expert_index, "expert_count": layer.expert_count},
            )
        existing_n_tokens = layer_token_totals.setdefault(item.layer_index, item.n_tokens)
        if existing_n_tokens != item.n_tokens:
            raise TopologyMismatchError(
                "activation stats layer token totals are inconsistent",
                layer_index=item.layer_index,
                details={"expected_n_tokens": existing_n_tokens, "actual_n_tokens": item.n_tokens},
            )
        existing_weighted = layer_weighted_totals.setdefault(item.layer_index, item.weighted_n_tokens)
        if existing_weighted is None:
            if item.weighted_n_tokens is not None:
                raise TopologyMismatchError(
                    "activation stats layer weighted token totals are inconsistent",
                    layer_index=item.layer_index,
                    details={"expected_weighted_n_tokens": None, "actual_weighted_n_tokens": item.weighted_n_tokens},
                )
        elif item.weighted_n_tokens is None or abs(existing_weighted - item.weighted_n_tokens) > 1e-12:
            raise TopologyMismatchError(
                "activation stats layer weighted token totals are inconsistent",
                layer_index=item.layer_index,
                details={
                    "expected_weighted_n_tokens": existing_weighted,
                    "actual_weighted_n_tokens": item.weighted_n_tokens,
                },
            )
    return sort_activation_stats(stats)


__all__ = [
    "align_activation_stats",
    "build_layer_topology_index",
]
