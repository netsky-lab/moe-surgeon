"""Pure static router metric helpers for deterministic expert ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import cast

import torch

from moe_surgeon.schemas import CANONICAL_FLOAT_EPSILON, ExpertStats, sort_experts

_METRIC_DTYPE = torch.float64


@dataclass(frozen=True)
class RouterMetricSummary:
    """Aggregate deterministic summary for one MoE router layer."""

    layer_index: int
    expert_count: int
    total_static_gate_mass: float
    total_top_k_mass_proxy: float
    entropy: float
    normalized_entropy: float


def _upcast_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Compute metrics in a stable floating dtype regardless of checkpoint precision."""

    return tensor.detach().to(dtype=_METRIC_DTYPE)


def _require_finite_non_negative(values: torch.Tensor, *, name: str) -> torch.Tensor:
    """Fail fast if a derived metric tensor is not finite and non-negative."""

    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if (values < 0).any():
        raise ValueError(f"{name} must be non-negative")
    return values


def _router_probabilities(router_proj_weight: torch.Tensor) -> torch.Tensor:
    """Normalize router projection weights over the expert axis."""

    if router_proj_weight.ndim != 2:
        raise ValueError("router_proj_weight must be rank-2 [num_experts, hidden_size]")
    probabilities = torch.softmax(_upcast_tensor(router_proj_weight), dim=0, dtype=_METRIC_DTYPE)
    return _require_finite_non_negative(probabilities, name="router_probabilities")


def _entropy_contribution(distribution: torch.Tensor) -> torch.Tensor:
    """Return the per-expert entropy contribution ``-p * log(p)``."""

    safe_distribution = _require_finite_non_negative(distribution, name="distribution").clamp_min(
        CANONICAL_FLOAT_EPSILON
    )
    contribution = -(safe_distribution * safe_distribution.log())
    return _require_finite_non_negative(contribution, name="entropy_contribution")


def static_expert_distribution(router_proj_weight: torch.Tensor) -> torch.Tensor:
    """Derive a deterministic static expert distribution from router projection weights.

    The Gemma4 router projection is shaped ``[num_experts, hidden_size]``.
    We apply softmax over the expert axis for each hidden feature, then average
    across hidden features so the returned distribution sums to one.
    """

    probabilities = _router_probabilities(router_proj_weight)
    return _require_finite_non_negative(
        probabilities.mean(dim=1),
        name="static_expert_distribution",
    )


def per_expert_scale_norm(per_expert_scale: torch.Tensor | None) -> torch.Tensor | None:
    """Return a non-negative per-expert norm derived from router bias/scale state."""

    if per_expert_scale is None:
        return None
    if per_expert_scale.ndim == 0:
        raise ValueError("per_expert_scale must expose an expert axis")
    reshaped = _upcast_tensor(per_expert_scale).reshape(per_expert_scale.shape[0], -1)
    return cast(torch.Tensor, torch.linalg.vector_norm(reshaped, ord=2, dim=1, dtype=_METRIC_DTYPE))


def top_k_mass_proxy(router_proj_weight: torch.Tensor, *, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate top-k retention mass and feature wins from static router weights."""

    if router_proj_weight.ndim != 2:
        raise ValueError("router_proj_weight must be rank-2 [num_experts, hidden_size]")
    num_experts = int(router_proj_weight.shape[0])
    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be in the range [1, num_experts]")

    probabilities = _router_probabilities(router_proj_weight)
    _, sorted_indices = torch.sort(probabilities, dim=0, descending=True, stable=True)
    topk_indices = sorted_indices[:top_k, :]
    selection_mask = torch.zeros_like(probabilities)
    selection_mask.scatter_(0, topk_indices, 1.0)
    retained_mass = _require_finite_non_negative(
        (probabilities * selection_mask).mean(dim=1),
        name="top_k_mass_proxy",
    )
    feature_win_count = torch.bincount(topk_indices.reshape(-1), minlength=num_experts).to(dtype=_METRIC_DTYPE)
    return retained_mass, _require_finite_non_negative(feature_win_count, name="feature_win_count")


def summarize_layer_metrics(
    *,
    layer_index: int,
    distribution: torch.Tensor,
    top_k_proxy: torch.Tensor,
) -> RouterMetricSummary:
    """Build an aggregate summary from per-expert layer metrics."""

    distribution_f = _upcast_tensor(distribution)
    top_k_proxy_f = _upcast_tensor(top_k_proxy)
    expert_count = int(distribution_f.shape[0])
    entropy = float(_entropy_contribution(distribution_f).sum().item())
    entropy_denom = log(expert_count) if expert_count > 1 else 1.0
    normalized_entropy = 0.0 if expert_count <= 1 else entropy / entropy_denom
    return RouterMetricSummary(
        layer_index=layer_index,
        expert_count=expert_count,
        total_static_gate_mass=float(distribution_f.sum().item()),
        total_top_k_mass_proxy=float(top_k_proxy_f.sum().item()),
        entropy=entropy,
        normalized_entropy=float(normalized_entropy),
    )


def build_expert_stats(
    *,
    layer_index: int,
    router_proj_weight: torch.Tensor,
    top_k: int,
    per_expert_scale: torch.Tensor | None = None,
) -> tuple[tuple[ExpertStats, ...], RouterMetricSummary]:
    """Compute deterministic static expert metrics for one router layer."""

    distribution = static_expert_distribution(router_proj_weight)
    top_k_proxy, feature_win_count = top_k_mass_proxy(router_proj_weight, top_k=top_k)
    bias_norm = per_expert_scale_norm(per_expert_scale)
    expert_count = int(distribution.shape[0])
    entropy_denom = log(expert_count) if expert_count > 1 else 1.0
    entropy_contribution = _entropy_contribution(distribution)

    stats: list[ExpertStats] = []
    ranking_inputs: list[tuple[int, float, float, int]] = []
    for expert_index in range(expert_count):
        mass = float(distribution[expert_index].item())
        entropy = float(entropy_contribution[expert_index].item())
        entropy_norm = 0.0 if expert_count <= 1 else entropy / entropy_denom
        bias_value = None if bias_norm is None else float(bias_norm[expert_index].item())
        top_k_mass = float(top_k_proxy[expert_index].item())
        bias_adjusted_score = mass if bias_value is None else mass * (1.0 + bias_value)
        ranking_inputs.append(
            (
                layer_index,
                bias_adjusted_score if bias_value is not None else mass,
                top_k_mass,
                expert_index,
            )
        )
        stats.append(
            ExpertStats(
                layer_index=layer_index,
                expert_index=expert_index,
                static_gate_mass=mass,
                static_gate_entropy=entropy,
                static_gate_entropy_norm=float(entropy_norm),
                router_bias_norm=bias_value,
                metadata={
                    "top_k_mass_proxy": top_k_mass,
                    "feature_count_proxy": int(feature_win_count[expert_index].item()),
                    "bias_adjusted_score": float(bias_adjusted_score),
                },
            )
        )

    ranked_indices = {
        expert_index: rank
        for rank, (_, _, _, expert_index) in enumerate(
            cast(
                list[tuple[int, float, float, int]],
                sort_experts(ranking_inputs),
            )
        )
    }
    ordered = sorted(
        stats,
        key=lambda stat: (
            ranked_indices[stat.expert_index],
            stat.expert_index,
        ),
    )
    ranked: list[ExpertStats] = []
    for item in ordered:
        assert isinstance(item, ExpertStats)
        ranked.append(
            ExpertStats(
                layer_index=item.layer_index,
                expert_index=item.expert_index,
                static_gate_mass=item.static_gate_mass,
                static_gate_entropy=item.static_gate_entropy,
                static_gate_entropy_norm=item.static_gate_entropy_norm,
                router_bias_norm=item.router_bias_norm,
                static_rank=ranked_indices[item.expert_index],
                ffn_param_count=item.ffn_param_count,
                metadata=dict(item.metadata),
            )
        )

    return tuple(ranked), summarize_layer_metrics(
        layer_index=layer_index,
        distribution=distribution,
        top_k_proxy=top_k_proxy,
    )


__all__ = [
    "RouterMetricSummary",
    "build_expert_stats",
    "per_expert_scale_norm",
    "static_expert_distribution",
    "summarize_layer_metrics",
    "top_k_mass_proxy",
]
