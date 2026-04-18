"""Pruning package with strategy, planning, and apply contracts."""

from moe_surgeon.prune.planner import LayerConstraintOverride, PlannerConstraints, build_prune_plan
from moe_surgeon.prune.strategy import (
    PruneStrategy,
    StrategyMetadata,
    StrategyName,
    StrategyRegistry,
    create_strategy,
    strategy_registry,
)

__all__ = [
    "ApplyLayerReport",
    "ApplyResult",
    "ApplyTensorDelta",
    "LayerConstraintOverride",
    "PlannerConstraints",
    "PruneStrategy",
    "StrategyMetadata",
    "StrategyName",
    "StrategyRegistry",
    "apply_prune_plan",
    "build_prune_plan",
    "create_strategy",
    "strategy_registry",
]


def __getattr__(name: str) -> object:
    """Lazily expose the heavier prune/apply surface."""

    if name not in {"ApplyLayerReport", "ApplyResult", "ApplyTensorDelta", "apply_prune_plan"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from moe_surgeon.prune.apply import (
        ApplyLayerReport,
        ApplyResult,
        ApplyTensorDelta,
        apply_prune_plan,
    )

    exports = {
        "ApplyLayerReport": ApplyLayerReport,
        "ApplyResult": ApplyResult,
        "ApplyTensorDelta": ApplyTensorDelta,
        "apply_prune_plan": apply_prune_plan,
    }
    return exports[name]
