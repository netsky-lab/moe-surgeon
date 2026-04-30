"""Package metadata for the lightweight moe-surgeon bootstrap."""

PACKAGE_NAME = "moe-surgeon"
PACKAGE_DESCRIPTION = "Python CLI for analyzing and pruning MoE models"
PACKAGE_LAYOUT = {
    "cli": "command graph and orchestration",
    "models": (
        "backend adapters, the offline safetensors checkpoint reader in "
        "src/moe_surgeon/models/checkpoints.py, the GGUF metadata reader in "
        "src/moe_surgeon/models/gguf.py, and topology/contracts"
    ),
    "analysis": "static router analysis",
    "runtime": "forward hook profiler",
    "prune": "strategy and plan generation (selection only)",
    "export": "artifact persistence and manifest output",
}
__version__ = "0.1.0"

__all__ = ["PACKAGE_DESCRIPTION", "PACKAGE_LAYOUT", "PACKAGE_NAME", "__version__"]
