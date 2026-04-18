from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path("src/moe_surgeon")


@dataclass(frozen=True)
class TrackedSourceInventory:
    """Tracked Python package/file layout rooted at ``src/moe_surgeon``."""

    packages: tuple[str, ...]
    modules: tuple[str, ...]


def _tracked_source_inventory() -> TrackedSourceInventory:
    result = subprocess.run(
        ["git", "ls-files", "--", "src/moe_surgeon/**/*.py", "src/moe_surgeon/*.py"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    tracked_paths = sorted(
        Path(path).as_posix()
        for path in result.stdout.splitlines()
        if path.startswith(f"{SRC_ROOT.as_posix()}/")
    )
    packages = sorted(
        path[: -len("/__init__.py")]
        for path in tracked_paths
        if path.endswith("/__init__.py")
    )
    return TrackedSourceInventory(packages=tuple(packages), modules=tuple(tracked_paths))


def _documented_source_layout() -> TrackedSourceInventory:
    return TrackedSourceInventory(
        packages=(
            "src/moe_surgeon",
            "src/moe_surgeon/analysis",
            "src/moe_surgeon/cli",
            "src/moe_surgeon/export",
            "src/moe_surgeon/models",
            "src/moe_surgeon/prune",
            "src/moe_surgeon/runtime",
        ),
        modules=(
            "src/moe_surgeon/__init__.py",
            "src/moe_surgeon/__main__.py",
            "src/moe_surgeon/analysis/__init__.py",
            "src/moe_surgeon/analysis/metrics.py",
            "src/moe_surgeon/analysis/scan.py",
            "src/moe_surgeon/cli/__init__.py",
            "src/moe_surgeon/cli/main.py",
            "src/moe_surgeon/export/__init__.py",
            "src/moe_surgeon/models/__init__.py",
            "src/moe_surgeon/models/backend.py",
            "src/moe_surgeon/models/checkpoints.py",
            "src/moe_surgeon/models/errors.py",
            "src/moe_surgeon/models/gemma4.py",
            "src/moe_surgeon/models/registry.py",
            "src/moe_surgeon/prune/__init__.py",
            "src/moe_surgeon/prune/apply.py",
            "src/moe_surgeon/prune/planner.py",
            "src/moe_surgeon/prune/strategies.py",
            "src/moe_surgeon/prune/strategy.py",
            "src/moe_surgeon/repo_metrics.py",
            "src/moe_surgeon/runtime/__init__.py",
            "src/moe_surgeon/runtime/bench.py",
            "src/moe_surgeon/runtime/profiler.py",
            "src/moe_surgeon/schemas.py",
            "src/moe_surgeon/test_env.py",
        ),
    )


def test_layout_tracked_inventory_matches_documented_src_tree() -> None:
    assert _tracked_source_inventory() == _documented_source_layout()
