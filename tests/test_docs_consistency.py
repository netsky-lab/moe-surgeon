from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from moe_surgeon import PACKAGE_LAYOUT
from moe_surgeon.models import PACKAGE_DESCRIPTION as MODELS_PACKAGE_DESCRIPTION


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("src/moe_surgeon")


@dataclass(frozen=True)
class SourceInventory:
    package_dirs: frozenset[str]
    module_files: frozenset[str]
    existing_paths: frozenset[str]


@dataclass(frozen=True)
class ReadmeClaims:
    package_buckets: tuple[str, ...]
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class ArchitectureClaims:
    text: str


def _tracked_source_inventory() -> SourceInventory:
    result = subprocess.run(
        ["git", "ls-files", "src/moe_surgeon"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    tracked_paths = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    package_dirs = {
        str(Path(path).parent.relative_to(SOURCE_ROOT.parent))
        for path in tracked_paths
        if path.endswith("/__init__.py")
    }
    module_files = {
        str(Path(path).relative_to(SOURCE_ROOT.parent))
        for path in tracked_paths
        if path.endswith(".py")
    }
    return SourceInventory(
        package_dirs=frozenset(package_dirs),
        module_files=frozenset(module_files),
        existing_paths=frozenset(
            path for path in tracked_paths if (REPO_ROOT / path).exists()
        ),
    )


def _readme_claims() -> ReadmeClaims:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    package_buckets = tuple(
        match.group(1)
        for match in re.finditer(r"^- ([a-z_]+/): ", readme_text, flags=re.MULTILINE)
    )
    source_files = tuple(
        sorted(set(re.findall(r"src/moe_surgeon/[a-z0-9_/]+\.py", readme_text)))
    )
    return ReadmeClaims(package_buckets=package_buckets, source_files=source_files)


def _architecture_claims() -> ArchitectureClaims:
    text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    return ArchitectureClaims(text=" ".join(text.split()))


def test_layout_tracked_source_inventory_matches_documented_package_layout() -> None:
    inventory = _tracked_source_inventory()

    assert inventory.package_dirs == frozenset(
        {
            "moe_surgeon",
            "moe_surgeon/analysis",
            "moe_surgeon/cli",
            "moe_surgeon/export",
            "moe_surgeon/models",
            "moe_surgeon/prune",
            "moe_surgeon/runtime",
        }
    )
    assert inventory.module_files.issuperset(
        {
            "moe_surgeon/__init__.py",
            "moe_surgeon/__main__.py",
            "moe_surgeon/analysis/__init__.py",
            "moe_surgeon/analysis/metrics.py",
            "moe_surgeon/analysis/scan.py",
            "moe_surgeon/cli/__init__.py",
            "moe_surgeon/cli/main.py",
            "moe_surgeon/export/__init__.py",
            "moe_surgeon/models/__init__.py",
            "moe_surgeon/models/backend.py",
            "moe_surgeon/models/checkpoints.py",
            "moe_surgeon/models/errors.py",
            "moe_surgeon/models/gemma4.py",
            "moe_surgeon/models/registry.py",
            "moe_surgeon/prune/__init__.py",
            "moe_surgeon/prune/apply.py",
            "moe_surgeon/prune/planner.py",
            "moe_surgeon/prune/strategies.py",
            "moe_surgeon/prune/strategy.py",
            "moe_surgeon/runtime/__init__.py",
            "moe_surgeon/runtime/bench.py",
            "moe_surgeon/runtime/profiler.py",
            "moe_surgeon/schemas.py",
        }
    )
    assert inventory.existing_paths.issuperset(
        {
            "src/moe_surgeon/__init__.py",
            "src/moe_surgeon/analysis/__init__.py",
            "src/moe_surgeon/cli/__init__.py",
            "src/moe_surgeon/export/__init__.py",
            "src/moe_surgeon/models/__init__.py",
            "src/moe_surgeon/models/checkpoints.py",
            "src/moe_surgeon/prune/__init__.py",
            "src/moe_surgeon/runtime/__init__.py",
        }
    )
    assert "moe_surgeon/models/checkpoints.py" in inventory.module_files


def test_readme_module_buckets_map_to_tracked_packages() -> None:
    inventory = _tracked_source_inventory()
    claims = _readme_claims()

    assert claims.package_buckets == (
        "cli/",
        "models/",
        "analysis/",
        "runtime/",
        "prune/",
        "export/",
    )
    assert {
        f"moe_surgeon/{bucket.removesuffix('/')}"
        for bucket in claims.package_buckets
    }.issubset(inventory.package_dirs)


def test_readme_source_file_claims_exist_in_tracked_layout() -> None:
    inventory = _tracked_source_inventory()
    claims = _readme_claims()

    assert claims.source_files == (
        "src/moe_surgeon/models/checkpoints.py",
        "src/moe_surgeon/schemas.py",
    )
    assert {
        str(Path(path).relative_to("src"))
        for path in claims.source_files
    }.issubset(inventory.module_files)
    assert set(claims.source_files).issubset(inventory.existing_paths)


def test_checkpoint_architecture_claim_tracks_reader_file_presence() -> None:
    inventory = _tracked_source_inventory()
    claims = _architecture_claims()

    assert "moe_surgeon/models/checkpoints.py" in inventory.module_files
    assert "checkpoints.py" in claims.text
    assert "offline-local `safetensors` checkpoint introspection" in claims.text
    assert "single-file and indexed sharded layouts" in claims.text
    assert "deterministic `state_keys` and targeted tensor reads" in claims.text
    assert "without importing `transformers` or materializing a full model" in claims.text


def test_package_descriptions_align_with_tracked_models_checkpoint_reader_role() -> None:
    inventory = _tracked_source_inventory()

    assert "moe_surgeon/models" in inventory.package_dirs
    assert "moe_surgeon/models/checkpoints.py" in inventory.module_files
    assert PACKAGE_LAYOUT["models"] == MODELS_PACKAGE_DESCRIPTION
    assert MODELS_PACKAGE_DESCRIPTION == (
        "backend adapters, lightweight checkpoint readers, and topology/contracts"
    )
