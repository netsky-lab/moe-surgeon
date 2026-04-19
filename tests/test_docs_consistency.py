from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from moe_surgeon import PACKAGE_LAYOUT
import moe_surgeon.models as models_package
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
    package_descriptions: dict[str, str]
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class ArchitectureClaims:
    text: str
    package_paths: dict[str, bool]


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
    package_descriptions = {
        match.group(1): _normalize_inline_code(match.group(2).replace("\n", " ").strip())
        for match in re.finditer(
            r"^- ([a-z_]+/): (.+?)(?=\n- [a-z_]+/: |\n## |\Z)",
            readme_text,
            flags=re.MULTILINE | re.DOTALL,
        )
    }
    source_files = tuple(
        sorted(set(re.findall(r"src/moe_surgeon/[a-z0-9_/]+\.py", readme_text)))
    )
    return ReadmeClaims(package_descriptions=package_descriptions, source_files=source_files)


def _architecture_claims() -> ArchitectureClaims:
    text = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    package_paths = _parse_architecture_package_structure(text)
    return ArchitectureClaims(text=" ".join(text.split()), package_paths=package_paths)


def _normalize_inline_code(text: str) -> str:
    return " ".join(text.replace("`", "").split())


def _parse_architecture_package_structure(text: str) -> dict[str, bool]:
    match = re.search(r"^## Package structure\n\n(?P<block>.*?)(?=^## )", text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, "ARCHITECTURE.md must contain a package-structure block"
    block = match.group("block").strip("\n")
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    assert lines and lines[0].strip() == "moe_surgeon/"
    package_paths: dict[str, bool] = {}
    stack: list[tuple[int, str]] = [(0, "moe_surgeon")]
    for line in lines[1:]:
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        assert stripped.startswith("- "), f"unexpected package-structure line: {line}"
        name = stripped[2:]
        future = name.endswith(" (future)")
        if future:
            name = name[: -len(" (future)")]
        while stack and indent <= stack[-1][0]:
            stack.pop()
        assert stack, f"invalid package-structure indentation: {line}"
        parent = stack[-1][1]
        path = f"{parent}/{name.removesuffix('/')}"
        package_paths[path] = future
        if name.endswith("/"):
            stack.append((indent, path))
    return package_paths


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

    assert tuple(claims.package_descriptions) == ("cli/", "models/", "analysis/", "runtime/", "prune/", "export/")
    assert {
        f"moe_surgeon/{bucket.removesuffix('/')}"
        for bucket in claims.package_descriptions
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


def test_architecture_package_structure_tracks_source_layout() -> None:
    inventory = _tracked_source_inventory()
    claims = _architecture_claims()

    existing_layout = set(inventory.package_dirs) | set(inventory.module_files)
    future_paths = {path for path, future in claims.package_paths.items() if future}
    active_paths = {path for path, future in claims.package_paths.items() if not future}

    assert future_paths.isdisjoint(inventory.module_files)
    assert active_paths.issubset(existing_layout)
    assert claims.package_paths["moe_surgeon/analysis/metrics.py"] is False
    assert claims.package_paths["moe_surgeon/runtime/bench.py"] is False
    assert claims.package_paths["moe_surgeon/prune/strategy.py"] is False
    assert claims.package_paths["moe_surgeon/export/manifest.py"] is False
    assert claims.package_paths["moe_surgeon/export/runner.py"] is False
    assert claims.package_paths["moe_surgeon/export/safetensors_writer.py"] is False


def test_checkpoint_architecture_claim_tracks_reader_file_presence() -> None:
    inventory = _tracked_source_inventory()
    claims = _architecture_claims()

    assert "moe_surgeon/models/checkpoints.py" in inventory.module_files
    assert "src/moe_surgeon/models/checkpoints.py" in claims.text
    assert "offline-local `safetensors` checkpoint introspection" in claims.text
    assert "single-file and indexed sharded layouts" in claims.text
    assert "deterministic `state_keys` and targeted tensor reads" in claims.text
    assert "without importing `transformers` or loading a full model" in claims.text


def test_package_descriptions_align_with_tracked_models_checkpoint_reader_role() -> None:
    inventory = _tracked_source_inventory()
    claims = _architecture_claims()
    expected_models_description = (
        "backend adapters, the offline safetensors checkpoint reader in "
        "src/moe_surgeon/models/checkpoints.py, and topology/contracts"
    )

    assert "moe_surgeon/models" in inventory.package_dirs
    assert "moe_surgeon/models/checkpoints.py" in inventory.module_files
    assert PACKAGE_LAYOUT["models"] == MODELS_PACKAGE_DESCRIPTION
    assert _readme_claims().package_descriptions["models/"] == expected_models_description
    assert MODELS_PACKAGE_DESCRIPTION == expected_models_description
    assert models_package.__doc__ == f"{MODELS_PACKAGE_DESCRIPTION}."
    assert "src/moe_surgeon/models/checkpoints.py" in MODELS_PACKAGE_DESCRIPTION
    assert "offline safetensors checkpoint reader" in MODELS_PACKAGE_DESCRIPTION
    assert "src/moe_surgeon/models/checkpoints.py" in claims.text
    assert "offline-local `safetensors` checkpoint introspection" in claims.text
    assert "deterministic `state_keys` and targeted tensor reads" in claims.text
    assert "without importing `transformers` or loading a full model" in claims.text
