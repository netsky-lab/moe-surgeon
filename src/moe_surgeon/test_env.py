"""Repo-local helpers for deterministic quality-gate subprocess isolation."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import MutableMapping


REPO_PROCESS_TEMP_SUBDIR = Path(".tmp") / "system"
REPO_PYTEST_TEMP_SUBDIR = Path(".tmp") / "pytest"
TEMP_ENV_KEYS = ("TMPDIR", "TMP", "TEMP")
REPO_ROOT_MARKERS = (
    "pyproject.toml",
    "AGENTS.md",
    ".supervisor/project.json",
    "src/moe_surgeon/repo_metrics.py",
)
FIXTURE_ROOT_MARKERS = (
    "pyproject.toml",
    "src/sitecustomize.py",
)


def bootstrap_repo_quality_gate_env(start_path: Path | None = None) -> None:
    """Apply repo-local quality-gate defaults when the current cwd is inside this repo."""

    candidate = Path.cwd() if start_path is None else start_path.resolve()
    root_path = find_repo_root(candidate)
    if root_path is None:
        return

    apply_quality_gate_env(root_path)


def find_repo_root(start_path: Path) -> Path | None:
    """Find the repository root by walking upward from ``start_path``."""

    search_path = start_path if start_path.is_dir() else start_path.parent
    for candidate in (search_path, *search_path.parents):
        if _matches_root_markers(candidate, REPO_ROOT_MARKERS) or _matches_root_markers(
            candidate, FIXTURE_ROOT_MARKERS
        ):
            return candidate
    return None


def apply_quality_gate_env(
    root_path: Path,
    *,
    env: MutableMapping[str, str] | None = None,
    prefer_pytest_temp: bool = False,
) -> MutableMapping[str, str]:
    """Apply repo-local tempdir and cache defaults for quality-gate subprocesses."""

    target_env = os.environ if env is None else env
    ensure_repo_src_import_path(root_path, env=target_env)
    target_env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    target_env.setdefault("RUFF_NO_CACHE", "true")
    target_env.setdefault("MYPY_CACHE_DIR", os.devnull)

    if not has_usable_tempdir(target_env, root_path=root_path):
        repo_temp_dir = (
            ensure_repo_pytest_tempdir(root_path)
            if prefer_pytest_temp
            else ensure_repo_process_tempdir(root_path)
        )
        repo_temp = str(repo_temp_dir)
        for key in TEMP_ENV_KEYS:
            target_env[key] = repo_temp

    return target_env


def ensure_repo_src_import_path(
    root_path: Path,
    *,
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Prefer the current checkout's ``src/`` tree over stale editable installs."""

    src_path = str((root_path / "src").resolve())
    if src_path not in sys.path:
        insert_at = 1 if sys.path and not sys.path[0] else 0
        active_python_path = os.environ.get("PYTHONPATH")
        if active_python_path:
            explicit_entries = [entry for entry in active_python_path.split(os.pathsep) if entry]
            for entry in explicit_entries:
                if entry in sys.path[insert_at:]:
                    insert_at = max(insert_at, sys.path.index(entry) + 1)
        sys.path.insert(insert_at, src_path)

    if env is None:
        return os.environ

    target_env = env
    python_path = target_env.get("PYTHONPATH")

    if not python_path:
        target_env["PYTHONPATH"] = src_path
        return target_env

    entries = [entry for entry in python_path.split(os.pathsep) if entry and entry != src_path]
    entries.insert(0, src_path)
    target_env["PYTHONPATH"] = os.pathsep.join(entries)
    return target_env


def apply_pytest_isolation(
    root_path: Path,
    *,
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Disable ambient pytest plugins and pin a repo-managed pytest temp dir."""

    target_env = apply_quality_gate_env(root_path, env=env, prefer_pytest_temp=True)
    repo_temp = str(ensure_repo_pytest_tempdir(root_path))
    for key in TEMP_ENV_KEYS:
        target_env[key] = repo_temp
    return target_env


def ensure_repo_process_tempdir(root_path: Path) -> Path:
    """Create and return the repo-managed process temp directory."""

    repo_temp_dir = root_path / REPO_PROCESS_TEMP_SUBDIR
    repo_temp_dir.mkdir(parents=True, exist_ok=True)
    return repo_temp_dir


def ensure_repo_pytest_tempdir(root_path: Path) -> Path:
    """Create and return the repo-managed pytest temp directory."""

    repo_temp_dir = root_path / REPO_PYTEST_TEMP_SUBDIR
    repo_temp_dir.mkdir(parents=True, exist_ok=True)
    return repo_temp_dir


def has_usable_tempdir(
    env: MutableMapping[str, str],
    *,
    root_path: Path | None = None,
) -> bool:
    """Return whether the current tempdir environment points to a writable dir."""

    for key in TEMP_ENV_KEYS:
        value = env.get(key)
        if not value:
            continue
        temp_path = Path(value).expanduser()
        if not _is_writable_directory(temp_path):
            continue
        if root_path is not None and not _is_relative_to(temp_path.resolve(), root_path.resolve()):
            continue
        return True
    return False


def tempdir_matches_root(root_path: Path, env: MutableMapping[str, str]) -> bool:
    """Return whether a configured tempdir already points inside ``root_path``."""
    return has_usable_tempdir(env, root_path=root_path)


def _matches_root_markers(candidate: Path, markers: tuple[str, ...]) -> bool:
    return all((candidate / marker).exists() for marker in markers)


def _is_writable_directory(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


def _is_relative_to(path: Path, root_path: Path) -> bool:
    try:
        path.relative_to(root_path)
    except ValueError:
        return False
    return True
