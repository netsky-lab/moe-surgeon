"""Repo-local helpers for deterministic pytest subprocess isolation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


REPO_PYTEST_TEMP_SUBDIR = Path(".tmp") / "pytest"


def apply_pytest_isolation(
    root_path: Path,
    *,
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Disable ambient pytest plugins and provide a repo-managed temp dir."""

    target_env = os.environ if env is None else env
    target_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    if not has_usable_tempdir(target_env):
        repo_temp_dir = ensure_repo_tempdir(root_path)
        repo_temp = str(repo_temp_dir)
        target_env["TMPDIR"] = repo_temp
        target_env["TMP"] = repo_temp
        target_env["TEMP"] = repo_temp

    return target_env


def ensure_repo_tempdir(root_path: Path) -> Path:
    """Create and return the repo-managed pytest temp directory."""

    repo_temp_dir = root_path / REPO_PYTEST_TEMP_SUBDIR
    repo_temp_dir.mkdir(parents=True, exist_ok=True)
    return repo_temp_dir


def has_usable_tempdir(env: MutableMapping[str, str]) -> bool:
    """Return whether the current tempdir environment points to a writable dir."""

    for key in ("TMPDIR", "TMP", "TEMP"):
        value = env.get(key)
        if value and _is_writable_directory(Path(value).expanduser()):
            return True
    return False


def _is_writable_directory(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False
