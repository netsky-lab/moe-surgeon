"""Editable-install startup bootstrap for hermetic quality-gate commands."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_TEMP_SUBDIR = Path(".tmp") / "system"
TEMP_ENV_KEYS = ("TMPDIR", "TMP", "TEMP")


def _bootstrap_quality_gate_env() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("RUFF_NO_CACHE", "true")

    if _has_usable_tempdir():
        return

    repo_temp_dir = REPO_ROOT / REPO_TEMP_SUBDIR
    repo_temp_dir.mkdir(parents=True, exist_ok=True)
    repo_temp = str(repo_temp_dir)
    for key in TEMP_ENV_KEYS:
        os.environ[key] = repo_temp


def _has_usable_tempdir() -> bool:
    for key in TEMP_ENV_KEYS:
        value = os.environ.get(key)
        if value and _is_writable_directory(Path(value).expanduser()):
            return True
    return False


def _is_writable_directory(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


_bootstrap_quality_gate_env()
