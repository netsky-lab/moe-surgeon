"""Repo-root startup bootstrap for commands launched from this checkout."""

from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = str((_REPO_ROOT / "src").resolve())
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from moe_surgeon.test_env import bootstrap_repo_quality_gate_env


bootstrap_repo_quality_gate_env(_REPO_ROOT)
