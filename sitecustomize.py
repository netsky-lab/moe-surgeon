"""Repo-root startup bootstrap for commands launched from this checkout."""

from __future__ import annotations

import os
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = str((_REPO_ROOT / "src").resolve())


def _chain_pythonpath_sitecustomize() -> None:
    python_path = os.environ.get("PYTHONPATH")
    if not python_path:
        return

    current_file = Path(__file__).resolve()
    for entry in python_path.split(os.pathsep):
        if not entry:
            continue
        candidate = (Path(entry) / "sitecustomize.py").resolve()
        if candidate == current_file or not candidate.is_file():
            continue
        namespace = {
            "__file__": str(candidate),
            "__name__": "_moe_surgeon_chained_sitecustomize",
        }
        exec(candidate.read_text(encoding="utf-8"), namespace)
        return


_chain_pythonpath_sitecustomize()
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from moe_surgeon.test_env import bootstrap_repo_quality_gate_env


bootstrap_repo_quality_gate_env(_REPO_ROOT)
