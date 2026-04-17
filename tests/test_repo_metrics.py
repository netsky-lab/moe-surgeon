from __future__ import annotations

import json
from pathlib import Path


def test_package_scripts_cover_repo_quality_gates() -> None:
    package_json = Path(__file__).resolve().parents[1] / "package.json"
    payload = json.loads(package_json.read_text(encoding="utf-8"))

    assert payload["scripts"]["lint"] == "python -m ruff check src tests"
    assert payload["scripts"]["typecheck"] == "python -m mypy src"
    assert payload["scripts"]["test"] == "python -m pytest"
    assert payload["scripts"]["metrics"] == "npm run lint && npm run typecheck && npm test"
