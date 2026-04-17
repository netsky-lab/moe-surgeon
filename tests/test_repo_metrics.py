from __future__ import annotations

import subprocess
import sys

from moe_surgeon import repo_metrics


def test_repo_metrics_runs_all_checks_by_default(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(repo_metrics.subprocess, "run", fake_run)

    result = repo_metrics.main([])

    assert result == 0
    assert calls == [
        (sys.executable, "-m", "ruff", "check", "src", "tests"),
        (sys.executable, "-m", "mypy", "src"),
        (sys.executable, "-m", "pytest", "tests"),
    ]


def test_repo_metrics_dispatches_named_checks(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(repo_metrics.subprocess, "run", fake_run)

    assert repo_metrics.main(["--check", "lint"]) == 0
    assert repo_metrics.main(["--check", "typecheck"]) == 0
    assert repo_metrics.main(["--check", "tests"]) == 0
    assert calls == [
        (sys.executable, "-m", "ruff", "check", "src", "tests"),
        (sys.executable, "-m", "mypy", "src"),
        (sys.executable, "-m", "pytest", "tests"),
    ]
