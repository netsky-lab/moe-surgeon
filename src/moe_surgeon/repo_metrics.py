"""Repo-owned metrics collector for supervisor verify commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


@dataclass(frozen=True)
class VerifyConfig:
    """Configured repo verification commands loaded from ``.supervisor/project.json``."""

    lint_command: str | None
    typecheck_command: str | None
    build_command: str | None
    test_command: str | None
    coverage_command: str | None
    browser_test_command: str | None


@dataclass(frozen=True)
class MetricCheck:
    """One executed verification check."""

    category: str
    name: str
    command: str
    exit_code: int
    passed: bool
    duration_ms: int
    output: str


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate pass/fail counts for a metrics run."""

    total: int
    passed: int
    failed: int


@dataclass(frozen=True)
class MetricsReport:
    """Machine-readable report emitted by the repo metrics collector."""

    task_id: str | None
    collected_at: str
    root_path: str
    checks: tuple[MetricCheck, ...]
    summary: MetricSummary


class MetricsConfigurationError(ValueError):
    """Raised when requested repo metrics checks are missing or invalid."""


def load_verify_config(root_path: Path) -> VerifyConfig:
    """Load repo metrics commands from the repo-local project config."""

    config_path = root_path / ".supervisor" / "project.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    metrics_config = payload.get("repoMetricsConfig", payload.get("verifyConfig"))
    if not isinstance(metrics_config, dict):
        raise ValueError("Missing repoMetricsConfig mapping in .supervisor/project.json")

    return VerifyConfig(
        lint_command=_optional_command(metrics_config.get("lintCommand")),
        typecheck_command=_optional_command(metrics_config.get("typeCheckCommand")),
        build_command=_optional_command(metrics_config.get("buildCommand")),
        test_command=_optional_command(metrics_config.get("testCommand")),
        coverage_command=_optional_command(metrics_config.get("coverageCommand")),
        browser_test_command=_optional_command(metrics_config.get("browserTestCommand")),
    )


def collect_metrics(root_path: Path, *, timeout_seconds: int, selected_check: str | None = None) -> MetricsReport:
    """Run configured repo checks and return a deterministic metrics report."""

    config = load_verify_config(root_path)
    checks: list[MetricCheck] = []
    available_checks = _iter_checks(config)

    if selected_check is not None and all(name != selected_check for _, name, _ in available_checks):
        raise MetricsConfigurationError(f"Requested check '{selected_check}' is not configured in .supervisor/project.json")

    for category, name, command in available_checks:
        if selected_check is not None and name != selected_check:
            continue
        checks.append(_run_check(root_path, category=category, name=name, command=command, timeout_seconds=timeout_seconds))

    passed = sum(1 for check in checks if check.passed)
    return MetricsReport(
        task_id=None,
        collected_at=_timestamp(),
        root_path=str(root_path),
        checks=tuple(checks),
        summary=MetricSummary(total=len(checks), passed=passed, failed=len(checks) - passed),
    )


def write_report(report: MetricsReport, output_path: Path | None) -> None:
    """Write the metrics report to disk or stdout as canonical JSON."""

    document = json.dumps(asdict(report), indent=2, sort_keys=True)
    if output_path is None:
        sys.stdout.write(f"{document}\n")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{document}\n", encoding="utf-8")


def append_task_log(root_path: Path, *, task_id: str, report: MetricsReport, artifact_path: Path | None) -> None:
    """Append a fresh summary line for the task to the supervisor log."""

    short_id = task_id.split("-", maxsplit=1)[0]
    log_path = root_path / ".supervisor" / "logs" / f"task-{short_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = "info" if report.summary.failed == 0 else "warning"
    status = "success" if report.summary.failed == 0 else "warning"
    log_entries = [
        {
            "timestamp": _timestamp(),
            "phase": "execution",
            "level": level,
            "message": f"Phase ended: execution ({status}) — Metrics: {report.summary.passed}/{report.summary.total} passed",
        }
    ]
    if artifact_path is not None:
        log_entries.append(
            {
                "timestamp": _timestamp(),
                "phase": "execution",
                "level": "info",
                "message": f"Repo metrics artifact: {artifact_path.relative_to(root_path)}",
            }
        )

    with log_path.open("a", encoding="utf-8") as handle:
        for entry in log_entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the repo metrics collector CLI."""

    parser = argparse.ArgumentParser(description="Run repo verify checks and emit a machine-readable metrics report.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root path. Defaults to the current working directory.")
    parser.add_argument("--task-id", help="Optional supervisor task id used to choose default artifact and log paths.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON metrics artifact.")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Per-check timeout in seconds.")
    parser.add_argument(
        "--check",
        choices=("lint", "typecheck", "build", "tests", "coverage", "browser_tests"),
        help="Optional single check to run via the collector entrypoint.",
    )
    args = parser.parse_args(argv)

    root_path = args.root.resolve()
    try:
        report = collect_metrics(root_path, timeout_seconds=args.timeout_seconds, selected_check=args.check)
    except MetricsConfigurationError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    final_report = MetricsReport(
        task_id=args.task_id,
        collected_at=report.collected_at,
        root_path=report.root_path,
        checks=report.checks,
        summary=report.summary,
    )
    output_path = _resolve_output_path(root_path, task_id=args.task_id, output_path=args.output)
    write_report(final_report, output_path)
    if args.task_id:
        append_task_log(root_path, task_id=args.task_id, report=final_report, artifact_path=output_path)

    return 0 if final_report.summary.failed == 0 else 1


def _optional_command(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Verify commands must be strings or null")
    stripped = value.strip()
    return stripped or None


def _iter_checks(config: VerifyConfig) -> tuple[tuple[str, str, str], ...]:
    checks: list[tuple[str, str, str]] = []
    if config.lint_command:
        checks.append(("code_quality", "lint", config.lint_command))
    if config.typecheck_command:
        checks.append(("code_quality", "typecheck", config.typecheck_command))
    if config.build_command:
        checks.append(("code_quality", "build", config.build_command))
    if config.test_command:
        checks.append(("tests", "tests", config.test_command))
    if config.coverage_command:
        checks.append(("tests", "coverage", config.coverage_command))
    if config.browser_test_command:
        checks.append(("browser", "browser_tests", config.browser_test_command))
    return tuple(checks)


def _run_check(root_path: Path, *, category: str, name: str, command: str, timeout_seconds: int) -> MetricCheck:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        cwd=root_path,
        timeout=timeout_seconds,
    )
    completed_at = datetime.now(timezone.utc)
    output = (result.stdout + result.stderr).strip()
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
    return MetricCheck(
        category=category,
        name=name,
        command=command,
        exit_code=result.returncode,
        passed=result.returncode == 0,
        duration_ms=duration_ms,
        output=output,
    )


def _resolve_output_path(root_path: Path, *, task_id: str | None, output_path: Path | None) -> Path | None:
    if output_path is not None:
        return output_path.resolve()
    if task_id is None:
        return None
    short_id = task_id.split("-", maxsplit=1)[0]
    return root_path / ".supervisor" / "logs" / f"task-{short_id}.metrics.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
