# Changelog

## Unreleased

- Added repository planning documentation for Python-first Moe-surgeon implementation.
- Introduced `AGENTS.md` with collaboration and safety conventions.
- Added `ROADMAP.md` backlog with priority-ordered tasks:
  1) bootstrap, 2) model loader, 3) router analyzer, 4) runtime profiler,
  5) pruner, 6) exporter, 7) CLI commands, 8) tests and docs.
- Added `ARCHITECTURE.md` with proposed module boundaries and data flow.
- Refined canonical schema contracts in `src/moe_surgeon/schemas.py` with stricter
  shape diagnostics, stable topology reference handling, and additional
  deterministic validation/tests for P1 ordering/contracts.
- Tightened the bootstrap CLI entrypoint so the installed `moe-surgeon` script
  calls an explicit lightweight `main()` wrapper around the Click group while
  preserving `python -m moe_surgeon` help-only execution.
- Expanded CLI regression coverage for package metadata exposure, wrapper-based
  help rendering, and heavy dependency avoidance on the help path.

## 2026-04-17
- Completed P2 package bootstrap wiring in `pyproject.toml`, `src/moe_surgeon/cli/main.py`, and `src/moe_surgeon/__main__.py` with an installable `moe-surgeon` script, module execution support, and placeholder Click subcommands (`scan`, `bench`, `prune`, `export`).
- Added CLI regression tests in `tests/test_cli.py` covering `python -m moe_surgeon --help` and a lightweight help path that avoids importing `torch`, `transformers`, and `safetensors`.
- Updated README and architecture notes to document the lightweight bootstrap CLI behavior and entrypoint usage.
- Completed P1 canonical schema implementation in `src/moe_surgeon/schemas.py` with typed dataclasses, canonical JSON round-trip helpers, deterministic ordering, and validation/invariant checks.
- Added schema-focused regression tests in `tests/test_schemas.py` for importability, deterministic sort behavior, and JSON compatibility.
- Reworked canonical contract layer in `src/moe_surgeon/schemas.py` for explicit tie-safe comparators (`sort_experts`, `sort_plan_items`, `sort_topology`), epsilon-bucketed numeric ordering, strict shape/invariant validation, and deterministic JSON metadata envelopes.
- Added additional schema regression tests for tie-breaking and nested manifest round-trip (`tests/test_schemas.py`).
