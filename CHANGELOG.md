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

## 2026-04-17
- Completed P1 canonical schema implementation in `src/moe_surgeon/schemas.py` with typed dataclasses, canonical JSON round-trip helpers, deterministic ordering, and validation/invariant checks.
- Added schema-focused regression tests in `tests/test_schemas.py` for importability, deterministic sort behavior, and JSON compatibility.
- Reworked canonical contract layer in `src/moe_surgeon/schemas.py` for explicit tie-safe comparators (`sort_experts`, `sort_plan_items`, `sort_topology`), epsilon-bucketed numeric ordering, strict shape/invariant validation, and deterministic JSON metadata envelopes.
- Added additional schema regression tests for tie-breaking and nested manifest round-trip (`tests/test_schemas.py`).
