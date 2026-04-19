# AGENTS.md

This repository is coordinated via Codex-style collaboration. All contributors and agents must follow this contract.

## Scope

- Build a Python-first CLI tool for MoE analysis and pruning.
- Primary target model is Gemma 4 26B-A4B.
- Default stack is PyTorch, Transformers, and safetensors.

## Required reading

Before touching architecture or implementation decisions, read:

- README.md
- AGENTS.md
- ROADMAP.md
- ARCHITECTURE.md
- CHANGELOG.md

## Conventions

- Python-first layout under src/.
- Typed signatures and docstrings for public functions.
- Dataclasses for structured outputs.
- Keep packages isolated by concern: cli/, models/, analysis/, runtime/, prune/, export/.
- No hidden mutation of model checkpoints.
- Deterministic behavior is mandatory: fixed seed support, stable sorting, canonical JSON serialization.
- Deterministic fixtures and stable tensor payloads are required for offline tests.
- Reproducibility claims must be testable: if a command or manifest is
  described as deterministic, add or update regression coverage that exercises
  that exact contract offline when feasible.

## Safety and correctness rules

- Never mutate source checkpoint files in place.
- No checkpoint mutation, including temporary in-place rewrites during tests or export staging.
- Always write outputs to a new directory.
- Fail fast on unsupported topology or shape mismatch.
- Preserve pre-prune metadata snapshots for auditability.
- Validate expert remaps before and after pruning; include detailed diagnostics.
- Safe fallback behavior means refusing unsupported or malformed inputs with explicit domain errors instead of silently skipping validation.
- Treat scan, bench, prune, apply, and export artifacts as immutable inputs once
  written; downstream stages should derive new outputs instead of patching
  prior artifacts in place.
- Shape validation is required both before tensor slicing/remap work and after
  remapped outputs are assembled.

## Operational notes

- The default quality bar is offline and deterministic: prefer tiny fixture
  backends, canonical manifests, and fixed-seed assertions over live-service
  dependence.
- Integration coverage is opt-in and should document the exact runtime/version
  expectation it relies on.
- When a safety check triggers, surface the specific domain error and context;
  do not substitute warnings, auto-repair, or best-effort continuation.

## Error design

Use explicit domain errors with context:
- unsupported model family
- backend mismatch
- topology mismatch
- mapping and shape invariant violations

## Quality gates

- python -m ruff check src tests
- python -m mypy src
- python -m pytest
- Optional: python -m pytest -m integration
