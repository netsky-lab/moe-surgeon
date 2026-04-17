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

## Safety and correctness rules

- Never mutate source checkpoint files in place.
- Always write outputs to a new directory.
- Fail fast on unsupported topology or shape mismatch.
- Preserve pre-prune metadata snapshots for auditability.
- Validate expert remaps before and after pruning; include detailed diagnostics.

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
- Optional: python -m pytest -k integration
