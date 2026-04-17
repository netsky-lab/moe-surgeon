# Architecture

## Design goals

- Reliable model analysis and pruning decisions.
- Deterministic, reproducible outputs.
- Strong safety guardrails around checkpoint writes.
- Extensibility to other MoE families via backend adapters.

## Package structure

moe_surgeon/
  - __init__.py
  - __main__.py
  - schemas.py
  - cli/
    - __init__.py
    - main.py
  - models/
    - __init__.py
    - backend.py
    - registry.py
    - errors.py
    - gemma4.py
  - analysis/
    - __init__.py
    - scan.py
    - metrics.py (future)
  - runtime/
    - __init__.py
    - profiler.py
  - prune/
    - __init__.py
    - strategies.py
    - planner.py
    - apply.py (future)
  - export/
    - __init__.py
    - safetensors_writer.py (future)
    - manifest.py

## Core contracts

### ModelBackend protocol

The backend contract exposes:

- supports(BackendSignature) -> bool
- load(...) -> LoadedBackendBundle
- iter_layers(...) / extract_topology(...) -> ordered LayerTopology sequence
- extract_router_state(...) -> RouterState
- extract_expert_state(...) -> tensor metadata mapping
- validate_bundle(...) / validate_layer(...) -> fail-fast invariants

A registry resolves one backend per model/config signature with deterministic
priority ordering, duplicate-name protection, and explicit unsupported or
ambiguous-match diagnostics.

Gemma 4 discovery is config-first: lightweight signatures route to the backend
without importing ML runtimes, then runtime loading performs explicit
Transformers capability checks before attempting model construction.

### Strategy pattern in prune

Pruning uses a pure strategy layer first, followed by a planner:

- prune/strategies.py: registry-backed strategies emit PruneCandidate lists.
- prune/planner.py: applies constraints and budgets to emit PrunePlan.

Mutations are performed in a later apply layer after planning.

## Data flow

1. scan: discover topology and static routing scores.
2. bench: capture live routing into ActivationStats.
   Runtime profiling is backend-driven and offline-safe: hooks attach only to
   backend-resolved router modules, aggregation is mask-aware, and cleanup runs
   in `finally`/context-manager exit paths.
3. prune: merge signals and create PrunePlan.
4. apply: remap tensors and validate invariants.
5. export: write deterministic outputs and manifests.

Analysis and runtime modules never mutate weights.

## Design decisions

1. Schema-first: all modules exchange typed contracts for decoupling and reproducibility.
2. Adapter isolation: model family details are contained in models adapters only.
3. Immutable transforms: pruning creates derived artifacts, never edits source files.
4. Canonical ordering: explicit tie-breakers for all ranking operations.
5. Runtime plus static fusion: combine static and live routing signals.
6. Lightweight bootstrap: package import and help output must not import heavy model/runtime dependencies.
