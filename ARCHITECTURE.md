# Architecture

## Design goals

- Reliable model analysis and pruning decisions.
- Deterministic, reproducible outputs.
- Strong safety guardrails around checkpoint writes.
- Extensibility to other MoE families via backend adapters.

## Package structure

moe_surgeon/
  - __init__.py
  - schemas.py
  - cli/
    - __init__.py
    - main.py (future)
  - models/
    - __init__.py
    - backend.py
    - registry.py
    - errors.py
    - gemma4.py (future)
  - analysis/
    - __init__.py
    - scan.py (future)
    - metrics.py (future)
  - runtime/
    - __init__.py
    - profiler.py (future)
  - prune/
    - __init__.py
    - strategy.py
    - planner.py
    - apply.py (future)
  - export/
    - __init__.py
    - safetensors_writer.py (future)
    - manifest.py

## Core contracts

### ModelBackend protocol

The backend contract exposes:

- supports(model_or_config) -> bool
- load_model(...) -> loaded model handle
- iter_moe_layers(...) -> iterable of LayerTopology
- extract_router_state(...) -> RouterState
- extract_expert_state(...) -> expert tensor mapping
- validate_layer_mapping(...) -> invariants report

A registry resolves one backend per model and reports missing or ambiguous matches clearly.

### Strategy pattern in prune

Pruning uses a pure strategy layer first, followed by a planner:

- prune/strategy.py: each strategy emits PruneCandidate lists.
- prune/planner.py: applies constraints and budgets to emit PrunePlan.

Mutations are performed in a later apply layer after planning.

## Data flow

1. scan: discover topology and static routing scores.
2. bench: capture live routing into ActivationStats.
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
