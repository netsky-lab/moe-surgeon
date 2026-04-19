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
    - checkpoints.py
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
    - apply.py
  - export/
    - __init__.py
    - safetensors_writer.py
    - runner.py
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

The `models/checkpoints.py` helper provides offline-local `safetensors`
checkpoint introspection for single-file and indexed sharded layouts, exposing
deterministic `state_keys` and targeted tensor reads without importing
`transformers` or materializing a full model.
Static scan consumes that same reader for local-checkpoint analysis: topology
discovery is driven from checkpoint key metadata and per-layer router metrics
load only the router tensors required for scoring.

### Strategy pattern in prune

Pruning uses a pure strategy layer first, followed by a planner:

- prune/strategies.py: registry-backed strategies emit PruneCandidate lists.
- prune/planner.py: applies constraints and budgets to emit PrunePlan.
- Planner traceability is deterministic: canonicalized constraints, resolved
  budget bounds, source run identity, and candidate digests are embedded in the
  plan payload so repeated identical inputs produce byte-stable JSON.

Mutations are performed in a later apply layer after planning.
The apply layer consumes validated `PrunePlan` items plus checkpoint/topology
context, computes deterministic expert remaps, rewrites only the targeted MoE
tensors into a derived checkpoint tree, and revalidates the remapped tensor
layouts before any output is materialized.

## Data flow

1. scan: discover topology and static routing scores.
   The CLI persists the scan payload plus a sidecar run manifest and treats the
   scan artifact as the canonical topology snapshot for downstream steps.
2. bench: capture live routing into ActivationStats.
   Runtime profiling is backend-driven and offline-safe: hooks attach only to
   backend-resolved router modules, aggregation is mask-aware, and cleanup runs
   in `finally`/context-manager exit paths.
   Bench preflight validates the runtime-loaded topology against the persisted
   scan artifact before any prompt execution.
3. prune: merge persisted scan + bench signals and create PrunePlan.
   The CLI writes `prune-plan.json` at the prune root and materializes the
   derived checkpoint under `applied-checkpoint/` so export can reuse apply
   results directly.
4. apply: remap tensors and validate invariants.
5. export: write deterministic outputs, compatibility metadata, and manifests.

Analysis and runtime modules never mutate weights.

## Design decisions

1. Schema-first: all modules exchange typed contracts for decoupling and reproducibility.
2. Adapter isolation: model family details are contained in models adapters only.
3. Immutable transforms: pruning creates derived artifacts, never edits source files.
4. Canonical ordering: explicit tie-breakers for all ranking operations.
5. Runtime plus static fusion: combine static and live routing signals.
6. Lightweight bootstrap: package import and help output must not import heavy model/runtime dependencies.
