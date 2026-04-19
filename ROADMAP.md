# Roadmap

## Mission

- Deliver a deterministic, auditable CLI for MoE analysis and pruning focused on Gemma 4 26B-A4B first.
- Preserve safety guarantees: no source checkpoint mutation, canonical ordering, strict topology validation, and explicit diagnostics.
- Keep architecture extensible to additional MoE backends.

## Backlog (implementation tasks)

1. P1 — Define canonical data schemas and ordering contracts (Priority 1)
Files: `src/moe_surgeon/schemas.py`
Description: Finalize `ModelHandle`, `LayerTopology`, `RouterState`, `ExpertStats`, `ActivationStats`, `PruneCandidate`, `PrunePlanItem`, `PrunePlan`, and `RunArtifactManifest` with explicit schema version, deterministic sort keys, stable identity fields, and invariant validation helpers.
Acceptance criteria: import is lightweight with no `torch`/`transformers` imports; comparator ordering is deterministic and documented; JSON canonicalization is reversible; invalid indices, negative counts, or malformed layer references raise clear domain errors.

2. P2 — Create Python package skeleton and CLI bootstrap (Priority 2)
Files: `pyproject.toml`, `src/moe_surgeon/__init__.py`, `src/moe_surgeon/cli/__init__.py`
Description: Set package metadata and dependency boundaries for Python-first execution, create stable package namespace directories (`analysis/`, `runtime/`, `prune/`, `export/`, `models/`, `cli/`), and keep non-runtime imports side-effect free.
Acceptance criteria: package metadata includes `torch`, `transformers`, `safetensors`, `click`; `python -m moe_surgeon` imports cleanly; package directories resolve as modules without runtime side effects.

3. P3 — Define backend protocol, registry, and errors (Priority 3)
Files: `src/moe_surgeon/models/backend.py`, `src/moe_surgeon/models/registry.py`, `src/moe_surgeon/models/errors.py`
Description: Introduce `ModelBackend` protocol, deterministic backend registration/selection contract, and explicit domain errors (`UnsupportedModelError`, `BackendMismatchError`, `TopologyMismatchError`, `ShapeInvariantViolationError`) used across scan/bench/prune/export.
Acceptance criteria: resolver returns exactly one backend for valid Gemma4 signatures; resolver fails fast with actionable messaging for no-match and ambiguous-match conditions; protocol methods cover all required operations for scanning, profiling, planning, and exporting.

4. P4 — Implement Gemma4 backend and topology loader (Priority 4)
Files: `src/moe_surgeon/models/gemma4.py`
Description: Implement adapter for `Gemma4ForConditionalGeneration`, model loading/tokenizer metadata capture, MoE layer discovery, and topology validation against expected config and tensor key patterns used by Gemma4 (`router.proj`, `router.scale`, `router.per_expert_scale`, `experts.gate_up_proj`, `experts.down_proj`).
Acceptance criteria: recognises Gemma4 MoE checkpoints, captures topology fields (`num_layers`, `num_experts`, `top_k`), validates dense+MoE hybrid invariants, and returns deterministic per-layer expert ordering.

5. P5 — Implement static router analyzer (`scan`) (Priority 5)
Files: `src/moe_surgeon/analysis/scan.py`, `src/moe_surgeon/analysis/metrics.py`
Description: Read static router weight state and route logits to produce per-expert `ExpertStats` per layer (`gate_mass`, entropy proxy, activation confidence, raw counts) that drives deterministic ranking baseline.
Acceptance criteria: outputs are deterministic and layer-order stable; per-layer `gate_mass` normalizes to expected sum tolerance; produced `ExpertStats` maps one-to-one with backend `LayerTopology` and serializes to `RunArtifactManifest`.

6. P6 — Implement runtime profiler (`bench`) (Priority 6)
Files: `src/moe_surgeon/runtime/profiler.py`
Description: Add router hook instrumentation capturing `top_k_indices`, `top_k_weights`, and optional logits during real token flows with sequence/pad masking, then aggregate into `ActivationStats`.
Acceptance criteria: only non-pad tokens contribute; output is reproducible with fixed seed and same prompt set; profiler results preserve the same layer ordering and expert indexing as `LayerTopology`.

7. P7 — Implement pruning strategies and planner (Priority 7)
Files: `src/moe_surgeon/prune/strategies.py`, `src/moe_surgeon/prune/planner.py`
Description: Introduce a pluggable strategy registry and implementations (`frequency`, `router_mass`, `combined`) and produce canonical `PrunePlan` with per-layer keep/drop sets under user constraints and minimum-expert rules.
Acceptance criteria: same input always produces identical JSON output; invalid budgets and constraints fail fast with actionable validation messages; output plan includes traceability metadata (`source_run_id`, threshold parameters, computed budgets).

8. P8 — Implement prune apply engine and remapping (Priority 8)
Files: `src/moe_surgeon/prune/apply.py`
Description: Apply `PrunePlan` by slicing and remapping expert tensors and router tensors (`gate_up_proj`, `down_proj`, `router.proj`, `router.per_expert_scale`) into a derived output tree while preserving all non-MoE model weights.
Acceptance criteria: source checkpoints are never mutated; dry-run reports planned index maps and tensor deltas only; post-apply shape and remap invariants are validated before writing artifacts.

9. P9 — Implement deterministic export pipeline (Priority 9)
Files: `src/moe_surgeon/export/safetensors_writer.py`, `src/moe_surgeon/export/manifest.py`
Description: Write pruned model in `safetensors` format, regenerate compatible config/index metadata, and persist manifest with canonical hash chain and schema revision.
Acceptance criteria: exported artifact can be reloaded by Transformers Gemma4-compatible path; repeated runs with same inputs are byte-stable; manifest includes source fingerprint, pre/pruned topology, and plan hash.

10. P10 — Wire CLI workflow and command orchestration (Priority 10)
Files: `src/moe_surgeon/cli/main.py`, `src/moe_surgeon/cli/commands/*`
Description: Implement `scan`, `bench`, `prune`, and `export` commands with shared context, schema persistence, fail-fast validation, and explicit output paths for artifacts.
Acceptance criteria: command dependency edges are enforced (`scan` output required for `bench`, scan/bench outputs required for `prune`, and prune manifest for `export`); invalid input manifests are rejected clearly; full run manifest is produced for each command.

11. P11 — Add tests, docs, and hardening (Priority 11)
Files: `tests/*`, `README.md`, `CHANGELOG.md`
Description: Add offline deterministic unit tests for schema invariants, backend discovery, scan metrics, profiler aggregation, planner validation, and apply/export contract; expand docs with safety/caveat guidance.
Acceptance criteria: validation-error branches are tested for each domain error; synthetic tiny-MoE fixtures are used for reproducibility; docs describe pruning trade-offs and expected failure modes.

## Implementation phase mapping

- Phase 0: P1, P2
- Phase 1: P3, P4
- Phase 2: P5
- Phase 3: P6
- Phase 4: P7, P8
- Phase 5: P9, P10
- Phase 6: P11

## Milestones

- Foundation (complete when P1-P2 done): schema/CLI bootstrap done.
- Backend + Scan (complete when P3-P5 done): topology discovery and static utilization outputs done.
- Runtime signal (complete when P6 done): reproducible activation profiler done.
- Planning (complete when P7 done): deterministic strategy and plan outputs done.
- Prune + Export (complete when P8-P10 done): end-to-end prunable/exportable workflow done.
- Hardening (complete when P11 done): test-backed, documented baseline ready.

## Implementation audit notes

- The lightweight local safetensors checkpoint reader was delivered in
  `99534567` (`src/moe_surgeon/models/checkpoints.py`,
  `tests/test_models_checkpoints.py`).
- Static scan's targeted router-only checkpoint reads were delivered in
  `e4ce49a9` (`src/moe_surgeon/analysis/scan.py`,
  `tests/test_analysis_scan.py`) on top of that reader.
- Repo tempdir/bootstrap fixes in `src/moe_surgeon/test_env.py` and
  `tests/test_repo_metrics.py` are quality-gate/test-environment hardening and
  should not be used as completion evidence for the checkpoint-reader or
  targeted static-scan backlog items.
- P11 hardening now depends on the tracked `tests/fixtures/tiny_gemma_like.py`
  scaffold plus offline CLI/export manifest regressions so scan, planner, and
  export contracts can be exercised without live model downloads.

## Execution notes and limitations

- Deterministic execution is a delivery constraint, not a best-effort goal:
  fixed seeds, canonical JSON, stable ranking ties, and reproducible fixtures
  must remain intact when new backends or strategies are added.
- No checkpoint mutation is permitted during apply/export execution notes,
  including temporary write-back shortcuts in tooling or tests.
- Execution order is part of the contract: `scan` establishes the topology
  snapshot, `bench` validates against it, `prune` records the canonical plan,
  and apply/export only consume derived artifacts from those prior stages.
- Safe fallback means explicit refusal on malformed manifests, backend
  mismatches, unsupported topologies, or tensor shape violations; roadmap
  completion does not include best-effort partial outputs.
- Known limitation: offline tests cover contract and manifest correctness, but
  they do not certify real-model throughput or routing quality; those remain in
  explicit integration coverage.
- Known limitation: current implementation and docs are intentionally optimized
  around Gemma 4 26B-A4B semantics, so additional MoE families require backend
  and validation extensions rather than inferred compatibility.
