# Changelog

## Unreleased

- Added CLI version smoke coverage in `tests/test_cli.py` for both
  `python -m moe_surgeon --version` and the installed `moe-surgeon --version`
  entrypoint by running each path in a fresh process and asserting the version
  path does not import `torch`, `transformers`, or `safetensors`.
- Replaced the installed-supervisor collector regression with a repo-config
  integration test in `tests/test_repo_metrics.py` that resolves this repo's
  `.supervisor/project.json` through the live collector path and fails if the
  resolved checks ever drop `typecheck`.
- Replayed the supervisor task-metrics persistence path for task
  `97333f84-e4ed-4f1a-b3d5-0930408fa389` after the verify-config fix so the
  stored authoritative task record and `.supervisor/logs/task-97333f84.log`
  now report `lint`, `typecheck`, and `test_suite` with a `3/3` summary.
- Reproduced the authoritative supervisor collector path that previously
  reported `2/2` for this repo, confirmed the root cause was a stale persisted
  supervisor `verifyConfig` with `typeCheckCommand: null`, and added regression
  coverage in `tests/test_repo_metrics.py` for both the repo-local `3/3`
  fallback path and the stale persisted-config shadowing path.
- Added `src/moe_surgeon/repo_metrics.py`, a lightweight repo-owned metrics
  collector that reads `.supervisor/project.json`, runs the configured checks
  as named `lint`, `typecheck`, and `tests` metrics, and writes a canonical
  JSON artifact instead of depending on stale supervisor logs alone.
- Updated `package.json`, `.supervisor/project.json`, and
  `.github/workflows/metrics.yml` so package scripts, supervisor verification,
  and CI all execute through the same repo metrics collector entrypoint.
- Expanded `tests/test_repo_metrics.py` to verify the collector-backed
  supervisor wiring, emitted metrics artifacts, and single-check collector mode.
- Updated `README.md` to describe the repo-owned metrics collector and its
  machine-readable output.
- Implemented the Gemma 4 backend adapter in `src/moe_surgeon/models/gemma4.py`
  with strict config validation, deterministic MoE layer discovery, required
  tensor-key checks, and router/expert tensor diagnostics for the documented
  `router.proj`, `router.scale`, `router.per_expert_scale`,
  `experts.gate_up_proj`, and `experts.down_proj` families.
- Added lazy default backend registration helpers in
  `src/moe_surgeon/models/backend.py` so Gemma 4 can be resolved from either a
  lightweight config mapping or an explicit `BackendSignature` without making
  `moe_surgeon.models.backend` import heavy runtime dependencies.
- Hardened model-domain diagnostic formatting for sequence-valued details in
  `src/moe_surgeon/models/errors.py`.
- Added focused Gemma 4 regression coverage in `tests/test_models_gemma4.py`
  covering lightweight imports, backend support matching, deterministic layer
  ordering, missing-key diagnostics, synthetic metadata capture, and the
  explicit unsupported-Transformers runtime guard for local `transformers 4.51.3`.
- Implemented explicit Gemma 4 MoE layer traversal in `src/moe_surgeon/models/gemma4.py`, including deterministic ordered MoE layer enumeration, config-vs-state tensor-key discovery, and fail-fast diagnostics for unexpected or incomplete MoE layer key sets.
- Expanded `tests/test_models_gemma4.py` with offline regression coverage for ordered MoE key traversal, non-MoE layer rejection, and unexpected layer-prefix mismatch handling.
- Hardened `src/moe_surgeon/models/gemma4.py` to require Gemma4 hybrid decoder-layer companion tensors (`mlp.*` and feedforward norms) alongside router/expert tensors during topology validation and to enforce exact expert tensor layouts from `moe_intermediate_size`.
- Extended `tests/test_models_gemma4.py` so synthetic Gemma4 layers model the published hybrid topology and regressions cover missing dense hybrid keys, wrong `moe_intermediate_size`, and invalid expert tensor rank failures.

## 2026-04-17
- Registered the Gemma4 backend through a canonical default-registry entry in
  `src/moe_surgeon/models/gemma4.py` and updated
  `src/moe_surgeon/models/backend.py` to consume default backend entries in a
  deterministic priority/name order.
- Expanded Gemma4 backend tests to cover the default registry entry metadata
  and resolution from lightweight explicit `BackendSignature` inputs alongside
  config mappings.
- Restored the `moe_surgeon.models.backend.BackendRegistry` compatibility
  import as a lazy export so the legacy import path still works without
  recreating the `backend.py`/`registry.py` circular import or adding import
  weight to plain `import moe_surgeon.models.backend`.
- Added fresh-process regression coverage for `moe_surgeon.models.backend`,
  `moe_surgeon.models.registry`, and the compatibility `BackendRegistry` import
  path while asserting `torch`, `transformers`, and `safetensors` stay unloaded.
- Tightened `models/backend.py` and `models/registry.py` so backend resolution
  accepts either lightweight config mappings or explicit signatures while
  preserving deterministic priority/name ordering and explicit compatibility
  failures.
- Expanded `models/errors.py` diagnostic helpers so model, layer, tensor, and
  shape context is normalized through one shared domain-error path.
- Added focused backend-registry tests for plain-config resolution, invalid
  compatibility responses, and priority validation.
- Added lightweight backend contracts in `src/moe_surgeon/models/backend.py`
  with `BackendSignature`, `LoadedBackendBundle`, `TensorMetadata`, and the
  `ModelBackend` protocol aligned to shared schema dataclasses.
- Added deterministic backend registration and resolution in
  `src/moe_surgeon/models/registry.py`, including duplicate-name checks and
  explicit unsupported/ambiguous backend diagnostics.
- Centralized model-domain errors in `src/moe_surgeon/models/errors.py`,
  preserved schema-compatible imports for topology/shape violations, and added
  focused backend registry/error regression tests.
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
- Hardened the executable module entrypoint in `src/moe_surgeon/__main__.py` by exposing a stable module-level `main()` wrapper that delegates to the lightweight Click bootstrap with the canonical `python -m moe_surgeon` program name.
- Added CLI regression coverage in `tests/test_cli.py` for importing the module entrypoint directly and invoking its help path without pulling in heavy runtime dependencies.
- Completed P2 package bootstrap wiring in `pyproject.toml`, `src/moe_surgeon/cli/main.py`, and `src/moe_surgeon/__main__.py` with an installable `moe-surgeon` script, module execution support, and placeholder Click subcommands (`scan`, `bench`, `prune`, `export`).
- Added CLI regression tests in `tests/test_cli.py` covering `python -m moe_surgeon --help` and a lightweight help path that avoids importing `torch`, `transformers`, and `safetensors`.
- Updated README and architecture notes to document the lightweight bootstrap CLI behavior and entrypoint usage.
- Completed P1 canonical schema implementation in `src/moe_surgeon/schemas.py` with typed dataclasses, canonical JSON round-trip helpers, deterministic ordering, and validation/invariant checks.
- Added schema-focused regression tests in `tests/test_schemas.py` for importability, deterministic sort behavior, and JSON compatibility.
- Reworked canonical contract layer in `src/moe_surgeon/schemas.py` for explicit tie-safe comparators (`sort_experts`, `sort_plan_items`, `sort_topology`), epsilon-bucketed numeric ordering, strict shape/invariant validation, and deterministic JSON metadata envelopes.
- Added additional schema regression tests for tie-breaking and nested manifest round-trip (`tests/test_schemas.py`).
