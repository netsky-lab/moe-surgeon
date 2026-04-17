# Changelog

## Unreleased

- Preserved the pinned `tiny-random/gemma-4-moe` snapshot revision in the live Gemma4 profiler integration helper, strengthened generation-path assertions so live forward and generation captures both validate the full router output contract, updated Gemma4 router-scale validation to accept the hidden-size vector shape used by live Transformers Gemma4 routers, relaxed runtime aggregation to accept signed finite router weights emitted after learned per-expert scaling, centralized the backend-owned Gemma4 runtime contract so both offline and live checks share the same `transformers>=5.5.0` floor, support date, and remediation guidance, tightened the offline Gemma4 backend tests so both unsupported-runtime branches assert the same canonical diagnostics and guidance text, and rewired the live Gemma4 profiler gate to call that same shared backend contract instead of maintaining its own version/symbol skip logic.
- Added offline-safe runtime profiling utilities in
  `src/moe_surgeon/runtime/profiler.py` and `src/moe_surgeon/runtime/bench.py`,
  including context-managed router hook attach/detach, backend-driven router
  module resolution, mask-aware activation aggregation, and deterministic bench
  artifact manifest generation keyed by prompt/config digests instead of
  timestamps.
- Exported runtime profiler entrypoints from `src/moe_surgeon/runtime/__init__.py`
  and added activation/topology ordering helpers in
  `src/moe_surgeon/analysis/scan.py`.
- Extended the lightweight CLI `bench` placeholder with prompt batching and
  profiler-option parsing in `src/moe_surgeon/cli/main.py`.
- Extended `src/moe_surgeon/runtime/profiler.py` with deterministic prompt
  batching helpers so bench flows can derive canonical attention-mask-aware
  prompt batches without requiring live Gemma4 inference.
- Tightened runtime activation aggregation to persist both unweighted
  active-token totals (`n_tokens`) and weighted layer totals
  (`weighted_n_tokens`) while continuing to mask out padding and
  sequence-prefix positions.
- Hardened `src/moe_surgeon/analysis/scan.py` validation so activation payloads
  fail fast when per-layer weighted or unweighted totals disagree across
  experts.
- Updated the bench CLI placeholder to count newline-delimited prompt files
  consistently with prompt batching.
- Added offline regression coverage for prompt batching, weighted layer totals,
  and scan-layer total consistency.
- Added offline regression coverage in `tests/test_runtime_profiler.py`,
  `tests/test_analysis_scan.py`, `tests/test_models_gemma4.py`, `tests/test_cli.py`,
  and `tests/test_schemas.py` for hook cleanup, aggregation semantics, topology
  alignment, deterministic manifest generation, and bench CLI option handling.
- Extended `tests/test_repo_metrics.py` with explicit missing-check regression
  coverage for absent `lintCommand` and `typeCheckCommand`, asserting
  `python -m moe_surgeon.repo_metrics --check <name>` fails with the named
  `Requested check ... is not configured` diagnostic.
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
- Added canonical static scan artifact helpers in `src/moe_surgeon/analysis/scan.py` for sorted JSON payload emission, timestamp-free content digests, and byte-identical repeated writes from identical inputs.
- Added timestamp-independent `RunArtifactManifest.canonical_digest` support in `src/moe_surgeon/schemas.py` and surfaced stable scan manifest metadata for `model_fingerprint`, `canonical_manifest_digest`, and `canonical_artifact_digest`.
- Updated `src/moe_surgeon/cli/main.py` scan placeholder text to point at the canonical scan artifact helpers and expanded regression coverage in `tests/test_analysis_scan.py` and `tests/test_schemas.py`.
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
## 2026-04-17
- Added static router metric helpers in `src/moe_surgeon/analysis/metrics.py` that upcast router math to stable float64, derive deterministic expert distributions from `router.proj.weight`, and compute per-expert mass, entropy, top-k proxy, and optional `router.per_expert_scale` norms without using deprecated PyTorch norm APIs.
- Added backend-driven static scan assembly in `src/moe_surgeon/analysis/scan.py` that reads ordered Gemma4 MoE layers from backend topology, loads router tensors strictly through `LayerTopology.module_paths`, emits one `RouterState` per MoE layer, and fails fast when only topology metadata is available without materialized numeric tensors.
- Added focused scan regression coverage in `tests/test_analysis_scan.py` for deterministic ranking, finite metrics, normalized per-layer mass, and the topology-only metadata failure path.

- Hardened `src/moe_surgeon/models/gemma4.py` to require Gemma4 hybrid decoder-layer companion tensors (`mlp.*` and feedforward norms) alongside router/expert tensors during topology validation and to enforce exact expert tensor layouts from `moe_intermediate_size`.
- Extended `tests/test_models_gemma4.py` so synthetic Gemma4 layers model the published hybrid topology and regressions cover missing dense hybrid keys, wrong `moe_intermediate_size`, and invalid expert tensor rank failures.

## 2026-04-17
- Extended `src/moe_surgeon/runtime/profiler.py` benchmark artifact output with
  explicit canonical `profiler_config` payloads, stable prompt/input checksum
  tracking, and direct JSON serialization helpers for deterministic bench
  artifacts that stay aligned with topology/activation ordering.
- Expanded `tests/test_runtime_profiler.py` with regression coverage for
  input-payload hashing, profiler-config hashing, and canonical JSON artifact
  writing.

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
- Added capability-gated live Gemma4 profiler coverage in `tests/test_runtime_profiler.py` using the pinned public MoE fixture `tiny-random/gemma-4-moe` to validate real router hook captures during both forward and generation paths once the local Transformers environment exposes Gemma4 support.
- The live profiler test currently skips on environments such as `transformers 4.51.3` where `transformers.models.gemma4` and `Gemma4ForConditionalGeneration` are not yet installed, preserving the existing offline/unit test baseline while automatically activating after the dependency upgrade.

## 2026-04-17
- Completed P5 static router metric hardening in `src/moe_surgeon/analysis/metrics.py` and `src/moe_surgeon/analysis/scan.py` by deriving deterministic softmax-based expert distributions, replacing unstable `topk` tie handling with stable sorting, and adding an aggregate scan summary over ordered MoE layers.
- Expanded `tests/test_analysis_scan.py` with deterministic tie-case coverage, aggregate-summary assertions, and scan regression checks for finite non-negative expert metrics.

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
