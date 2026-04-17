# moe-surgeon

moe-surgeon is a Python-first CLI for analyzing and pruning Mixture-of-Experts (MoE) models with reproducible, auditable behavior.

## Mission

Identify low-utility experts and produce smaller checkpoints without editing source weights in place.

## Primary target

- Model: Gemma 4 26B-A4B
- Architecture: Gemma4ForConditionalGeneration
- Text transformer layers: 30
- Experts per MoE layer: 128
- Top-k experts per token: 8

## Research-backed architecture notes

### Gemma4 topology

The target checkpoint uses text_config and a MoE-enabled decoder stack with:

- num_hidden_layers: 30
- enable_moe_block: true
- num_experts: 128
- top_k_experts: 8
- moe_intermediate_size: 704
- hidden_size: 2816
- intermediate_size: 2112

Gemma4 text layers are hybrid: dense FFN modules are present and MoE experts are integrated alongside routing paths.

### Router mechanism

Behavior is a top-k soft routing pattern:

1. Project token hidden states to expert logits.
2. Apply normalization and scale.
3. Softmax over experts.
4. Keep top_k_experts outputs.
5. Reweight and apply per_expert_scale.

### Key checkpoint key families

Observed key families for the Gemma4 language stack:

- model.language_model.layers.{L}.router.proj.weight
- model.language_model.layers.{L}.router.scale
- model.language_model.layers.{L}.router.per_expert_scale
- model.language_model.layers.{L}.experts.gate_up_proj
- model.language_model.layers.{L}.experts.down_proj

This pattern supports deterministic topology discovery and safe remapping.

## Canonical data schema

All modules use shared dataclasses in src/moe_surgeon/schemas.py before any tensor mutation:

- ModelHandle
- LayerTopology
- RouterState
- ExpertStats
- ActivationStats
- PruneCandidate
- PrunePlanItem
- PrunePlan
- RunArtifactManifest

All ranking is deterministic with tie-breakers on score, secondary metric, and expert index.

## Module structure

- cli/: command graph and orchestration (scan, bench, prune, export)
- models/: backend adapters and topology/contracts
- analysis/: static router analysis
- runtime/: forward hook profiler
- prune/: strategy and plan generation (selection only)
- export/: artifact persistence and manifest output

## Safety and reproducibility policy

- Never mutate source checkpoints.
- Write all outputs to a new artifact path.
- Validate expert index mapping before and after prune operations.
- Deterministic sorting and stable serialization for all manifests.
- Strict shape diagnostics before write-back.

## CLI bootstrap

- Install in editable mode with `python -m pip install -e .`.
- Gemma4 execution requires `transformers>=5.5.0`. Hugging Face documents
  Gemma4 support as added on 2026-04-01, and the first PyPI release carrying
  that support is `transformers 5.5.0` from 2026-04-02.
- Run help with `python -m moe_surgeon --help` or the installed `moe-surgeon --help` script.
- The bootstrap CLI exposes placeholder `scan`, `bench`, `prune`, and `export` commands without importing model/runtime backends during help parsing.

## Quality commands

- Run `npm run lint` for the Ruff gate defined in `AGENTS.md`.
- Run `npm run typecheck` for the `python -m mypy src` gate.
- Run `npm test` for the `python -m pytest` suite.
- `.supervisor/project.json` routes supervisor checks through the repo metrics
  collector and stores the underlying raw lint, typecheck, and test commands in
  one place.
- Run `npm run metrics` to execute that config through the repo-owned collector
  and emit a machine-readable summary of named `lint`, `typecheck`, and `tests`
  checks.
