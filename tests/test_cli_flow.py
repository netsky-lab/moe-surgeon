from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from moe_surgeon.analysis.scan import load_scan_artifact
from moe_surgeon.cli.main import cli
from moe_surgeon.models.backend import LoadedBackendBundle
from moe_surgeon.runtime.bench import load_benchmark_artifact
from moe_surgeon.schemas import LayerTopology, ModelHandle, RouterState

from tests.test_prune_apply import _write_checkpoint


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class _FakeHookHandle:
    module: "_FakeRouterModule"
    hook: object
    removed: bool = False

    def remove(self) -> None:
        if self.removed:
            return
        self.removed = True
        self.module.hooks.remove(self.hook)


@dataclass
class _FakeRouterModule:
    name: str
    hooks: list[object] = field(default_factory=list)

    def register_forward_hook(self, hook: object) -> _FakeHookHandle:
        self.hooks.append(hook)
        return _FakeHookHandle(module=self, hook=hook)

    def run(self, output: object) -> None:
        for hook in tuple(self.hooks):
            hook(self, (), output)


class _FakeTokenizer:
    def __call__(self, prompts: tuple[str, ...], **_: object) -> dict[str, object]:
        max_length = max(len(prompt) for prompt in prompts)
        attention_mask: list[list[int]] = []
        input_ids: list[list[int]] = []
        for prompt in prompts:
            token_count = len(prompt)
            padding = max_length - token_count
            attention_mask.append(([1] * token_count) + ([0] * padding))
            input_ids.append(list(range(token_count)) + ([0] * padding))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _router_state(layer: LayerTopology) -> RouterState:
    return RouterState(
        layer_index=layer.layer_index,
        num_experts=layer.expert_count,
        top_k=layer.top_k,
        logits_shape=(0, layer.expert_count),
        top_k_indices_shape=(1, layer.top_k),
        top_k_weights_shape=(1, layer.top_k),
        projection_shape=(layer.expert_count, layer.hidden_size),
        per_expert_scale_shape=(layer.expert_count,),
        has_raw_logits_capture=True,
        route_scale_present=True,
    )


class _FakeRuntimeModel:
    def __init__(self, modules: dict[int, _FakeRouterModule]) -> None:
        self._modules = modules

    def eval(self) -> "_FakeRuntimeModel":
        return self

    def __call__(self, **kwargs: object) -> object:
        attention_mask = kwargs["attention_mask"]
        assert isinstance(attention_mask, list)
        for layer_index in sorted(self._modules):
            top_k_indices = [[[0, 1] for _ in prompt_mask] for prompt_mask in attention_mask]
            top_k_weights = [[[0.7, 0.3] for _ in prompt_mask] for prompt_mask in attention_mask]
            router_scores = [
                [[0.7, 0.3, 0.0, 0.0] for _ in prompt_mask]
                for prompt_mask in attention_mask
            ]
            self._modules[layer_index].run(
                {
                    "top_k_indices": top_k_indices,
                    "top_k_weights": top_k_weights,
                    "router_scores": router_scores,
                }
            )
        return object()


class _FakeRuntimeBackend:
    def __init__(self, topology: tuple[LayerTopology, ...]) -> None:
        self._topology = topology
        self._router_states = {
            layer.layer_index: _router_state(layer)
            for layer in topology
        }
        self._modules = {
            layer.layer_index: _FakeRouterModule(name=layer.layer_name)
            for layer in topology
        }

    def extract_topology(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        del bundle
        return self._topology

    def extract_router_state(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> RouterState:
        del bundle
        return self._router_states[layer.layer_index]

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        del bundle
        active_state = self._router_states[layer.layer_index] if router_state is None else router_state
        if active_state.top_k != layer.top_k:
            raise AssertionError("fake runtime backend top-k mismatch")

    def resolve_router_module(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> _FakeRouterModule:
        del bundle
        return self._modules[layer.layer_index]

    def bundle_from_scan_artifact(
        self,
        *,
        model_handle: ModelHandle,
        source_path: Path,
    ) -> LoadedBackendBundle:
        return LoadedBackendBundle(
            backend_name=model_handle.backend_name or "gemma4",
            model_handle=ModelHandle(
                model_id=model_handle.model_id,
                revision=model_handle.revision,
                backend_name=model_handle.backend_name,
                source_path=str(source_path),
                seed=model_handle.seed,
            ),
            model=_FakeRuntimeModel(self._modules),
            config={},
            tokenizer=_FakeTokenizer(),
        )


def test_cli_flow_runs_full_offline_chain_with_stubbed_runtime_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)
    artifact_root = tmp_path / "artifacts"

    shared_args = [
        "--artifact-root",
        str(artifact_root),
        "--model-id",
        "google/gemma-4-27b",
        "--source-path",
        str(checkpoint_root),
        "--revision",
        "rev-123",
        "--backend",
        "gemma4",
        "--seed",
        "7",
    ]

    scan_result = runner.invoke(cli, [*shared_args, "scan"])
    assert scan_result.exit_code == 0, scan_result.output
    scan_path = artifact_root / "scan" / "scan.json"
    scan_run_manifest_path = artifact_root / "scan" / "scan.run-manifest.json"
    assert scan_path.is_file()
    assert scan_run_manifest_path.is_file()
    scan_artifact = load_scan_artifact(scan_path)
    scan_run_manifest = _load_json(scan_run_manifest_path)
    assert scan_artifact.manifest.seed == 7
    assert scan_artifact.manifest.model_handle is not None
    assert scan_artifact.manifest.model_handle.seed == 7
    assert scan_run_manifest["command"] == "scan"
    assert scan_run_manifest["seed"] == 7
    assert scan_run_manifest["model_handle"]["model_id"] == "google/gemma-4-27b"
    assert scan_run_manifest["model_handle"]["revision"] == "rev-123"
    assert scan_run_manifest["model_handle"]["backend_name"] == "gemma4"
    assert scan_run_manifest["output_paths"] == {"scan_artifact": str(scan_path)}
    assert scan_run_manifest["metadata"]["run_manifest_path"] == str(scan_run_manifest_path)

    runtime_backend = _FakeRuntimeBackend(scan_artifact.layers)
    runtime_bundle = runtime_backend.bundle_from_scan_artifact(
        model_handle=scan_artifact.manifest.model_handle,
        source_path=checkpoint_root,
    )
    monkeypatch.setattr(
        "moe_surgeon.cli.main._load_runtime_bundle",
        lambda shared: (runtime_backend, runtime_bundle),
    )

    bench_result = runner.invoke(
        cli,
        [
            *shared_args,
            "bench",
            "--scan-artifact",
            str(scan_path),
            "--prompt",
            "alpha",
            "--prompt",
            "beta",
            "--batch-size",
            "2",
            "--capture-router-scores",
        ],
    )
    assert bench_result.exit_code == 0, bench_result.output
    bench_path = artifact_root / "bench" / "bench.json"
    bench_run_manifest_path = artifact_root / "bench" / "bench.run-manifest.json"
    assert bench_path.is_file()
    assert bench_run_manifest_path.is_file()
    bench_artifact = load_benchmark_artifact(bench_path)
    bench_run_manifest = _load_json(bench_run_manifest_path)
    assert bench_artifact.manifest.seed == 7
    assert bench_artifact.manifest.model_handle is not None
    assert bench_artifact.manifest.model_handle.seed == 7
    assert bench_artifact.profiler_config == {
        "batch_size": 2,
        "capture_router_scores": True,
    }
    assert bench_run_manifest["command"] == "bench"
    assert bench_run_manifest["seed"] == 7
    assert bench_run_manifest["parent_artifacts"] == [str(scan_path)]
    assert bench_run_manifest["output_paths"] == {"benchmark_artifact": str(bench_path)}
    assert bench_run_manifest["metadata"]["run_manifest_path"] == str(bench_run_manifest_path)
    assert len(bench_artifact.activation_stats) == 4
    assert "output_path=" + str(bench_path) in bench_result.output
    assert "prompt_inputs=2" in bench_result.output
    assert "prompt_batches=1" in bench_result.output
    assert "batch_size=2" in bench_result.output

    prune_result = runner.invoke(
        cli,
        [
            *shared_args,
            "prune",
            "--scan-artifact",
            str(scan_path),
            "--bench-artifact",
            str(bench_path),
            "--target-experts",
            "2",
        ],
    )
    assert prune_result.exit_code == 0, prune_result.output
    prune_root = artifact_root / "prune"
    apply_artifact_dir = prune_root / "applied-checkpoint"
    prune_run_manifest_path = prune_root / "prune-run-manifest.json"
    assert (prune_root / "prune-plan.json").is_file()
    assert prune_run_manifest_path.is_file()
    assert (apply_artifact_dir / "apply-manifest.json").is_file()
    prune_run_manifest = _load_json(prune_run_manifest_path)
    assert prune_run_manifest["command"] == "prune"
    assert prune_run_manifest["seed"] == 7
    assert prune_run_manifest["parent_artifacts"] == [str(scan_path), str(bench_path)]
    assert prune_run_manifest["output_paths"] == {
        "apply_artifact_dir": str(apply_artifact_dir),
        "prune_plan": str(prune_root / "prune-plan.json"),
    }
    assert prune_run_manifest["metadata"]["apply_artifact_dir"] == str(apply_artifact_dir)
    assert "apply_artifact_dir=" + str(apply_artifact_dir) in prune_result.output

    export_result = runner.invoke(
        cli,
        [
            *shared_args,
            "export",
            "--apply-artifact-dir",
            str(apply_artifact_dir),
        ],
    )
    assert export_result.exit_code == 0, export_result.output
    export_root = artifact_root / "export"
    export_manifest = _load_json(export_root / "run-manifest.json")
    apply_manifest = _load_json(apply_artifact_dir / "apply-manifest.json")
    apply_run_manifest = _load_json(apply_artifact_dir / "run-manifest.json")

    assert (export_root / "SHA256SUMS").is_file()
    assert export_manifest["command"] == "export"
    assert export_manifest["parent_artifacts"] == ["apply-audit.json", "apply-manifest.json"]
    assert export_manifest["seed"] == 7
    assert export_manifest["model_handle"]["seed"] == 7
    assert export_manifest["metadata"]["apply_id"] == apply_manifest["apply_id"]
    assert export_manifest["metadata"]["plan_id"] == apply_manifest["plan_id"]
    assert apply_run_manifest["seed"] == 7
    assert apply_run_manifest["model_handle"]["seed"] == 7
    assert apply_run_manifest["command"] == "export"
    assert apply_run_manifest["metadata"]["apply_id"] == apply_manifest["apply_id"]
    assert list(export_manifest["output_paths"]) == sorted(export_manifest["output_paths"])
    assert "output_dir=" + str(export_root) in export_result.output
