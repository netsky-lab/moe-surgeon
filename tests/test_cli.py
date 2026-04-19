from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from click.testing import CliRunner

from moe_surgeon import PACKAGE_DESCRIPTION, PACKAGE_NAME, __version__
from moe_surgeon.analysis.scan import load_scan_artifact
from moe_surgeon.__main__ import main as module_main
from moe_surgeon.cli.main import cli
from moe_surgeon.models.backend import LoadedBackendBundle
from moe_surgeon.schemas import ModelHandle
from test_prune_apply import _write_checkpoint
from test_runtime_profiler import FakeBackend, FakeRouterModule, FakeTokenizer, _router_state

FORBIDDEN_RUNTIME_MODULES = ("torch", "transformers", "safetensors")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _repo_python_env(*, extra_pythonpath: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    python_path_entries: list[str] = []
    if extra_pythonpath is not None:
        python_path_entries.append(str(extra_pythonpath))
    python_path_entries.append(str(_SRC_ROOT))
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return env


def _run_cli_with_import_probe(tmp_path: Path, command: list[str]) -> tuple[subprocess.CompletedProcess[str], set[str]]:
    probe_path = tmp_path / "import-probe"
    probe_path.mkdir()
    output_path = probe_path / "imports.json"
    sitecustomize_path = probe_path / "sitecustomize.py"
    sitecustomize_path.write_text(
        """
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import sys


def _dump_imports() -> None:
    output_path = os.environ["MOE_SURGEON_IMPORT_PROBE_OUTPUT"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(sys.modules), handle)


atexit.register(_dump_imports)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    env = _repo_python_env(extra_pythonpath=probe_path)
    env["MOE_SURGEON_IMPORT_PROBE_OUTPUT"] = str(output_path)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    imported_modules = set(json.loads(output_path.read_text(encoding="utf-8")))
    return result, imported_modules


def test_click_group_metadata_is_exposed_without_runtime_imports() -> None:
    assert cli.name == PACKAGE_NAME
    assert PACKAGE_DESCRIPTION == "Python CLI for analyzing and pruning MoE models"
    assert __version__ == "0.1.0"


def test_main_wrapper_runs_click_group_help() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from moe_surgeon.cli.main import main; main()", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_repo_python_env(),
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "scan" in result.stdout
    assert "bench" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout


def test_module_help_lists_placeholder_subcommands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "moe_surgeon", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_repo_python_env(),
    )

    assert result.returncode == 0
    assert "Usage: python -m moe_surgeon" in result.stdout
    assert "scan" in result.stdout
    assert "bench" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout


def test_root_help_lists_shared_options() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "moe_surgeon", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_repo_python_env(),
    )

    assert result.returncode == 0, result.stderr
    assert "--model-id" in result.stdout
    assert "--source-path DIRECTORY" in result.stdout
    assert "Local checkpoint directory containing config.json" in result.stdout
    assert "and safetensors weights." in result.stdout
    assert "--backend" in result.stdout
    assert "--seed" in result.stdout
    assert "--artifact-root" in result.stdout


def test_scan_help_lists_directory_only_source_path_contract() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "moe_surgeon", "scan", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_repo_python_env(),
    )

    assert result.returncode == 0, result.stderr
    assert "Usage: python -m moe_surgeon scan" in result.stdout
    assert "--source-path DIRECTORY" in result.stdout
    assert "Local checkpoint directory containing config.json" in result.stdout
    assert "and safetensors weights." in result.stdout


def test_module_main_wrapper_runs_click_group_help() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from moe_surgeon.__main__ import main; main()", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_repo_python_env(),
    )

    assert result.returncode == 0
    assert "Usage: python -m moe_surgeon" in result.stdout
    assert "scan" in result.stdout
    assert "bench" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout


def test_cli_help_does_not_import_heavy_dependencies() -> None:
    probe = """
import sys
from click.testing import CliRunner
from moe_surgeon.cli.main import cli

result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0, result.output
forbidden = [name for name in ("torch", "transformers", "safetensors") if name in sys.modules]
assert not forbidden, forbidden
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=_repo_python_env(),
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_module_main_is_importable() -> None:
    assert callable(module_main)


def test_shared_root_context_is_available_to_subcommands() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        checkpoint_root = Path("checkpoint")
        checkpoint_root.mkdir()
        _write_checkpoint(checkpoint_root)
        result = runner.invoke(
            cli,
            [
                "--model-id",
                "google/gemma-4-27b",
                "--source-path",
                str(checkpoint_root),
                "scan",
                "--output",
                "artifacts/scan.json",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "command=scan" in result.output
        assert "source_path=checkpoint" in result.output
        assert "output_path=artifacts/scan.json" in result.output


def test_scan_command_writes_manifest_sidecar(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    output_path = tmp_path / "scan" / "scan.json"
    result = runner.invoke(cli, ["--source-path", str(checkpoint_root), "scan", "--output", str(output_path)])

    assert result.exit_code == 0, result.output
    assert output_path.is_file()
    assert output_path.with_name("scan.run-manifest.json").is_file()


def test_scan_command_rejects_existing_output_path(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    output_path = tmp_path / "scan.json"
    output_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(cli, ["--source-path", str(checkpoint_root), "scan", "--output", str(output_path)])

    assert result.exit_code == 24
    assert "error[24:artifact_validation]" in result.output
    assert "scan output_path must not already exist" in result.output


def test_scan_command_rejects_file_source_path_at_cli_parse_time(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_file = tmp_path / "model.safetensors"
    checkpoint_file.write_bytes(b"not-a-directory")

    result = runner.invoke(
        cli,
        ["--source-path", str(checkpoint_file), "scan", "--output", str(tmp_path / "scan.json")],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--source-path'" in result.output
    assert "Directory" in result.output
    assert str(checkpoint_file) in result.output


@pytest.mark.parametrize(
    ("command", "expected_output"),
    [
        ([sys.executable, "-m", "moe_surgeon", "--version"], "python -m moe_surgeon, version 0.1.0"),
        (["moe-surgeon", "--version"], "moe-surgeon, version 0.1.0"),
    ],
)
def test_cli_version_commands_are_lightweight(
    tmp_path: Path, command: list[str], expected_output: str
) -> None:
    if command[0] == "moe-surgeon":
        script_path = shutil.which(command[0])
        assert script_path is not None
        command = [script_path, *command[1:]]

    result, imported_modules = _run_cli_with_import_probe(tmp_path, command)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_output
    forbidden = [
        module_name
        for module_name in imported_modules
        if any(
            module_name == forbidden_name or module_name.startswith(f"{forbidden_name}.")
            for forbidden_name in FORBIDDEN_RUNTIME_MODULES
        )
    ]
    assert not forbidden, forbidden


def test_bench_command_accepts_prompt_batching_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    scan_path = tmp_path / "scan.json"
    scan_result = runner.invoke(cli, ["--source-path", str(checkpoint_root), "scan", "--output", str(scan_path)])
    assert scan_result.exit_code == 0, scan_result.output
    scan_artifact = load_scan_artifact(scan_path)

    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def __call__(self, **kwargs: object) -> object:
            attention_mask = kwargs["attention_mask"]
            assert isinstance(attention_mask, list)
            top_k_indices = [
                [[0, 1] for _ in prompt_mask]
                for prompt_mask in attention_mask
            ]
            top_k_weights = [
                [[0.6, 0.4] for _ in prompt_mask]
                for prompt_mask in attention_mask
            ]
            router_scores = [
                [[0.6, 0.4, 0.0, 0.0] for _ in prompt_mask]
                for prompt_mask in attention_mask
            ]
            module.run(
                {
                    "top_k_indices": top_k_indices,
                    "top_k_weights": top_k_weights,
                    "router_scores": router_scores,
                }
            )
            return object()

    fake_bundle = LoadedBackendBundle(
        backend_name="fake",
        model_handle=ModelHandle(
            model_id=scan_artifact.manifest.model_handle.model_id,
            revision=scan_artifact.manifest.model_handle.revision,
            backend_name=scan_artifact.manifest.model_handle.backend_name,
            seed=scan_artifact.manifest.model_handle.seed,
        ),
        model=FakeModel(),
        config={},
        tokenizer=FakeTokenizer(),
    )
    fake_topology = scan_artifact.layers

    def _fake_load_runtime_bundle(shared: object) -> tuple[FakeBackend, LoadedBackendBundle]:
        del shared
        return backend, fake_bundle

    monkeypatch.setattr("moe_surgeon.cli.main._load_runtime_bundle", _fake_load_runtime_bundle)
    monkeypatch.setattr(
        backend,
        "extract_topology",
        lambda bundle: fake_topology,
        raising=False,
    )

    result = runner.invoke(
        cli,
        [
            "--model-id",
            "google/gemma-4-27b",
            "bench",
            "--scan-artifact",
            str(scan_path),
            "--prompt",
            "alpha",
            "--prompt",
            "beta",
            "--batch-size",
            "2",
            "--seed",
            "7",
            "--capture-router-scores",
            "--output",
            str(tmp_path / "bench.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "prompt_inputs=2" in result.output
    assert "prompt_batches=1" in result.output
    assert "batch_size=2" in result.output
    assert (tmp_path / "bench.json").is_file()


def test_bench_command_rejects_conflicting_seed_from_scan_artifact(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    scan_path = tmp_path / "scan.json"
    scan_result = runner.invoke(
        cli,
        ["--source-path", str(checkpoint_root), "--seed", "3", "scan", "--output", str(scan_path)],
    )
    assert scan_result.exit_code == 0, scan_result.output

    result = runner.invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_root),
            "--seed",
            "7",
            "bench",
            "--scan-artifact",
            str(scan_path),
            "--prompt",
            "alpha",
            "--output",
            str(tmp_path / "bench.json"),
        ],
    )

    assert result.exit_code == 24
    assert "error[24:artifact_validation]" in result.output
    assert "bench seed must be deterministic across CLI context and artifacts" in result.output


def test_bench_command_maps_missing_scan_artifact_to_artifact_validation(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    missing_scan = tmp_path / "missing-scan.json"
    result = runner.invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_root),
            "bench",
            "--scan-artifact",
            str(missing_scan),
            "--prompt",
            "alpha",
            "--output",
            str(tmp_path / "bench.json"),
        ],
    )

    assert result.exit_code == 24
    assert "error[24:artifact_validation]" in result.output
    assert "scan artifact does not exist" in result.output


def test_prune_command_maps_malformed_bench_artifact_to_artifact_validation(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    scan_path = tmp_path / "scan.json"
    scan_result = runner.invoke(cli, ["--source-path", str(checkpoint_root), "scan", "--output", str(scan_path)])
    assert scan_result.exit_code == 0, scan_result.output

    malformed_bench_path = tmp_path / "bench.json"
    malformed_bench_path.write_text("[]", encoding="utf-8")
    result = runner.invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_root),
            "prune",
            "--scan-artifact",
            str(scan_path),
            "--bench-artifact",
            str(malformed_bench_path),
            "--output-dir",
            str(tmp_path / "prune"),
            "--target-experts",
            "2",
        ],
    )

    assert result.exit_code == 24
    assert "error[24:artifact_validation]" in result.output
    assert "benchmark artifact payload must be a JSON object" in result.output


def test_bench_command_rejects_runtime_backend_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    scan_path = tmp_path / "scan.json"
    scan_result = runner.invoke(cli, ["--source-path", str(checkpoint_root), "scan", "--output", str(scan_path)])
    assert scan_result.exit_code == 0, scan_result.output
    scan_artifact = load_scan_artifact(scan_path)

    module = FakeRouterModule(name="layer-0")
    backend = FakeBackend(router_states={0: _router_state(0)}, modules={0: module})
    fake_bundle = LoadedBackendBundle(
        backend_name="fake",
        model_handle=ModelHandle(model_id="google/gemma-4-27b", backend_name="fake-other"),
        model=object(),
        config={},
        tokenizer=FakeTokenizer(),
    )

    def _fake_load_runtime_bundle(shared: object) -> tuple[FakeBackend, LoadedBackendBundle]:
        del shared
        return backend, fake_bundle

    monkeypatch.setattr("moe_surgeon.cli.main._load_runtime_bundle", _fake_load_runtime_bundle)
    monkeypatch.setattr(backend, "extract_topology", lambda bundle: scan_artifact.layers, raising=False)

    result = runner.invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_root),
            "bench",
            "--scan-artifact",
            str(scan_path),
            "--prompt",
            "alpha",
            "--output",
            str(tmp_path / "bench.json"),
        ],
    )

    assert result.exit_code == 21
    assert "error[21:backend_mismatch]" in result.output
    assert "scan artifact does not match runtime-loaded model/backend identity" in result.output


def test_prune_and_export_commands_chain_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    scan_path = tmp_path / "scan.json"
    scan_result = runner.invoke(cli, ["--source-path", str(checkpoint_root), "scan", "--output", str(scan_path)])
    assert scan_result.exit_code == 0, scan_result.output
    scan_artifact = load_scan_artifact(scan_path)

    bench_path = tmp_path / "bench.json"
    bench_payload = {
        "manifest": {
            "__schema_type": "RunArtifactManifest",
            "__schema_version": "1.0.0",
            "run_id": "bench-test",
            "command": "bench",
            "model_handle": scan_artifact.manifest.model_handle.to_dict(),
            "top_k": 2,
            "prompt_count": 1,
            "seed": 0,
            "prompt_set_hash": None,
            "started_at": "1970-01-01T00:00:00+00:00",
            "finished_at": None,
            "input_checksums": {},
            "output_paths": {},
            "parent_artifacts": [],
            "run_plan": None,
            "metadata": {},
        },
        "topology": [
            layer.to_dict() for layer in scan_artifact.layers
        ],
        "activation_stats": [
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 0,
                "token_count": 10,
                "weighted_token_count": 8.0,
                "mass_sum": 8.0,
                "mean_weight": 0.8,
                "entropy": 0.1,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 5.0,
                "density": 0.5,
                "metadata": {},
            },
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 1,
                "token_count": 6,
                "weighted_token_count": 5.0,
                "mass_sum": 5.0,
                "mean_weight": 0.83,
                "entropy": 0.2,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 3.0,
                "density": 0.3,
                "metadata": {},
            },
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 2,
                "token_count": 4,
                "weighted_token_count": 3.0,
                "mass_sum": 3.0,
                "mean_weight": 0.75,
                "entropy": 0.3,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 2.0,
                "density": 0.2,
                "metadata": {},
            },
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 3,
                "token_count": 2,
                "weighted_token_count": 1.0,
                "mass_sum": 1.0,
                "mean_weight": 0.5,
                "entropy": 0.4,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 1.0,
                "density": 0.1,
                "metadata": {},
            },
        ],
        "profiler_config": {"batch_size": 1},
        "input_payload_hash": None,
    }
    bench_path.write_text(json.dumps(bench_payload), encoding="utf-8")

    prune_root = tmp_path / "prune"
    prune_result = runner.invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_root),
            "prune",
            "--scan-artifact",
            str(scan_path),
            "--bench-artifact",
            str(bench_path),
            "--output-dir",
            str(prune_root),
            "--target-experts",
            "2",
        ],
    )
    assert prune_result.exit_code == 0, prune_result.output
    assert (prune_root / "prune-plan.json").is_file()
    assert (prune_root / "prune-run-manifest.json").is_file()
    assert (prune_root / "applied-checkpoint" / "apply-manifest.json").is_file()

    export_root = tmp_path / "export"
    export_result = runner.invoke(
        cli,
        [
            "export",
            "--apply-artifact-dir",
            str(prune_root / "applied-checkpoint"),
            "--output-dir",
            str(export_root),
        ],
    )
    assert export_result.exit_code == 0, export_result.output
    assert (export_root / "run-manifest.json").is_file()


def test_export_command_maps_malformed_apply_artifact_to_artifact_validation(tmp_path: Path) -> None:
    runner = CliRunner()
    apply_root = tmp_path / "apply"
    apply_root.mkdir()
    (apply_root / "apply-manifest.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "export",
            "--apply-artifact-dir",
            str(apply_root),
            "--output-dir",
            str(tmp_path / "export"),
        ],
    )

    assert result.exit_code == 24
    assert "error[24:artifact_validation]" in result.output
    assert "apply manifest payload must be a JSON object" in result.output


def test_prune_and_export_preserve_non_zero_seed_across_manifests(tmp_path: Path) -> None:
    runner = CliRunner()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    _write_checkpoint(checkpoint_root)

    scan_path = tmp_path / "scan.json"
    scan_result = runner.invoke(
        cli,
        ["--source-path", str(checkpoint_root), "--seed", "7", "scan", "--output", str(scan_path)],
    )
    assert scan_result.exit_code == 0, scan_result.output
    scan_artifact = load_scan_artifact(scan_path)

    bench_path = tmp_path / "bench.json"
    bench_payload = {
        "manifest": {
            "__schema_type": "RunArtifactManifest",
            "__schema_version": "1.0.0",
            "run_id": "bench-test-seed",
            "command": "bench",
            "model_handle": scan_artifact.manifest.model_handle.to_dict(),
            "top_k": 2,
            "prompt_count": 1,
            "seed": 7,
            "prompt_set_hash": None,
            "started_at": "1970-01-01T00:00:00+00:00",
            "finished_at": None,
            "input_checksums": {},
            "output_paths": {},
            "parent_artifacts": [],
            "run_plan": None,
            "metadata": {},
        },
        "topology": [layer.to_dict() for layer in scan_artifact.layers],
        "activation_stats": [
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 0,
                "token_count": 10,
                "weighted_token_count": 8.0,
                "mass_sum": 8.0,
                "mean_weight": 0.8,
                "entropy": 0.1,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 5.0,
                "density": 0.5,
                "metadata": {},
            },
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 1,
                "token_count": 6,
                "weighted_token_count": 5.0,
                "mass_sum": 5.0,
                "mean_weight": 0.83,
                "entropy": 0.2,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 3.0,
                "density": 0.3,
                "metadata": {},
            },
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 2,
                "token_count": 4,
                "weighted_token_count": 3.0,
                "mass_sum": 3.0,
                "mean_weight": 0.75,
                "entropy": 0.3,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 2.0,
                "density": 0.2,
                "metadata": {},
            },
            {
                "__schema_type": "ActivationStats",
                "__schema_version": "1.0.0",
                "layer_index": 0,
                "expert_index": 3,
                "token_count": 2,
                "weighted_token_count": 1.0,
                "mass_sum": 1.0,
                "mean_weight": 0.5,
                "entropy": 0.4,
                "n_tokens": 20,
                "weighted_n_tokens": 20.0,
                "timestamp_span": None,
                "top1_mass": 1.0,
                "density": 0.1,
                "metadata": {},
            },
        ],
        "profiler_config": {"batch_size": 1},
        "input_payload_hash": None,
    }
    bench_path.write_text(json.dumps(bench_payload), encoding="utf-8")

    prune_root = tmp_path / "prune"
    prune_result = runner.invoke(
        cli,
        [
            "--source-path",
            str(checkpoint_root),
            "prune",
            "--scan-artifact",
            str(scan_path),
            "--bench-artifact",
            str(bench_path),
            "--output-dir",
            str(prune_root),
            "--target-experts",
            "2",
        ],
    )
    assert prune_result.exit_code == 0, prune_result.output

    apply_manifest = json.loads((prune_root / "applied-checkpoint" / "apply-manifest.json").read_text(encoding="utf-8"))
    materialized_run_manifest = json.loads(
        (prune_root / "applied-checkpoint" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert apply_manifest["model_handle"]["seed"] == 7
    assert materialized_run_manifest["seed"] == 7
    assert materialized_run_manifest["model_handle"]["seed"] == 7

    export_root = tmp_path / "export"
    export_result = runner.invoke(
        cli,
        [
            "export",
            "--apply-artifact-dir",
            str(prune_root / "applied-checkpoint"),
            "--output-dir",
            str(export_root),
        ],
    )
    assert export_result.exit_code == 0, export_result.output

    exported_manifest = json.loads((export_root / "run-manifest.json").read_text(encoding="utf-8"))
    assert exported_manifest["seed"] == 7
    assert exported_manifest["model_handle"]["seed"] == 7
