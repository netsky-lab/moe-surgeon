from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from moe_surgeon.analysis.scan import load_scan_artifact
from moe_surgeon.cli.main import cli

from tests.test_prune_apply import _write_checkpoint


def test_cli_flow_scan_prune_export_chains_artifacts_deterministically(tmp_path: Path) -> None:
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
    assert (prune_root / "prune-plan.json").is_file()
    assert (prune_root / "prune-run-manifest.json").is_file()
    assert (prune_root / "applied-checkpoint" / "apply-manifest.json").is_file()
    assert (export_root / "SHA256SUMS").is_file()
    assert exported_manifest["seed"] == 7
    assert exported_manifest["model_handle"]["seed"] == 7
    assert list(exported_manifest["output_paths"]) == sorted(exported_manifest["output_paths"])
