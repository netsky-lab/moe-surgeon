from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from moe_surgeon.export import run_export
from moe_surgeon.models.backend import LoadedBackendBundle, resolve_backend
from moe_surgeon.models.checkpoints import open_local_safetensors_checkpoint
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.prune import apply_prune_plan
from moe_surgeon.schemas import ModelHandle

from test_prune_apply import _plan, _write_checkpoint, _write_sharded_checkpoint


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def test_run_export_rejects_tensor_shapes_inconsistent_with_pruned_config(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")
    invalid_state = dict(applied.derived_state_dict or {})
    invalid_state["model.language_model.layers.0.router.proj.weight"] = source[
        "model.language_model.layers.0.router.proj.weight"
    ]
    invalid = replace(applied, derived_state_dict=invalid_state)
    export_dir = tmp_path / "invalid-export"

    with pytest.raises(TopologyMismatchError, match="router projection shape mismatch"):
        run_export(invalid, output_dir=export_dir)

    assert not export_dir.exists()


def test_run_export_rejects_dense_passthrough_topology_damage_before_writing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")
    invalid_state = dict(applied.derived_state_dict or {})
    invalid_state["model.language_model.layers.0.mlp.down_proj.weight"] = invalid_state[
        "model.language_model.layers.0.mlp.down_proj.weight"
    ].new_zeros((4, 6))
    invalid = replace(applied, derived_state_dict=invalid_state)
    export_dir = tmp_path / "invalid-export"

    with pytest.raises(ShapeInvariantViolationError, match="mlp.down_proj.weight shape mismatch"):
        run_export(invalid, output_dir=export_dir)

    assert not export_dir.exists()


def test_run_export_is_byte_stable_across_repeated_runs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sharded_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    first_dir = tmp_path / "export-one"
    second_dir = tmp_path / "export-two"
    run_export(applied, output_dir=first_dir)
    run_export(applied, output_dir=second_dir)

    first_manifest = (first_dir / "run-manifest.json").read_bytes()
    second_manifest = (second_dir / "run-manifest.json").read_bytes()
    assert first_manifest == second_manifest
    assert (first_dir / "SHA256SUMS").read_bytes() == (second_dir / "SHA256SUMS").read_bytes()

    first_payload = json.loads(first_manifest)
    second_payload = json.loads(second_manifest)
    assert first_payload["metadata"]["canonical_manifest_digest"] == second_payload["metadata"][
        "canonical_manifest_digest"
    ]
    assert all(not str(value).startswith("/") for value in first_payload["output_paths"].values())


def test_run_export_sha256sums_matches_literal_file_digests(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    recorded = {}
    for line in (export_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        recorded[filename] = digest

    actual_files = {
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(recorded) == actual_files
    assert "run-manifest.json" in recorded
    assert recorded["run-manifest.json"] == _sha256_file(export_dir / "run-manifest.json")
    for relative_name, digest in recorded.items():
        assert digest == _sha256_file(export_dir / relative_name)


def test_run_export_rewrites_config_to_match_exported_tensor_topology(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    reopened = open_local_safetensors_checkpoint(export_dir)
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    router = reopened.load_tensors(("model.language_model.layers.0.router.proj.weight",))

    assert config["text_config"]["num_experts"] == 2
    assert tuple(router["model.language_model.layers.0.router.proj.weight"].shape) == (2, 3)

    backend = resolve_backend(reopened.to_backend_signature())
    assert isinstance(backend, Gemma4Backend)
    topology = backend.extract_topology(
        LoadedBackendBundle(
            backend_name=backend.name,
            model_handle=ModelHandle(
                model_id=reopened.model_id,
                revision=reopened.revision,
                backend_name=backend.name,
                source_path=str(reopened.checkpoint_dir),
            ),
            model=object(),
            config=reopened.config,
            metadata={
                "state_dict": {item.tensor_key: item for item in reopened.tensor_metadata()},
                "backend_version": backend.backend_version,
            },
        )
    )

    assert len(topology) == 1
    assert topology[0].expert_count == 2
    assert topology[0].top_k == 2


def test_run_export_preserves_sharded_transformers_compatible_topology(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sharded_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    reopened = open_local_safetensors_checkpoint(export_dir)
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))

    assert (export_dir / "model.safetensors.index.json").is_file()
    assert len(set(reopened.weight_map.values())) == 2
    assert config["text_config"]["num_experts"] == 2

    backend = resolve_backend(reopened.to_backend_signature())
    assert isinstance(backend, Gemma4Backend)
    topology = backend.extract_topology(
        LoadedBackendBundle(
            backend_name=backend.name,
            model_handle=ModelHandle(
                model_id=reopened.model_id,
                revision=reopened.revision,
                backend_name=backend.name,
                source_path=str(reopened.checkpoint_dir),
            ),
            model=object(),
            config=reopened.config,
            metadata={
                "state_dict": {item.tensor_key: item for item in reopened.tensor_metadata()},
                "backend_version": backend.backend_version,
            },
        )
    )

    assert len(topology) == 1
    assert topology[0].expert_count == 2


def test_run_export_returns_sorted_shard_filenames_for_deterministic_manifests(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sharded_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    result = run_export(applied, output_dir=export_dir)

    assert result.sharded is True
    assert result.weight_files == tuple(sorted(result.weight_files))
    assert result.weight_files == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )


def test_run_export_rejects_non_apply_result_contract(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ApplyResult"):
        run_export({"derived_state_dict": {}}, output_dir=tmp_path / "export")
