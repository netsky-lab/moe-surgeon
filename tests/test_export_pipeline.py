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


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }


def _extract_topology(checkpoint_dir: Path) -> tuple[int, int, int]:
    reopened = open_local_safetensors_checkpoint(checkpoint_dir)
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
    return (len(topology), topology[0].expert_count, topology[0].top_k)


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


def test_run_export_rejects_config_inconsistent_dense_passthrough_width_before_writing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")
    invalid_state = dict(applied.derived_state_dict or {})
    invalid_state["model.language_model.layers.0.mlp.down_proj.weight"] = invalid_state[
        "model.language_model.layers.0.mlp.down_proj.weight"
    ].new_zeros((3, 5))
    invalid_state["model.language_model.layers.0.mlp.gate_proj.weight"] = invalid_state[
        "model.language_model.layers.0.mlp.gate_proj.weight"
    ].new_zeros((5, 3))
    invalid_state["model.language_model.layers.0.mlp.up_proj.weight"] = invalid_state[
        "model.language_model.layers.0.mlp.up_proj.weight"
    ].new_zeros((5, 3))
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

    first_tree = _tree_bytes(first_dir)
    second_tree = _tree_bytes(second_dir)
    assert first_tree == second_tree

    first_payload = json.loads(first_tree["run-manifest.json"])
    second_payload = json.loads(second_tree["run-manifest.json"])
    assert first_payload["metadata"]["canonical_manifest_digest"] == second_payload["metadata"][
        "canonical_manifest_digest"
    ]
    assert first_payload["metadata"]["weight_map_digest"] == second_payload["metadata"]["weight_map_digest"]
    assert all(not str(value).startswith("/") for value in first_payload["output_paths"].values())
    assert first_tree["model.safetensors.index.json"] == second_tree["model.safetensors.index.json"]


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


def test_run_export_preserves_assets_and_excludes_stale_weight_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    (source_root / "tokenizer.json").write_text('{"type":"dummy"}\n', encoding="utf-8")
    (source_root / "assets").mkdir()
    (source_root / "assets" / "vocab.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (source_root / "stale.safetensors").write_bytes(b"obsolete")

    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")
    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    assert (export_dir / "tokenizer.json").read_text(encoding="utf-8") == '{"type":"dummy"}\n'
    assert (export_dir / "assets" / "vocab.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert not (export_dir / "stale.safetensors").exists()


def test_run_export_rewrites_config_to_match_exported_tensor_topology(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    reopened = open_local_safetensors_checkpoint(export_dir)
    router = reopened.load_tensors(("model.language_model.layers.0.router.proj.weight",))

    assert config["text_config"]["num_experts"] == 2
    assert not (export_dir / "model.safetensors.index.json").exists()
    assert tuple(router["model.language_model.layers.0.router.proj.weight"].shape) == (2, 3)
    assert _extract_topology(export_dir) == (1, 2, 2)


def test_run_export_preserves_sharded_transformers_compatible_topology(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sharded_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    reopened = open_local_safetensors_checkpoint(export_dir)

    assert (export_dir / "model.safetensors.index.json").is_file()
    assert len(set(reopened.weight_map.values())) == 2
    assert config["text_config"]["num_experts"] == 2
    assert _extract_topology(export_dir) == (1, 2, 2)
    assert reopened.state_keys() == tuple(sorted((applied.derived_state_dict or {}).keys()))


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


def test_run_export_manifest_contains_provenance_and_topology_digests(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sharded_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    export_result = run_export(applied, output_dir=export_dir)
    manifest = json.loads((export_dir / "run-manifest.json").read_text(encoding="utf-8"))

    assert manifest["command"] == "export"
    assert manifest["run_id"] == export_result.export_id
    assert manifest["parent_artifacts"] == ["apply-audit.json", "apply-manifest.json"]
    assert manifest["input_checksums"]["source_metadata_digest"] == applied.source_metadata_digest
    assert manifest["input_checksums"]["plan_canonical_digest"] == applied.metadata["plan_canonical_digest"]
    assert manifest["metadata"]["apply_id"] == applied.apply_id
    assert manifest["metadata"]["plan_id"] == applied.plan_id
    assert manifest["metadata"]["plan_versioned_manifest_id"] == applied.metadata["plan_versioned_manifest_id"]
    assert manifest["metadata"]["plan_canonical_digest"] == applied.metadata["plan_canonical_digest"]
    assert manifest["metadata"]["source_checkpoint_fingerprint"] == applied.source_checkpoint_fingerprint
    assert manifest["metadata"]["source_expert_count"] == 4
    assert manifest["metadata"]["target_expert_count"] == 2
    assert manifest["metadata"]["source_layer_count"] == 1
    assert manifest["metadata"]["target_layer_count"] == 1
    assert manifest["metadata"]["weight_files"] == ",".join(export_result.weight_files)
    assert manifest["metadata"]["export_payload_digest"] == export_result.canonical_digest
    assert isinstance(manifest["metadata"]["weight_map_digest"], str)
    assert isinstance(manifest["metadata"]["source_topology_digest"], str)
    assert isinstance(manifest["metadata"]["target_topology_digest"], str)
    assert isinstance(manifest["metadata"]["canonical_manifest_digest"], str)
    assert manifest["metadata"]["source_topology_digest"] != manifest["metadata"]["target_topology_digest"]


def test_run_export_rejects_non_apply_result_contract(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ApplyResult"):
        run_export({"derived_state_dict": {}}, output_dir=tmp_path / "export")
