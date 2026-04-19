from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from moe_surgeon.export import run_export
from moe_surgeon.export.runner import run_export_from_apply_artifact
from moe_surgeon.prune import apply_prune_plan

from tests.test_prune_apply import _plan, _write_checkpoint


def test_export_manifest_links_apply_artifacts_and_digests(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    export_result = run_export(applied, output_dir=export_dir)
    payload = json.loads((export_dir / "run-manifest.json").read_text(encoding="utf-8"))

    assert payload["command"] == "export"
    assert payload["parent_artifacts"] == ["apply-audit.json", "apply-manifest.json"]
    assert payload["input_checksums"] == {
        "apply_audit_digest": sha256(applied.audit_json().encode("utf-8")).hexdigest(),
        "plan_canonical_digest": applied.metadata["plan_canonical_digest"],
        "source_metadata_digest": applied.source_metadata_digest,
    }
    assert payload["metadata"]["apply_id"] == applied.apply_id
    assert payload["metadata"]["plan_id"] == applied.plan_id
    assert payload["metadata"]["export_payload_digest"] == export_result.canonical_digest
    assert payload["metadata"]["source_expert_count"] == 4
    assert payload["metadata"]["target_expert_count"] == 2
    assert payload["metadata"]["source_checkpoint_fingerprint"] == applied.source_checkpoint_fingerprint
    assert isinstance(payload["metadata"]["canonical_manifest_digest"], str)
    assert len(payload["metadata"]["canonical_manifest_digest"]) == 64
    assert payload["output_paths"]["apply_audit"] == "apply-audit.json"
    assert payload["output_paths"]["apply_manifest"] == "apply-manifest.json"
    assert list(payload["output_paths"]) == sorted(payload["output_paths"])
    for parent_name in payload["parent_artifacts"]:
        assert (export_dir / parent_name).is_file()
    for relative_name in payload["output_paths"].values():
        assert (export_dir / relative_name).is_file()


def test_export_manifest_is_deterministic_when_reloaded_from_apply_artifact(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    apply_root = tmp_path / "applied"
    apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=apply_root)

    first_dir = tmp_path / "export-one"
    second_dir = tmp_path / "export-two"
    run_export_from_apply_artifact(apply_root, output_dir=first_dir)
    run_export_from_apply_artifact(apply_root, output_dir=second_dir)

    first_files = sorted(
        path.relative_to(first_dir).as_posix()
        for path in first_dir.rglob("*")
        if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second_dir).as_posix()
        for path in second_dir.rglob("*")
        if path.is_file()
    )

    assert first_files == second_files
    for relative_name in first_files:
        assert (first_dir / relative_name).read_bytes() == (second_dir / relative_name).read_bytes()


def test_export_manifest_matches_direct_and_reloaded_apply_exports(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    apply_root = tmp_path / "applied"
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=apply_root)

    direct_dir = tmp_path / "export-direct"
    reload_dir = tmp_path / "export-reloaded"
    direct_result = run_export(applied, output_dir=direct_dir)
    reload_result = run_export_from_apply_artifact(apply_root, output_dir=reload_dir)

    assert direct_result.export_id == reload_result.export_id
    assert direct_result.canonical_digest == reload_result.canonical_digest
    direct_manifest = json.loads((direct_dir / "run-manifest.json").read_text(encoding="utf-8"))
    reload_manifest = json.loads((reload_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert direct_manifest["metadata"] == reload_manifest["metadata"]
    assert direct_manifest["output_paths"] == reload_manifest["output_paths"]
    assert direct_manifest["model_handle"]["source_path"] == str(source_root.resolve())
    assert reload_manifest["model_handle"]["source_path"] == str(apply_root.resolve())
