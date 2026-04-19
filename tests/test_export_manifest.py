from __future__ import annotations

import json
from pathlib import Path

from moe_surgeon.export import run_export
from moe_surgeon.prune import apply_prune_plan

from tests.test_prune_apply import _plan, _write_checkpoint


def test_export_manifest_links_apply_artifacts_and_digests(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    export_dir = tmp_path / "export"
    run_export(applied, output_dir=export_dir)

    payload = json.loads((export_dir / "run-manifest.json").read_text(encoding="utf-8"))

    assert payload["command"] == "export"
    assert payload["parent_artifacts"] == ["apply-audit.json", "apply-manifest.json"]
    assert list(payload["output_paths"]) == sorted(payload["output_paths"])
    assert payload["metadata"]["apply_id"] == applied.apply_id
    assert payload["metadata"]["plan_id"] == applied.plan_id
    assert payload["metadata"]["source_expert_count"] == 4
    assert payload["metadata"]["target_expert_count"] == 2
    assert isinstance(payload["metadata"]["canonical_manifest_digest"], str)
    assert len(payload["metadata"]["canonical_manifest_digest"]) == 64


def test_export_manifest_is_deterministic_across_repeated_exports(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_checkpoint(source_root)
    applied = apply_prune_plan(source_root, plan=_plan(), dry_run=False, output_dir=tmp_path / "applied")

    first_dir = tmp_path / "export-one"
    second_dir = tmp_path / "export-two"
    run_export(applied, output_dir=first_dir)
    run_export(applied, output_dir=second_dir)

    assert (first_dir / "run-manifest.json").read_bytes() == (second_dir / "run-manifest.json").read_bytes()
    assert (first_dir / "SHA256SUMS").read_bytes() == (second_dir / "SHA256SUMS").read_bytes()
