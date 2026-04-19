"""Canonical export manifest and checksum helpers."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from moe_surgeon.schemas import RunArtifactManifest, to_json

if TYPE_CHECKING:
    from moe_surgeon.export.safetensors_writer import ExportResult
    from moe_surgeon.prune.apply import ApplyResult


_MANIFEST_FILENAME = "run-manifest.json"
_SHA256SUMS_FILENAME = "SHA256SUMS"


def write_export_manifest(
    *,
    output_dir: str | Path,
    export_result: ExportResult,
    apply_result: ApplyResult,
) -> RunArtifactManifest:
    """Write the canonical run manifest and literal checksum listing."""

    output_root = Path(output_dir).expanduser().resolve()
    source_expert_count, target_expert_count = _expert_count_snapshot(apply_result=apply_result)
    manifest = RunArtifactManifest(
        run_id=export_result.export_id,
        command="export",
        model_handle=apply_result.model_handle,
        input_checksums={
            "source_metadata_digest": apply_result.source_metadata_digest,
            "plan_canonical_digest": str(apply_result.metadata["plan_canonical_digest"]),
            "apply_audit_digest": sha256(apply_result.audit_json().encode("utf-8")).hexdigest(),
        },
        output_paths=dict(sorted(export_result.artifact_filenames.items())),
        parent_artifacts=("apply-audit.json", "apply-manifest.json"),
        metadata={
            "apply_id": apply_result.apply_id,
            "plan_id": apply_result.plan_id,
            "plan_versioned_manifest_id": str(apply_result.metadata["plan_versioned_manifest_id"]),
            "plan_canonical_digest": str(apply_result.metadata["plan_canonical_digest"]),
            "source_metadata_digest": apply_result.source_metadata_digest,
            "source_checkpoint_fingerprint": apply_result.source_checkpoint_fingerprint,
            "export_payload_digest": export_result.canonical_digest,
            "source_expert_count": source_expert_count,
            "target_expert_count": target_expert_count,
        },
    )
    manifest = replace(
        manifest,
        metadata={
            **dict(manifest.metadata),
            "canonical_manifest_digest": manifest.canonical_digest,
        },
    )
    manifest_path = output_root / _MANIFEST_FILENAME
    manifest_path.write_text(to_json(manifest), encoding="utf-8")
    _write_sha256sums(output_root=output_root)
    return manifest


def _write_sha256sums(*, output_root: Path) -> None:
    entries: list[tuple[str, str]] = []
    for file_path in sorted(path for path in output_root.rglob("*") if path.is_file()):
        relative_name = file_path.relative_to(output_root).as_posix()
        if relative_name == _SHA256SUMS_FILENAME:
            continue
        entries.append((_sha256_file(file_path), relative_name))
    lines = "".join(f"{digest}  {name}\n" for digest, name in entries)
    (output_root / _SHA256SUMS_FILENAME).write_text(lines, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expert_count_snapshot(*, apply_result: ApplyResult) -> tuple[int, int]:
    source_counts = {report.source_expert_count for report in apply_result.layer_reports}
    target_counts = {report.target_expert_count for report in apply_result.layer_reports}
    if len(source_counts) != 1 or len(target_counts) != 1:
        raise ValueError("export manifest requires uniform Gemma4 expert counts across layer reports")
    return next(iter(source_counts)), next(iter(target_counts))


__all__ = ["write_export_manifest"]
