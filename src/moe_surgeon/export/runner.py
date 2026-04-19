"""Top-level export orchestration."""

from __future__ import annotations

from pathlib import Path

from moe_surgeon.export.manifest import write_export_manifest
from moe_surgeon.export.safetensors_writer import ExportResult, write_safetensors_artifact


def run_export(apply_result, *, output_dir: str | Path) -> ExportResult:
    """Run the deterministic export pipeline for a validated apply result."""

    export_result = write_safetensors_artifact(apply_result, output_dir=output_dir)
    write_export_manifest(
        output_dir=output_dir,
        export_result=export_result,
        apply_result=apply_result,
    )
    return export_result


__all__ = ["run_export", "ExportResult"]
