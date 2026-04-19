"""Top-level export orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from moe_surgeon.export.manifest import write_export_manifest
from moe_surgeon.export.safetensors_writer import ExportResult, write_safetensors_artifact

if TYPE_CHECKING:
    from moe_surgeon.prune.apply import ApplyResult


def run_export(apply_result: ApplyResult, *, output_dir: str | Path) -> ExportResult:
    """Run the deterministic export pipeline for a validated apply result."""

    from moe_surgeon.prune.apply import ApplyResult

    if not isinstance(apply_result, ApplyResult):
        raise TypeError("run_export requires an ApplyResult from apply_prune_plan")
    export_result = write_safetensors_artifact(apply_result, output_dir=output_dir)
    write_export_manifest(
        output_dir=output_dir,
        export_result=export_result,
        apply_result=apply_result,
    )
    return export_result


__all__ = ["run_export", "ExportResult"]
