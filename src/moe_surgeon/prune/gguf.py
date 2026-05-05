"""Static GGUF prune/export path for packed MoE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from moe_surgeon.analysis.scan import StaticScanResult
from moe_surgeon.models.errors import ArtifactValidationError, TopologyMismatchError
from moe_surgeon.models.gguf import open_local_gguf_checkpoint
from moe_surgeon.schemas import CANONICAL_DEFAULT_TIMESTAMP, ExpertStats, LayerTopology, to_json


_CHECKSUM_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class GgufInspectResult:
    """Deterministic GGUF inventory summary."""

    model_id: str
    checkpoint_path: str
    architecture: str
    file_size_bytes: int
    tensor_count: int
    expert_count: int | None
    top_k: int | None
    block_count: int | None
    hidden_size: int | None
    model_size_label: str | None
    quantization_version: int | None

    def to_payload(self) -> dict[str, object]:
        """Return canonical JSON-friendly inspection payload."""

        return {
            "model_id": self.model_id,
            "checkpoint_path": self.checkpoint_path,
            "architecture": self.architecture,
            "file_size_bytes": self.file_size_bytes,
            "tensor_count": self.tensor_count,
            "expert_count": self.expert_count,
            "top_k": self.top_k,
            "block_count": self.block_count,
            "hidden_size": self.hidden_size,
            "model_size_label": self.model_size_label,
            "quantization_version": self.quantization_version,
        }


@dataclass(frozen=True)
class GgufPruneLayerReport:
    """Per-layer GGUF prune diagnostics."""

    layer_index: int
    source_expert_count: int
    target_expert_count: int
    keep_indices: tuple[int, ...]
    drop_indices: tuple[int, ...]


@dataclass(frozen=True)
class GgufPruneResult:
    """Deterministic metadata for a materialized GGUF prune output."""

    prune_id: str
    source_path: str
    output_path: str | None
    source_fingerprint: str
    source_sha256: str
    output_sha256: str | None
    target_experts: int
    created_at: str
    dry_run: bool
    layer_reports: tuple[GgufPruneLayerReport, ...]
    rewritten_tensor_count: int
    copied_tensor_count: int
    rewritten_tensor_keys: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        """Return canonical JSON-friendly prune metadata."""

        return {
            "prune_id": self.prune_id,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_fingerprint": self.source_fingerprint,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "target_experts": self.target_experts,
            "created_at": self.created_at,
            "dry_run": self.dry_run,
            "rewritten_tensor_count": self.rewritten_tensor_count,
            "copied_tensor_count": self.copied_tensor_count,
            "rewritten_tensor_keys": list(self.rewritten_tensor_keys),
            "layer_reports": [
                {
                    "layer_index": report.layer_index,
                    "source_expert_count": report.source_expert_count,
                    "target_expert_count": report.target_expert_count,
                    "keep_indices": list(report.keep_indices),
                    "drop_indices": list(report.drop_indices),
                }
                for report in self.layer_reports
            ],
        }


def prune_gguf_static(
    checkpoint_path: str | Path,
    *,
    scan_result: StaticScanResult,
    target_experts: int,
    output_path: str | Path | None,
    dry_run: bool = False,
) -> GgufPruneResult:
    """Write a statically pruned GGUF MoE checkpoint.

    Expert selection is derived from static scan ranks. This path rewrites only
    tensors whose first raw data axis is the expert axis and updates
    architecture-specific expert-count metadata. It does not attempt runtime
    activation profiling.
    """

    checkpoint = open_local_gguf_checkpoint(checkpoint_path)
    layers = tuple(sorted(scan_result.layers, key=lambda layer: layer.layer_index))
    reports = _build_layer_reports(
        layers=layers,
        expert_stats=scan_result.expert_stats,
        target_experts=target_experts,
        model_id=checkpoint.model_id,
    )
    rewritten_keys = _planned_rewritten_tensor_keys(layers)
    copied_count = len(checkpoint.tensors) - len(rewritten_keys)
    source_sha256 = _sha256_file(checkpoint.checkpoint_path)
    output: Path | None = None
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
    if dry_run:
        seed = _prune_seed(
            source_fingerprint=checkpoint.file_fingerprint,
            target_experts=target_experts,
            reports=reports,
            rewritten=len(rewritten_keys),
            copied=copied_count,
            dry_run=True,
        )
        return GgufPruneResult(
            prune_id=f"gguf-prune-{sha256(to_json(seed).encode('utf-8')).hexdigest()[:16]}",
            source_path=str(checkpoint.checkpoint_path),
            output_path=str(output) if output is not None else None,
            source_fingerprint=checkpoint.file_fingerprint,
            source_sha256=source_sha256,
            output_sha256=None,
            target_experts=target_experts,
            created_at=CANONICAL_DEFAULT_TIMESTAMP,
            dry_run=True,
            layer_reports=reports,
            rewritten_tensor_count=len(rewritten_keys),
            copied_tensor_count=copied_count,
            rewritten_tensor_keys=rewritten_keys,
        )
    if output is None:
        raise ArtifactValidationError(
            "GGUF prune output_path is required unless dry_run is enabled",
            model_id=checkpoint.model_id,
        )
    if output.exists():
        raise ArtifactValidationError(
            "GGUF prune output_path must not already exist",
            model_id=checkpoint.model_id,
            details={"output_path": str(output)},
        )
    if output == checkpoint.checkpoint_path:
        raise ArtifactValidationError(
            "GGUF prune output_path must differ from source checkpoint",
            model_id=checkpoint.model_id,
            details={"output_path": str(output)},
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    reports_by_layer = {report.layer_index: report for report in reports}
    tmp_output = output.with_name(f".{output.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    rewritten = 0
    copied = 0
    try:
        from gguf import GGUFReader, GGUFWriter

        reader = GGUFReader(checkpoint.checkpoint_path, mode="r")
        writer = GGUFWriter(tmp_output, arch=checkpoint.architecture)
        _write_fields(
            writer,
            checkpoint.fields,
            architecture=checkpoint.architecture,
            target_experts=target_experts,
        )
        for tensor in reader.tensors:
            name = str(tensor.name)
            layer_index = _parse_layer_index(name)
            if layer_index is not None and layer_index in reports_by_layer:
                report = reports_by_layer[layer_index]
                transformed = _slice_tensor_if_needed(
                    name=name,
                    data=tensor.data,
                    logical_shape=tuple(int(item) for item in tensor.shape),
                    tensor_type=tensor.tensor_type,
                    report=report,
                )
                if transformed.rewritten:
                    rewritten += 1
                else:
                    copied += 1
                _add_tensor(
                    writer,
                    name=name,
                    data=transformed.data,
                    logical_shape=transformed.logical_shape,
                    tensor_type=tensor.tensor_type,
                )
                continue
            copied += 1
            _add_tensor(
                writer,
                name=name,
                data=tensor.data,
                logical_shape=tuple(int(item) for item in tensor.shape),
                tensor_type=tensor.tensor_type,
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        tmp_output.replace(output)
        _validate_pruned_output(output, reports=reports, source_tensor_count=len(checkpoint.tensors))
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    output_sha256 = _sha256_file(output)
    seed = _prune_seed(
        source_fingerprint=checkpoint.file_fingerprint,
        target_experts=target_experts,
        reports=reports,
        rewritten=rewritten,
        copied=copied,
        dry_run=False,
    )
    prune_id = f"gguf-prune-{sha256(to_json(seed).encode('utf-8')).hexdigest()[:16]}"
    result = GgufPruneResult(
        prune_id=prune_id,
        source_path=str(checkpoint.checkpoint_path),
        output_path=str(output),
        source_fingerprint=checkpoint.file_fingerprint,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        target_experts=target_experts,
        created_at=CANONICAL_DEFAULT_TIMESTAMP,
        dry_run=False,
        layer_reports=reports,
        rewritten_tensor_count=rewritten,
        copied_tensor_count=copied,
        rewritten_tensor_keys=tuple(sorted(rewritten_keys)),
    )
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        to_json(result.to_payload()),
        encoding="utf-8",
    )
    return result


def inspect_gguf(checkpoint_path: str | Path) -> GgufInspectResult:
    """Return deterministic metadata for a local GGUF checkpoint."""

    checkpoint = open_local_gguf_checkpoint(checkpoint_path)
    fields = checkpoint.fields
    return GgufInspectResult(
        model_id=checkpoint.model_id,
        checkpoint_path=str(checkpoint.checkpoint_path),
        architecture=checkpoint.architecture,
        file_size_bytes=checkpoint.checkpoint_path.stat().st_size,
        tensor_count=len(checkpoint.tensors),
        expert_count=_field_int(fields, _expert_count_metadata_key(checkpoint.architecture)),
        top_k=_field_int(fields, _expert_used_count_metadata_key(checkpoint.architecture)),
        block_count=_field_int(fields, _block_count_metadata_key(checkpoint.architecture)),
        hidden_size=_field_int(fields, _embedding_length_metadata_key(checkpoint.architecture)),
        model_size_label=_field_str(fields, "general.size_label"),
        quantization_version=_field_int(fields, "general.quantization_version"),
    )


@dataclass(frozen=True)
class _TensorTransform:
    data: np.ndarray[Any, Any]
    logical_shape: tuple[int, ...]
    rewritten: bool


def _build_layer_reports(
    *,
    layers: Sequence[LayerTopology],
    expert_stats: Sequence[ExpertStats],
    target_experts: int,
    model_id: str,
) -> tuple[GgufPruneLayerReport, ...]:
    if isinstance(target_experts, bool) or target_experts <= 0:
        raise TopologyMismatchError(
            "GGUF target_experts must be positive",
            model_id=model_id,
            details={"target_experts": target_experts},
        )
    stats_by_layer: dict[int, list[ExpertStats]] = {}
    for stat in expert_stats:
        stats_by_layer.setdefault(stat.layer_index, []).append(stat)
    reports: list[GgufPruneLayerReport] = []
    for layer in layers:
        if target_experts > layer.expert_count:
            raise TopologyMismatchError(
                "GGUF target_experts cannot exceed source expert count",
                model_id=model_id,
                layer_index=layer.layer_index,
                details={
                    "target_experts": target_experts,
                    "source_expert_count": layer.expert_count,
                },
            )
        if target_experts < layer.top_k:
            raise TopologyMismatchError(
                "GGUF target_experts cannot be below layer top_k",
                model_id=model_id,
                layer_index=layer.layer_index,
                details={"target_experts": target_experts, "top_k": layer.top_k},
            )
        layer_stats = stats_by_layer.get(layer.layer_index, [])
        if len(layer_stats) != layer.expert_count:
            raise TopologyMismatchError(
                "GGUF scan expert stats do not cover layer",
                model_id=model_id,
                layer_index=layer.layer_index,
                details={
                    "expected_expert_count": layer.expert_count,
                    "actual_expert_stats": len(layer_stats),
                },
            )
        ranked = sorted(
            layer_stats,
            key=lambda stat: (
                stat.static_rank if stat.static_rank is not None else layer.expert_count,
                stat.expert_index,
            ),
        )
        keep = tuple(sorted(stat.expert_index for stat in ranked[:target_experts]))
        drop = tuple(index for index in range(layer.expert_count) if index not in set(keep))
        reports.append(
            GgufPruneLayerReport(
                layer_index=layer.layer_index,
                source_expert_count=layer.expert_count,
                target_expert_count=target_experts,
                keep_indices=keep,
                drop_indices=drop,
            )
        )
    return tuple(reports)


def _planned_rewritten_tensor_keys(layers: Sequence[LayerTopology]) -> tuple[str, ...]:
    prunable_roles = {
        "experts_down_proj",
        "experts_gate_proj",
        "experts_gate_up_proj",
        "experts_up_proj",
        "router_per_expert_scale",
        "router_proj",
    }
    return tuple(
        sorted(
            tensor_key
            for layer in layers
            for role, tensor_key in layer.module_paths.items()
            if role in prunable_roles
        )
    )


def _prune_seed(
    *,
    source_fingerprint: str,
    target_experts: int,
    reports: Sequence[GgufPruneLayerReport],
    rewritten: int,
    copied: int,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "source_fingerprint": source_fingerprint,
        "target_experts": target_experts,
        "reports": [
            {
                "layer_index": report.layer_index,
                "keep_indices": list(report.keep_indices),
                "drop_indices": list(report.drop_indices),
            }
            for report in reports
        ],
        "rewritten": rewritten,
        "copied": copied,
        "dry_run": dry_run,
    }


def _validate_pruned_output(
    output_path: Path,
    *,
    reports: Sequence[GgufPruneLayerReport],
    source_tensor_count: int,
) -> None:
    checkpoint = open_local_gguf_checkpoint(output_path)
    target_counts = {report.target_expert_count for report in reports}
    if len(target_counts) != 1:
        raise TopologyMismatchError("GGUF prune validation requires uniform target experts")
    target_experts = next(iter(target_counts))
    expert_count_key = _expert_count_metadata_key(checkpoint.architecture)
    if checkpoint.fields.get(expert_count_key) != target_experts:
        raise TopologyMismatchError(
            "GGUF prune output expert_count metadata mismatch",
            model_id=checkpoint.model_id,
            details={
                "expected_expert_count": target_experts,
                "actual_expert_count": checkpoint.fields.get(expert_count_key),
                "metadata_key": expert_count_key,
            },
        )
    if len(checkpoint.tensors) != source_tensor_count:
        raise TopologyMismatchError(
            "GGUF prune output tensor count mismatch",
            model_id=checkpoint.model_id,
            details={
                "expected_tensor_count": source_tensor_count,
                "actual_tensor_count": len(checkpoint.tensors),
            },
        )
    for report in reports:
        if checkpoint.architecture == "gemma4":
            _validate_gemma4_pruned_layer(checkpoint.tensors, report=report)
        elif checkpoint.architecture == "qwen35moe":
            _validate_qwen35moe_pruned_layer(checkpoint.tensors, report=report)
        else:
            raise TopologyMismatchError(
                "GGUF prune validation does not support architecture",
                model_id=checkpoint.model_id,
                details={"architecture": checkpoint.architecture},
            )


def _validate_tensor_axis(shape: Sequence[int], expected_experts: int, *, axis: int) -> None:
    if not shape or shape[axis] != expected_experts:
        raise TopologyMismatchError(
            "GGUF prune output expert-axis shape mismatch",
            expected_shape=(expected_experts,),
            actual_shape=tuple(int(item) for item in shape),
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHECKSUM_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_int(fields: Mapping[str, object], key: str) -> int | None:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _field_str(fields: Mapping[str, object], key: str) -> str | None:
    value = fields.get(key)
    return value if isinstance(value, str) else None


def _write_fields(
    writer: Any,
    fields: Mapping[str, object],
    *,
    architecture: str,
    target_experts: int,
) -> None:
    expert_count_key = _expert_count_metadata_key(architecture)
    for key, value in sorted(fields.items()):
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        output_value = target_experts if key == expert_count_key else value
        _add_field(writer, key, output_value)


def _add_field(writer: Any, key: str, value: object) -> None:
    if isinstance(value, bool):
        writer.add_bool(key, value)
        return
    if isinstance(value, int):
        writer.add_uint32(key, value) if value >= 0 else writer.add_int32(key, value)
        return
    if isinstance(value, float):
        writer.add_float32(key, value)
        return
    if isinstance(value, str):
        writer.add_string(key, value)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > 0:
            writer.add_array(key, list(value))
        return
    writer.add_string(key, str(value))


def _parse_layer_index(tensor_name: str) -> int | None:
    if not tensor_name.startswith("blk."):
        return None
    parts = tensor_name.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _slice_tensor_if_needed(
    *,
    name: str,
    data: np.ndarray[Any, Any],
    logical_shape: tuple[int, ...],
    tensor_type: object,
    report: GgufPruneLayerReport,
) -> _TensorTransform:
    if not _is_expert_axis_tensor(name):
        return _TensorTransform(data=data, logical_shape=logical_shape, rewritten=False)
    if data.shape[0] != report.source_expert_count:
        raise TopologyMismatchError(
            "GGUF expert tensor raw axis does not match source expert count",
            layer_index=report.layer_index,
            tensor_key=name,
            expected_shape=(report.source_expert_count,),
            actual_shape=tuple(int(item) for item in data.shape),
            details={"tensor_type": str(getattr(tensor_type, "name", tensor_type))},
        )
    keep = np.asarray(report.keep_indices, dtype=np.int64)
    sliced = np.take(data, keep, axis=0).copy()
    target_shape: tuple[int, ...]
    if len(logical_shape) == 1:
        target_shape = (report.target_expert_count,)
    else:
        target_shape = (*logical_shape[:-1], report.target_expert_count)
    return _TensorTransform(data=sliced, logical_shape=target_shape, rewritten=True)


def _add_tensor(
    writer: Any,
    *,
    name: str,
    data: np.ndarray[Any, Any],
    logical_shape: tuple[int, ...],
    tensor_type: object,
) -> None:
    if str(getattr(tensor_type, "name", tensor_type)) == "F32":
        writer.add_tensor(name, data)
        return
    writer.add_tensor(name, data, raw_shape=data.shape, raw_dtype=tensor_type)


def _is_expert_axis_tensor(name: str) -> bool:
    return name.endswith(
        (
            ".ffn_gate_inp.weight",
            ".ffn_down_exps.scale",
            ".ffn_gate_exps.weight",
            ".ffn_gate_up_exps.weight",
            ".ffn_up_exps.weight",
            ".ffn_down_exps.weight",
        )
    )


def _validate_gemma4_pruned_layer(
    tensors: Mapping[str, object],
    *,
    report: GgufPruneLayerReport,
) -> None:
    prefix = f"blk.{report.layer_index}"
    expected = report.target_expert_count
    _validate_tensor_axis(
        tensors[f"{prefix}.ffn_gate_inp.weight"].data_shape,  # type: ignore[attr-defined]
        expected,
        axis=0,
    )
    _validate_tensor_axis(
        tensors[f"{prefix}.ffn_down_exps.scale"].data_shape,  # type: ignore[attr-defined]
        expected,
        axis=-1,
    )
    _validate_tensor_axis(
        tensors[f"{prefix}.ffn_gate_up_exps.weight"].shape,  # type: ignore[attr-defined]
        expected,
        axis=-1,
    )
    _validate_tensor_axis(
        tensors[f"{prefix}.ffn_down_exps.weight"].shape,  # type: ignore[attr-defined]
        expected,
        axis=-1,
    )


def _validate_qwen35moe_pruned_layer(
    tensors: Mapping[str, object],
    *,
    report: GgufPruneLayerReport,
) -> None:
    prefix = f"blk.{report.layer_index}"
    expected = report.target_expert_count
    _validate_tensor_axis(
        tensors[f"{prefix}.ffn_gate_inp.weight"].data_shape,  # type: ignore[attr-defined]
        expected,
        axis=0,
    )
    for tensor_name in (
        f"{prefix}.ffn_gate_exps.weight",
        f"{prefix}.ffn_up_exps.weight",
        f"{prefix}.ffn_down_exps.weight",
    ):
        _validate_tensor_axis(
            tensors[tensor_name].shape,  # type: ignore[attr-defined]
            expected,
            axis=-1,
        )


def _expert_count_metadata_key(architecture: str) -> str:
    if architecture == "gemma4":
        return "gemma4.expert_count"
    if architecture == "qwen35moe":
        return "qwen35moe.expert_count"
    return f"{architecture}.expert_count"


def _expert_used_count_metadata_key(architecture: str) -> str:
    if architecture == "gemma4":
        return "gemma4.expert_used_count"
    if architecture == "qwen35moe":
        return "qwen35moe.expert_used_count"
    return f"{architecture}.expert_used_count"


def _block_count_metadata_key(architecture: str) -> str:
    if architecture == "gemma4":
        return "gemma4.block_count"
    if architecture == "qwen35moe":
        return "qwen35moe.block_count"
    return f"{architecture}.block_count"


def _embedding_length_metadata_key(architecture: str) -> str:
    if architecture == "gemma4":
        return "gemma4.embedding_length"
    if architecture == "qwen35moe":
        return "qwen35moe.embedding_length"
    return f"{architecture}.embedding_length"


__all__ = [
    "GgufInspectResult",
    "GgufPruneLayerReport",
    "GgufPruneResult",
    "inspect_gguf",
    "prune_gguf_static",
]
