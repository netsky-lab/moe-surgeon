"""Deterministic safetensors export helpers for pruned checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Mapping, Sequence

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
import torch

from moe_surgeon.models.backend import LoadedBackendBundle, resolve_backend
from moe_surgeon.models.checkpoints import LocalSafetensorsCheckpoint, open_local_safetensors_checkpoint
from moe_surgeon.models.errors import BackendMismatchError, TopologyMismatchError
from moe_surgeon.models.gemma4 import Gemma4Backend
from moe_surgeon.schemas import CANONICAL_DEFAULT_TIMESTAMP, LayerTopology, ModelHandle, to_json

if TYPE_CHECKING:
    from moe_surgeon.prune.apply import ApplyLayerReport, ApplyResult


_APPLY_MANIFEST_FILENAME = "apply-manifest.json"
_APPLY_AUDIT_FILENAME = "apply-audit.json"
_CONFIG_FILENAME = "config.json"
_INDEX_FILENAME = "model.safetensors.index.json"
_SINGLE_WEIGHTS_FILENAME = "model.safetensors"


@dataclass(frozen=True)
class ExportResult:
    """Deterministic export metadata for one written checkpoint tree."""

    export_id: str
    apply_id: str
    plan_id: str
    source_metadata_digest: str
    output_checkpoint_dir: str
    weight_files: tuple[str, ...]
    artifact_filenames: Mapping[str, str]
    weight_map: Mapping[str, str]
    sharded: bool
    created_at: str = CANONICAL_DEFAULT_TIMESTAMP

    def to_payload(self) -> dict[str, object]:
        """Return the canonical payload used by the export manifest."""

        return {
            "export_id": self.export_id,
            "apply_id": self.apply_id,
            "plan_id": self.plan_id,
            "source_metadata_digest": self.source_metadata_digest,
            "weight_files": list(self.weight_files),
            "artifact_filenames": dict(sorted(self.artifact_filenames.items())),
            "weight_map": [[tensor_key, shard_name] for tensor_key, shard_name in sorted(self.weight_map.items())],
            "sharded": self.sharded,
            "created_at": self.created_at,
        }

    @property
    def canonical_digest(self) -> str:
        return sha256(to_json(self.to_payload()).encode("utf-8")).hexdigest()


def write_safetensors_artifact(
    apply_result: ApplyResult,
    *,
    output_dir: str | Path,
) -> ExportResult:
    """Write a deterministic export tree from a validated apply result."""

    if apply_result.dry_run:
        raise TopologyMismatchError(
            "export requires a materialized apply result",
            model_id=apply_result.model_handle.model_id,
            details={"apply_id": apply_result.apply_id},
        )
    if apply_result.derived_state_dict is None:
        raise TopologyMismatchError(
            "export requires derived_state_dict from apply",
            model_id=apply_result.model_handle.model_id,
            details={"apply_id": apply_result.apply_id},
        )
    derived_state_dict = apply_result.derived_state_dict

    source_checkpoint = open_local_safetensors_checkpoint(apply_result.source_checkpoint_dir)
    backend = _resolve_export_backend(source_checkpoint)
    validation_bundle = _build_validation_bundle(
        checkpoint=source_checkpoint,
        backend=backend,
        state=derived_state_dict,
    )
    topology = backend.extract_topology(
        _build_validation_bundle(
            checkpoint=source_checkpoint,
            backend=backend,
            state={item.tensor_key: item for item in source_checkpoint.tensor_metadata()},
        )
    )
    _validate_export_state(
        apply_result,
        checkpoint=source_checkpoint,
        backend=backend,
        topology=topology,
        validation_bundle=validation_bundle,
    )

    output_root = _prepare_output_root(
        output_dir=output_dir,
        source_checkpoint=source_checkpoint,
        model_id=apply_result.model_handle.model_id,
    )
    derived_state = {
        key: derived_state_dict[key].detach().cpu().contiguous()
        for key in sorted(derived_state_dict)
    }
    copied_artifacts = _copy_non_weight_artifacts(source_checkpoint=source_checkpoint, output_root=output_root)
    written_config = _write_config(
        source_config=source_checkpoint.config,
        layer_reports=apply_result.layer_reports,
        output_root=output_root,
    )
    written_sidecars = _write_apply_sidecars(apply_result=apply_result, output_root=output_root)
    weight_files, weight_map, wrote_index = _write_weights(
        source_checkpoint=source_checkpoint,
        derived_state=derived_state,
        output_root=output_root,
    )
    artifact_filenames = {
        "config": written_config,
        "apply_manifest": written_sidecars[0],
        "apply_audit": written_sidecars[1],
        **copied_artifacts,
    }
    if wrote_index:
        artifact_filenames["index"] = _INDEX_FILENAME
    for index, filename in enumerate(weight_files, start=1):
        artifact_filenames[f"weights_{index}"] = filename

    export_seed = {
        "apply_id": apply_result.apply_id,
        "plan_id": apply_result.plan_id,
        "source_metadata_digest": apply_result.source_metadata_digest,
        "weight_map": [[key, value] for key, value in sorted(weight_map.items())],
        "artifact_filenames": dict(sorted(artifact_filenames.items())),
    }
    export_id = f"export-{sha256(to_json(export_seed).encode('utf-8')).hexdigest()[:16]}"
    return ExportResult(
        export_id=export_id,
        apply_id=apply_result.apply_id,
        plan_id=apply_result.plan_id,
        source_metadata_digest=apply_result.source_metadata_digest,
        output_checkpoint_dir=str(output_root),
        weight_files=weight_files,
        artifact_filenames=artifact_filenames,
        weight_map=weight_map,
        sharded=wrote_index,
    )


def _resolve_export_backend(checkpoint: LocalSafetensorsCheckpoint) -> Gemma4Backend:
    backend = resolve_backend(checkpoint.to_backend_signature())
    if not isinstance(backend, Gemma4Backend):
        raise BackendMismatchError(
            "export requires a Gemma4 backend with prune tensor helpers",
            model_id=checkpoint.model_id,
            backend_name=getattr(backend, "name", None),
        )
    return backend


def _build_validation_bundle(
    *,
    checkpoint: LocalSafetensorsCheckpoint,
    backend: Gemma4Backend,
    state: Mapping[str, object],
) -> LoadedBackendBundle:
    return LoadedBackendBundle(
        backend_name=backend.name,
        model_handle=ModelHandle(
            model_id=checkpoint.model_id,
            revision=checkpoint.revision,
            backend_name=backend.name,
            source_path=str(checkpoint.checkpoint_dir),
        ),
        model=object(),
        config=checkpoint.config,
        metadata={"state_dict": state, "backend_version": backend.backend_version},
    )


def _validate_export_state(
    apply_result: ApplyResult,
    *,
    checkpoint: LocalSafetensorsCheckpoint,
    backend: Gemma4Backend,
    topology: Sequence[LayerTopology],
    validation_bundle: LoadedBackendBundle,
) -> None:
    source_keys = checkpoint.state_keys()
    derived_state_dict = apply_result.derived_state_dict
    assert derived_state_dict is not None
    derived_keys = tuple(sorted(derived_state_dict))
    if derived_keys != source_keys:
        raise TopologyMismatchError(
            "export derived tensors must preserve the checkpoint key set",
            model_id=checkpoint.model_id,
            details={
                "source_tensor_count": len(source_keys),
                "derived_tensor_count": len(derived_keys),
            },
        )

    reports_by_layer = {report.layer_index: report for report in apply_result.layer_reports}
    expected_layers = tuple(layer.layer_index for layer in topology)
    if tuple(sorted(reports_by_layer)) != expected_layers:
        raise TopologyMismatchError(
            "export apply result layer coverage does not match checkpoint topology",
            model_id=checkpoint.model_id,
            details={
                "expected_layer_indices": ",".join(str(index) for index in expected_layers),
                "report_layer_indices": ",".join(str(index) for index in sorted(reports_by_layer)),
            },
        )

    for layer in topology:
        report = reports_by_layer[layer.layer_index]
        tensor_keys = backend.resolve_prune_tensor_keys(validation_bundle, layer_index=layer.layer_index)
        for tensor_role, tensor_key in tensor_keys.items():
            backend.validate_prune_tensor(
                validation_bundle,
                layer=_target_layer(layer, report=report),
                tensor_role=tensor_role,
                tensor_key=tensor_key,
                tensor_value=derived_state_dict[tensor_key],
                target_expert_count=report.target_expert_count,
            )


def _target_layer(layer: LayerTopology, *, report: ApplyLayerReport) -> LayerTopology:
    return LayerTopology(
        layer_index=layer.layer_index,
        layer_name=layer.layer_name,
        layer_type=layer.layer_type,
        expert_count=report.target_expert_count,
        top_k=layer.top_k,
        hidden_size=layer.hidden_size,
        moe_intermediate_size=layer.moe_intermediate_size,
        expert_dim=layer.expert_dim,
        ffn_in_features=layer.ffn_in_features,
        ffn_out_features=layer.ffn_out_features,
        layer_ref=layer.layer_ref,
        module_paths=dict(layer.module_paths),
        metadata=dict(layer.metadata),
    )


def _prepare_output_root(
    *,
    output_dir: str | Path,
    source_checkpoint: LocalSafetensorsCheckpoint,
    model_id: str,
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    if output_root == source_checkpoint.checkpoint_dir:
        raise TopologyMismatchError(
            "output_dir must differ from source checkpoint directory",
            model_id=model_id,
            details={"output_dir": str(output_root)},
        )
    if output_root.exists():
        if not output_root.is_dir():
            raise TopologyMismatchError(
                "output_dir must be a directory path",
                model_id=model_id,
                details={"output_dir": str(output_root)},
            )
        if any(output_root.iterdir()):
            raise TopologyMismatchError(
                "output_dir must be empty",
                model_id=model_id,
                details={"output_dir": str(output_root)},
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _copy_non_weight_artifacts(
    *,
    source_checkpoint: LocalSafetensorsCheckpoint,
    output_root: Path,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for entry in sorted(source_checkpoint.checkpoint_dir.iterdir(), key=lambda path: path.name):
        if _is_weight_artifact(entry.name) or entry.name in {
            _APPLY_AUDIT_FILENAME,
            _APPLY_MANIFEST_FILENAME,
            "run-manifest.json",
            "SHA256SUMS",
        }:
            continue
        target = output_root / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
        copied[f"copied_{entry.name}"] = entry.name
    return copied


def _write_config(
    *,
    source_config: Mapping[str, object],
    layer_reports: Sequence[ApplyLayerReport],
    output_root: Path,
) -> str:
    target_expert_count = _uniform_target_expert_count(layer_reports)
    payload = _with_updated_num_experts(source_config, target_expert_count=target_expert_count)
    (output_root / _CONFIG_FILENAME).write_text(to_json(payload), encoding="utf-8")
    return _CONFIG_FILENAME


def _uniform_target_expert_count(layer_reports: Sequence[ApplyLayerReport]) -> int:
    counts = {report.target_expert_count for report in layer_reports}
    if not counts:
        raise TopologyMismatchError("export requires at least one layer report")
    if len(counts) != 1:
        raise TopologyMismatchError(
            "export requires a uniform target expert count across Gemma4 MoE layers",
            details={"target_expert_counts": ",".join(str(count) for count in sorted(counts))},
        )
    return next(iter(counts))


def _with_updated_num_experts(
    source_config: Mapping[str, object],
    *,
    target_expert_count: int,
) -> dict[str, object]:
    payload = json.loads(json.dumps(source_config))
    if not isinstance(payload, dict):
        raise TopologyMismatchError("checkpoint config must be a JSON object")
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        _validate_top_k_experts(text_config, target_expert_count=target_expert_count)
        text_config["num_experts"] = target_expert_count
    _validate_top_k_experts(payload, target_expert_count=target_expert_count)
    if isinstance(payload.get("num_experts"), int):
        payload["num_experts"] = target_expert_count
    return payload


def _validate_top_k_experts(
    config_payload: Mapping[str, object],
    *,
    target_expert_count: int,
) -> None:
    top_k_experts = config_payload.get("top_k_experts")
    if isinstance(top_k_experts, int) and top_k_experts > target_expert_count:
        raise TopologyMismatchError(
            "export config top_k_experts cannot exceed pruned num_experts",
            details={
                "top_k_experts": top_k_experts,
                "target_expert_count": target_expert_count,
            },
        )


def _write_apply_sidecars(*, apply_result: ApplyResult, output_root: Path) -> tuple[str, str]:
    (output_root / _APPLY_MANIFEST_FILENAME).write_text(apply_result.manifest_json(), encoding="utf-8")
    (output_root / _APPLY_AUDIT_FILENAME).write_text(apply_result.audit_json(), encoding="utf-8")
    return (_APPLY_MANIFEST_FILENAME, _APPLY_AUDIT_FILENAME)


def _write_weights(
    *,
    source_checkpoint: LocalSafetensorsCheckpoint,
    derived_state: Mapping[str, torch.Tensor],
    output_root: Path,
) -> tuple[tuple[str, ...], Mapping[str, str], bool]:
    sorted_state = dict(sorted(derived_state.items()))
    shard_names = tuple(sorted(set(source_checkpoint.weight_map.values())))
    source_is_single_file = shard_names == (_SINGLE_WEIGHTS_FILENAME,)
    split = split_torch_state_dict_into_shards(
        sorted_state,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=_target_max_shard_size(
            source_checkpoint=source_checkpoint,
            derived_state=sorted_state,
            source_is_single_file=source_is_single_file,
        ),
    )
    for filename, tensor_keys in split.filename_to_tensors.items():
        shard = {tensor_key: sorted_state[tensor_key] for tensor_key in tensor_keys}
        save_file(shard, str(output_root / filename), metadata={"format": "pt"})
    if not split.is_sharded:
        return ((_SINGLE_WEIGHTS_FILENAME,), dict(sorted(split.tensor_to_filename.items())), False)

    index_payload = {
        "metadata": dict(sorted(split.metadata.items())),
        "weight_map": dict(sorted(split.tensor_to_filename.items())),
    }
    (output_root / _INDEX_FILENAME).write_text(to_json(index_payload), encoding="utf-8")
    return (
        tuple(split.filename_to_tensors),
        dict(sorted(split.tensor_to_filename.items())),
        True,
    )


def _target_max_shard_size(
    *,
    source_checkpoint: LocalSafetensorsCheckpoint,
    derived_state: Mapping[str, torch.Tensor],
    source_is_single_file: bool,
) -> int:
    total_size = _total_tensor_bytes(derived_state)
    if source_is_single_file:
        return total_size

    source_shard_sizes = _source_shard_sizes(source_checkpoint=source_checkpoint)
    max_source_shard_size = max(source_shard_sizes.values())
    if total_size > max_source_shard_size:
        return max_source_shard_size

    largest_tensor = max(
        int(tensor.nelement()) * int(tensor.element_size()) for tensor in derived_state.values()
    )
    if total_size > largest_tensor:
        return total_size - 1
    return largest_tensor


def _source_shard_sizes(*, source_checkpoint: LocalSafetensorsCheckpoint) -> Mapping[str, int]:
    sizes: dict[str, int] = {}
    for metadata in source_checkpoint.tensor_metadata():
        sizes.setdefault(metadata.shard_filename, 0)
        sizes[metadata.shard_filename] += _tensor_nbytes_from_shape(
            shape=metadata.shape,
            dtype=metadata.dtype,
        )
    return sizes


def _tensor_nbytes_from_shape(*, shape: tuple[int, ...], dtype: str) -> int:
    return _dtype_nbytes(dtype=dtype) * _numel(shape=shape)


def _numel(*, shape: tuple[int, ...]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _dtype_nbytes(*, dtype: str) -> int:
    probe = torch.empty((), dtype=_parse_torch_dtype(dtype))
    return int(probe.element_size())


def _parse_torch_dtype(dtype: str) -> torch.dtype:
    normalized = dtype
    safetensors_aliases: dict[str, torch.dtype] = {
        "BOOL": torch.bool,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
    }
    if normalized in safetensors_aliases:
        return safetensors_aliases[normalized]
    if normalized.startswith("torch."):
        normalized = normalized.split(".", 1)[1]
    candidate = getattr(torch, normalized, None)
    if not isinstance(candidate, torch.dtype):
        raise TopologyMismatchError(
            "export could not resolve source tensor dtype for deterministic sharding",
            details={"dtype": dtype},
        )
    return candidate


def _total_tensor_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(int(tensor.nelement()) * int(tensor.element_size()) for tensor in state.values())


def _is_weight_artifact(filename: str) -> bool:
    return filename.endswith(".safetensors") or filename == _INDEX_FILENAME


__all__ = [
    "ExportResult",
    "write_safetensors_artifact",
]
