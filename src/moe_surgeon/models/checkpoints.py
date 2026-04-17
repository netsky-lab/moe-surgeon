"""Lightweight local safetensors checkpoint readers for topology-only analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from safetensors import safe_open

from moe_surgeon.models.backend import BackendSignature
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError

if TYPE_CHECKING:
    import torch


_CONFIG_FILENAME = "config.json"
_SINGLE_FILE_NAME = "model.safetensors"
_INDEX_FILENAME = "model.safetensors.index.json"
_PICKLE_FILENAMES = (
    "pytorch_model.bin",
    "model.bin",
    "pytorch_model.pt",
    "model.ckpt",
)


@dataclass(frozen=True)
class CheckpointTensorMetadata:
    """Minimal deterministic metadata for one checkpoint tensor."""

    tensor_key: str
    shape: tuple[int, ...]
    dtype: str
    shard_filename: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalSafetensorsCheckpoint:
    """Parsed local checkpoint metadata and targeted tensor accessors."""

    checkpoint_dir: Path
    config: Mapping[str, object]
    model_id: str
    revision: str | None
    tensors: Mapping[str, CheckpointTensorMetadata]
    weight_map: Mapping[str, str]

    def state_keys(self) -> tuple[str, ...]:
        """Return checkpoint tensor keys in canonical sorted order."""

        return tuple(sorted(self.tensors))

    def tensor_metadata(
        self,
        tensor_names: Sequence[str] | None = None,
    ) -> tuple[CheckpointTensorMetadata, ...]:
        """Return deterministic metadata for all or a requested subset of tensors."""

        names = self._normalize_tensor_names(tensor_names)
        return tuple(self.tensors[name] for name in names)

    def load_tensors(self, tensor_names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Materialize only the requested tensor subset via deterministic shard reads."""

        names = self._normalize_tensor_names(tensor_names)
        shard_to_keys: dict[str, list[str]] = {}
        for name in names:
            shard_to_keys.setdefault(self.weight_map[name], []).append(name)

        loaded: dict[str, torch.Tensor] = {}
        for shard_filename in sorted(shard_to_keys):
            shard_path = _resolve_shard_path(self.checkpoint_dir, shard_filename)
            with safe_open(  # type: ignore[no-untyped-call,attr-defined]
                str(shard_path), framework="pt", device="cpu"
            ) as handle:
                for tensor_key in sorted(shard_to_keys[shard_filename]):
                    try:
                        loaded[tensor_key] = handle.get_tensor(tensor_key)
                    except Exception as exc:
                        raise TopologyMismatchError(
                            "checkpoint shard is missing indexed tensor payload",
                            model_id=self.model_id,
                            tensor_key=tensor_key,
                            details={
                                "checkpoint_path": str(self.checkpoint_dir),
                                "shard_filename": shard_filename,
                            },
                        ) from exc
        return {name: loaded[name] for name in names}

    def to_backend_signature(self) -> BackendSignature:
        """Return a lightweight backend-dispatch signature for this checkpoint."""

        return BackendSignature.from_mapping(
            self.config,
            model_id=self.model_id,
            source_path=self.checkpoint_dir,
        )

    def _normalize_tensor_names(self, tensor_names: Sequence[str] | None) -> tuple[str, ...]:
        names = self.state_keys() if tensor_names is None else tuple(sorted({str(name) for name in tensor_names}))
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise TopologyMismatchError(
                "checkpoint tensor key is missing",
                model_id=self.model_id,
                tensor_key=missing[0],
                details={
                    "checkpoint_path": str(self.checkpoint_dir),
                    "missing_tensor_keys": ",".join(missing),
                },
            )
        return names


def open_local_safetensors_checkpoint(checkpoint_dir: str | Path) -> LocalSafetensorsCheckpoint:
    """Open a local single-file or sharded safetensors checkpoint without loading a model."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise TopologyMismatchError(
            "checkpoint path must be an existing directory",
            model_id=str(root),
            details={"checkpoint_path": str(root)},
        )

    config_path = root / _CONFIG_FILENAME
    if not config_path.is_file():
        raise TopologyMismatchError(
            "checkpoint config.json is missing",
            model_id=str(root),
            details={"checkpoint_path": str(root)},
        )
    config = _load_json_object(config_path, error_message="checkpoint config.json is malformed")
    model_id = _config_model_id(config, checkpoint_dir=root)
    revision = _optional_string(config.get("_commit_hash")) or _optional_string(config.get("revision"))

    index_path = root / _INDEX_FILENAME
    single_path = root / _SINGLE_FILE_NAME
    if index_path.is_file():
        weight_map = _load_index_weight_map(index_path=index_path, checkpoint_dir=root, model_id=model_id)
    elif single_path.is_file():
        weight_map = _build_single_file_weight_map(single_path=single_path, checkpoint_dir=root, model_id=model_id)
    else:
        _raise_missing_layout_error(root=root, model_id=model_id)

    tensors = _read_tensor_metadata(weight_map=weight_map, checkpoint_dir=root, model_id=model_id)
    return LocalSafetensorsCheckpoint(
        checkpoint_dir=root,
        config=config,
        model_id=model_id,
        revision=revision,
        tensors=tensors,
        weight_map=weight_map,
    )


def _raise_missing_layout_error(*, root: Path, model_id: str) -> None:
    for filename in _PICKLE_FILENAMES:
        if (root / filename).exists():
            raise TopologyMismatchError(
                "pickle-only checkpoints are unsupported; expected safetensors weights",
                model_id=model_id,
                details={"checkpoint_path": str(root), "unsupported_weight_file": filename},
            )
    raise TopologyMismatchError(
        "checkpoint safetensors weights are missing",
        model_id=model_id,
        details={
            "checkpoint_path": str(root),
            "expected_files": f"{_SINGLE_FILE_NAME},{_INDEX_FILENAME}",
        },
    )


def _config_model_id(config: Mapping[str, object], *, checkpoint_dir: Path) -> str:
    return _optional_string(config.get("_name_or_path")) or _optional_string(config.get("model_id")) or str(
        checkpoint_dir
    )


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _load_json_object(path: Path, *, error_message: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ShapeInvariantViolationError(
            error_message,
            model_id=str(path.parent),
            details={"checkpoint_path": str(path.parent), "file_path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise ShapeInvariantViolationError(
            error_message,
            model_id=str(path.parent),
            details={"checkpoint_path": str(path.parent), "file_path": str(path)},
        )
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate key: {key}")
        payload[key] = value
    return payload


def _load_index_weight_map(
    *,
    index_path: Path,
    checkpoint_dir: Path,
    model_id: str,
) -> dict[str, str]:
    try:
        payload = json.loads(
            index_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except ValueError as exc:
        raise ShapeInvariantViolationError(
            "checkpoint index.json is malformed",
            model_id=model_id,
            details={"checkpoint_path": str(checkpoint_dir), "file_path": str(index_path)},
        ) from exc
    if not isinstance(payload, dict):
        raise ShapeInvariantViolationError(
            "checkpoint index.json is malformed",
            model_id=model_id,
            details={"checkpoint_path": str(checkpoint_dir), "file_path": str(index_path)},
        )
    raw_weight_map = payload.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ShapeInvariantViolationError(
            "checkpoint index weight_map must be an object",
            model_id=model_id,
            details={"checkpoint_path": str(checkpoint_dir), "file_path": str(index_path)},
        )

    weight_map: dict[str, str] = {}
    for tensor_key, shard_filename in sorted(raw_weight_map.items()):
        if not isinstance(tensor_key, str) or not tensor_key.strip():
            raise ShapeInvariantViolationError(
                "checkpoint index tensor key must be a non-empty string",
                model_id=model_id,
                details={"checkpoint_path": str(checkpoint_dir), "file_path": str(index_path)},
            )
        if not isinstance(shard_filename, str) or not shard_filename.strip():
            raise ShapeInvariantViolationError(
                "checkpoint index shard filename must be a non-empty string",
                model_id=model_id,
                tensor_key=tensor_key,
                details={"checkpoint_path": str(checkpoint_dir), "file_path": str(index_path)},
            )
        _resolve_shard_path(checkpoint_dir, shard_filename)
        weight_map[tensor_key] = shard_filename

    if not weight_map:
        raise ShapeInvariantViolationError(
            "checkpoint index weight_map must not be empty",
            model_id=model_id,
            details={"checkpoint_path": str(checkpoint_dir), "file_path": str(index_path)},
        )
    return weight_map


def _build_single_file_weight_map(
    *,
    single_path: Path,
    checkpoint_dir: Path,
    model_id: str,
) -> dict[str, str]:
    try:
        with safe_open(  # type: ignore[no-untyped-call,attr-defined]
            str(single_path), framework="pt", device="cpu"
        ) as handle:
            keys = tuple(sorted(str(key) for key in handle.keys()))
    except Exception as exc:
        raise TopologyMismatchError(
            "checkpoint safetensors file could not be opened",
            model_id=model_id,
            details={"checkpoint_path": str(checkpoint_dir), "shard_filename": single_path.name},
        ) from exc
    if not keys:
        raise ShapeInvariantViolationError(
            "checkpoint safetensors file contains no tensors",
            model_id=model_id,
            details={"checkpoint_path": str(checkpoint_dir), "shard_filename": single_path.name},
        )
    return {key: single_path.name for key in keys}


def _read_tensor_metadata(
    *,
    weight_map: Mapping[str, str],
    checkpoint_dir: Path,
    model_id: str,
) -> dict[str, CheckpointTensorMetadata]:
    shard_to_keys: dict[str, list[str]] = {}
    for tensor_key, shard_filename in weight_map.items():
        shard_to_keys.setdefault(shard_filename, []).append(tensor_key)

    tensors: dict[str, CheckpointTensorMetadata] = {}
    for shard_filename in sorted(shard_to_keys):
        shard_path = _resolve_shard_path(checkpoint_dir, shard_filename)
        expected_keys = tuple(sorted(shard_to_keys[shard_filename]))
        try:
            with safe_open(  # type: ignore[no-untyped-call,attr-defined]
                str(shard_path), framework="pt", device="cpu"
            ) as handle:
                actual_keys = {str(key) for key in handle.keys()}
                for tensor_key in expected_keys:
                    if tensor_key not in actual_keys:
                        raise TopologyMismatchError(
                            "checkpoint shard is missing indexed tensor key",
                            model_id=model_id,
                            tensor_key=tensor_key,
                            details={
                                "checkpoint_path": str(checkpoint_dir),
                                "shard_filename": shard_filename,
                            },
                        )
                    tensor_slice = handle.get_slice(tensor_key)
                    shape = tuple(int(dimension) for dimension in tensor_slice.get_shape())
                    tensors[tensor_key] = CheckpointTensorMetadata(
                        tensor_key=tensor_key,
                        shape=shape,
                        dtype=str(tensor_slice.get_dtype()),
                        shard_filename=shard_filename,
                        metadata={"checkpoint_path": str(checkpoint_dir)},
                    )
        except TopologyMismatchError:
            raise
        except Exception as exc:
            raise TopologyMismatchError(
                "checkpoint safetensors shard could not be read",
                model_id=model_id,
                details={"checkpoint_path": str(checkpoint_dir), "shard_filename": shard_filename},
            ) from exc
    return dict(sorted(tensors.items()))


def _resolve_shard_path(checkpoint_dir: Path, shard_filename: str) -> Path:
    root = checkpoint_dir.resolve()
    shard_path = Path(shard_filename)
    if shard_path.is_absolute():
        raise ShapeInvariantViolationError(
            "checkpoint index shard filename must be relative",
            model_id=str(root),
            details={"checkpoint_path": str(root), "shard_filename": shard_filename},
        )
    resolved = (root / shard_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ShapeInvariantViolationError(
            "checkpoint index shard path escapes checkpoint directory",
            model_id=str(root),
            details={"checkpoint_path": str(root), "shard_filename": shard_filename},
        ) from exc
    if not resolved.is_file():
        raise TopologyMismatchError(
            "checkpoint shard file is missing",
            model_id=str(root),
            details={"checkpoint_path": str(root), "shard_filename": shard_filename},
        )
    return resolved


__all__ = [
    "CheckpointTensorMetadata",
    "LocalSafetensorsCheckpoint",
    "open_local_safetensors_checkpoint",
]
