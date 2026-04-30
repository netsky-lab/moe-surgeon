"""GGUF checkpoint reader and Gemma4 topology adapter.

This module keeps GGUF support metadata-first. It can scan router tensors that
are stored as plain F32 GGUF tensors, while quantized expert tensor rewrites are
left to a dedicated apply/export path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from moe_surgeon.models.backend import BackendSignature, LoadedBackendBundle, TensorMetadata
from moe_surgeon.models.errors import (
    BackendMismatchError,
    ShapeInvariantViolationError,
    TopologyMismatchError,
)
from moe_surgeon.schemas import LayerTopology, ModelHandle, RouterState

_BACKEND_VERSION = "0.1.0"
_GGUF_SUFFIX = ".gguf"
_SUPPORTED_ARCHITECTURE = "gemma4"
DEFAULT_REGISTRY_PRIORITY = 200


@dataclass(frozen=True)
class GgufTensorMetadata:
    """Minimal deterministic descriptor for one GGUF tensor."""

    tensor_key: str
    shape: tuple[int, ...]
    data_shape: tuple[int, ...]
    dtype: str
    tensor_type: str
    n_bytes: int
    data_offset: int


@dataclass(frozen=True)
class LocalGgufCheckpoint:
    """Lightweight local GGUF checkpoint handle."""

    checkpoint_path: Path
    fields: Mapping[str, object]
    tensors: Mapping[str, GgufTensorMetadata]
    _tensor_payload_cache: Mapping[str, object] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def model_id(self) -> str:
        name = self.fields.get("general.name")
        if isinstance(name, str) and name.strip():
            return name
        return self.checkpoint_path.stem

    @property
    def revision(self) -> str | None:
        source_hash = self.fields.get("general.source.hash")
        if isinstance(source_hash, str) and source_hash.strip():
            return source_hash
        return None

    @property
    def architecture(self) -> str:
        value = self.fields.get("general.architecture")
        return value if isinstance(value, str) else "unknown"

    @property
    def file_fingerprint(self) -> str:
        stat = self.checkpoint_path.stat()
        payload = (
            f"{self.checkpoint_path.name}:{stat.st_size}:"
            f"{self.fields.get('GGUF.version')}:{self.fields.get('GGUF.tensor_count')}"
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def state_keys(self) -> tuple[str, ...]:
        """Return tensor keys in deterministic order."""

        return tuple(sorted(self.tensors))

    def tensor_metadata(self, tensor_names: Sequence[str] | None = None) -> tuple[GgufTensorMetadata, ...]:
        """Return deterministic metadata for all or a requested tensor subset."""

        names = self._normalize_tensor_names(tensor_names)
        return tuple(self.tensors[name] for name in names)

    def load_tensors(self, tensor_names: Sequence[str]) -> dict[str, object]:
        """Load only requested unquantized GGUF tensors as torch tensors."""

        import torch

        names = self._normalize_tensor_names(tensor_names)
        by_name = self._tensor_payloads()
        loaded: dict[str, object] = {}
        for name in names:
            tensor = by_name.get(name)
            if tensor is None:
                raise TopologyMismatchError(
                    "GGUF tensor is missing from checkpoint payload",
                    model_id=self.model_id,
                    tensor_key=name,
                    details={"checkpoint_path": str(self.checkpoint_path)},
                )
            tensor_payload = cast(Any, tensor)
            tensor_type = _tensor_type_name(tensor_payload)
            if tensor_type != "F32":
                raise ShapeInvariantViolationError(
                    "GGUF scan can only materialize F32 tensors",
                    model_id=self.model_id,
                    tensor_key=name,
                    details={"tensor_type": tensor_type},
                )
            loaded[name] = torch.from_numpy(tensor_payload.data.copy())
        return loaded

    def _tensor_payloads(self) -> Mapping[str, object]:
        cached = self._tensor_payload_cache
        if cached is not None:
            return cached
        from gguf import GGUFReader

        reader = GGUFReader(self.checkpoint_path, mode="r")
        payloads = {str(tensor.name): tensor for tensor in reader.tensors}
        object.__setattr__(self, "_tensor_payload_cache", payloads)
        return payloads

    def to_backend_signature(self) -> BackendSignature:
        """Return the backend resolver signature for this GGUF checkpoint."""

        return BackendSignature(
            model_id=self.model_id,
            architecture=self.architecture,
            model_type=self.architecture,
            revision=self.revision,
            source_path=str(self.checkpoint_path),
            config=dict(self.fields),
            metadata={"format": "gguf", "checkpoint_path": str(self.checkpoint_path)},
        )

    def _normalize_tensor_names(self, tensor_names: Sequence[str] | None) -> tuple[str, ...]:
        if tensor_names is None:
            return self.state_keys()
        names = tuple(str(name) for name in tensor_names)
        missing = tuple(name for name in names if name not in self.tensors)
        if missing:
            raise TopologyMismatchError(
                "GGUF checkpoint is missing requested tensor",
                model_id=self.model_id,
                tensor_key=missing[0],
                details={"checkpoint_path": str(self.checkpoint_path)},
            )
        return tuple(sorted(names))


class Gemma4GgufBackend:
    """Backend adapter for Gemma4 GGUF checkpoints."""

    name = "gemma4-gguf"
    backend_version = _BACKEND_VERSION

    def supports(self, signature: BackendSignature) -> bool:
        """Return whether this backend can handle the provided signature."""

        if signature.metadata.get("format") != "gguf":
            return False
        architecture = signature.architecture or signature.model_type
        return architecture == _SUPPORTED_ARCHITECTURE

    def load(
        self,
        signature: BackendSignature,
        *,
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
    ) -> LoadedBackendBundle:
        """Load a local GGUF file into a lightweight bundle."""

        if signature.source_path is None:
            raise TopologyMismatchError("GGUF backend requires source_path")
        checkpoint = open_local_gguf_checkpoint(signature.source_path)
        bundle = LoadedBackendBundle(
            backend_name=self.name,
            model_handle=ModelHandle(
                model_id=checkpoint.model_id,
                revision=checkpoint.revision,
                backend_name=self.name,
                source_path=str(checkpoint.checkpoint_path),
                framework_version="gguf",
                device=device,
                dtype=dtype,
                seed=seed,
                metadata={
                    "format": "gguf",
                    "file_fingerprint": checkpoint.file_fingerprint,
                    "tensor_count": len(checkpoint.tensors),
                },
            ),
            model=object(),
            config=checkpoint.fields,
            metadata={
                "gguf_checkpoint": checkpoint,
                "state_keys": checkpoint.state_keys(),
                "state_dict": {item.tensor_key: item for item in checkpoint.tensor_metadata()},
                "backend_version": self.backend_version,
            },
        )
        self.validate_bundle(bundle)
        return bundle

    def iter_layers(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        """Yield deterministic GGUF MoE layer topology metadata."""

        return self.extract_topology(bundle)

    def extract_topology(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        """Return the full deterministic topology snapshot."""

        config = bundle.config
        block_count = _required_int(config, "gemma4.block_count", model_id=bundle.model_handle.model_id)
        hidden_size = _required_int(config, "gemma4.embedding_length", model_id=bundle.model_handle.model_id)
        expert_count = _required_int(config, "gemma4.expert_count", model_id=bundle.model_handle.model_id)
        top_k = _required_int(config, "gemma4.expert_used_count", model_id=bundle.model_handle.model_id)
        moe_width = _required_int(
            config,
            "gemma4.expert_feed_forward_length",
            model_id=bundle.model_handle.model_id,
        )
        dense_width = _optional_int(config, "gemma4.feed_forward_length")
        layers: list[LayerTopology] = []
        for layer_index in range(block_count):
            paths = _layer_paths(layer_index)
            layers.append(
                LayerTopology(
                    layer_index=layer_index,
                    layer_name=f"blk.{layer_index}",
                    layer_type="gemma4_gguf_moe",
                    expert_count=expert_count,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_width,
                    expert_dim=moe_width,
                    ffn_in_features=hidden_size,
                    ffn_out_features=hidden_size,
                    layer_ref=f"layer_{layer_index}",
                    module_paths=paths,
                    metadata={
                        "format": "gguf",
                        "dense_feed_forward_length": dense_width,
                    },
                )
            )
        return tuple(layers)

    def extract_router_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> RouterState:
        """Extract router metadata for a single GGUF layer."""

        tensor_index = _tensor_index(bundle)
        router_proj_key = layer.module_paths["router_proj"]
        per_expert_key = layer.module_paths["router_per_expert_scale"]
        router_proj = _required_tensor(tensor_index, router_proj_key, bundle=bundle, layer=layer)
        per_expert = _required_tensor(tensor_index, per_expert_key, bundle=bundle, layer=layer)
        return RouterState(
            layer_index=layer.layer_index,
            num_experts=layer.expert_count,
            top_k=layer.top_k,
            logits_shape=(1, layer.expert_count),
            top_k_indices_shape=(1, layer.top_k),
            top_k_weights_shape=(1, layer.top_k),
            projection_shape=router_proj.data_shape,
            per_expert_scale_shape=per_expert.data_shape,
            has_router_probabilities=True,
            route_scale_present=True,
            metadata={
                "router_proj_key": router_proj_key,
                "router_per_expert_scale_key": per_expert_key,
                "format": "gguf",
            },
        )

    def extract_expert_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> Mapping[str, TensorMetadata]:
        """Return per-layer expert tensor descriptors."""

        tensor_index = _tensor_index(bundle)
        out: dict[str, TensorMetadata] = {}
        for role in ("experts_gate_up_proj", "experts_down_proj"):
            key = layer.module_paths[role]
            tensor = _required_tensor(tensor_index, key, bundle=bundle, layer=layer)
            out[role] = TensorMetadata(
                tensor_key=key,
                shape=tensor.shape,
                dtype=tensor.tensor_type,
                metadata={"format": "gguf", "n_bytes": tensor.n_bytes},
            )
        return out

    def validate_bundle(self, bundle: LoadedBackendBundle) -> None:
        """Validate required Gemma4 GGUF metadata and layer tensors."""

        if bundle.config.get("general.architecture") != _SUPPORTED_ARCHITECTURE:
            raise BackendMismatchError(
                "GGUF backend requires general.architecture=gemma4",
                model_id=bundle.model_handle.model_id,
                backend_name=self.name,
            )
        for layer in self.extract_topology(bundle):
            self.validate_layer(bundle, layer=layer, router_state=self.extract_router_state(bundle, layer=layer))

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        """Validate GGUF tensor presence and router shapes for one layer."""

        tensor_index = _tensor_index(bundle)
        for role, tensor_key in layer.module_paths.items():
            _required_tensor(tensor_index, tensor_key, bundle=bundle, layer=layer, role=role)

        router_proj = tensor_index[layer.module_paths["router_proj"]]
        if router_proj.data_shape != (layer.expert_count, layer.hidden_size):
            raise TopologyMismatchError(
                "Gemma4 GGUF router projection shape mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=router_proj.tensor_key,
                expected_shape=(layer.expert_count, layer.hidden_size),
                actual_shape=router_proj.data_shape,
            )
        per_expert = tensor_index[layer.module_paths["router_per_expert_scale"]]
        if per_expert.data_shape != (layer.expert_count,):
            raise TopologyMismatchError(
                "Gemma4 GGUF per-expert scale shape mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=per_expert.tensor_key,
                expected_shape=(layer.expert_count,),
                actual_shape=per_expert.data_shape,
            )
        if router_state is not None and router_state.projection_shape != router_proj.data_shape:
            raise TopologyMismatchError(
                "Gemma4 GGUF router_state projection shape mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )

    def resolve_router_module(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> object:
        """GGUF files do not expose Python router modules."""

        raise TopologyMismatchError(
            "GGUF backend does not expose runtime router modules for bench",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
        )


def open_local_gguf_checkpoint(path: str | Path) -> LocalGgufCheckpoint:
    """Open a local GGUF file and read deterministic metadata."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.suffix.lower() != _GGUF_SUFFIX:
        raise TopologyMismatchError(
            "GGUF checkpoint path must point to a .gguf file",
            details={"checkpoint_path": str(checkpoint_path)},
        )
    try:
        from gguf import GGUFReader

        reader = GGUFReader(checkpoint_path, mode="r")
    except Exception as exc:
        raise TopologyMismatchError(
            "GGUF checkpoint could not be opened",
            details={"checkpoint_path": str(checkpoint_path)},
        ) from exc

    fields = {str(name): _field_value(field) for name, field in reader.fields.items()}
    tensors = {
        str(tensor.name): GgufTensorMetadata(
            tensor_key=str(tensor.name),
            shape=tuple(int(item) for item in tensor.shape),
            data_shape=tuple(int(item) for item in getattr(tensor.data, "shape", ())),
            dtype=str(getattr(tensor.data, "dtype", "unknown")),
            tensor_type=_tensor_type_name(tensor),
            n_bytes=int(tensor.n_bytes),
            data_offset=int(tensor.data_offset),
        )
        for tensor in reader.tensors
    }
    if not tensors:
        raise ShapeInvariantViolationError(
            "GGUF checkpoint contains no tensors",
            model_id=str(fields.get("general.name") or checkpoint_path.stem),
            details={"checkpoint_path": str(checkpoint_path)},
        )
    return LocalGgufCheckpoint(checkpoint_path=checkpoint_path, fields=fields, tensors=tensors)


def default_registry_entry() -> tuple[Gemma4GgufBackend, int]:
    """Return the default registry entry for GGUF Gemma4 checkpoints."""

    return Gemma4GgufBackend(), DEFAULT_REGISTRY_PRIORITY


def _field_value(field: object) -> object:
    contents = getattr(field, "contents", None)
    if callable(contents):
        value = contents()
    else:
        value = field
    return _json_scalar_or_sequence(value)


def _json_scalar_or_sequence(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_scalar_or_sequence(tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_json_scalar_or_sequence(item) for item in value)
    return str(value)


def _tensor_type_name(tensor: object) -> str:
    tensor_type = getattr(tensor, "tensor_type", None)
    name = getattr(tensor_type, "name", None)
    return str(name if name is not None else tensor_type)


def _required_int(config: Mapping[str, object], key: str, *, model_id: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TopologyMismatchError(
            "GGUF checkpoint is missing required integer metadata",
            model_id=model_id,
            details={"metadata_key": key},
        )
    if value <= 0:
        raise TopologyMismatchError(
            "GGUF integer metadata must be positive",
            model_id=model_id,
            details={"metadata_key": key, "value": value},
        )
    return value


def _optional_int(config: Mapping[str, object], key: str) -> int | None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _layer_paths(layer_index: int) -> dict[str, str]:
    prefix = f"blk.{layer_index}"
    return {
        "router_proj": f"{prefix}.ffn_gate_inp.weight",
        "router_scale": f"{prefix}.ffn_gate_inp.scale",
        "router_per_expert_scale": f"{prefix}.ffn_down_exps.scale",
        "experts_gate_up_proj": f"{prefix}.ffn_gate_up_exps.weight",
        "experts_down_proj": f"{prefix}.ffn_down_exps.weight",
        "mlp_down_proj": f"{prefix}.ffn_down.weight",
        "mlp_gate_proj": f"{prefix}.ffn_gate.weight",
        "mlp_up_proj": f"{prefix}.ffn_up.weight",
        "pre_feedforward_layernorm_2": f"{prefix}.pre_ffw_norm_2.weight",
        "post_feedforward_layernorm": f"{prefix}.post_ffw_norm.weight",
        "post_feedforward_layernorm_1": f"{prefix}.post_ffw_norm_1.weight",
        "post_feedforward_layernorm_2": f"{prefix}.post_ffw_norm_2.weight",
    }


def _tensor_index(bundle: LoadedBackendBundle) -> Mapping[str, GgufTensorMetadata]:
    state = bundle.metadata.get("state_dict")
    if not isinstance(state, Mapping):
        raise TopologyMismatchError(
            "GGUF backend bundle requires tensor metadata state_dict",
            model_id=bundle.model_handle.model_id,
        )
    out: dict[str, GgufTensorMetadata] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, GgufTensorMetadata):
            raise TopologyMismatchError(
                "GGUF tensor metadata state_dict is malformed",
                model_id=bundle.model_handle.model_id,
            )
        out[key] = value
    return out


def _required_tensor(
    tensor_index: Mapping[str, GgufTensorMetadata],
    tensor_key: str,
    *,
    bundle: LoadedBackendBundle,
    layer: LayerTopology,
    role: str | None = None,
) -> GgufTensorMetadata:
    tensor = tensor_index.get(tensor_key)
    if tensor is None:
        raise TopologyMismatchError(
            "Gemma4 GGUF layer tensor is missing",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            tensor_key=tensor_key,
            details={} if role is None else {"tensor_role": role},
        )
    return tensor


__all__ = [
    "Gemma4GgufBackend",
    "GgufTensorMetadata",
    "LocalGgufCheckpoint",
    "default_registry_entry",
    "open_local_gguf_checkpoint",
]
