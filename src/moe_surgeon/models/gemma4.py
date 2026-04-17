"""Gemma 4 backend adapter with config-first topology discovery.

This module stays import-light on import. Runtime-heavy dependencies are only
loaded inside ``load()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Sequence, cast

from moe_surgeon.models.backend import BackendSignature, LoadedBackendBundle, TensorMetadata
from moe_surgeon.models.errors import (
    ShapeInvariantViolationError,
    TopologyMismatchError,
    UnsupportedModelError,
)
from moe_surgeon.schemas import LayerTopology, ModelHandle, RouterState

_BACKEND_VERSION = "1.0.0"
_SUPPORTED_MODEL_TYPE = "gemma4"
_SUPPORTED_ARCHITECTURE = "Gemma4ForConditionalGeneration"
_SUPPORT_ADDED_DATE = "2026-04-01"
_DEFAULT_LAYER_PREFIX = "model.language_model.layers.{layer_index}"
_REQUIRED_LAYER_KEYS = {
    "router_proj": "router.proj.weight",
    "router_scale": "router.scale",
    "router_per_expert_scale": "router.per_expert_scale",
    "experts_gate_up_proj": "experts.gate_up_proj",
    "experts_down_proj": "experts.down_proj",
}


@dataclass(frozen=True)
class Gemma4TopologyConfig:
    """Normalized Gemma 4 text-stack topology metadata."""

    model_id: str
    revision: str | None
    num_hidden_layers: int
    hidden_size: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int | None
    moe_layer_indices: tuple[int, ...]
    raw_config: Mapping[str, object]


class Gemma4Backend:
    """Backend adapter for Gemma 4 conditional-generation checkpoints."""

    name = "gemma4"
    backend_version = _BACKEND_VERSION

    def supports(self, signature: BackendSignature) -> bool:
        """Return whether the provided lightweight signature matches Gemma 4."""

        if signature.model_type == _SUPPORTED_MODEL_TYPE:
            return True
        if signature.architecture == _SUPPORTED_ARCHITECTURE:
            return True

        raw_architectures = signature.config.get("architectures")
        if isinstance(raw_architectures, Sequence) and not isinstance(raw_architectures, (str, bytes)):
            return any(
                isinstance(item, str) and item == _SUPPORTED_ARCHITECTURE for item in raw_architectures
            )
        return False

    def load(
        self,
        signature: BackendSignature,
        *,
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
    ) -> LoadedBackendBundle:
        """Load a Gemma 4 checkpoint with explicit runtime capability guards."""

        topology = self._parse_topology_config(signature)
        source = signature.source_path or signature.model_id
        transformers_version = self._installed_version("transformers")
        torch_version = self._installed_version("torch")

        try:
            transformers_module = import_module("transformers")
        except Exception as exc:  # pragma: no cover - exercised by environment issues
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=(self.name,),
                details={
                    "backend_name": self.name,
                    "reason": "transformers import failed",
                    "source": source,
                    "support_added_on": _SUPPORT_ADDED_DATE,
                },
            ) from exc

        model_class = getattr(transformers_module, _SUPPORTED_ARCHITECTURE, None)
        tokenizer_class = getattr(transformers_module, "AutoTokenizer", None)
        if model_class is None or tokenizer_class is None:
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=(self.name,),
                details={
                    "backend_name": self.name,
                    "installed_transformers_version": transformers_version or "unknown",
                    "required_symbol": _SUPPORTED_ARCHITECTURE,
                    "support_added_on": _SUPPORT_ADDED_DATE,
                    "guidance": (
                        "Upgrade transformers to a release published on or after "
                        f"{_SUPPORT_ADDED_DATE} with Gemma4 support."
                    ),
                    "source": source,
                },
            )

        torch_module = import_module("torch")
        resolved_dtype = self._resolve_torch_dtype(torch_module, dtype)
        tokenizer = tokenizer_class.from_pretrained(source, revision=signature.revision)
        model = model_class.from_pretrained(
            source,
            revision=signature.revision,
            torch_dtype=resolved_dtype,
        )
        raw_config = getattr(model, "config", None)
        bundle_config = self._coerce_runtime_config(raw_config, signature=signature)
        self._parse_topology_config(
            BackendSignature(
                model_id=signature.model_id,
                architecture=signature.architecture,
                model_type=signature.model_type,
                revision=signature.revision,
                source_path=signature.source_path,
                config=bundle_config,
                metadata=signature.metadata,
            )
        )

        model_dtype = getattr(model, "dtype", None)
        handle_dtype = str(model_dtype) if model_dtype is not None else dtype
        model_handle = ModelHandle(
            model_id=signature.model_id,
            revision=topology.revision,
            backend_name=self.name,
            source_path=signature.source_path,
            tokenizer_id=source,
            framework_version=transformers_version,
            device=device,
            dtype=handle_dtype,
            seed=seed,
            metadata={
                "backend_version": self.backend_version,
                "torch_dtype": handle_dtype or "unknown",
                "torch_version": torch_version or "unknown",
            },
        )
        return LoadedBackendBundle(
            backend_name=self.name,
            model_handle=model_handle,
            model=model,
            config=bundle_config,
            tokenizer=tokenizer,
            metadata={
                "backend_version": self.backend_version,
                "transformers_version": transformers_version or "unknown",
            },
        )

    def iter_layers(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        """Return the deterministic Gemma 4 MoE layer sequence."""

        return self.extract_topology(bundle)

    def extract_topology(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        """Discover MoE decoder layers from config and state-key metadata."""

        topology = self._parse_bundle_topology(bundle)
        layers: list[LayerTopology] = []
        for layer_index in topology.moe_layer_indices:
            tensor_keys = self.resolve_layer_tensor_keys(bundle, layer_index=layer_index)
            layer_name = self._layer_prefix(bundle, layer_index=layer_index)
            layers.append(
                LayerTopology(
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_type="gemma4_moe_decoder",
                    expert_count=topology.num_experts,
                    top_k=topology.top_k,
                    hidden_size=topology.hidden_size,
                    moe_intermediate_size=topology.moe_intermediate_size,
                    expert_dim=topology.moe_intermediate_size,
                    ffn_in_features=topology.hidden_size,
                    ffn_out_features=topology.hidden_size,
                    layer_ref=f"layer_{layer_index}",
                    module_paths=dict(tensor_keys),
                    metadata={"backend_version": self.backend_version},
                )
            )
        return tuple(layers)

    def resolve_layer_tensor_keys(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer_index: int,
    ) -> Mapping[str, str]:
        """Resolve and validate required tensor keys for one Gemma 4 MoE layer."""

        available = self._state_index(bundle)
        prefix = self._layer_prefix(bundle, layer_index=layer_index)
        resolved = {name: f"{prefix}.{suffix}" for name, suffix in _REQUIRED_LAYER_KEYS.items()}
        missing = [tensor_key for tensor_key in resolved.values() if tensor_key not in available]
        if missing:
            raise TopologyMismatchError(
                "missing Gemma4 MoE tensor keys",
                model_id=bundle.model_handle.model_id,
                layer_index=layer_index,
                details={
                    "missing_keys": ",".join(sorted(missing)),
                    "layer_prefix": prefix,
                },
            )
        return dict(sorted(resolved.items()))

    def extract_router_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> RouterState:
        """Return router metadata for a validated Gemma 4 MoE layer."""

        topology = self._parse_bundle_topology(bundle)
        tensor_keys = self.resolve_layer_tensor_keys(bundle, layer_index=layer.layer_index)
        state_index = self._state_index(bundle)
        projection_shape = self._tensor_shape(state_index[tensor_keys["router_proj"]])
        scale_shape = self._tensor_shape(state_index[tensor_keys["router_scale"]], allow_scalar=True)
        per_expert_scale_shape = self._tensor_shape(state_index[tensor_keys["router_per_expert_scale"]])

        self._validate_router_projection(
            projection_shape,
            bundle=bundle,
            layer=layer,
            expected_experts=topology.num_experts,
            expected_hidden_size=topology.hidden_size,
            tensor_key=tensor_keys["router_proj"],
        )
        self._validate_per_expert_scale(
            per_expert_scale_shape,
            bundle=bundle,
            layer=layer,
            expected_experts=topology.num_experts,
            tensor_key=tensor_keys["router_per_expert_scale"],
        )
        self._validate_router_scale(scale_shape, bundle=bundle, layer=layer, tensor_key=tensor_keys["router_scale"])

        return RouterState(
            layer_index=layer.layer_index,
            num_experts=topology.num_experts,
            top_k=topology.top_k,
            logits_shape=(0, topology.num_experts),
            projection_shape=projection_shape,
            per_expert_scale_shape=per_expert_scale_shape,
            route_scale_present=True,
            metadata={
                "router_proj_key": tensor_keys["router_proj"],
                "router_scale_key": tensor_keys["router_scale"],
                "router_per_expert_scale_key": tensor_keys["router_per_expert_scale"],
            },
        )

    def extract_expert_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> Mapping[str, TensorMetadata]:
        """Return expert tensor metadata for a validated Gemma 4 MoE layer."""

        topology = self._parse_bundle_topology(bundle)
        tensor_keys = self.resolve_layer_tensor_keys(bundle, layer_index=layer.layer_index)
        state_index = self._state_index(bundle)
        gate_up_shape = self._tensor_shape(state_index[tensor_keys["experts_gate_up_proj"]])
        down_shape = self._tensor_shape(state_index[tensor_keys["experts_down_proj"]])
        gate_up_dtype = self._tensor_dtype(state_index[tensor_keys["experts_gate_up_proj"]])
        down_dtype = self._tensor_dtype(state_index[tensor_keys["experts_down_proj"]])

        self._validate_expert_tensor(
            gate_up_shape,
            bundle=bundle,
            layer=layer,
            expected_experts=topology.num_experts,
            expected_hidden_size=topology.hidden_size,
            tensor_key=tensor_keys["experts_gate_up_proj"],
        )
        self._validate_expert_tensor(
            down_shape,
            bundle=bundle,
            layer=layer,
            expected_experts=topology.num_experts,
            expected_hidden_size=topology.hidden_size,
            tensor_key=tensor_keys["experts_down_proj"],
        )

        return {
            "down_proj": TensorMetadata(
                tensor_key=tensor_keys["experts_down_proj"],
                shape=down_shape,
                dtype=down_dtype,
                metadata={"backend_version": self.backend_version, "layer_index": layer.layer_index},
            ),
            "gate_up_proj": TensorMetadata(
                tensor_key=tensor_keys["experts_gate_up_proj"],
                shape=gate_up_shape,
                dtype=gate_up_dtype,
                metadata={"backend_version": self.backend_version, "layer_index": layer.layer_index},
            ),
        }

    def validate_bundle(self, bundle: LoadedBackendBundle) -> None:
        """Validate Gemma 4 bundle-level topology invariants."""

        layers = self.extract_topology(bundle)
        if not layers:
            raise TopologyMismatchError(
                "Gemma4 bundle must expose at least one MoE layer",
                model_id=bundle.model_handle.model_id,
            )
        for layer in layers:
            self.validate_layer(bundle, layer=layer)

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        """Validate one Gemma 4 MoE layer against topology and tensor invariants."""

        active_router_state = router_state or self.extract_router_state(bundle, layer=layer)
        if active_router_state.num_experts != layer.expert_count:
            raise TopologyMismatchError(
                "router expert count does not match layer topology",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                details={
                    "router_num_experts": active_router_state.num_experts,
                    "layer_expert_count": layer.expert_count,
                },
            )
        if active_router_state.top_k != layer.top_k:
            raise TopologyMismatchError(
                "router top_k does not match layer topology",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                details={"router_top_k": active_router_state.top_k, "layer_top_k": layer.top_k},
            )
        self.extract_expert_state(bundle, layer=layer)

    def _parse_bundle_topology(self, bundle: LoadedBackendBundle) -> Gemma4TopologyConfig:
        return self._parse_topology_mapping(
            model_id=bundle.model_handle.model_id,
            revision=bundle.model_handle.revision,
            config=bundle.config,
        )

    def _parse_topology_config(self, signature: BackendSignature) -> Gemma4TopologyConfig:
        if not self.supports(signature):
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=(self.name,),
                details={
                    "backend_name": self.name,
                    "architecture": signature.architecture or "unknown",
                    "model_type": signature.model_type or "unknown",
                },
            )
        return self._parse_topology_mapping(
            model_id=signature.model_id,
            revision=signature.revision,
            config=signature.config,
        )

    def _parse_topology_mapping(
        self,
        *,
        model_id: str,
        revision: str | None,
        config: Mapping[str, object],
    ) -> Gemma4TopologyConfig:
        top_level = dict(config)
        text_config = self._text_config(top_level)
        architecture = self._architecture(top_level)
        if architecture != _SUPPORTED_ARCHITECTURE:
            raise UnsupportedModelError(
                model_id,
                available_backends=(self.name,),
                details={"backend_name": self.name, "architecture": architecture or "unknown"},
            )

        model_type = top_level.get("model_type")
        if model_type != _SUPPORTED_MODEL_TYPE:
            raise UnsupportedModelError(
                model_id,
                available_backends=(self.name,),
                details={"backend_name": self.name, "model_type": str(model_type or "unknown")},
            )

        num_hidden_layers = self._required_int(text_config, "num_hidden_layers", model_id=model_id)
        hidden_size = self._required_int(text_config, "hidden_size", model_id=model_id)
        num_experts = self._required_int(
            text_config,
            "num_experts",
            model_id=model_id,
            aliases=("num_local_experts",),
        )
        top_k = self._required_int(
            text_config,
            "top_k_experts",
            model_id=model_id,
            aliases=("top_k",),
        )
        enable_moe_block = self._required_bool(text_config, "enable_moe_block", model_id=model_id)
        if not enable_moe_block:
            raise TopologyMismatchError(
                "Gemma4 text_config.enable_moe_block must be true",
                model_id=model_id,
            )
        if top_k > num_experts:
            raise TopologyMismatchError(
                "top_k cannot exceed experts count",
                model_id=model_id,
                details={"top_k": top_k, "num_experts": num_experts},
            )

        moe_intermediate_size = self._optional_int(
            text_config,
            "moe_intermediate_size",
            model_id=model_id,
        )
        moe_layer_indices = self._moe_layer_indices(
            text_config=text_config,
            num_hidden_layers=num_hidden_layers,
            model_id=model_id,
        )
        return Gemma4TopologyConfig(
            model_id=model_id,
            revision=revision,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            moe_intermediate_size=moe_intermediate_size,
            moe_layer_indices=moe_layer_indices,
            raw_config=top_level,
        )

    def _text_config(self, config: Mapping[str, object]) -> Mapping[str, object]:
        text_config = config.get("text_config")
        if not isinstance(text_config, Mapping):
            raise TopologyMismatchError("Gemma4 config must include text_config mapping")
        return text_config

    def _architecture(self, config: Mapping[str, object]) -> str | None:
        raw_architectures = config.get("architectures")
        if isinstance(raw_architectures, Sequence) and not isinstance(raw_architectures, (str, bytes)):
            for item in raw_architectures:
                if isinstance(item, str) and item:
                    return item
        architecture = config.get("architecture")
        if isinstance(architecture, str) and architecture:
            return architecture
        return None

    def _required_int(
        self,
        payload: Mapping[str, object],
        field_name: str,
        *,
        model_id: str,
        aliases: Sequence[str] = (),
    ) -> int:
        raw_value: object | None = None
        actual_name = field_name
        for candidate in (field_name, *aliases):
            if candidate in payload:
                raw_value = payload[candidate]
                actual_name = candidate
                break
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
            raise TopologyMismatchError(
                f"Gemma4 config field {actual_name} must be positive int",
                model_id=model_id,
            )
        return raw_value

    def _optional_int(
        self,
        payload: Mapping[str, object],
        field_name: str,
        *,
        model_id: str,
    ) -> int | None:
        raw_value = payload.get(field_name)
        if raw_value is None:
            return None
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
            raise TopologyMismatchError(
                f"Gemma4 config field {field_name} must be positive int when provided",
                model_id=model_id,
            )
        return raw_value

    def _required_bool(self, payload: Mapping[str, object], field_name: str, *, model_id: str) -> bool:
        raw_value = payload.get(field_name)
        if not isinstance(raw_value, bool):
            raise TopologyMismatchError(
                f"Gemma4 config field {field_name} must be bool",
                model_id=model_id,
            )
        return raw_value

    def _moe_layer_indices(
        self,
        *,
        text_config: Mapping[str, object],
        num_hidden_layers: int,
        model_id: str,
    ) -> tuple[int, ...]:
        raw_layer_indices = text_config.get("moe_layer_indices")
        if raw_layer_indices is None:
            return tuple(range(num_hidden_layers))
        if not isinstance(raw_layer_indices, Sequence) or isinstance(raw_layer_indices, (str, bytes)):
            raise TopologyMismatchError("Gemma4 moe_layer_indices must be a sequence", model_id=model_id)
        normalized: list[int] = []
        for index, value in enumerate(raw_layer_indices):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TopologyMismatchError(
                    "Gemma4 moe_layer_indices entries must be int",
                    model_id=model_id,
                    details={"entry_index": index},
                )
            if value < 0 or value >= num_hidden_layers:
                raise TopologyMismatchError(
                    "Gemma4 moe_layer_indices entry is out of range",
                    model_id=model_id,
                    details={"entry_index": index, "layer_index": value, "num_hidden_layers": num_hidden_layers},
                )
            normalized.append(value)
        ordered = tuple(sorted(set(normalized)))
        if not ordered:
            raise TopologyMismatchError("Gemma4 moe_layer_indices cannot be empty", model_id=model_id)
        return ordered

    def _coerce_runtime_config(
        self,
        raw_config: object,
        *,
        signature: BackendSignature,
    ) -> Mapping[str, object]:
        if raw_config is None:
            return dict(signature.config)
        if hasattr(raw_config, "to_dict"):
            candidate = raw_config.to_dict()
            if isinstance(candidate, Mapping):
                return dict(candidate)
        if isinstance(raw_config, Mapping):
            return dict(raw_config)
        return dict(signature.config)

    def _installed_version(self, package_name: str) -> str | None:
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            return None

    def _resolve_torch_dtype(self, torch_module: Any, dtype: str | None) -> object | None:
        if dtype is None:
            return None
        torch_dtype = getattr(torch_module, dtype, None)
        if torch_dtype is None:
            raise UnsupportedModelError(
                "torch",
                available_backends=(self.name,),
                details={"backend_name": self.name, "invalid_dtype": dtype},
            )
        return cast(object, torch_dtype)

    def _state_index(self, bundle: LoadedBackendBundle) -> Mapping[str, object]:
        metadata_state = bundle.metadata.get("state_dict")
        if isinstance(metadata_state, Mapping):
            return metadata_state

        metadata_keys = bundle.metadata.get("state_keys")
        if isinstance(metadata_keys, Sequence) and not isinstance(metadata_keys, (str, bytes)):
            return {str(key): None for key in metadata_keys}

        state_dict = getattr(bundle.model, "state_dict", None)
        if callable(state_dict):
            loaded_state = state_dict()
            if isinstance(loaded_state, Mapping):
                return loaded_state

        raise TopologyMismatchError(
            "Gemma4 topology discovery requires state_dict or state_keys metadata",
            model_id=bundle.model_handle.model_id,
        )

    def _layer_prefix(self, bundle: LoadedBackendBundle, *, layer_index: int) -> str:
        template = bundle.metadata.get("layer_prefix_template", _DEFAULT_LAYER_PREFIX)
        if not isinstance(template, str) or "{layer_index}" not in template:
            raise TopologyMismatchError(
                "Gemma4 layer prefix template must contain {layer_index}",
                model_id=bundle.model_handle.model_id,
            )
        return template.format(layer_index=layer_index)

    def _tensor_shape(self, value: object, *, allow_scalar: bool = False) -> tuple[int, ...]:
        shape = getattr(value, "shape", value)
        if shape is None:
            if allow_scalar:
                return ()
            raise ShapeInvariantViolationError("tensor metadata must expose shape")
        if isinstance(shape, tuple):
            raw_shape = shape
        elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
            raw_shape = tuple(shape)
        else:
            raise ShapeInvariantViolationError("tensor shape must be sequence")
        normalized = tuple(int(dimension) for dimension in raw_shape)
        if any(dimension < 0 for dimension in normalized):
            raise ShapeInvariantViolationError("tensor shape dimensions must be non-negative")
        if not allow_scalar and not normalized:
            raise ShapeInvariantViolationError("tensor shape cannot be scalar")
        return normalized

    def _tensor_dtype(self, value: object) -> str | None:
        raw_dtype = getattr(value, "dtype", None)
        if raw_dtype is None:
            return None
        return str(raw_dtype)

    def _validate_router_projection(
        self,
        shape: tuple[int, ...],
        *,
        bundle: LoadedBackendBundle,
        layer: LayerTopology,
        expected_experts: int,
        expected_hidden_size: int,
        tensor_key: str,
    ) -> None:
        expected_shape = (expected_experts, expected_hidden_size)
        if shape != expected_shape:
            raise TopologyMismatchError(
                "Gemma4 router projection shape mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=expected_shape,
                actual_shape=shape,
            )

    def _validate_per_expert_scale(
        self,
        shape: tuple[int, ...],
        *,
        bundle: LoadedBackendBundle,
        layer: LayerTopology,
        expected_experts: int,
        tensor_key: str,
    ) -> None:
        expected_shape = (expected_experts,)
        if shape != expected_shape:
            raise TopologyMismatchError(
                "Gemma4 per-expert scale shape mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=expected_shape,
                actual_shape=shape,
            )

    def _validate_router_scale(
        self,
        shape: tuple[int, ...],
        *,
        bundle: LoadedBackendBundle,
        layer: LayerTopology,
        tensor_key: str,
    ) -> None:
        if shape not in ((), (1,)):
            raise TopologyMismatchError(
                "Gemma4 router scale must be scalar",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=(),
                actual_shape=shape,
            )

    def _validate_expert_tensor(
        self,
        shape: tuple[int, ...],
        *,
        bundle: LoadedBackendBundle,
        layer: LayerTopology,
        expected_experts: int,
        expected_hidden_size: int,
        tensor_key: str,
    ) -> None:
        if len(shape) < 2:
            raise ShapeInvariantViolationError(
                "Gemma4 expert tensor rank must be >= 2",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                actual_shape=shape,
            )
        if shape[0] != expected_experts:
            raise ShapeInvariantViolationError(
                "Gemma4 expert tensor expert axis mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=(expected_experts,),
                actual_shape=(shape[0],),
            )
        if expected_hidden_size not in shape[1:]:
            raise ShapeInvariantViolationError(
                "Gemma4 expert tensor must include hidden_size on a non-expert axis",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                details={"hidden_size": expected_hidden_size, "actual_shape": "x".join(map(str, shape))},
            )


__all__ = ["Gemma4Backend", "Gemma4TopologyConfig"]
