"""Gemma 4 backend adapter with config-first topology discovery.

This module stays import-light on import. Runtime-heavy dependencies are only
loaded inside ``load()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
from packaging.version import InvalidVersion, Version
import re
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
_MINIMUM_TRANSFORMERS_VERSION = "5.5.0"
_MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE = "2026-04-02"
GEMMA4_MIN_TRANSFORMERS_VERSION = _MINIMUM_TRANSFORMERS_VERSION
GEMMA4_SUPPORT_ADDED_ON = _SUPPORT_ADDED_DATE
_DEFAULT_LAYER_PREFIX = "model.language_model.layers.{layer_index}"
DEFAULT_REGISTRY_PRIORITY = 100
_REQUIRED_LAYER_KEYS = {
    "router_proj": "router.proj.weight",
    "router_scale": "router.scale",
    "router_per_expert_scale": "router.per_expert_scale",
    "experts_gate_up_proj": "experts.gate_up_proj",
    "experts_down_proj": "experts.down_proj",
    "mlp_down_proj": "mlp.down_proj.weight",
    "mlp_gate_proj": "mlp.gate_proj.weight",
    "mlp_up_proj": "mlp.up_proj.weight",
    "pre_feedforward_layernorm": "pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2": "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm": "post_feedforward_layernorm.weight",
    "post_feedforward_layernorm_1": "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2": "post_feedforward_layernorm_2.weight",
}


def gemma4_runtime_guidance(installed_transformers_version: str | None) -> str:
    """Return the canonical upgrade guidance for Gemma4 runtime support."""

    installed_version = installed_transformers_version or "unknown"
    return (
        "Upgrade transformers to a release published on or after "
        f"{GEMMA4_SUPPORT_ADDED_ON} with Gemma4 support "
        f"(>={GEMMA4_MIN_TRANSFORMERS_VERSION}; PyPI release date "
        f"{_MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE}); installed version is {installed_version}"
    )


def check_gemma4_runtime_support(
    *,
    model_id: str,
    source: str,
    installed_transformers_version: str | None,
) -> tuple[Any, Any]:
    """Validate the installed Transformers runtime and return required Gemma4 symbols."""

    try:
        transformers_module = import_module("transformers")
    except Exception as exc:  # pragma: no cover - exercised by environment issues
        raise UnsupportedModelError(
            model_id,
            available_backends=("gemma4",),
            details={
                "backend_name": "gemma4",
                "reason": "transformers import failed",
                "source": source,
                "minimum_transformers_version": GEMMA4_MIN_TRANSFORMERS_VERSION,
                "support_added_on": GEMMA4_SUPPORT_ADDED_ON,
                "guidance": gemma4_runtime_guidance(installed_transformers_version),
            },
        ) from exc

    if not _gemma4_meets_minimum_transformers_version(installed_transformers_version):
        raise UnsupportedModelError(
            model_id,
            available_backends=("gemma4",),
            details=_gemma4_runtime_support_details(
                source=source,
                transformers_version=installed_transformers_version,
                required_symbol=_SUPPORTED_ARCHITECTURE,
            ),
        )

    try:
        model_class = getattr(transformers_module, _SUPPORTED_ARCHITECTURE, None)
        tokenizer_class = getattr(transformers_module, "AutoTokenizer", None)
    except Exception as exc:
        raise UnsupportedModelError(
            model_id,
            available_backends=("gemma4",),
            details={
                **_gemma4_runtime_support_details(
                    source=source,
                    transformers_version=installed_transformers_version,
                    required_symbol=_SUPPORTED_ARCHITECTURE,
                ),
                "guidance": (
                    f"{gemma4_runtime_guidance(installed_transformers_version)} "
                    "If you already installed a supported transformers release, "
                    "reinstall a standard Hugging Face transformers build because the Gemma4 symbols are missing."
                ),
                "symbol_name": _SUPPORTED_ARCHITECTURE,
                "symbol_resolution_error": f"{exc.__class__.__name__}: {exc}",
            },
        ) from exc
    if model_class is None or tokenizer_class is None:
        required_symbol = _SUPPORTED_ARCHITECTURE if model_class is None else "AutoTokenizer"
        raise UnsupportedModelError(
            model_id,
            available_backends=("gemma4",),
            details={
                **_gemma4_runtime_support_details(
                    source=source,
                    transformers_version=installed_transformers_version,
                    required_symbol=required_symbol,
                ),
                "guidance": (
                    f"{gemma4_runtime_guidance(installed_transformers_version)} "
                    "If you already installed a supported transformers release, "
                    "reinstall a standard Hugging Face transformers build because the Gemma4 symbols are missing."
                ),
            },
        )
    return cast(Any, model_class), cast(Any, tokenizer_class)


def _gemma4_runtime_support_details(
    *,
    source: str,
    transformers_version: str | None,
    required_symbol: str,
) -> Mapping[str, object]:
    return {
        "backend_name": "gemma4",
        "installed_transformers_version": transformers_version or "unknown",
        "minimum_transformers_version": GEMMA4_MIN_TRANSFORMERS_VERSION,
        "minimum_transformers_pypi_release_date": _MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE,
        "required_symbol": required_symbol,
        "support_added_on": GEMMA4_SUPPORT_ADDED_ON,
        "guidance": gemma4_runtime_guidance(transformers_version),
        "source": source,
    }


def _gemma4_meets_minimum_transformers_version(installed_version: str | None) -> bool:
    if installed_version is None:
        return False
    try:
        return Version(installed_version) >= Version(GEMMA4_MIN_TRANSFORMERS_VERSION)
    except InvalidVersion:
        return False


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

        model_class, tokenizer_class = check_gemma4_runtime_support(
            model_id=signature.model_id,
            source=source,
            installed_transformers_version=transformers_version,
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

    def iter_moe_layer_indices(self, bundle: LoadedBackendBundle) -> tuple[int, ...]:
        """Return the ordered decoder-layer indices that are expected to host MoE blocks."""

        topology = self._parse_bundle_topology(bundle)
        return topology.moe_layer_indices

    def iter_moe_layer_tensor_keys(
        self,
        bundle: LoadedBackendBundle,
    ) -> tuple[tuple[int, Mapping[str, str]], ...]:
        """Return ordered per-layer MoE tensor keys after full topology validation."""

        topology = self._parse_bundle_topology(bundle)
        discovered = self._discover_layer_tensor_keys(bundle, topology=topology)
        return tuple((layer_index, dict(discovered[layer_index])) for layer_index in topology.moe_layer_indices)

    def extract_topology(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        """Discover MoE decoder layers from config and state-key metadata."""

        topology = self._parse_bundle_topology(bundle)
        layer_tensor_keys = self.iter_moe_layer_tensor_keys(bundle)
        layers: list[LayerTopology] = []
        for layer_index, tensor_keys in layer_tensor_keys:
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

        topology = self._parse_bundle_topology(bundle)
        if layer_index not in topology.moe_layer_indices:
            raise TopologyMismatchError(
                "requested Gemma4 layer is not configured as MoE",
                model_id=bundle.model_handle.model_id,
                layer_index=layer_index,
                details={"moe_layer_indices": ",".join(str(index) for index in topology.moe_layer_indices)},
            )
        available = self._state_index(bundle)
        prefix = self._layer_prefix(bundle, layer_index=layer_index)
        resolved = {name: f"{prefix}.{suffix}" for name, suffix in _REQUIRED_LAYER_KEYS.items()}
        missing = [tensor_key for tensor_key in resolved.values() if tensor_key not in available]
        if missing:
            raise TopologyMismatchError(
                "missing Gemma4 hybrid layer tensor keys",
                model_id=bundle.model_handle.model_id,
                layer_index=layer_index,
                details={
                    "missing_keys": ",".join(sorted(missing)),
                    "layer_prefix": prefix,
                },
            )
        return dict(sorted(resolved.items()))

    def resolve_prune_tensor_keys(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer_index: int,
    ) -> Mapping[str, str]:
        """Return the MoE tensor keys rewritten by prune/apply for one layer."""

        tensor_keys = self.resolve_layer_tensor_keys(bundle, layer_index=layer_index)
        return {
            "experts_down_proj": tensor_keys["experts_down_proj"],
            "experts_gate_up_proj": tensor_keys["experts_gate_up_proj"],
            "router_per_expert_scale": tensor_keys["router_per_expert_scale"],
            "router_proj": tensor_keys["router_proj"],
            "router_scale": tensor_keys["router_scale"],
        }

    def expected_prune_tensor_shapes(
        self,
        *,
        layer: LayerTopology,
        target_expert_count: int,
    ) -> Mapping[str, tuple[int, ...]]:
        """Return the expected Gemma4 prune tensor shapes for a target expert count."""

        if layer.moe_intermediate_size is None:
            raise TopologyMismatchError(
                "Gemma4 config must include moe_intermediate_size for expert validation",
                layer_index=layer.layer_index,
            )
        if target_expert_count <= 0:
            raise TopologyMismatchError(
                "target expert count must be positive",
                layer_index=layer.layer_index,
                details={"target_expert_count": target_expert_count},
            )
        return {
            "experts_down_proj": self._expected_expert_shape(
                tensor_name="experts.down_proj",
                expected_experts=target_expert_count,
                expected_hidden_size=layer.hidden_size,
                expected_moe_intermediate_size=layer.moe_intermediate_size,
            ),
            "experts_gate_up_proj": self._expected_expert_shape(
                tensor_name="experts.gate_up_proj",
                expected_experts=target_expert_count,
                expected_hidden_size=layer.hidden_size,
                expected_moe_intermediate_size=layer.moe_intermediate_size,
            ),
            "router_per_expert_scale": (target_expert_count,),
            "router_proj": (target_expert_count, layer.hidden_size),
            "router_scale": (),
        }

    def validate_prune_tensor(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        tensor_role: str,
        tensor_key: str,
        tensor_value: object,
        target_expert_count: int,
    ) -> tuple[int, ...]:
        """Validate one prune/apply tensor against Gemma4 layer layout rules."""

        shape = self._tensor_shape(tensor_value, allow_scalar=tensor_role == "router_scale")
        if tensor_role == "router_proj":
            self._validate_router_projection(
                shape,
                bundle=bundle,
                layer=layer,
                expected_experts=target_expert_count,
                expected_hidden_size=layer.hidden_size,
                tensor_key=tensor_key,
            )
            return shape
        if tensor_role == "router_per_expert_scale":
            self._validate_per_expert_scale(
                shape,
                bundle=bundle,
                layer=layer,
                expected_experts=target_expert_count,
                tensor_key=tensor_key,
            )
            return shape
        if tensor_role == "router_scale":
            self._validate_router_scale(
                shape,
                bundle=bundle,
                layer=layer,
                expected_hidden_size=layer.hidden_size,
                tensor_key=tensor_key,
            )
            return shape
        if tensor_role == "experts_gate_up_proj":
            self._validate_expert_tensor(
                shape,
                bundle=bundle,
                layer=layer,
                expected_experts=target_expert_count,
                expected_moe_intermediate_size=layer.moe_intermediate_size,
                expected_hidden_size=layer.hidden_size,
                tensor_key=tensor_key,
                tensor_name="experts.gate_up_proj",
            )
            return shape
        if tensor_role == "experts_down_proj":
            self._validate_expert_tensor(
                shape,
                bundle=bundle,
                layer=layer,
                expected_experts=target_expert_count,
                expected_moe_intermediate_size=layer.moe_intermediate_size,
                expected_hidden_size=layer.hidden_size,
                tensor_key=tensor_key,
                tensor_name="experts.down_proj",
            )
            return shape
        raise TopologyMismatchError(
            "unsupported prune tensor role",
            model_id=bundle.model_handle.model_id,
            layer_index=layer.layer_index,
            tensor_key=tensor_key,
            details={"tensor_role": tensor_role},
        )

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
        self._validate_router_scale(
            scale_shape,
            bundle=bundle,
            layer=layer,
            expected_hidden_size=topology.hidden_size,
            tensor_key=tensor_keys["router_scale"],
        )

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
            expected_moe_intermediate_size=topology.moe_intermediate_size,
            expected_hidden_size=topology.hidden_size,
            tensor_key=tensor_keys["experts_gate_up_proj"],
            tensor_name="experts.gate_up_proj",
        )
        self._validate_expert_tensor(
            down_shape,
            bundle=bundle,
            layer=layer,
            expected_experts=topology.num_experts,
            expected_moe_intermediate_size=topology.moe_intermediate_size,
            expected_hidden_size=topology.hidden_size,
            tensor_key=tensor_keys["experts_down_proj"],
            tensor_name="experts.down_proj",
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

    def resolve_router_module(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> object:
        """Resolve the live router module for one Gemma 4 MoE layer."""

        self.validate_layer(bundle, layer=layer)
        router_path = self.resolve_router_module_path(bundle, layer=layer)
        return self._resolve_object_path(bundle.model, router_path, bundle=bundle, layer=layer)

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

    def _validate_transformers_runtime_support(
        self,
        *,
        signature: BackendSignature,
        installed_transformers_version: str | None,
        transformers_module: object,
        source: str,
    ) -> tuple[object, object]:
        normalized_version = installed_transformers_version or "unknown"
        if not self._is_supported_transformers_version(installed_transformers_version):
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=(self.name,),
                details={
                    "backend_name": self.name,
                    "installed_transformers_version": normalized_version,
                    "minimum_transformers_version": _MINIMUM_TRANSFORMERS_VERSION,
                    "minimum_transformers_pypi_release_date": _MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE,
                    "required_symbol": _SUPPORTED_ARCHITECTURE,
                    "support_added_on": _SUPPORT_ADDED_DATE,
                    "guidance": self._upgrade_guidance(),
                    "source": source,
                },
            )

        model_class = self._lookup_transformers_symbol(
            signature=signature,
            transformers_module=transformers_module,
            installed_transformers_version=normalized_version,
            source=source,
            symbol_name=_SUPPORTED_ARCHITECTURE,
        )
        tokenizer_class = self._lookup_transformers_symbol(
            signature=signature,
            transformers_module=transformers_module,
            installed_transformers_version=normalized_version,
            source=source,
            symbol_name="AutoTokenizer",
        )
        if model_class is None or tokenizer_class is None:
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=(self.name,),
                details={
                    "backend_name": self.name,
                    "installed_transformers_version": normalized_version,
                    "minimum_transformers_version": _MINIMUM_TRANSFORMERS_VERSION,
                    "minimum_transformers_pypi_release_date": _MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE,
                    "required_symbol": _SUPPORTED_ARCHITECTURE,
                    "support_added_on": _SUPPORT_ADDED_DATE,
                    "guidance": (
                        f"{self._upgrade_guidance()} If you already installed "
                        f"transformers>={_MINIMUM_TRANSFORMERS_VERSION}, reinstall a standard Hugging Face "
                        "transformers build because the Gemma4 symbols are missing."
                    ),
                    "source": source,
                },
            )
        return model_class, tokenizer_class

    def _lookup_transformers_symbol(
        self,
        *,
        signature: BackendSignature,
        transformers_module: object,
        installed_transformers_version: str,
        source: str,
        symbol_name: str,
    ) -> object | None:
        try:
            return getattr(transformers_module, symbol_name, None)
        except Exception as exc:
            raise UnsupportedModelError(
                signature.model_id,
                available_backends=(self.name,),
                details={
                    "backend_name": self.name,
                    "installed_transformers_version": installed_transformers_version,
                    "minimum_transformers_version": _MINIMUM_TRANSFORMERS_VERSION,
                    "minimum_transformers_pypi_release_date": _MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE,
                    "required_symbol": _SUPPORTED_ARCHITECTURE,
                    "support_added_on": _SUPPORT_ADDED_DATE,
                    "guidance": (
                        f"{self._upgrade_guidance()} If you already installed "
                        f"transformers>={_MINIMUM_TRANSFORMERS_VERSION}, reinstall a standard Hugging Face "
                        "transformers build because the Gemma4 symbols are missing."
                    ),
                    "source": source,
                    "symbol_resolution_error": f"{exc.__class__.__name__}: {exc}",
                    "symbol_name": symbol_name,
                },
            ) from exc

    def _is_supported_transformers_version(self, version: str | None) -> bool:
        if version is None:
            return False
        installed_parts = self._version_key(version)
        required_parts = self._version_key(_MINIMUM_TRANSFORMERS_VERSION)
        if installed_parts is None or required_parts is None:
            return False
        return installed_parts >= required_parts

    def _version_key(self, version: str) -> tuple[int, int, int] | None:
        match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", version)
        if match is None:
            return None
        major, minor, patch = match.groups()
        return (int(major), int(minor), int(patch))

    def _upgrade_guidance(self) -> str:
        return (
            f"Install transformers>={_MINIMUM_TRANSFORMERS_VERSION}. Gemma4 support was added on "
            f"{_SUPPORT_ADDED_DATE} and first shipped on PyPI in {_MINIMUM_TRANSFORMERS_VERSION} "
            f"({_MINIMUM_TRANSFORMERS_PYPI_RELEASE_DATE})."
        )

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

    def _resolve_runtime_components(
        self,
        transformers_module: object,
        *,
        model_id: str,
        source: str,
        transformers_version: str | None,
    ) -> tuple[Any, Any]:
        self._validate_transformers_runtime(
            model_id=model_id,
            source=source,
            transformers_version=transformers_version,
            transformers_module=transformers_module,
        )
        model_class = getattr(transformers_module, _SUPPORTED_ARCHITECTURE, None)
        tokenizer_class = getattr(transformers_module, "AutoTokenizer", None)
        return cast(Any, model_class), cast(Any, tokenizer_class)

    def _validate_transformers_runtime(
        self,
        *,
        model_id: str,
        source: str,
        transformers_version: str | None,
        transformers_module: object,
    ) -> None:
        if not self._meets_minimum_transformers_version(transformers_version):
            raise UnsupportedModelError(
                model_id,
                available_backends=(self.name,),
                details=self._runtime_support_details(
                    source=source,
                    transformers_version=transformers_version,
                    required_symbol=_SUPPORTED_ARCHITECTURE,
                ),
            )

        model_class = getattr(transformers_module, _SUPPORTED_ARCHITECTURE, None)
        tokenizer_class = getattr(transformers_module, "AutoTokenizer", None)
        if model_class is None or tokenizer_class is None:
            required_symbol = _SUPPORTED_ARCHITECTURE if model_class is None else "AutoTokenizer"
            raise UnsupportedModelError(
                model_id,
                available_backends=(self.name,),
                details=self._runtime_support_details(
                    source=source,
                    transformers_version=transformers_version,
                    required_symbol=required_symbol,
                ),
            )

    def _runtime_support_details(
        self,
        *,
        source: str,
        transformers_version: str | None,
        required_symbol: str,
    ) -> Mapping[str, object]:
        return {
            "backend_name": self.name,
            "installed_transformers_version": transformers_version or "unknown",
            "minimum_transformers_version": GEMMA4_MIN_TRANSFORMERS_VERSION,
            "required_symbol": required_symbol,
            "support_added_on": GEMMA4_SUPPORT_ADDED_ON,
            "guidance": gemma4_runtime_guidance(transformers_version),
            "source": source,
        }

    def _meets_minimum_transformers_version(self, installed_version: str | None) -> bool:
        if installed_version is None:
            return False
        try:
            return Version(installed_version) >= Version(GEMMA4_MIN_TRANSFORMERS_VERSION)
        except InvalidVersion:
            return False

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

    def resolve_router_module_path(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> str:
        """Resolve the canonical router module path for one validated layer."""

        tensor_keys = layer.module_paths or self.resolve_layer_tensor_keys(bundle, layer_index=layer.layer_index)
        router_proj_key = tensor_keys.get("router_proj")
        if router_proj_key is None:
            raise TopologyMismatchError(
                "Gemma4 layer topology missing router_proj module path",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
            )
        suffix = ".proj.weight"
        if not router_proj_key.endswith(suffix):
            raise TopologyMismatchError(
                "Gemma4 router projection key must end with .proj.weight",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=router_proj_key,
            )
        return router_proj_key[: -len(suffix)]

    def _layer_prefix(self, bundle: LoadedBackendBundle, *, layer_index: int) -> str:
        template = self._layer_prefix_template(bundle)
        return template.format(layer_index=layer_index)

    def _resolve_object_path(
        self,
        root: object,
        path: str,
        *,
        bundle: LoadedBackendBundle,
        layer: LayerTopology,
    ) -> object:
        current = root
        traversed: list[str] = []
        for segment in path.split("."):
            traversed.append(segment)
            if isinstance(current, Mapping):
                if segment in current:
                    current = current[segment]
                    continue
                raise TopologyMismatchError(
                    "Gemma4 router module path could not be resolved",
                    model_id=bundle.model_handle.model_id,
                    layer_index=layer.layer_index,
                    details={"module_path": path, "missing_segment": segment},
                )
            if segment.isdigit():
                index = int(segment)
                try:
                    current = cast(Any, current)[index]
                except Exception as exc:
                    raise TopologyMismatchError(
                        "Gemma4 router module path could not be resolved",
                        model_id=bundle.model_handle.model_id,
                        layer_index=layer.layer_index,
                        details={"module_path": path, "missing_segment": segment},
                    ) from exc
                continue
            if hasattr(current, segment):
                current = getattr(current, segment)
                continue
            try:
                current = cast(Any, current)[segment]
            except Exception as exc:
                raise TopologyMismatchError(
                    "Gemma4 router module path could not be resolved",
                    model_id=bundle.model_handle.model_id,
                    layer_index=layer.layer_index,
                    details={"module_path": path, "missing_segment": segment},
                ) from exc
        return current

    def _layer_prefix_template(self, bundle: LoadedBackendBundle) -> str:
        template = bundle.metadata.get("layer_prefix_template", _DEFAULT_LAYER_PREFIX)
        if not isinstance(template, str) or "{layer_index}" not in template:
            raise TopologyMismatchError(
                "Gemma4 layer prefix template must contain {layer_index}",
                model_id=bundle.model_handle.model_id,
            )
        return template

    def _layer_tensor_pattern(self, bundle: LoadedBackendBundle) -> re.Pattern[str]:
        template = self._layer_prefix_template(bundle)
        prefix_before, prefix_after = template.split("{layer_index}", maxsplit=1)
        suffixes = "|".join(re.escape(suffix) for suffix in sorted(_REQUIRED_LAYER_KEYS.values()))
        return re.compile(
            rf"^{re.escape(prefix_before)}(?P<layer_index>\d+){re.escape(prefix_after)}\.(?P<suffix>{suffixes})$"
        )

    def _discover_layer_tensor_keys(
        self,
        bundle: LoadedBackendBundle,
        *,
        topology: Gemma4TopologyConfig,
    ) -> Mapping[int, Mapping[str, str]]:
        pattern = self._layer_tensor_pattern(bundle)
        suffix_to_name = {suffix: name for name, suffix in _REQUIRED_LAYER_KEYS.items()}
        matched: dict[int, dict[str, str]] = {}
        for tensor_key in sorted(str(key) for key in self._state_index(bundle)):
            match = pattern.match(tensor_key)
            if match is None:
                continue
            layer_index = int(match.group("layer_index"))
            tensor_name = suffix_to_name[match.group("suffix")]
            matched.setdefault(layer_index, {})[tensor_name] = tensor_key

        expected = set(topology.moe_layer_indices)
        unexpected_layers = tuple(sorted(layer_index for layer_index in matched if layer_index not in expected))
        missing_layer_indices: list[int] = []
        missing_by_layer: list[str] = []
        for layer_index in topology.moe_layer_indices:
            layer_keys = matched.get(layer_index, {})
            missing = [name for name in sorted(_REQUIRED_LAYER_KEYS) if name not in layer_keys]
            if missing:
                missing_layer_indices.append(layer_index)
                missing_by_layer.append(f"{layer_index}:{','.join(missing)}")

        if unexpected_layers or missing_by_layer:
            if not unexpected_layers and len(missing_layer_indices) == 1:
                self.resolve_layer_tensor_keys(bundle, layer_index=missing_layer_indices[0])
            details: dict[str, object] = {
                "expected_moe_layers": ",".join(str(index) for index in topology.moe_layer_indices),
                "layer_prefix_template": self._layer_prefix_template(bundle),
            }
            if unexpected_layers:
                details["unexpected_moe_layers"] = ",".join(str(index) for index in unexpected_layers)
            if missing_by_layer:
                details["missing_keys_by_layer"] = ";".join(missing_by_layer)
            raise TopologyMismatchError(
                "Gemma4 MoE layer tensor topology mismatch",
                model_id=bundle.model_handle.model_id,
                details=details,
            )

        return {layer_index: dict(sorted(layer_keys.items())) for layer_index, layer_keys in sorted(matched.items())}

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
        expected_hidden_size: int,
        tensor_key: str,
    ) -> None:
        if shape not in ((), (1,), (expected_hidden_size,)):
            raise TopologyMismatchError(
                "Gemma4 router scale must be scalar or hidden-size vector",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=(expected_hidden_size,),
                actual_shape=shape,
            )

    def _validate_expert_tensor(
        self,
        shape: tuple[int, ...],
        *,
        bundle: LoadedBackendBundle,
        layer: LayerTopology,
        expected_experts: int,
        expected_moe_intermediate_size: int | None,
        expected_hidden_size: int,
        tensor_key: str,
        tensor_name: str,
    ) -> None:
        if expected_moe_intermediate_size is None:
            raise TopologyMismatchError(
                "Gemma4 config must include moe_intermediate_size for expert validation",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
            )
        expected_shape = self._expected_expert_shape(
            tensor_name=tensor_name,
            expected_experts=expected_experts,
            expected_hidden_size=expected_hidden_size,
            expected_moe_intermediate_size=expected_moe_intermediate_size,
        )
        if len(shape) != 3:
            raise ShapeInvariantViolationError(
                f"Gemma4 {tensor_name} rank must be 3",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=expected_shape,
                actual_shape=shape,
                details={"expected_layout": self._expected_expert_layout(tensor_name=tensor_name)},
            )
        if shape != expected_shape:
            raise ShapeInvariantViolationError(
                f"Gemma4 {tensor_name} shape mismatch",
                model_id=bundle.model_handle.model_id,
                layer_index=layer.layer_index,
                tensor_key=tensor_key,
                expected_shape=expected_shape,
                actual_shape=shape,
                details={"expected_layout": self._expected_expert_layout(tensor_name=tensor_name)},
            )

    def _expected_expert_shape(
        self,
        *,
        tensor_name: str,
        expected_experts: int,
        expected_hidden_size: int,
        expected_moe_intermediate_size: int,
    ) -> tuple[int, int, int]:
        if tensor_name == "experts.gate_up_proj":
            return (expected_experts, expected_moe_intermediate_size * 2, expected_hidden_size)
        if tensor_name == "experts.down_proj":
            return (expected_experts, expected_hidden_size, expected_moe_intermediate_size)
        raise ValueError(f"unsupported expert tensor name: {tensor_name}")

    def _expected_expert_layout(self, *, tensor_name: str) -> str:
        if tensor_name == "experts.gate_up_proj":
            return "(num_experts, 2 * moe_intermediate_size, hidden_size)"
        if tensor_name == "experts.down_proj":
            return "(num_experts, hidden_size, moe_intermediate_size)"
        raise ValueError(f"unsupported expert tensor name: {tensor_name}")


def default_registry_entry() -> tuple[Gemma4Backend, int]:
    """Return the canonical default-registry registration for Gemma 4."""

    return Gemma4Backend(), DEFAULT_REGISTRY_PRIORITY


__all__ = [
    "DEFAULT_REGISTRY_PRIORITY",
    "Gemma4Backend",
    "Gemma4TopologyConfig",
    "default_registry_entry",
]
