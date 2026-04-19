from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Mapping

import torch

from moe_surgeon.models.backend import LoadedBackendBundle, TensorMetadata
from moe_surgeon.schemas import ActivationStats, ExpertStats, LayerTopology, ModelHandle, RouterState


def tiny_gemma_config(*, moe_layer_indices: tuple[int, ...] = (0, 1)) -> dict[str, object]:
    """Return a deterministic Gemma4-like config mapping for offline tests."""

    return {
        "_name_or_path": "tests/tiny-gemma-like",
        "_commit_hash": "tiny-rev-1",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "num_hidden_layers": 4,
            "hidden_size": 3,
            "enable_moe_block": True,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 2,
            "moe_layer_indices": list(moe_layer_indices),
        },
    }


def tiny_layer_topology(*, layer_index: int) -> LayerTopology:
    """Return one deterministic tiny MoE topology entry."""

    prefix = f"model.language_model.layers.{layer_index}"
    return LayerTopology(
        layer_index=layer_index,
        layer_name=prefix,
        layer_type="gemma4_moe",
        expert_count=4,
        top_k=2,
        hidden_size=3,
        layer_ref=f"layer_{layer_index}",
        module_paths={
            "router_proj": f"{prefix}.router.proj.weight",
            "router_scale": f"{prefix}.router.scale",
            "router_per_expert_scale": f"{prefix}.router.per_expert_scale",
            "experts_gate_up_proj": f"{prefix}.experts.gate_up_proj",
            "experts_down_proj": f"{prefix}.experts.down_proj",
        },
    )


def tiny_topology() -> tuple[LayerTopology, ...]:
    """Return the ordered tiny topology used across offline tests."""

    return tuple(tiny_layer_topology(layer_index=index) for index in (0, 1))


def tiny_router_state(*, layer_index: int) -> RouterState:
    """Return deterministic router metadata for one tiny MoE layer."""

    return RouterState(
        layer_index=layer_index,
        num_experts=4,
        top_k=2,
        logits_shape=(0, 4),
        top_k_indices_shape=(1, 2),
        top_k_weights_shape=(1, 2),
        projection_shape=(4, 3),
        per_expert_scale_shape=(4,),
        has_raw_logits_capture=True,
        route_scale_present=True,
        metadata={"fixture": "tiny_gemma_like"},
    )


def tiny_router_states() -> dict[int, RouterState]:
    """Return router metadata indexed by layer."""

    return {layer.layer_index: tiny_router_state(layer_index=layer.layer_index) for layer in tiny_topology()}


def tiny_layer_state(*, layer_index: int, shift: float = 0.0) -> dict[str, torch.Tensor]:
    """Return one layer's deterministic tensor payload."""

    prefix = f"model.language_model.layers.{layer_index}"
    return {
        f"{prefix}.router.proj.weight": torch.tensor(
            [
                [3.0 + shift, 0.0, -2.0],
                [1.0, 2.0 + shift, 0.0],
                [-2.0, 1.0, 3.0 + shift],
                [0.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        f"{prefix}.router.scale": torch.tensor(1.0 + shift, dtype=torch.float32),
        f"{prefix}.router.per_expert_scale": torch.tensor(
            [0.2, 0.4 + shift, 0.1, 0.3],
            dtype=torch.float32,
        ),
        f"{prefix}.experts.gate_up_proj": torch.arange(48, dtype=torch.float32).reshape(4, 4, 3) + shift,
        f"{prefix}.experts.down_proj": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2) + shift,
    }


def tiny_state_dict() -> dict[str, torch.Tensor]:
    """Return a deterministic two-layer state dict for scan/apply/export tests."""

    state: dict[str, torch.Tensor] = {}
    for layer_index, shift in ((0, 0.0), (1, 0.5)):
        state.update(tiny_layer_state(layer_index=layer_index, shift=shift))
    return state


def tiny_expert_stats() -> tuple[ExpertStats, ...]:
    """Return deterministic static metrics aligned to ``tiny_topology``."""

    return (
        ExpertStats(layer_index=0, expert_index=0, static_gate_mass=0.60, static_gate_entropy=0.10, static_gate_entropy_norm=0.10),
        ExpertStats(layer_index=0, expert_index=1, static_gate_mass=0.20, static_gate_entropy=0.20, static_gate_entropy_norm=0.20),
        ExpertStats(layer_index=0, expert_index=2, static_gate_mass=0.15, static_gate_entropy=0.30, static_gate_entropy_norm=0.30),
        ExpertStats(layer_index=0, expert_index=3, static_gate_mass=0.05, static_gate_entropy=0.40, static_gate_entropy_norm=0.40),
        ExpertStats(layer_index=1, expert_index=0, static_gate_mass=0.25, static_gate_entropy=0.30, static_gate_entropy_norm=0.30),
        ExpertStats(layer_index=1, expert_index=1, static_gate_mass=0.40, static_gate_entropy=0.20, static_gate_entropy_norm=0.20),
        ExpertStats(layer_index=1, expert_index=2, static_gate_mass=0.20, static_gate_entropy=0.10, static_gate_entropy_norm=0.10),
        ExpertStats(layer_index=1, expert_index=3, static_gate_mass=0.15, static_gate_entropy=0.50, static_gate_entropy_norm=0.50),
    )


def tiny_activation_stats() -> tuple[ActivationStats, ...]:
    """Return deterministic runtime metrics aligned to ``tiny_topology``."""

    return (
        ActivationStats(layer_index=0, expert_index=0, token_count=12, weighted_token_count=9.0, mass_sum=9.0, mean_weight=0.75, entropy=0.10, n_tokens=20, weighted_n_tokens=14.0, top1_mass=7.0, density=0.60),
        ActivationStats(layer_index=0, expert_index=1, token_count=8, weighted_token_count=5.0, mass_sum=5.0, mean_weight=0.625, entropy=0.20, n_tokens=20, weighted_n_tokens=14.0, top1_mass=4.0, density=0.40),
        ActivationStats(layer_index=0, expert_index=2, token_count=5, weighted_token_count=3.0, mass_sum=3.0, mean_weight=0.60, entropy=0.30, n_tokens=20, weighted_n_tokens=14.0, top1_mass=2.0, density=0.25),
        ActivationStats(layer_index=0, expert_index=3, token_count=3, weighted_token_count=2.0, mass_sum=2.0, mean_weight=0.67, entropy=0.40, n_tokens=20, weighted_n_tokens=14.0, top1_mass=1.0, density=0.15),
        ActivationStats(layer_index=1, expert_index=0, token_count=7, weighted_token_count=4.0, mass_sum=4.0, mean_weight=0.57, entropy=0.25, n_tokens=20, weighted_n_tokens=14.0, top1_mass=2.0, density=0.35),
        ActivationStats(layer_index=1, expert_index=1, token_count=11, weighted_token_count=7.0, mass_sum=7.0, mean_weight=0.64, entropy=0.20, n_tokens=20, weighted_n_tokens=14.0, top1_mass=5.0, density=0.55),
        ActivationStats(layer_index=1, expert_index=2, token_count=6, weighted_token_count=2.0, mass_sum=2.0, mean_weight=0.33, entropy=0.15, n_tokens=20, weighted_n_tokens=14.0, top1_mass=1.0, density=0.30),
        ActivationStats(layer_index=1, expert_index=3, token_count=4, weighted_token_count=1.0, mass_sum=1.0, mean_weight=0.25, entropy=0.35, n_tokens=20, weighted_n_tokens=14.0, top1_mass=0.5, density=0.20),
    )


def tiny_bundle(
    *,
    source_path: str | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> LoadedBackendBundle:
    """Return a deterministic loaded bundle backed by the tiny fixture tensors."""

    materialized_state = dict(tiny_state_dict() if state_dict is None else state_dict)
    return LoadedBackendBundle(
        backend_name="tiny-gemma-like",
        model_handle=ModelHandle(
            model_id="tests/tiny-gemma-like",
            revision="tiny-rev-1",
            backend_name="tiny-gemma-like",
            source_path=source_path,
            seed=7,
        ),
        model=SimpleNamespace(),
        config=tiny_gemma_config(),
        metadata={
            "backend_version": "fixture-1",
            "state_dict": materialized_state,
            "state_keys": tuple(sorted(materialized_state)),
        },
    )


@dataclass
class TinyHookHandle:
    module: "TinyRouterModule"
    hook: object
    removed: bool = False

    def remove(self) -> None:
        if self.removed:
            return
        self.removed = True
        self.module.hooks.remove(self.hook)


@dataclass
class TinyRouterModule:
    name: str
    hooks: list[object] = field(default_factory=list)

    def register_forward_hook(self, hook: object) -> TinyHookHandle:
        self.hooks.append(hook)
        return TinyHookHandle(module=self, hook=hook)

    def emit(self, output: object) -> None:
        for hook in tuple(self.hooks):
            hook(self, (), output)


class TinyMockBackend:
    """Minimal offline backend used by test-only scan and bench flows."""

    name = "tiny-gemma-like"

    def __init__(
        self,
        *,
        topology: tuple[LayerTopology, ...] | None = None,
        router_states: Mapping[int, RouterState] | None = None,
        state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._topology = tiny_topology() if topology is None else tuple(topology)
        self._router_states = dict(tiny_router_states() if router_states is None else router_states)
        self._state_dict = dict(tiny_state_dict() if state_dict is None else state_dict)
        self._modules = {
            layer.layer_index: TinyRouterModule(name=f"router-{layer.layer_index}") for layer in self._topology
        }

    def supports(self, signature: object) -> bool:
        return getattr(signature, "model_type", None) == "gemma4"

    def load(
        self,
        signature: object,
        *,
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
    ) -> LoadedBackendBundle:
        del signature
        bundle = tiny_bundle(state_dict=self._state_dict)
        return LoadedBackendBundle(
            backend_name=self.name,
            model_handle=ModelHandle(
                model_id=bundle.model_handle.model_id,
                revision=bundle.model_handle.revision,
                backend_name=self.name,
                device=device,
                dtype=dtype,
                seed=seed,
            ),
            model=bundle.model,
            config=bundle.config,
            metadata=bundle.metadata,
        )

    def iter_layers(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        del bundle
        return self._topology

    def extract_topology(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        del bundle
        return self._topology

    def extract_router_state(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> RouterState:
        del bundle
        return self._router_states[layer.layer_index]

    def extract_expert_state(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> dict[str, TensorMetadata]:
        del bundle
        return {
            "gate_up_proj": TensorMetadata(
                tensor_key=layer.module_paths["experts_gate_up_proj"],
                shape=tuple(int(dimension) for dimension in self._state_dict[layer.module_paths["experts_gate_up_proj"]].shape),
            ),
            "down_proj": TensorMetadata(
                tensor_key=layer.module_paths["experts_down_proj"],
                shape=tuple(int(dimension) for dimension in self._state_dict[layer.module_paths["experts_down_proj"]].shape),
            ),
        }

    def validate_bundle(self, bundle: LoadedBackendBundle) -> None:
        del bundle

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        del bundle
        active_state = self._router_states[layer.layer_index] if router_state is None else router_state
        if active_state.num_experts != layer.expert_count:
            raise AssertionError("fixture topology mismatch")
        if active_state.top_k != layer.top_k:
            raise AssertionError("fixture router-state mismatch")

    def resolve_router_module(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> TinyRouterModule:
        del bundle
        return self._modules[layer.layer_index]


__all__ = [
    "TinyHookHandle",
    "TinyMockBackend",
    "TinyRouterModule",
    "tiny_activation_stats",
    "tiny_bundle",
    "tiny_expert_stats",
    "tiny_gemma_config",
    "tiny_layer_state",
    "tiny_layer_topology",
    "tiny_router_state",
    "tiny_router_states",
    "tiny_state_dict",
    "tiny_topology",
]
