"""Backend protocol contracts for lightweight model-family adapters.

This module intentionally stays import-light. The legacy
``from moe_surgeon.models.backend import BackendRegistry`` import path is
preserved through a lazy module attribute to avoid a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from moe_surgeon.schemas import LayerTopology, ModelHandle, RouterState

if TYPE_CHECKING:
    from moe_surgeon.models.registry import BackendRegistry as BackendRegistry


@dataclass(frozen=True)
class BackendSignature:
    """Lightweight model/config signature used for backend dispatch."""

    model_id: str
    architecture: str | None = None
    model_type: str | None = None
    revision: str | None = None
    source_path: str | None = None
    config: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        model_id: str | None = None,
        source_path: str | Path | None = None,
    ) -> "BackendSignature":
        """Build a dispatch signature from a plain config-like mapping."""

        config = dict(payload)
        resolved_model_id = model_id or str(config.get("_name_or_path") or config.get("model_id") or "unknown")
        architecture = None
        raw_architectures = config.get("architectures")
        if isinstance(raw_architectures, Sequence) and not isinstance(raw_architectures, (str, bytes)):
            for item in raw_architectures:
                if isinstance(item, str) and item.strip():
                    architecture = item
                    break
        if architecture is None:
            raw_architecture = config.get("architecture")
            if isinstance(raw_architecture, str) and raw_architecture.strip():
                architecture = raw_architecture
        model_type = config.get("model_type")
        revision = config.get("_commit_hash") or config.get("revision")
        return cls(
            model_id=resolved_model_id,
            architecture=architecture if isinstance(architecture, str) else None,
            model_type=model_type if isinstance(model_type, str) else None,
            revision=revision if isinstance(revision, str) else None,
            source_path=str(source_path) if source_path is not None else None,
            config=config,
        )


BackendSignatureInput = BackendSignature | Mapping[str, object]


def coerce_backend_signature(
    signature: BackendSignatureInput,
    *,
    model_id: str | None = None,
    source_path: str | Path | None = None,
) -> BackendSignature:
    """Normalize a resolver input into a lightweight backend signature."""

    if isinstance(signature, BackendSignature):
        return signature
    return BackendSignature.from_mapping(signature, model_id=model_id, source_path=source_path)


@dataclass(frozen=True)
class TensorMetadata:
    """Minimal tensor descriptor returned by backend state accessors."""

    tensor_key: str
    shape: tuple[int, ...]
    dtype: str | None = None
    expert_index: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedBackendBundle:
    """Opaque loaded backend bundle shared across analysis stages."""

    backend_name: str
    model_handle: ModelHandle
    model: object
    config: Mapping[str, object]
    tokenizer: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class ModelBackend(Protocol):
    """Static protocol that constrains model-family backend adapters."""

    name: str

    def supports(self, signature: BackendSignature) -> bool:
        """Return whether this backend can handle the provided signature."""

    def load(
        self,
        signature: BackendSignature,
        *,
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
    ) -> LoadedBackendBundle:
        """Load a checkpoint/config into an opaque backend bundle."""

    def iter_layers(self, bundle: LoadedBackendBundle) -> Sequence[LayerTopology]:
        """Yield deterministic layer topology metadata for the model."""

    def extract_topology(self, bundle: LoadedBackendBundle) -> Sequence[LayerTopology]:
        """Return the full deterministic topology snapshot."""

    def extract_router_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> RouterState:
        """Extract router metadata for a single layer."""

    def extract_expert_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> Mapping[str, TensorMetadata]:
        """Return per-layer expert tensor descriptors used by pruning/export."""

    def validate_bundle(self, bundle: LoadedBackendBundle) -> None:
        """Validate backend-level topology/loading invariants."""

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        """Validate layer-level topology and routing invariants."""


@lru_cache(maxsize=1)
def build_backend_registry() -> "BackendRegistry":
    """Build the default deterministic backend registry lazily."""

    from moe_surgeon.models.gemma4 import default_registry_entry
    from moe_surgeon.models.registry import BackendRegistry

    registry = BackendRegistry()
    default_entries = (default_registry_entry(),)
    for backend, priority in sorted(default_entries, key=lambda entry: (-entry[1], entry[0].name)):
        registry.register(backend, priority=priority)
    return registry


def resolve_backend(
    signature: BackendSignatureInput,
    *,
    model_id: str | None = None,
    source_path: str | Path | None = None,
) -> ModelBackend:
    """Resolve a backend from the default deterministic registry."""

    return build_backend_registry().resolve(signature, model_id=model_id, source_path=source_path)


def __getattr__(name: str) -> object:
    """Lazily resolve compatibility exports without importing them on module load."""

    if name != "BackendRegistry":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from moe_surgeon.models.registry import BackendRegistry

    return BackendRegistry


__all__ = [
    "BackendRegistry",
    "BackendSignatureInput",
    "BackendSignature",
    "build_backend_registry",
    "coerce_backend_signature",
    "LoadedBackendBundle",
    "ModelBackend",
    "resolve_backend",
    "TensorMetadata",
]
