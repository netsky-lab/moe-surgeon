from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pytest

from moe_surgeon.models.backend import (
    BackendSignature,
    LoadedBackendBundle,
    TensorMetadata,
    coerce_backend_signature,
)
from moe_surgeon.models.errors import (
    BackendMismatchError,
    ShapeInvariantViolationError,
    TopologyMismatchError,
    UnsupportedModelError,
)
from moe_surgeon.models.registry import BackendRegistry
from moe_surgeon.schemas import LayerTopology, ModelHandle, RouterState


@dataclass
class StubBackend:
    name: str
    supported_model_type: str | None = None

    def supports(self, signature: BackendSignature) -> bool:
        return signature.model_type == self.supported_model_type

    def load(
        self,
        signature: BackendSignature,
        *,
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
    ) -> LoadedBackendBundle:
        return LoadedBackendBundle(
            backend_name=self.name,
            model_handle=ModelHandle(model_id=signature.model_id, backend_name=self.name, device=device, dtype=dtype, seed=seed),
            model=object(),
            config=signature.config,
        )

    def iter_layers(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        return self.extract_topology(bundle)

    def extract_topology(self, bundle: LoadedBackendBundle) -> tuple[LayerTopology, ...]:
        return (
            LayerTopology(
                layer_index=0,
                layer_name="model.layers.0",
                layer_type="moe",
                expert_count=4,
                top_k=2,
                hidden_size=128,
            ),
        )

    def extract_router_state(self, bundle: LoadedBackendBundle, *, layer: LayerTopology) -> RouterState:
        return RouterState(
            layer_index=layer.layer_index,
            num_experts=layer.expert_count,
            top_k=layer.top_k,
            logits_shape=(8, layer.expert_count),
            projection_shape=(layer.expert_count, layer.hidden_size),
        )

    def extract_expert_state(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
    ) -> dict[str, TensorMetadata]:
        return {
            "gate_up_proj": TensorMetadata(
                tensor_key=f"{layer.layer_name}.experts.gate_up_proj",
                shape=(layer.expert_count, layer.hidden_size, layer.hidden_size),
            )
        }

    def validate_bundle(self, bundle: LoadedBackendBundle) -> None:
        return None


@dataclass
class InvalidSupportsBackend(StubBackend):
    def supports(self, signature: BackendSignature) -> bool:  # type: ignore[override]
        return "yes"  # type: ignore[return-value]

    def validate_layer(
        self,
        bundle: LoadedBackendBundle,
        *,
        layer: LayerTopology,
        router_state: RouterState | None = None,
    ) -> None:
        return None


def test_backend_signature_from_mapping_uses_lightweight_config_fields() -> None:
    signature = BackendSignature.from_mapping(
        {
            "_name_or_path": "google/gemma",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "_commit_hash": "abc123",
        }
    )

    assert signature.model_id == "google/gemma"
    assert signature.architecture == "Gemma4ForConditionalGeneration"
    assert signature.model_type == "gemma4"
    assert signature.revision == "abc123"


def test_coerce_backend_signature_accepts_plain_config_mappings() -> None:
    signature = coerce_backend_signature(
        {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"},
        model_id="google/gemma",
        source_path=Path("/tmp/checkpoint"),
    )

    assert signature.model_id == "google/gemma"
    assert signature.architecture == "Gemma4ForConditionalGeneration"
    assert signature.model_type == "gemma4"
    assert signature.source_path == "/tmp/checkpoint"


@pytest.mark.parametrize(
    ("module_name", "import_statement", "forbidden_modules"),
    (
        (
            "moe_surgeon.models.backend",
            "import moe_surgeon.models.backend",
            ("moe_surgeon.models.registry",),
        ),
        ("moe_surgeon.models.registry", "import moe_surgeon.models.registry", ()),
    ),
)
def test_model_modules_import_in_fresh_process_without_heavy_dependencies(
    module_name: str,
    import_statement: str,
    forbidden_modules: tuple[str, ...],
) -> None:
    probe = f"""
import sys

{import_statement}

forbidden = [name for name in ("torch", "transformers", "safetensors") if name in sys.modules]
assert not forbidden, forbidden
unexpected = [name for name in {forbidden_modules!r} if name in sys.modules]
assert not unexpected, unexpected
print("{module_name}")
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == module_name


def test_backend_registry_compatibility_alias_imports_in_fresh_process() -> None:
    probe = """
import sys

from moe_surgeon.models.backend import BackendRegistry
from moe_surgeon.models.registry import BackendRegistry as CanonicalBackendRegistry

assert BackendRegistry is CanonicalBackendRegistry
forbidden = [name for name in ("torch", "transformers", "safetensors") if name in sys.modules]
assert not forbidden, forbidden
print(BackendRegistry.__name__)
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "BackendRegistry"


def test_registry_resolves_single_backend_deterministically() -> None:
    registry = BackendRegistry()
    fallback = StubBackend(name="fallback", supported_model_type="gemma4")
    preferred = StubBackend(name="preferred", supported_model_type="gemma4")
    registry.register(fallback, priority=10)
    registry.register(preferred, priority=20)

    resolved = registry.resolve(BackendSignature(model_id="model", model_type="gemma4"))

    assert resolved is preferred
    assert registry.names() == ("preferred", "fallback")


def test_registry_resolves_from_plain_config_mapping() -> None:
    registry = BackendRegistry()
    backend = StubBackend(name="gemma4", supported_model_type="gemma4")
    registry.register(backend, priority=5)

    resolved = registry.resolve(
        {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"},
        model_id="google/gemma-4",
        source_path="/models/gemma-4",
    )

    assert resolved is backend


def test_registry_rejects_duplicate_backend_names() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend(name="gemma4", supported_model_type="gemma4"))

    with pytest.raises(BackendMismatchError, match="duplicate backend registration"):
        registry.register(StubBackend(name="gemma4", supported_model_type="gemma4"))


def test_registry_rejects_non_integer_priority() -> None:
    registry = BackendRegistry()

    with pytest.raises(BackendMismatchError, match="backend priority must be int"):
        registry.register(StubBackend(name="gemma4", supported_model_type="gemma4"), priority=True)


def test_registry_raises_unsupported_model_with_context() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend(name="gemma4", supported_model_type="gemma4"))

    with pytest.raises(UnsupportedModelError) as exc_info:
        registry.resolve(BackendSignature(model_id="unknown-model", model_type="llama"))

    message = str(exc_info.value)
    assert "unsupported model family" in message
    assert "model_id=unknown-model" in message
    assert "available_backends=gemma4" in message


def test_registry_raises_ambiguous_resolution_for_same_priority_matches() -> None:
    registry = BackendRegistry()
    registry.register(StubBackend(name="alpha", supported_model_type="gemma4"), priority=5)
    registry.register(StubBackend(name="beta", supported_model_type="gemma4"), priority=5)

    with pytest.raises(BackendMismatchError, match="ambiguous backend resolution") as exc_info:
        registry.resolve(BackendSignature(model_id="model", model_type="gemma4"))

    assert "candidate_backends=alpha,beta" in str(exc_info.value)


def test_registry_raises_when_backend_supports_contract_is_invalid() -> None:
    registry = BackendRegistry()
    registry.register(InvalidSupportsBackend(name="broken", supported_model_type="gemma4"))

    with pytest.raises(BackendMismatchError, match="backend supports\\(\\) must return bool") as exc_info:
        registry.resolve(BackendSignature(model_id="model", model_type="gemma4"))

    assert "backend=broken" in str(exc_info.value)


def test_backend_protocol_contract_returns_schema_types() -> None:
    backend = StubBackend(name="gemma4", supported_model_type="gemma4")
    signature = BackendSignature(model_id="model", model_type="gemma4")

    bundle = backend.load(signature)
    layers = backend.iter_layers(bundle)
    router_state = backend.extract_router_state(bundle, layer=layers[0])
    expert_state = backend.extract_expert_state(bundle, layer=layers[0])

    assert isinstance(bundle.model_handle, ModelHandle)
    assert isinstance(layers[0], LayerTopology)
    assert isinstance(router_state, RouterState)
    assert expert_state["gate_up_proj"].shape == (4, 128, 128)


def test_model_errors_are_reexported_from_schemas() -> None:
    from moe_surgeon.schemas import ShapeInvariantViolationError as schema_shape_error
    from moe_surgeon.schemas import TopologyMismatchError as schema_topology_error

    assert schema_shape_error is ShapeInvariantViolationError
    assert schema_topology_error is TopologyMismatchError


def test_topology_mismatch_and_shape_violation_messages_include_context() -> None:
    topology_error = TopologyMismatchError(
        "router shape mismatch",
        model_id="gemma",
        layer_index=3,
        tensor_key="router.proj.weight",
        expected_shape=(128, 2816),
        actual_shape=(127, 2816),
    )
    shape_error = ShapeInvariantViolationError(
        "invalid tensor shape",
        model_id="gemma",
        layer_index=3,
        tensor_key="experts.down_proj",
        expected_shape=(128, 704, 2816),
        actual_shape=(128, 703, 2816),
    )

    assert "layer_index=3" in str(topology_error)
    assert "expected_shape=128x2816" in str(topology_error)
    assert "actual_shape=127x2816" in str(topology_error)
    assert "tensor_key=experts.down_proj" in str(shape_error)


def test_topology_mismatch_error_accepts_legacy_positional_context() -> None:
    error = TopologyMismatchError("x", "y", "z", "a")

    assert "legacy_arg_1=y" in str(error)
    assert "legacy_arg_2=z" in str(error)
    assert "legacy_arg_3=a" in str(error)
