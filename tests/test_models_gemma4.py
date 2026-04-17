from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import subprocess
import sys

import pytest

from moe_surgeon.models.backend import (
    BackendSignature,
    LoadedBackendBundle,
    build_backend_registry,
    resolve_backend,
)
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError, UnsupportedModelError
from moe_surgeon.models.gemma4 import DEFAULT_REGISTRY_PRIORITY, Gemma4Backend, default_registry_entry
from moe_surgeon.schemas import ModelHandle


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str = "float32"


def _gemma4_config(*, moe_layer_indices: list[int] | None = None) -> dict[str, object]:
    text_config: dict[str, object] = {
        "num_hidden_layers": 5,
        "hidden_size": 2816,
        "enable_moe_block": True,
        "num_experts": 128,
        "top_k_experts": 8,
        "moe_intermediate_size": 704,
    }
    if moe_layer_indices is not None:
        text_config["moe_layer_indices"] = moe_layer_indices
    return {
        "_name_or_path": "google/gemma-4-27b",
        "_commit_hash": "rev-123",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": text_config,
    }


def _layer_state(layer_index: int) -> dict[str, FakeTensor]:
    prefix = f"model.language_model.layers.{layer_index}"
    return {
        f"{prefix}.router.proj.weight": FakeTensor((128, 2816)),
        f"{prefix}.router.scale": FakeTensor(()),
        f"{prefix}.router.per_expert_scale": FakeTensor((128,)),
        f"{prefix}.experts.gate_up_proj": FakeTensor((128, 1408, 2816)),
        f"{prefix}.experts.down_proj": FakeTensor((128, 2816, 704)),
        f"{prefix}.mlp.down_proj.weight": FakeTensor((2816, 2112)),
        f"{prefix}.mlp.gate_proj.weight": FakeTensor((2112, 2816)),
        f"{prefix}.mlp.up_proj.weight": FakeTensor((2112, 2816)),
        f"{prefix}.pre_feedforward_layernorm.weight": FakeTensor((2816,)),
        f"{prefix}.pre_feedforward_layernorm_2.weight": FakeTensor((2816,)),
        f"{prefix}.post_feedforward_layernorm.weight": FakeTensor((2816,)),
        f"{prefix}.post_feedforward_layernorm_1.weight": FakeTensor((2816,)),
        f"{prefix}.post_feedforward_layernorm_2.weight": FakeTensor((2816,)),
    }


def _bundle(*, config: dict[str, object], state_dict: dict[str, FakeTensor]) -> LoadedBackendBundle:
    return LoadedBackendBundle(
        backend_name="gemma4",
        model_handle=ModelHandle(model_id="google/gemma-4-27b", revision="rev-123", backend_name="gemma4"),
        model=object(),
        config=config,
        metadata={"state_dict": state_dict, "backend_version": "1.0.0"},
    )


def test_gemma4_module_import_is_lightweight_in_fresh_process() -> None:
    probe = """
import sys
import moe_surgeon.models.gemma4
forbidden = [name for name in ("torch", "transformers", "safetensors") if name in sys.modules]
assert not forbidden, forbidden
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"


def test_gemma4_backend_supports_lightweight_signatures() -> None:
    backend = Gemma4Backend()

    assert backend.supports(BackendSignature(model_id="m", model_type="gemma4"))
    assert backend.supports(
        BackendSignature(model_id="m", architecture="Gemma4ForConditionalGeneration")
    )
    assert not backend.supports(BackendSignature(model_id="m", model_type="llama"))


def test_gemma4_default_registry_entry_uses_canonical_priority() -> None:
    backend, priority = default_registry_entry()

    assert backend.name == "gemma4"
    assert priority == DEFAULT_REGISTRY_PRIORITY


def test_gemma4_backend_extract_topology_returns_sorted_moe_layers() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[4, 1, 3])
    state_dict: dict[str, FakeTensor] = {}
    for layer_index in (1, 3, 4):
        state_dict.update(_layer_state(layer_index))
    bundle = _bundle(config=config, state_dict=state_dict)

    layers = backend.extract_topology(bundle)

    assert [layer.layer_index for layer in layers] == [1, 3, 4]
    assert layers[0].module_paths["router_proj"] == "model.language_model.layers.1.router.proj.weight"
    assert layers[2].module_paths["experts_down_proj"].endswith("layers.4.experts.down_proj")


def test_gemma4_backend_iterates_only_configured_moe_layers_and_tensor_keys() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[3, 1])
    state_dict: dict[str, FakeTensor] = {}
    for layer_index in (1, 3):
        state_dict.update(_layer_state(layer_index))
    bundle = _bundle(config=config, state_dict=state_dict)

    layer_indices = backend.iter_moe_layer_indices(bundle)
    layer_tensor_keys = backend.iter_moe_layer_tensor_keys(bundle)

    assert layer_indices == (1, 3)
    assert [layer_index for layer_index, _ in layer_tensor_keys] == [1, 3]
    assert layer_tensor_keys[0][1]["router_proj"] == "model.language_model.layers.1.router.proj.weight"
    assert layer_tensor_keys[1][1]["experts_down_proj"] == "model.language_model.layers.3.experts.down_proj"


def test_gemma4_backend_missing_required_tensor_key_raises_topology_mismatch() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0)
    del state_dict["model.language_model.layers.0.router.per_expert_scale"]
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(TopologyMismatchError, match="missing Gemma4 hybrid layer tensor keys") as exc_info:
        backend.extract_topology(bundle)

    assert "router.per_expert_scale" in str(exc_info.value)


def test_gemma4_backend_validate_bundle_rejects_missing_dense_hybrid_keys() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0)
    del state_dict["model.language_model.layers.0.mlp.down_proj.weight"]
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(TopologyMismatchError, match="missing Gemma4 hybrid layer tensor keys") as exc_info:
        backend.validate_bundle(bundle)

    assert "mlp.down_proj.weight" in str(exc_info.value)


def test_gemma4_backend_detects_unexpected_moe_layer_tensor_prefixes() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[1, 3])
    state_dict: dict[str, FakeTensor] = {}
    for layer_index in (1, 2, 3):
        state_dict.update(_layer_state(layer_index))
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(TopologyMismatchError, match="Gemma4 MoE layer tensor topology mismatch") as exc_info:
        backend.iter_moe_layer_tensor_keys(bundle)

    assert "unexpected_moe_layers=2" in str(exc_info.value)


def test_gemma4_backend_rejects_resolving_non_moe_layer_keys() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[1, 3])
    state_dict: dict[str, FakeTensor] = {}
    for layer_index in (1, 3):
        state_dict.update(_layer_state(layer_index))
    bundle = _bundle(config=config, state_dict=state_dict)

    with pytest.raises(TopologyMismatchError, match="requested Gemma4 layer is not configured as MoE") as exc_info:
        backend.resolve_layer_tensor_keys(bundle, layer_index=2)

    assert "moe_layer_indices=1,3" in str(exc_info.value)


def test_gemma4_backend_router_and_expert_state_validate_shapes() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    bundle = _bundle(config=config, state_dict=_layer_state(0))

    layer = backend.extract_topology(bundle)[0]
    router_state = backend.extract_router_state(bundle, layer=layer)
    expert_state = backend.extract_expert_state(bundle, layer=layer)

    assert router_state.projection_shape == (128, 2816)
    assert router_state.per_expert_scale_shape == (128,)
    assert expert_state["gate_up_proj"].shape == (128, 1408, 2816)
    backend.validate_layer(bundle, layer=layer, router_state=router_state)


def test_gemma4_backend_resolves_runtime_router_module_from_topology() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                layers=[SimpleNamespace(router=SimpleNamespace(register_forward_hook=lambda hook: None))]
            )
        )
    )
    bundle = LoadedBackendBundle(
        backend_name="gemma4",
        model_handle=ModelHandle(model_id="google/gemma-4-27b", revision="rev-123", backend_name="gemma4"),
        model=model,
        config=config,
        metadata={"state_dict": _layer_state(0), "backend_version": "1.0.0"},
    )

    layer = backend.extract_topology(bundle)[0]
    resolved = backend.resolve_router_module(bundle, layer=layer)

    assert resolved is model.model.language_model.layers[0].router


def test_gemma4_backend_expert_shape_mismatch_raises_shape_invariant_error() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0)
    state_dict["model.language_model.layers.0.experts.down_proj"] = FakeTensor((127, 2816, 704))
    bundle = _bundle(config=config, state_dict=state_dict)
    layer = backend.extract_topology(bundle)[0]

    with pytest.raises(ShapeInvariantViolationError, match="experts.down_proj shape mismatch") as exc_info:
        backend.extract_expert_state(bundle, layer=layer)

    assert "expected_shape=128x2816x704" in str(exc_info.value)
    assert "actual_shape=127x2816x704" in str(exc_info.value)


def test_gemma4_backend_rejects_wrong_moe_intermediate_size() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0)
    state_dict["model.language_model.layers.0.experts.gate_up_proj"] = FakeTensor((128, 1998, 2816))
    state_dict["model.language_model.layers.0.experts.down_proj"] = FakeTensor((128, 2816, 999))
    bundle = _bundle(config=config, state_dict=state_dict)
    layer = backend.extract_topology(bundle)[0]

    with pytest.raises(ShapeInvariantViolationError, match="experts.gate_up_proj shape mismatch") as exc_info:
        backend.extract_expert_state(bundle, layer=layer)

    message = str(exc_info.value)
    assert "expected_shape=128x1408x2816" in message
    assert "actual_shape=128x1998x2816" in message
    assert "expected_layout=(num_experts, 2 * moe_intermediate_size, hidden_size)" in message


def test_gemma4_backend_rejects_unexpected_expert_tensor_rank() -> None:
    backend = Gemma4Backend()
    config = _gemma4_config(moe_layer_indices=[0])
    state_dict = _layer_state(0)
    state_dict["model.language_model.layers.0.experts.down_proj"] = FakeTensor((128, 2816))
    bundle = _bundle(config=config, state_dict=state_dict)
    layer = backend.extract_topology(bundle)[0]

    with pytest.raises(ShapeInvariantViolationError, match="experts.down_proj rank must be 3") as exc_info:
        backend.extract_expert_state(bundle, layer=layer)

    message = str(exc_info.value)
    assert "expected_shape=128x2816x704" in message
    assert "actual_shape=128x2816" in message


def test_gemma4_backend_load_populates_model_handle_metadata_with_monkeypatched_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Gemma4Backend()
    signature = BackendSignature.from_mapping(_gemma4_config())

    class FakeConfig:
        def to_dict(self) -> dict[str, object]:
            return _gemma4_config()

    class FakeModel:
        config = FakeConfig()
        dtype = "torch.bfloat16"

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, source: str, revision: str | None = None, torch_dtype: object | None = None) -> FakeModel:
            assert source == "google/gemma-4-27b"
            assert revision == "rev-123"
            assert torch_dtype == "fake-bfloat16"
            return FakeModel()

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, source: str, revision: str | None = None) -> object:
            assert source == "google/gemma-4-27b"
            assert revision == "rev-123"
            return object()

    def fake_import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(
                Gemma4ForConditionalGeneration=FakeModelClass,
                AutoTokenizer=FakeTokenizerClass,
            )
        if name == "torch":
            return SimpleNamespace(bfloat16="fake-bfloat16")
        raise AssertionError(name)

    monkeypatch.setattr("moe_surgeon.models.gemma4.import_module", fake_import_module)
    monkeypatch.setattr(
        Gemma4Backend,
        "_installed_version",
        lambda self, package_name: {"transformers": "4.60.0", "torch": "2.5.1"}.get(package_name),
    )

    bundle = backend.load(signature, dtype="bfloat16", seed=7)

    assert bundle.model_handle.revision == "rev-123"
    assert bundle.model_handle.framework_version == "4.60.0"
    assert bundle.model_handle.dtype == "torch.bfloat16"
    assert bundle.model_handle.metadata["backend_version"] == backend.backend_version
    assert bundle.model_handle.metadata["torch_dtype"] == "torch.bfloat16"
    assert bundle.metadata["backend_version"] == backend.backend_version


def test_gemma4_backend_load_raises_actionable_error_when_runtime_support_is_missing() -> None:
    backend = Gemma4Backend()
    signature = BackendSignature.from_mapping(_gemma4_config())

    with pytest.raises(UnsupportedModelError, match="unsupported model family") as exc_info:
        backend.load(signature)

    message = str(exc_info.value)
    assert "installed_transformers_version=4.51.3" in message
    assert "required_symbol=Gemma4ForConditionalGeneration" in message
    assert "support_added_on=2026-04-01" in message


def test_default_registry_resolves_gemma4_backend_from_mapping_and_signature() -> None:
    mapping = _gemma4_config()
    signature = BackendSignature.from_mapping(mapping)
    model_type_only = BackendSignature(model_id="google/gemma-4-27b", model_type="gemma4")

    registry = build_backend_registry()

    assert registry.names() == ("gemma4",)
    assert registry.resolve(mapping).name == "gemma4"
    assert resolve_backend(signature).name == "gemma4"
    assert resolve_backend(model_type_only).name == "gemma4"
