from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from moe_surgeon.analysis.scan import load_local_scan_bundle, scan_model
from moe_surgeon.models.backend import BackendSignature, LoadedBackendBundle, build_backend_registry
from moe_surgeon.models.errors import ShapeInvariantViolationError, TopologyMismatchError
from moe_surgeon.models.gguf import (
    QWEN35MOE_REGISTRY_PRIORITY,
    GgufTensorMetadata,
    Qwen35MoeGgufBackend,
    open_local_gguf_checkpoint,
    qwen35moe_registry_entry,
)
from moe_surgeon.prune.gguf import inspect_gguf, prune_gguf_static
from moe_surgeon.schemas import ModelHandle


def _tensor(name: str, shape: tuple[int, ...], *, tensor_type: str = "F32") -> GgufTensorMetadata:
    return GgufTensorMetadata(
        tensor_key=name,
        shape=shape,
        data_shape=tuple(reversed(shape)) if len(shape) == 2 else shape,
        dtype="float32",
        tensor_type=tensor_type,
        n_bytes=1,
        data_offset=0,
    )


def _qwen_config() -> dict[str, object]:
    return {
        "general.architecture": "qwen35moe",
        "general.name": "Qwen3.6-35B-A3B",
        "qwen35moe.block_count": 2,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.expert_feed_forward_length": 512,
        "qwen35moe.expert_shared_feed_forward_length": 512,
        "qwen35moe.full_attention_interval": 4,
    }


def _layer_tensors(layer_index: int) -> dict[str, GgufTensorMetadata]:
    prefix = f"blk.{layer_index}"
    return {
        f"{prefix}.ffn_gate_inp.weight": GgufTensorMetadata(
            tensor_key=f"{prefix}.ffn_gate_inp.weight",
            shape=(2048, 256),
            data_shape=(256, 2048),
            dtype="float32",
            tensor_type="F32",
            n_bytes=1,
            data_offset=0,
        ),
        f"{prefix}.ffn_gate_inp_shexp.weight": _tensor(
            f"{prefix}.ffn_gate_inp_shexp.weight",
            (2048,),
        ),
        f"{prefix}.ffn_gate_exps.weight": _tensor(
            f"{prefix}.ffn_gate_exps.weight",
            (2048, 512, 256),
            tensor_type="Q4_K",
        ),
        f"{prefix}.ffn_up_exps.weight": _tensor(
            f"{prefix}.ffn_up_exps.weight",
            (2048, 512, 256),
            tensor_type="Q4_K",
        ),
        f"{prefix}.ffn_down_exps.weight": _tensor(
            f"{prefix}.ffn_down_exps.weight",
            (512, 2048, 256),
            tensor_type="Q5_K",
        ),
        f"{prefix}.ffn_gate_shexp.weight": _tensor(
            f"{prefix}.ffn_gate_shexp.weight",
            (2048, 512),
            tensor_type="Q8_0",
        ),
        f"{prefix}.ffn_up_shexp.weight": _tensor(
            f"{prefix}.ffn_up_shexp.weight",
            (2048, 512),
            tensor_type="Q8_0",
        ),
        f"{prefix}.ffn_down_shexp.weight": _tensor(
            f"{prefix}.ffn_down_shexp.weight",
            (512, 2048),
            tensor_type="Q8_0",
        ),
    }


def _bundle(state: dict[str, GgufTensorMetadata] | None = None) -> LoadedBackendBundle:
    tensors = _state() if state is None else state
    return LoadedBackendBundle(
        backend_name="qwen35moe-gguf",
        model_handle=ModelHandle(
            model_id="Qwen3.6-35B-A3B",
            backend_name="qwen35moe-gguf",
            framework_version="gguf",
        ),
        model=object(),
        config=_qwen_config(),
        metadata={"state_dict": tensors, "backend_version": "0.1.0"},
    )


def _state() -> dict[str, GgufTensorMetadata]:
    tensors: dict[str, GgufTensorMetadata] = {}
    for layer_index in range(2):
        tensors.update(_layer_tensors(layer_index))
    return tensors


def _write_tiny_qwen35moe_gguf(path: Path) -> Path:
    writer = GGUFWriter(path, arch="qwen35moe")
    writer.add_uint32("qwen35moe.block_count", 1)
    writer.add_uint32("qwen35moe.context_length", 32)
    writer.add_uint32("qwen35moe.embedding_length", 5)
    writer.add_uint32("qwen35moe.expert_count", 4)
    writer.add_uint32("qwen35moe.expert_used_count", 2)
    writer.add_uint32("qwen35moe.expert_feed_forward_length", 3)
    writer.add_uint32("qwen35moe.expert_shared_feed_forward_length", 3)
    writer.add_uint32("qwen35moe.full_attention_interval", 4)
    writer.add_tensor(
        "blk.0.ffn_gate_inp.weight",
        np.arange(20, dtype=np.float32).reshape(4, 5),
    )
    writer.add_tensor("blk.0.ffn_gate_inp_shexp.weight", np.ones((5,), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate_exps.weight", np.ones((4, 3, 5), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_up_exps.weight", np.ones((4, 3, 5), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down_exps.weight", np.ones((4, 5, 3), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_gate_shexp.weight", np.ones((3, 5), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_up_shexp.weight", np.ones((3, 5), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down_shexp.weight", np.ones((5, 3), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_qwen35moe_gguf_backend_supports_architecture_signature() -> None:
    backend = Qwen35MoeGgufBackend()

    assert backend.supports(
        BackendSignature(
            model_id="qwen",
            architecture="qwen35moe",
            model_type="qwen35moe",
            metadata={"format": "gguf"},
        )
    )
    assert not backend.supports(BackendSignature(model_id="qwen", architecture="qwen35moe"))
    assert not backend.supports(
        BackendSignature(model_id="gemma", architecture="gemma4", metadata={"format": "gguf"})
    )


def test_qwen35moe_gguf_default_registry_entry_uses_canonical_priority() -> None:
    backend, priority = qwen35moe_registry_entry()

    assert backend.name == "qwen35moe-gguf"
    assert priority == QWEN35MOE_REGISTRY_PRIORITY


def test_default_registry_resolves_qwen35moe_gguf_backend() -> None:
    registry = build_backend_registry()

    assert "qwen35moe-gguf" in registry.names()
    resolved = registry.resolve(
        BackendSignature(
            model_id="qwen",
            architecture="qwen35moe",
            model_type="qwen35moe",
            metadata={"format": "gguf"},
        )
    )

    assert resolved.name == "qwen35moe-gguf"


def test_qwen35moe_gguf_extracts_hybrid_topology_metadata() -> None:
    backend = Qwen35MoeGgufBackend()
    layers = backend.extract_topology(_bundle())

    assert len(layers) == 2
    assert layers[0].expert_count == 256
    assert layers[0].top_k == 8
    assert layers[0].hidden_size == 2048
    assert layers[0].moe_intermediate_size == 512
    assert layers[0].metadata["attention_type"] == "linear_attention"
    assert layers[0].metadata["has_ssm"] is True
    assert layers[0].module_paths["router_proj"] == "blk.0.ffn_gate_inp.weight"
    assert layers[0].module_paths["shared_expert_gate_proj"] == "blk.0.ffn_gate_shexp.weight"


def test_qwen35moe_gguf_router_and_expert_state_validate_real_shapes() -> None:
    backend = Qwen35MoeGgufBackend()
    bundle = _bundle()
    layer = backend.extract_topology(bundle)[0]

    router_state = backend.extract_router_state(bundle, layer=layer)
    expert_state = backend.extract_expert_state(bundle, layer=layer)

    assert router_state.projection_shape == (256, 2048)
    assert router_state.per_expert_scale_shape is None
    assert expert_state["experts_gate_proj"].shape == (2048, 512, 256)
    assert expert_state["shared_expert_down_proj"].shape == (512, 2048)
    backend.validate_layer(bundle, layer=layer, router_state=router_state)


def test_qwen35moe_gguf_rejects_missing_required_tensor() -> None:
    backend = Qwen35MoeGgufBackend()
    state = _state()
    del state["blk.0.ffn_up_exps.weight"]
    bundle = _bundle(state=state)
    layer = backend.extract_topology(bundle)[0]

    with pytest.raises(TopologyMismatchError, match="GGUF layer tensor is missing") as exc_info:
        backend.validate_layer(bundle, layer=layer)

    assert "blk.0.ffn_up_exps.weight" in str(exc_info.value)


def test_qwen35moe_gguf_rejects_expert_shape_mismatch() -> None:
    backend = Qwen35MoeGgufBackend()
    state = _state()
    state["blk.0.ffn_gate_exps.weight"] = _tensor(
        "blk.0.ffn_gate_exps.weight",
        (2048, 513, 256),
        tensor_type="Q4_K",
    )
    bundle = _bundle(state=state)
    layer = backend.extract_topology(bundle)[0]

    with pytest.raises(ShapeInvariantViolationError, match="Qwen3.5-MoE GGUF tensor shape mismatch") as exc_info:
        backend.validate_layer(bundle, layer=layer)

    assert "tensor_role=experts_gate_proj" in str(exc_info.value)


def test_qwen35moe_gguf_scan_uses_router_tensor_without_per_expert_scale(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_qwen35moe_gguf(tmp_path / "tiny-qwen.gguf")
    bundle, backend = load_local_scan_bundle(checkpoint_path)

    result = scan_model(bundle, backend=backend)

    assert result.manifest.model_handle is not None
    assert result.manifest.model_handle.backend_name == "qwen35moe-gguf"
    assert len(result.layers) == 1
    assert len(result.expert_stats) == 4
    assert result.router_states[0].per_expert_scale_shape is None


def test_prune_gguf_static_writes_qwen35moe_expert_axis_tensors(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_qwen35moe_gguf(tmp_path / "tiny-qwen.gguf")
    bundle, backend = load_local_scan_bundle(checkpoint_path)
    scan_result = scan_model(bundle, backend=backend)

    output_path = tmp_path / "tiny-qwen-pruned.gguf"
    result = prune_gguf_static(
        checkpoint_path,
        scan_result=scan_result,
        target_experts=3,
        output_path=output_path,
    )
    pruned = open_local_gguf_checkpoint(output_path)

    assert result.dry_run is False
    assert result.rewritten_tensor_count == 4
    assert result.copied_tensor_count == len(open_local_gguf_checkpoint(checkpoint_path).tensors) - 4
    assert pruned.fields["qwen35moe.expert_count"] == 3
    assert pruned.tensors["blk.0.ffn_gate_inp.weight"].data_shape == (3, 5)
    assert pruned.tensors["blk.0.ffn_gate_exps.weight"].shape[-1] == 3
    assert pruned.tensors["blk.0.ffn_up_exps.weight"].shape[-1] == 3
    assert pruned.tensors["blk.0.ffn_down_exps.weight"].shape[-1] == 3
    assert pruned.tensors["blk.0.ffn_gate_inp_shexp.weight"].shape == (5,)
    assert pruned.tensors["blk.0.ffn_gate_shexp.weight"].shape == (5, 3)
    assert (tmp_path / "tiny-qwen-pruned.gguf.manifest.json").is_file()


def test_inspect_gguf_returns_qwen35moe_inventory(tmp_path: Path) -> None:
    checkpoint_path = _write_tiny_qwen35moe_gguf(tmp_path / "tiny-qwen.gguf")

    result = inspect_gguf(checkpoint_path)

    assert result.architecture == "qwen35moe"
    assert result.expert_count == 4
    assert result.top_k == 2
    assert result.block_count == 1
    assert result.hidden_size == 5
